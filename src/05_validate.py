"""05. AI 분류 검증: 분석자가 직접 분류한 검증 라벨(층화 표본 200건)과 비교.

프롬프트 확정·전체 분류 완료 후에만 실행한다 (D13).

입력
  data/labels/labels_*.xlsx       분석자 검증 라벨 (첫 파일이 주 라벨, 2개 이상이면 라벨러 간 kappa)
  data/labels/sample_ids.csv      번호 ↔ 행 id, 층 정보
  data/processed/classified.parquet  AI 최종 분류 (Haiku → low만 Sonnet)
  outputs/llm_cache/consistency_*_v1_2.csv  같은 단위를 다시 분류한 결과 (일관성 검사)
출력 (outputs/validation/, 커밋 대상)
  validation_report.md, summary.csv, confusion_matrix.csv, per_category.csv, by_group.csv,
  examples.csv (그림 6 분류 예시), disagreements.csv (불일치 목록, 분석자 메모·AI 이유 포함)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
CLASSIFIED = ROOT / "data" / "processed" / "classified.parquet"
CACHE = ROOT / "outputs" / "llm_cache"
OUT = ROOT / "outputs" / "validation"
CATS = list("ABCDEFGH")
GROUP = {"A": "자동차(A·B)", "B": "자동차(A·B)", "C": "사람 중심(C·D·E)", "D": "사람 중심(C·D·E)",
         "E": "사람 중심(C·D·E)", "F": "기타 교통(F·G)", "G": "기타 교통(F·G)", "H": "범위 밖(H)"}


def read_labels(path: Path) -> pd.DataFrame:
    ws = openpyxl.load_workbook(path, read_only=True)["라벨링"]
    rows = [r[:7] for r in ws.iter_rows(min_row=3, values_only=True) if r[0] is not None]
    d = pd.DataFrame(rows, columns=["번호", "사업명", "분야", "부문", "범주", "확신도", "메모"])
    d["번호"] = d["번호"].astype(int)
    d["범주"] = d["범주"].astype(str).str.strip().str.upper()
    d["확신도"] = d["확신도"].astype(str).str.strip().str.lower()
    return d


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return m - c, m + c


def kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def consistency_categories() -> pd.Series:
    """일관성 검사의 최종 범주 (Haiku 재분류 → low는 Sonnet 재분류로 덮어씀)."""
    h = pd.read_csv(CACHE / "consistency_haiku_v1_2.csv", dtype=str).drop_duplicates("id", keep="last")
    final = h.set_index("id")["category"]
    sp = CACHE / "consistency_sonnet_v1_2.csv"
    if sp.exists():
        s = pd.read_csv(sp, dtype=str).drop_duplicates("id", keep="last")
        final.update(s.set_index("id")["category"])
    return final


def main() -> None:
    files = sorted(LABEL_DIR.glob("labels_*.xlsx"))
    if not files:
        raise SystemExit("검증 라벨 파일(data/labels/labels_*.xlsx)이 없습니다")
    labels = [read_labels(f) for f in files]
    sample = pd.read_csv(LABEL_DIR / "sample_ids.csv")
    rows = pd.read_parquet(CLASSIFIED, columns=["id", "단위id", "지출액", "category", "confidence", "reason",
                                                "category_haiku", "confidence_haiku", "model"])
    d = sample.merge(labels[0][["번호", "범주", "확신도", "메모"]], on="번호").merge(rows, on="id", how="left")
    d = d.rename(columns={"범주": "분석자", "확신도": "분석자_확신도", "category": "AI", "confidence": "AI_확신도"})
    if d["AI"].isna().any():
        raise SystemExit("AI 분류가 없는 표본이 있습니다. 04를 먼저 완료하세요")
    d["일치"] = d["분석자"] == d["AI"]
    d["가중치"] = d["층 모집단 행 수"] / d["층 표본 수"]
    n, k = len(d), int(d["일치"].sum())
    lo, hi = wilson(k, n)

    summary = {
        "표본 수": n,
        "정확도(단순)": d["일치"].mean(),
        "정확도 95% 신뢰구간 하한": lo, "정확도 95% 신뢰구간 상한": hi,
        "정확도(층별 모집단 가중)": np.average(d["일치"], weights=d["가중치"]),
        "정확도(지출액 가중)": np.average(d["일치"], weights=d["지출액"].clip(lower=0) + 1),
        "정확도(Haiku 1단계만)": (d["분석자"] == d["category_haiku"]).mean(),
        "Cohen kappa(분석자 vs AI)": kappa(d["분석자"], d["AI"]),
        "지표 묶음 일치율(사람 중심/자동차/기타 교통/범위 밖)": (d["분석자"].map(GROUP) == d["AI"].map(GROUP)).mean(),
    }
    if len(labels) > 1:
        m = labels[0].merge(labels[1], on="번호", suffixes=("_1", "_2"))
        summary["라벨러 간 kappa"] = kappa(m["범주_1"], m["범주_2"])
    cons = consistency_categories() if (CACHE / "consistency_haiku_v1_2.csv").exists() else None
    if cons is not None:
        again = d["단위id"].map(cons)
        ok = again.notna()
        summary["일관성: 재분류 일치율"] = (again[ok] == d.loc[ok, "AI"]).mean()
        summary["일관성: 재분류 kappa"] = kappa(again[ok], d.loc[ok, "AI"])
        summary["일관성: 재분류 후 분석자 대비 정확도"] = (again[ok] == d.loc[ok, "분석자"]).mean()

    cm = pd.crosstab(d["분석자"], d["AI"]).reindex(index=CATS, columns=CATS, fill_value=0)
    per = []
    for c in CATS:
        tp = cm.loc[c, c]
        pred, true = cm[c].sum(), cm.loc[c].sum()
        prec = tp / pred if pred else np.nan
        rec = tp / true if true else np.nan
        f1 = 2 * prec * rec / (prec + rec) if pred and true and (prec + rec) else np.nan
        per.append({"범주": c, "분석자 건수": int(true), "AI 건수": int(pred), "정밀도": prec, "재현율": rec, "F1": f1})
    per = pd.DataFrame(per)

    groups = []
    for col, name in (("연도", "연도"), ("추출단계", "추출단계"), ("AI_확신도", "AI 확신도"), ("분석자_확신도", "분석자 확신도")):
        for v, g in d.groupby(col):
            groups.append({"구분": name, "값": v, "표본 수": len(g), "정확도": g["일치"].mean()})
    groups = pd.DataFrame(groups)

    dis = d[~d["일치"]][["번호", "연도", "세부사업명", "분야", "부문", "분석자", "분석자_확신도", "메모", "AI", "AI_확신도",
                       "reason", "model"]].rename(columns={"reason": "AI 이유"})
    # 그림 6 분류 예시: 범주마다 일치 사례 하나(번호 순), 불일치 사례 둘
    ex = [d[d["일치"] & (d["분석자"] == c)].sort_values("번호").head(1) for c in CATS]
    ex.append(d[~d["일치"]].sort_values("번호").head(2))
    ex = pd.concat(ex)
    ex = pd.DataFrame({"사업명": ex["세부사업명"].str.slice(0, 26), "분석자": ex["분석자"], "AI": ex["AI"],
                       "일치": ex["일치"].map({True: "O", False: "X"})})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.Series(summary, name="값").rename_axis("항목").to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    cm.to_csv(OUT / "confusion_matrix.csv", encoding="utf-8-sig")
    per.to_csv(OUT / "per_category.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT / "by_group.csv", index=False, encoding="utf-8-sig")
    dis.to_csv(OUT / "disagreements.csv", index=False, encoding="utf-8-sig")
    ex.to_csv(OUT / "examples.csv", index=False, encoding="utf-8-sig")

    def f(v, key: str = ""):
        if "kappa" in key:
            return f"{v:.3f}"
        return f"{v:.1%}" if isinstance(v, float) and 0 <= v <= 1 else (f"{v:.3f}" if isinstance(v, float) else str(v))

    L = ["# 05 AI 분류 검증", "", "자동 생성: src/05_validate.py. 기준 v1.2, 분석자가 직접 분류한 검증 라벨 "
         f"{n}건(층화 표본, {', '.join(f.name for f in files)})과 AI 최종 분류 비교.", "",
         "## 요약", "", "| 항목 | 값 |", "| --- | ---: |"]
    L += [f"| {k} | {f(v, k)} |" for k, v in summary.items()]
    L += ["", "- 층별 모집단 가중: 층 가중치 = 층 모집단 행 수 / 층 표본 수 (D19 표본 설계 보정)",
          "- 지출액 가중: 표본 행의 지출액으로 가중 (지표가 금액 기준이므로 참고)",
          "- 지표 묶음 일치율: 범주가 달라도 지표 계산상 같은 묶음이면 일치로 봄", "",
          "## 범주별 정밀도·재현율", "", "| 범주 | 분석자 | AI | 정밀도 | 재현율 | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    L += [f"| {r.범주} | {r._2} | {r._3} | {f(r.정밀도)} | {f(r.재현율)} | {f(r.F1)} |" for r in per.itertuples()]
    L += ["", "## 혼동행렬 (행: 분석자, 열: AI)", "", "| | " + " | ".join(CATS) + " |", "| --- |" + " ---: |" * 8]
    L += [f"| {c} | " + " | ".join(str(cm.loc[c, x]) for x in CATS) + " |" for c in CATS]
    L += ["", "## 구분별 정확도", "", "| 구분 | 값 | 표본 수 | 정확도 |", "| --- | --- | ---: | ---: |"]
    L += [f"| {r.구분} | {r.값} | {r._3} | {f(r.정확도)} |" for r in groups.itertuples()]
    L += ["", f"불일치 {len(dis)}건 목록: outputs/validation/disagreements.csv (분석자 메모·AI 이유 포함)", ""]
    (OUT / "validation_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[5:5 + 4 + len(summary)]))


if __name__ == "__main__":
    main()
