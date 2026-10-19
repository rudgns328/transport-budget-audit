"""06. 지자체별 교통예산 지표 (CLAUDE.md 6장).

입력: data/processed/classified.parquet (04), data/processed/gov_aux.parquet (04b)
출력 (outputs/indicators/, 커밋 대상)
  gov_indicators.csv     연도 × (지역, 자치단체)별 범주 금액과 지표 (지출액·예산현액 기준)
  type_summary.csv       연도 × 유형별 분포 (중앙값·사분위)
  execution_rate.csv     광역·기초 × 범주별 집행률
  national_own_source.csv  자체 재원 기준 전국 합계 (이중 계상 방지)
  robustness.csv         결론 확인용 비교 (예산현액 기준, 2024·2025 평균, 출자 민감도)
  outputs/results_summary.md  핵심 수치와 출처(파일·열·계산식)

원칙: 광역 본청과 기초는 섞지 않는다. 비교는 같은 유형 안에서만, 지자체 순위는 만들지 않는다.
지표는 진단이며 인과를 뜻하지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIED = ROOT / "data" / "processed" / "classified.parquet"
AUX = ROOT / "data" / "processed" / "gov_aux.parquet"
OUT = ROOT / "outputs" / "indicators"
SUMMARY = ROOT / "outputs" / "results_summary.md"

CATS = list("ABCDEFGH")
TRANSPORT = list("ABCDEFG")          # 지표 분모 (H 제외)
PEOPLE = list("CDE")                 # 사람 중심
TYPE_ORDER = ["광역", "세종", "제주", "자치구", "시", "군"]
BASE_YEAR, COMPARE_YEAR, AUX_YEAR = 2025, 2019, 2024
EQUITY_ITEM = "서울교통공사 출자 지원"  # D25: 모델이 H로 본 출자금. C로 넣은 경우 민감도 확인


# ---------- 준비 ----------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    c = pd.read_parquet(CLASSIFIED)
    if c["category"].isna().any():
        raise SystemExit("미분류 행이 있습니다. 04를 먼저 완료하세요")
    # 2019 군위군은 경북 소속 → 2023 대구 편입 후 기준으로 맞춘다 (D16)
    gunwi = (c["자치단체"] == "군위군") & (c["지역"] == "경북")
    c.loc[gunwi, "지역"] = "대구"
    aux = pd.read_parquet(AUX)
    return c, aux


def amounts(c: pd.DataFrame, value: str) -> pd.DataFrame:
    """(연도, 지역, 자치단체) × 범주 금액."""
    p = c.pivot_table(index=["연도", "지역", "자치단체"], columns="category", values=value,
                      aggfunc="sum", fill_value=0).reindex(columns=CATS, fill_value=0)
    return p


def ratios(p: pd.DataFrame) -> pd.DataFrame:
    ag = p[TRANSPORT].sum(axis=1)
    out = pd.DataFrame(index=p.index)
    out["교통예산_AG"] = ag
    out["사람중심비율"] = p[PEOPLE].sum(axis=1) / ag
    out["자동차확대비율"] = p["A"] / ag
    for k in TRANSPORT:
        out[f"비중_{k}"] = p[k] / ag
    return out


def build_gov_table(c: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    spend, budget = amounts(c, "지출액"), amounts(c, "예산현액")
    r_s, r_b = ratios(spend), ratios(budget)
    t = spend.add_prefix("지출_").join(budget.add_prefix("예산_"))
    t = t.join(r_s.add_suffix("_지출")).join(r_b[["사람중심비율", "자동차확대비율"]].add_suffix("_예산"))
    t = t.reset_index()

    # 판단 불가 비중: H이면서 low인 행의 건수·지출액이 후보 전체에서 차지하는 비중
    c = c.assign(불가=(c["category"] == "H") & (c["confidence"] == "low"))
    g = c.groupby(["연도", "지역", "자치단체"])
    unk = pd.DataFrame({"판단불가_건수비중": g["불가"].mean(),
                        "판단불가_금액비중": g.apply(lambda d: d.loc[d["불가"], "지출액"].sum() / d["지출액"].sum()
                                             if d["지출액"].sum() else np.nan, include_groups=False)}).reset_index()
    t = t.merge(unk, on=["연도", "지역", "자치단체"], how="left")
    t = t.merge(aux, on=["지역", "자치단체"], how="left")
    t["1인당_보행자전거_원"] = t["지출_D"] / t["인구"]  # 인구는 2025.12 기준 (모든 연도 공통 분모)
    return t


# ---------- 요약 ----------

def q(s: pd.Series, p: float) -> float:
    return float(s.quantile(p)) if len(s.dropna()) else np.nan


def type_summary(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (y, typ), d in t.groupby(["연도", "유형"]):
        for col in ["사람중심비율_지출", "자동차확대비율_지출", "사람중심비율_예산", "비중_D_지출", "비중_E_지출",
                    "1인당_보행자전거_원", "판단불가_금액비중"]:
            s = d[col]
            rows.append({"연도": y, "유형": typ, "지표": col, "자치단체수": int(s.notna().sum()),
                         "중앙값": q(s, .5), "Q1": q(s, .25), "Q3": q(s, .75), "최소": s.min(), "최대": s.max()})
    out = pd.DataFrame(rows)
    out["유형"] = pd.Categorical(out["유형"], TYPE_ORDER, ordered=True)
    return out.sort_values(["지표", "연도", "유형"])


def pooled(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """같은 층위 안에서 금액을 합쳐 구한 비율 (광역과 기초는 섞지 않음)."""
    g = t.groupby(by)[[f"지출_{k}" for k in CATS]].sum()
    ag = g[[f"지출_{k}" for k in TRANSPORT]].sum(axis=1)
    return pd.DataFrame({"사람중심비율": g[[f"지출_{k}" for k in PEOPLE]].sum(axis=1) / ag,
                         "자동차확대비율": g["지출_A"] / ag, "교통예산_AG_억": ag / 1e8,
                         "H_억": g["지출_H"] / 1e8})


def execution_rate(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (y, lvl), d in c.groupby(["연도", "광역기초"]):
        for k in CATS:
            dk = d[d["category"] == k]
            rows.append({"연도": y, "광역기초": lvl, "범주": k, "예산현액_억": dk["예산현액"].sum() / 1e8,
                         "지출액_억": dk["지출액"].sum() / 1e8,
                         "집행률": dk["지출액"].sum() / dk["예산현액"].sum() if dk["예산현액"].sum() else np.nan})
        for name, ks in (("사람중심(C+D+E)", PEOPLE), ("자동차(A+B)", ["A", "B"])):
            dk = d[d["category"].isin(ks)]
            rows.append({"연도": y, "광역기초": lvl, "범주": name, "예산현액_억": dk["예산현액"].sum() / 1e8,
                         "지출액_억": dk["지출액"].sum() / 1e8, "집행률": dk["지출액"].sum() / dk["예산현액"].sum()})
    return pd.DataFrame(rows)


def national_own_source(c: pd.DataFrame) -> pd.DataFrame:
    """자체 재원 기준 전국 합계: 광역 = 시도비, 기초 = 시군구비 (예산현액 재원 구성 기준)."""
    own = np.where(c["광역기초"] == "광역", c["예산_시도비"], c["예산_시군구비"])
    d = c.assign(자체재원=own)
    p = d.pivot_table(index="연도", columns="category", values="자체재원", aggfunc="sum", fill_value=0)
    p = p.reindex(columns=CATS, fill_value=0) / 1e8
    p["A~G"] = p[TRANSPORT].sum(axis=1)
    p["사람중심비율"] = p[PEOPLE].sum(axis=1) / p["A~G"]
    p["자동차확대비율"] = p["A"] / p["A~G"]
    return p.reset_index()


def change(t: pd.DataFrame) -> pd.DataFrame:
    """2019 → 2025: 같은 자치단체의 변화량을 유형별로 요약 (순위는 만들지 않음)."""
    a = t[t["연도"] == COMPARE_YEAR].set_index(["지역", "자치단체"])
    b = t[t["연도"] == BASE_YEAR].set_index(["지역", "자치단체"])
    rows = []
    for col in ["사람중심비율_지출", "자동차확대비율_지출"]:
        diff = (b[col] - a[col]).dropna()
        typ = b.loc[diff.index, "유형"]
        for tp, s in diff.groupby(typ):
            rows.append({"지표": col, "유형": tp, "자치단체수": len(s), "변화_중앙값(%p)": s.median() * 100,
                         "증가한_자치단체_비율": (s > 0).mean(),
                         f"{COMPARE_YEAR}_중앙값": a.loc[s.index, col].median(), f"{BASE_YEAR}_중앙값": b.loc[s.index, col].median()})
    out = pd.DataFrame(rows)
    out["유형"] = pd.Categorical(out["유형"], TYPE_ORDER, ordered=True)
    return out.sort_values(["지표", "유형"])


def corr(x: pd.Series, y: pd.Series) -> float:
    """자치단체가 3곳 미만인 유형(세종·제주)은 상관을 계산하지 않는다."""
    return x.corr(y) if len(x) >= 3 else np.nan


def robustness(c: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = t[t["연도"] == BASE_YEAR].set_index(["지역", "자치단체"])
    # ① 예산현액 기준
    for tp, d in base.groupby("유형"):
        rows.append({"검사": "예산현액 기준", "유형": tp, "기준값_중앙값": d["사람중심비율_지출"].median(),
                     "비교값_중앙값": d["사람중심비율_예산"].median(),
                     "상관": corr(d["사람중심비율_지출"], d["사람중심비율_예산"])})
    # ② 2024·2025 2개년 합산 (D18)
    two = c[c["연도"].isin([AUX_YEAR, BASE_YEAR])].assign(연도=BASE_YEAR)
    r2 = ratios(amounts(two, "지출액")).reset_index().set_index(["지역", "자치단체"])
    for tp, d in base.groupby("유형"):
        rows.append({"검사": "2024·2025 합산", "유형": tp, "기준값_중앙값": d["사람중심비율_지출"].median(),
                     "비교값_중앙값": r2.loc[d.index, "사람중심비율"].median(),
                     "상관": corr(d["사람중심비율_지출"], r2.loc[d.index, "사람중심비율"])})
    # ③ 출자금 민감도 (D25): 해당 사업을 C로 넣었을 때
    hit = c[(c["연도"] == BASE_YEAR) & (c["세부사업명"] == EQUITY_ITEM)]
    for (rg, nm), d in hit.groupby(["지역", "자치단체"]):
        c2 = c[(c["연도"] == BASE_YEAR) & (c["지역"] == rg) & (c["자치단체"] == nm)].copy()
        c2.loc[c2["세부사업명"] == EQUITY_ITEM, "category"] = "C"
        r = ratios(amounts(c2, "지출액")).iloc[0]
        rows.append({"검사": f"출자금 C 처리 ({rg} {nm})", "유형": base.loc[(rg, nm), "유형"],
                     "기준값_중앙값": base.loc[(rg, nm), "사람중심비율_지출"], "비교값_중앙값": r["사람중심비율"], "상관": np.nan})
    return pd.DataFrame(rows)


def elderly_gap(t: pd.DataFrame) -> pd.DataFrame:
    """기초 자치단체 중 같은 유형 안에서 65세 이상 비율 상위 25%와 나머지의 D·E 비중 비교."""
    b = t[(t["연도"] == BASE_YEAR) & (t["광역기초"] == "기초")].copy()
    b["고령상위"] = b.groupby("유형")["고령비율"].transform(lambda s: s >= s.quantile(.75))
    rows = []
    for (tp, top), d in b.groupby(["유형", "고령상위"]):
        rows.append({"유형": tp, "집단": "고령비율 상위 25%" if top else "나머지", "자치단체수": len(d),
                     "고령비율_중앙값": d["고령비율"].median(), "D비중_중앙값": d["비중_D_지출"].median(),
                     "E비중_중앙값": d["비중_E_지출"].median(), "1인당D_중앙값_원": d["1인당_보행자전거_원"].median()})
    return pd.DataFrame(rows)


# ---------- 보고 ----------

def pct(x: float) -> str:
    return f"{x * 100:.1f}%" if pd.notna(x) else "-"


def write_summary(t, ts, ex, nat, ch, rb, eg) -> str:
    vpath = ROOT / "outputs" / "validation" / "summary.csv"
    if vpath.exists():
        v = pd.read_csv(vpath, index_col=0)["값"].astype(float)
        vnote = (f"> 분류는 AI(기준 v1.2) 결과다. 분석자가 직접 분류한 검증 라벨 200건 대비 정확도 {pct(v['정확도(단순)'])} "
                 f"(95% 신뢰구간 {pct(v['정확도 95% 신뢰구간 하한'])}부터 {pct(v['정확도 95% 신뢰구간 상한'])}, "
                 f"층별 가중 {pct(v['정확도(층별 모집단 가중)'])}, kappa {v['Cohen kappa(분석자 vs AI)']:.3f})"
                 + (f". 외부 협조자 2명 포함 사람 라벨러 3명 사이 kappa {v['사람–사람 kappa 평균']:.3f}, AI–사람 {v['AI–사람 kappa 평균']:.3f}, "
                    f"다수결 기준 정확도 {pct(v['정확도(다수결 기준)'])}" if "사람–사람 kappa 평균" in v.index else "")
                 + ". 출처: outputs/validation/validation_report.md")
    else:
        vnote = "> 분류는 AI(기준 v1.2) 결과이며 검증(05) 전 수치다."
    L = ["# 결과 요약 (자동 생성: src/06_indicators.py)", "", vnote,
         "> 지표는 진단이며 인과를 뜻하지 않는다. 광역 본청과 기초는 섞지 않고, 같은 유형 안에서만 비교한다.", "",
         "## 지표 정의", "",
         "- 사람 중심 교통예산 비율 = (C+D+E) / (A~G 합), 자동차 용량 확대 비율 = A / (A~G 합). 기본 금액은 지출액",
         "- 출처: outputs/indicators/gov_indicators.csv 열 `사람중심비율_지출`, `자동차확대비율_지출` (지출_C+지출_D+지출_E)/교통예산_AG_지출",
         ""]
    L += ["## 1. 유형별 사람 중심 비율 (지출액, 자치단체별 비율의 중앙값 [Q1, Q3])", "",
          "| 유형 | " + " | ".join(str(y) for y in sorted(t["연도"].unique())) + " |",
          "| --- |" + " ---: |" * t["연도"].nunique()]
    s = ts[ts["지표"] == "사람중심비율_지출"]
    for tp in TYPE_ORDER:
        cells = []
        for y in sorted(t["연도"].unique()):
            r = s[(s["유형"] == tp) & (s["연도"] == y)]
            cells.append(f"{pct(r['중앙값'].iloc[0])} [{pct(r['Q1'].iloc[0])}, {pct(r['Q3'].iloc[0])}]" if len(r) else "-")
        L.append(f"| {tp} | " + " | ".join(cells) + " |")
    L += ["", "출처: outputs/indicators/type_summary.csv (지표 = 사람중심비율_지출, 열 중앙값·Q1·Q3)", ""]

    L += ["## 2. 유형별 자동차 용량 확대 비율 (지출액, 중앙값)", "", "| 유형 | " + " | ".join(str(y) for y in sorted(t["연도"].unique())) + " |",
          "| --- |" + " ---: |" * t["연도"].nunique()]
    s = ts[ts["지표"] == "자동차확대비율_지출"]
    for tp in TYPE_ORDER:
        cells = [pct(s[(s["유형"] == tp) & (s["연도"] == y)]["중앙값"].iloc[0]) for y in sorted(t["연도"].unique())]
        L.append(f"| {tp} | " + " | ".join(cells) + " |")
    L += ["", "출처: outputs/indicators/type_summary.csv (지표 = 자동차확대비율_지출)", ""]

    L += [f"## 3. {COMPARE_YEAR} → {BASE_YEAR} 변화 (같은 자치단체의 변화량, 유형별)", "",
          "| 지표 | 유형 | 자치단체 수 | 변화 중앙값 | 증가한 자치단체 비율 |", "| --- | --- | ---: | ---: | ---: |"]
    for _, r in ch.iterrows():
        L.append(f"| {r['지표']} | {r['유형']} | {r['자치단체수']} | {r['변화_중앙값(%p)']:+.1f}%p | {pct(r['증가한_자치단체_비율'])} |")
    L += ["", "출처: outputs/indicators/change_2019_2025.csv (2025 값 − 2019 값, 군위군은 대구 기준으로 연결)", ""]

    L += ["## 4. 범주별 집행률 (지출액 / 예산현액)", "", "| 연도 | 층위 | 사람중심(C+D+E) | 자동차(A+B) | A | B | C | D | E |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (y, lvl), d in ex.groupby(["연도", "광역기초"]):
        v = d.set_index("범주")["집행률"]
        L.append(f"| {y} | {lvl} | {pct(v['사람중심(C+D+E)'])} | {pct(v['자동차(A+B)'])} | "
                 + " | ".join(pct(v[k]) for k in "ABCDE") + " |")
    L += ["", "출처: outputs/indicators/execution_rate.csv (층위 안 범주별 지출액 합 / 예산현액 합)", ""]

    L += ["## 5. 전국 합계 (자체 재원 기준, 이중 계상 방지)", "",
          "| 연도 | A~G 합(억 원) | 사람중심비율 | 자동차확대비율 |", "| --- | ---: | ---: | ---: |"]
    for _, r in nat.iterrows():
        L.append(f"| {int(r['연도'])} | {r['A~G']:,.0f} | {pct(r['사람중심비율'])} | {pct(r['자동차확대비율'])} |")
    L += ["", "출처: outputs/indicators/national_own_source.csv (광역 행은 예산_시도비, 기초 행은 예산_시군구비 합, 예산현액 재원 기준)", ""]

    L += ["## 6. 결론 확인 (강건성)", "", "| 검사 | 유형 | 기준(2025 지출) | 비교 | 상관 |", "| --- | --- | ---: | ---: | ---: |"]
    for r in rb.itertuples():
        L.append(f"| {r.검사} | {r.유형} | {pct(r.기준값_중앙값)} | {pct(r.비교값_중앙값)} | "
                 + (f"{r.상관:.2f}" if pd.notna(r.상관) else "-") + " |")
    L += ["", "출처: outputs/indicators/robustness.csv", ""]

    L += ["## 7. 고령 지역 보조 분석 (2025 기초, 같은 유형 안 65세 이상 비율 상위 25% vs 나머지)", "",
          "| 유형 | 집단 | 자치단체 수 | 고령비율 | D 비중 | E 비중 | 1인당 D(원) |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in eg.iterrows():
        L.append(f"| {r['유형']} | {r['집단']} | {r['자치단체수']} | {pct(r['고령비율_중앙값'])} | {pct(r['D비중_중앙값'])} | "
                 f"{pct(r['E비중_중앙값'])} | {r['1인당D_중앙값_원']:,.0f} |")
    L += ["", "출처: outputs/indicators/elderly_gap.csv (D·E 비중 = 지출_D 또는 지출_E / 교통예산_AG)", ""]

    unk = t[t["연도"] == BASE_YEAR]
    L += ["## 8. 판단 불가 비중 (H 중 확신도 low)", "",
          f"- {BASE_YEAR} 자치단체별 금액 비중 중앙값 {pct(unk['판단불가_금액비중'].median())}, "
          f"최대 {pct(unk['판단불가_금액비중'].max())}",
          "- 출처: gov_indicators.csv 열 `판단불가_금액비중` (H·low 행 지출액 / 후보 전체 지출액)", ""]
    return "\n".join(L)


def main() -> None:
    c, aux = load()
    t = build_gov_table(c, aux)
    ts, ex, nat = type_summary(t), execution_rate(c), national_own_source(c)
    ch, rb, eg = change(t), robustness(c, t), elderly_gap(t)

    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "gov_indicators.csv", index=False, encoding="utf-8-sig")
    ts.to_csv(OUT / "type_summary.csv", index=False, encoding="utf-8-sig")
    ex.to_csv(OUT / "execution_rate.csv", index=False, encoding="utf-8-sig")
    nat.to_csv(OUT / "national_own_source.csv", index=False, encoding="utf-8-sig")
    ch.to_csv(OUT / "change_2019_2025.csv", index=False, encoding="utf-8-sig")
    rb.to_csv(OUT / "robustness.csv", index=False, encoding="utf-8-sig")
    eg.to_csv(OUT / "elderly_gap.csv", index=False, encoding="utf-8-sig")
    SUMMARY.write_text(write_summary(t, ts, ex, nat, ch, rb, eg), encoding="utf-8")
    print(f"자치단체 × 연도 {len(t):,}행 → {OUT.relative_to(ROOT)}, 요약 → {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
