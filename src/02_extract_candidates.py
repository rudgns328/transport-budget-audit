"""02. 교통 관련 후보 사업 2단계 추출.

1단계: 분야가 교통 분야(2024·2025 '교통및물류', 2019 '수송및교통', D15)인 행 전부
2단계: 그 외 분야에서 세부사업명에 키워드가 들어간 행

입력: data/processed/budget_{연도}.parquet
출력:
  data/processed/candidates.parquet      후보 행 (연도 통합, 추출단계·매칭키워드 열 추가)
  outputs/qa/02_extract_report.md        단계별 행 수, 고유 사업명 수, 키워드별 건수

실행: python src/02_extract_candidates.py [--years 2025 2024 2019]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
QA_DIR = ROOT / "outputs" / "qa"

TRANSPORT_FIELD = {2025: "교통및물류", 2024: "교통및물류", 2019: "수송및교통"}
KEYWORDS = ["도로", "교통", "보행", "보도", "횡단", "자전거", "따릉이", "공영자전거", "버스", "정류장", "정류소",
            "환승", "철도", "지하철", "주차", "어린이보호구역", "스쿨존", "노인보호구역", "이동지원",
            "특별교통수단", "바우처택시", "저상", "전기차", "수소차", "충전", "킥보드"]
# 교통 관련으로 보이는 특별회계. 추출 기준이 아니라 누락 점검용으로만 쓴다.
TRANSPORT_ACCOUNT_PATTERN = r"주차|교통|철도"


def matched_keywords(name: str) -> str:
    return "|".join(k for k in KEYWORDS if k in name)


def extract(year: int) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"budget_{year}.parquet")
    field = TRANSPORT_FIELD[year]
    if field not in set(df["분야"]):
        raise ValueError(f"{year}: 교통 분야명 '{field}'이 데이터에 없습니다")

    df["매칭키워드"] = df["세부사업명"].map(matched_keywords)
    stage1 = df["분야"] == field
    stage2 = ~stage1 & (df["매칭키워드"] != "")
    df["추출단계"] = pd.NA
    df.loc[stage1, "추출단계"] = 1
    df.loc[stage2, "추출단계"] = 2

    missed = df[~stage1 & ~stage2 & df["회계"].str.contains(TRANSPORT_ACCOUNT_PATTERN)]
    missed.attrs["year"] = year
    out = df[stage1 | stage2].copy()
    out["추출단계"] = out["추출단계"].astype("int8")
    out.attrs["missed"] = missed
    return out


def md_table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] + [" ---: "] * (len(header) - 1)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:,}" if isinstance(v, (int,)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def build_report(cands: pd.DataFrame, missed: dict[int, pd.DataFrame]) -> str:
    years = sorted(cands["연도"].unique(), reverse=True)
    out = ["# 02 후보 추출 보고", "", "자동 생성: src/02_extract_candidates.py", "",
           "## 연도·단계별 후보", ""]
    rows = []
    for y in years:
        c = cands[cands["연도"] == y]
        for stage in (1, 2):
            s = c[c["추출단계"] == stage]
            rows.append([str(y), stage, len(s), s["세부사업명"].nunique(), f"{s['지출액'].sum() / 1e8:,.0f}"])
        rows.append([str(y), "계", len(c), c["세부사업명"].nunique(), f"{c['지출액'].sum() / 1e8:,.0f}"])
    out += [md_table(rows, ["연도", "단계", "행 수", "고유 사업명", "지출액(억 원)"]), ""]

    # LLM 호출 수 추정: 연도를 합쳐 고유 사업명 기준
    names = {y: set(cands.loc[cands["연도"] == y, "세부사업명"]) for y in years}
    seen, rows = set(), []
    for y in years:
        new = names[y] - seen
        seen |= names[y]
        rows.append([str(y), len(names[y]), len(new), len(seen)])
    out += ["## 고유 사업명 (LLM 호출 수 추정, 정규화 전 문자열 기준)", "",
            md_table(rows, ["연도", "해당 연도 고유명", "앞 연도에 없던 새 이름", "누적 고유명"]), ""]

    s2 = cands[cands["추출단계"] == 2]
    kw = s2["매칭키워드"].str.split("|").explode().value_counts()
    out += ["## 2단계 키워드별 행 수 (한 행에 여러 키워드 가능)", "",
            md_table([[k, int(v)] for k, v in kw.items()], ["키워드", "행 수"]), ""]
    by_field = s2.groupby(["연도", "분야"]).size().reset_index(name="n").sort_values(["연도", "n"], ascending=[False, False])
    by_field["연도"] = by_field["연도"].astype(str)
    out += ["## 2단계 후보의 원래 분야", "",
            md_table(by_field.values.tolist(), ["연도", "분야", "행 수"]), ""]
    by_sector = cands[cands["추출단계"] == 1].groupby(["연도", "부문"]).size().reset_index(name="n").sort_values(
        ["연도", "n"], ascending=[False, False])
    by_sector["연도"] = by_sector["연도"].astype(str)
    out += ["## 1단계 후보의 부문", "", md_table(by_sector.values.tolist(), ["연도", "부문", "행 수"]), ""]

    # 범위 밖 가능성이 큰 키워드 매칭 예시 (제거하지 않고 LLM이 H로 분류하도록 둔다)
    probes = {"도로명주소": "도로명주소", "버스킹": "버스킹", "언론보도·보도자료": "언론보도|보도자료", "교통비(수당·지원금)": "교통비"}
    rows = [[label, int(s2["세부사업명"].str.contains(p).sum())] for label, p in probes.items()]
    out += ["## 2단계 오탐 가능 패턴 (참고, 제거하지 않음)", "", md_table(rows, ["패턴", "행 수"]), ""]

    an = cands[cands["지출액"] > cands["예산현액"]]
    out += ["## 후보 안의 이상치 (지출액 > 예산현액, D17)", "",
            md_table(an[["id", "지역", "자치단체", "세부사업명", "예산현액", "지출액"]].values.tolist(),
                     ["id", "지역", "자치단체", "세부사업명", "예산현액", "지출액"]) if len(an) else "없음", ""]

    rows = []
    for y, m in missed.items():
        rows.append([str(y), len(m), f"{m['지출액'].sum() / 1e8:,.0f}"])
    out += ["## 누락 점검: 교통 관련 회계(주차·교통·철도)인데 후보가 아닌 행", "",
            md_table(rows, ["연도", "행 수", "지출액(억 원)"]), ""]
    for y, m in missed.items():
        top = m.nlargest(15, "지출액")[["지역", "자치단체", "회계", "세부사업명", "분야", "지출액"]]
        out += [f"### {y} 지출액 상위 15", "", md_table(top.values.tolist(), list(top.columns)), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2019])
    args = ap.parse_args()

    frames, missed = [], {}
    for year in args.years:
        c = extract(year)
        missed[year] = c.attrs.pop("missed")
        frames.append(c)
        print(f"[{year}] 1단계 {(c['추출단계'] == 1).sum():,}행, 2단계 {(c['추출단계'] == 2).sum():,}행, "
              f"고유 사업명 {c['세부사업명'].nunique():,}")
    cands = pd.concat(frames, ignore_index=True)
    cands.to_parquet(PROCESSED_DIR / "candidates.parquet", index=False)
    print(f"전체 후보 {len(cands):,}행, 연도 통합 고유 사업명 {cands['세부사업명'].nunique():,}")

    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "02_extract_report.md").write_text(build_report(cands, missed), encoding="utf-8")
    print(f"보고서: {QA_DIR / '02_extract_report.md'}")


if __name__ == "__main__":
    main()
