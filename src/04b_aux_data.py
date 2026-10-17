"""04b. 인구·경계 보조 데이터 결합.

입력 (사용자가 직접 내려받음, 원칙 8)
  data/raw/population/101_DT_1B04006_*.xls   KOSIS 행정구역(시군구)별/1세별 주민등록인구 2025.12
                                             (행정안전부 주민등록인구통계, Excel 2003 XML 형식)
  data/raw/boundary/국가데이터처_SGIS 행정구역 통계 및 경계_20250630.zip  시군구 경계 2025년 2분기
  data/processed/budget_2025.parquet         매칭 대상 자치단체 목록 (지역, 자치단체)

출력
  data/processed/gov_aux.parquet         (지역, 자치단체)별 유형·인구·65세 이상·면적·인구밀도
  data/processed/sigungu_2025.gpkg       지도용 경계 (일반구는 시로 합침, 세종·제주는 시도 하나로)
  outputs/aux/gov_population_area.csv    위 표 (커밋 대상)
  outputs/qa/04b_aux_report.md           매칭 결과

매칭 규칙 (CLAUDE.md 3-3): 키는 (지역, 자치단체). 광역 본청은 시도 전체 인구·면적.
세종은 단층제라 본청만, 제주는 행정시(제주시·서귀포시)를 합쳐 본청 하나로 본다.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POP_DIR = ROOT / "data" / "raw" / "population"
BND_ZIP = ROOT / "data" / "raw" / "boundary" / "국가데이터처_SGIS 행정구역 통계 및 경계_20250630.zip"
BND_DIR = ROOT / "data" / "raw" / "boundary" / "extracted"
BND_STEM = "bnd_sigungu_00_2025_2Q"
BUDGET = ROOT / "data" / "processed" / "budget_2025.parquet"
OUT_PARQUET = ROOT / "data" / "processed" / "gov_aux.parquet"
OUT_GPKG = ROOT / "data" / "processed" / "sigungu_2025.gpkg"
OUT_CSV = ROOT / "outputs" / "aux" / "gov_population_area.csv"
REPORT = ROOT / "outputs" / "qa" / "04b_aux_report.md"

SIDO_FULL = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주",
    "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원특별자치도": "강원",
    "충청북도": "충북", "충청남도": "충남", "전북특별자치도": "전북", "전라남도": "전남", "경상북도": "경북",
    "경상남도": "경남", "제주특별자치도": "제주",
}
SGIS_SIDO = {"11": "서울", "21": "부산", "22": "대구", "23": "인천", "24": "광주", "25": "대전", "26": "울산",
             "29": "세종", "31": "경기", "32": "강원", "33": "충북", "34": "충남", "35": "전북", "36": "전남",
             "37": "경북", "38": "경남", "39": "제주"}
METRO = {"서울", "부산", "대구", "인천", "광주", "대전", "울산"}  # 구가 자치구인 시도
SINGLE_TIER = {"세종", "제주"}  # 기초 자치단체 없음 (본청만)
SS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}


# ---------- 인구 ----------

def read_kosis_xml(path: Path) -> list[list[str | None]]:
    """KOSIS Excel 2003 XML. 뒤쪽 '메타정보' 시트는 이스케이프되지 않은 '<'가 있어 잘라낸다."""
    txt = path.read_bytes().decode("cp949").strip()
    txt = re.sub(r"^<\?xml[^>]*\?>\s*", "", txt)
    cut = txt.find('<Worksheet ss:Name="메타정보">')
    if cut > 0:
        txt = txt[:cut] + "</Workbook>"
    sheet = ET.fromstring(txt).find("ss:Worksheet", SS)
    return [[(c.find("ss:Data", SS).text if c.find("ss:Data", SS) is not None else None)
             for c in r.findall("ss:Cell", SS)] for r in sheet.findall("ss:Table/ss:Row", SS)]


def load_population() -> tuple[pd.DataFrame, str]:
    files = sorted(POP_DIR.glob("101_DT_1B04006_*.xls"))
    if not files:
        raise SystemExit(f"{POP_DIR}에 KOSIS 인구 파일(101_DT_1B04006_*.xls)이 없습니다")
    rows = read_kosis_xml(files[-1])
    period = rows[1][4]
    if not period.startswith("2025.12"):
        raise SystemExit(f"인구 기준 시점이 2025.12가 아닙니다: {period}")

    # 행은 시도 → 그 아래 시군구 순서. 같은 이름(중구 등)을 시도별로 구분하려고 순서대로 읽는다.
    recs, sido = [], None
    for name, age, _item, _unit, value in (r[:5] for r in rows[2:] if len(r) >= 5):
        if name in SIDO_FULL:
            sido = SIDO_FULL[name]
        if name == "전국" or sido is None:
            continue
        recs.append({"지역": sido, "이름": name, "연령": age, "인구": int(value)})
    pop = pd.DataFrame(recs)
    # 세종은 시도 행과 그 아래 시군구 행이 같은 이름으로 두 번 나온다 → 한 번만 센다
    pop = pop.drop_duplicates(["지역", "이름", "연령"])
    pop["이름"] = pop["이름"].replace({"창원시(통합)": "창원시"})
    pop["나이"] = pop["연령"].str.extract(r"^(\d+)").astype(float)
    total = pop[pop["연령"] == "계"].groupby(["지역", "이름"], sort=False)["인구"].sum()
    old = pop[pop["나이"] >= 65].groupby(["지역", "이름"], sort=False)["인구"].sum()
    out = pd.DataFrame({"인구": total, "인구65": old}).reset_index()
    out["시도행"] = out["이름"].isin(SIDO_FULL)
    out.loc[out["시도행"], "이름"] = "본청"
    national = next(int(r[4]) for r in rows[2:] if r[0] == "전국" and r[1] == "계")
    return out, f"{files[-1].name} (기준 {period.strip()})", national


def keep_gov_units(pop: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """자치단체 단위만 남긴다: 출장소, 일반구(자치구가 아닌 구), 제주 행정시 제외."""
    dropped = []
    keep = []
    for r in pop.itertuples():
        reason = None
        if "출장소" in r.이름:
            reason = "출장소"
        elif r.이름 != "본청" and r.지역 in SINGLE_TIER:
            reason = "단층제·행정시"
        elif r.이름 != "본청" and r.지역 not in METRO and r.이름.endswith("구"):
            reason = "일반구"
        if reason:
            dropped.append(f"{r.지역} {r.이름}({reason})")
        keep.append(reason is None)
    return pop[keep].drop(columns="시도행").rename(columns={"이름": "자치단체"}), dropped


# ---------- 경계 ----------

def extract_boundary() -> Path:
    BND_DIR.mkdir(parents=True, exist_ok=True)
    shp = BND_DIR / f"{BND_STEM}.shp"
    if shp.exists():
        return shp
    with zipfile.ZipFile(BND_ZIP) as z:
        for info in z.infolist():
            base = Path(info.filename).name
            if base.startswith(BND_STEM):
                (BND_DIR / base).write_bytes(z.read(info))
    return shp


def load_boundary() -> gpd.GeoDataFrame:
    g = gpd.read_file(extract_boundary())
    g["지역"] = g["SIGUNGU_CD"].str[:2].map(SGIS_SIDO)
    # 일반구("수원시 장안구")는 시로 합치고, 세종·제주는 본청 하나로 합친다
    g["자치단체"] = g["SIGUNGU_NM"].str.split(" ").str[0]
    g.loc[g["지역"].isin(SINGLE_TIER), "자치단체"] = "본청"
    g = g.dissolve(by=["지역", "자치단체"], as_index=False)[["지역", "자치단체", "geometry"]]
    g["면적_km2"] = g.geometry.area / 1e6  # EPSG:5179 (미터)
    return g


def sido_area(g: gpd.GeoDataFrame) -> pd.Series:
    return g.groupby("지역")["면적_km2"].sum()


# ---------- 결합 ----------

def gov_type(region: str, name: str) -> str:
    if name == "본청":
        return "세종" if region == "세종" else "제주" if region == "제주" else "광역"
    if name.endswith("구"):
        return "자치구"
    if name.endswith("시"):
        return "시"
    return "군"


def main() -> None:
    govs = pd.read_parquet(BUDGET, columns=["지역", "자치단체"]).drop_duplicates().reset_index(drop=True)
    pop_all, pop_src, national = load_population()
    pop, dropped = keep_gov_units(pop_all)
    bnd = load_boundary()

    area = bnd.set_index(["지역", "자치단체"])["면적_km2"]
    sido = sido_area(bnd)

    aux = govs.merge(pop, on=["지역", "자치단체"], how="left")
    aux["면적_km2"] = [sido[r] if n == "본청" else area.get((r, n)) for r, n in zip(aux["지역"], aux["자치단체"])]
    aux["유형"] = [gov_type(r, n) for r, n in zip(aux["지역"], aux["자치단체"])]
    aux["광역기초"] = aux["자치단체"].eq("본청").map({True: "광역", False: "기초"})
    aux["고령비율"] = aux["인구65"] / aux["인구"]
    aux["인구밀도"] = aux["인구"] / aux["면적_km2"]
    aux = aux[["지역", "자치단체", "광역기초", "유형", "인구", "인구65", "고령비율", "면적_km2", "인구밀도"]]

    missing_pop = aux[aux["인구"].isna()]
    missing_area = aux[aux["면적_km2"].isna()]
    budget_keys = set(zip(govs["지역"], govs["자치단체"]))
    pop_only = sorted(f"{r} {n}" for r, n in zip(pop["지역"], pop["자치단체"]) if (r, n) not in budget_keys)
    bnd_only = sorted(f"{r} {n}" for r, n in zip(bnd["지역"], bnd["자치단체"])
                      if (r, n) not in budget_keys and n != "본청")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    aux.to_parquet(OUT_PARQUET, index=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    aux.round({"고령비율": 4, "면적_km2": 2, "인구밀도": 1}).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    bnd["geometry"] = bnd.geometry.simplify(200)  # 지도용 단순화 (200m)
    bnd.merge(aux[["지역", "자치단체", "유형"]], on=["지역", "자치단체"], how="left").to_file(OUT_GPKG, driver="GPKG")

    by_type = aux.groupby("유형").agg(자치단체수=("자치단체", "size"), 인구중앙값=("인구", "median"),
                                    고령비율중앙값=("고령비율", "median"), 인구밀도중앙값=("인구밀도", "median"))
    lines = [
        "# 04b 보조 데이터 결합 보고", "", "자동 생성: src/04b_aux_data.py", "",
        f"- 인구: {pop_src}", f"- 경계: {BND_ZIP.name} → {BND_STEM} (2025-06-30 기준)",
        f"- 예산 자치단체 {len(govs)}곳 중 인구 미매칭 {len(missing_pop)}곳, 면적 미매칭 {len(missing_area)}곳",
        f"- 인구 자료에서 제외: {len(dropped)}개 (출장소·일반구·제주 행정시)",
        f"- 인구 자료에만 있는 단위: {', '.join(pop_only) or '없음'}",
        f"- 경계 자료에만 있는 단위: {', '.join(bnd_only) or '없음'}",
        f"- 미매칭 목록: {', '.join(missing_pop['지역'] + ' ' + missing_pop['자치단체']) or '없음'}"
        f" / {', '.join(missing_area['지역'] + ' ' + missing_area['자치단체']) or '없음'}",
        f"- 전국 인구 합계(본청 행 합): {int(aux.loc[aux['광역기초'] == '광역', '인구'].sum()):,} (KOSIS 전국 값 {national:,})",
        "", "## 유형별 요약", "",
        "| 유형 | 자치단체 수 | 인구 중앙값 | 65세 이상 비율 중앙값 | 인구밀도 중앙값(명/km²) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for t, r in by_type.iterrows():
        lines.append(f"| {t} | {r.자치단체수} | {r.인구중앙값:,.0f} | {r.고령비율중앙값:.1%} | {r.인구밀도중앙값:,.0f} |")
    lines += ["", "## 제외한 인구 단위", "", ", ".join(dropped), ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[3:12]))


if __name__ == "__main__":
    main()
