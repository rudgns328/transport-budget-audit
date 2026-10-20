"""07. 보고서용 그림 (CLAUDE.md 7장).

출력: outputs/figures/fig1_flow.png ... fig6_validation.png (200dpi PNG)
  1 분석 흐름도  2 전국 시군구 지도(사람 중심 비율, 2025)  3 유형별 분포(상자그림)
  4 범주별 집행률  5 2019 → 2025 변화  6 AI 분류 검증(혼동행렬 + 분류 예시) — 05 결과가 있을 때만

그림에는 가제를 넣지 않는다 (CLAUDE.md 0장). 색: 범주형은 고정 순서(파랑, 주황), 크기는 파랑 단색 단계.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
IND = ROOT / "outputs" / "indicators"
VAL = ROOT / "outputs" / "validation"
GPKG = ROOT / "data" / "processed" / "sigungu_2025.gpkg"
FIG = ROOT / "outputs" / "figures"

# 팔레트 (dataviz 기준 팔레트, 라이트 모드)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
NODATA = "#f0efec"
CAT_NAME = {"A": "A 자동차 용량 확대", "B": "B 도로 유지·차량 안전", "C": "C 대중교통", "D": "D 보행·자전거",
            "E": "E 교통약자", "F": "F 친환경차", "G": "G 운영·행정", "H": "H 범위 밖·판단 불가"}
TYPES = ["광역", "자치구", "시", "군"]
CAT_TICK = {"A": "A\n자동차\n용량 확대", "B": "B\n도로 유지·\n차량 안전", "C": "C\n대중교통", "D": "D\n보행·\n자전거",
            "E": "E\n교통약자", "F": "F\n친환경차", "G": "G\n운영·행정"}


def setup_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("NanumGothic", "나눔고딕", "AppleGothic", "Noto Sans CJK KR", "Malgun Gothic"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    else:
        warnings.warn("한글 폰트를 찾지 못했습니다. 그림의 한글이 깨질 수 있습니다 (나눔고딕 설치 권장)")
        return ""
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt.rcParams["font.family"][0]


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: outputs/figures/{name}")


def hgrid(ax) -> None:
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------- 1. 흐름도 ----------

def fig1_flow(t: pd.DataFrame) -> None:
    cache = ROOT / "outputs" / "llm_cache"
    units = len(pd.read_csv(cache / "items.csv", usecols=["id"]))
    sonnet = len(pd.read_csv(cache / "sonnet_v1_2.csv", usecols=["id"]).drop_duplicates())
    steps = [
        ("데이터", "지방재정365\n세부사업별 세출현황\n2019·2024·2025"),
        ("후보 추출", "교통 분야 전체 +\n다른 분야 키워드\n후보 90,325건"),
        ("AI 분류", f"사업명 {units:,}개 단위\nHaiku 전체 분류 →\nlow {sonnet:,}개 Sonnet 재분류"),
        ("분석자 검증", "층화 표본 200건\n직접 분류해 AI와 비교\n(정확도·혼동행렬)"),
        ("지표", "사람 중심 비율 등\n243개 자치단체\n같은 유형 안 비교"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_axis_off()
    w, h, gap = 0.17, 0.62, 0.03
    for i, (head, body) in enumerate(steps):
        x = i * (w + gap) + 0.01
        accent = head == "AI 분류"
        ax.add_patch(FancyBboxPatch((x, 0.15), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="#eaf2fc" if accent else "white", ec=BLUE if accent else AXIS, lw=1.2,
                                    transform=ax.transAxes))
        ax.text(x + w / 2, 0.15 + h - 0.1, head, ha="center", va="center", fontsize=11, fontweight="bold",
                color=INK, transform=ax.transAxes)
        ax.text(x + w / 2, 0.15 + h / 2 - 0.07, body, ha="center", va="center", fontsize=8.8, color=INK2,
                linespacing=1.5, transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.004, 0.46), (x + w + gap - 0.004, 0.46),
                                         arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2,
                                         transform=ax.transAxes))
    save(fig, "fig1_flow.png")


# ---------- 2. 지도 ----------

def fig2_map(t: pd.DataFrame) -> None:
    g = gpd.read_file(GPKG)
    d = t[t["연도"] == 2025][["지역", "자치단체", "사람중심비율_지출"]]
    g = g.merge(d, on=["지역", "자치단체"], how="left")
    bins = [0, 0.15, 0.30, 0.45, 0.60, 1.0]
    cmap, norm = ListedColormap(SEQ), BoundaryNorm(bins, len(SEQ))
    labels = ["15% 미만", "15–30%", "30–45%", "45–60%", "60% 이상"]

    fig = plt.figure(figsize=(8.5, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    g.plot(ax=ax, column="사람중심비율_지출", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.3,
           missing_kwds={"color": NODATA})
    # 수도권 확대 (서울 자치구가 작아 본 지도에서 읽기 어려움)
    cap = g[g["지역"].isin(["서울", "인천", "경기"])]
    ins = fig.add_axes([0.66, 0.62, 0.33, 0.33])
    ins.set_axis_off()
    cap.plot(ax=ins, column="사람중심비율_지출", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.3)
    seoul = g[g["지역"] == "서울"].total_bounds
    ins.set_xlim(seoul[0] - 25000, seoul[2] + 25000)
    ins.set_ylim(seoul[1] - 20000, seoul[3] + 20000)
    ins.set_title("수도권 확대", fontsize=9, color=INK2)
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_edgecolor(AXIS)

    handles = [Patch(fc=c, ec="white", label=l) for c, l in zip(SEQ, labels)]
    ax.legend(handles=handles, title="사람 중심 교통예산 비율\n(C+D+E)/(A~G), 2025 지출액", loc="lower right",
              frameon=False, fontsize=9, title_fontsize=9.5, bbox_to_anchor=(0.98, 0.04))
    ax.text(0.02, 0.02, "기초 자치단체 226곳 + 세종·제주(단층제, 본청 값). 광역 본청은 지도에서 제외.\n"
            "자치구는 대중교통 예산을 광역시가 집행해 구조적으로 낮다 — 같은 유형 안에서 비교할 것.",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    save(fig, "fig2_map.png")


# ---------- 3. 유형별 분포 ----------

def fig3_box(t: pd.DataFrame) -> None:
    d = t[t["연도"] == 2025]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, [("사람중심비율_지출", "사람 중심 교통예산 비율 (C+D+E)"),
                                       ("자동차확대비율_지출", "자동차 용량 확대 비율 (A)")]):
        data = [d.loc[d["유형"] == tp, col].dropna() * 100 for tp in TYPES]
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": BLUE, "lw": 2},
                   boxprops={"color": AXIS}, whiskerprops={"color": AXIS}, capprops={"color": AXIS})
        for i, s in enumerate(data, 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(s)), s, s=9, color=INK2, alpha=0.45, lw=0)
            ax.text(i + 0.3, s.median(), f"{s.median():.1f}%", va="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(1, len(TYPES) + 1), [f"{tp}\n({len(s)})" for tp, s in zip(TYPES, data)])
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        hgrid(ax)
    axes[0].set_ylabel("교통예산(A~G) 중 비중, %")
    fig.text(0.01, -0.02, "2025 지출액 기준. 점 하나가 자치단체 하나, 상자는 사분위, 파란 선은 중앙값. 괄호는 자치단체 수.",
             fontsize=8, color=INK2)
    save(fig, "fig3_types.png")


# ---------- 4. 집행률 ----------

def fig4_exec(ex: pd.DataFrame) -> None:
    d = ex[(ex["연도"] == 2025) & ex["범주"].isin(list("ABCDEFG"))]
    cats = list("ABCDEFG")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(cats))
    for j, (lvl, color) in enumerate([("광역", BLUE), ("기초", ORANGE)]):
        v = d[d["광역기초"] == lvl].set_index("범주").loc[cats, "집행률"] * 100
        bars = ax.bar(x + (j - 0.5) * 0.36, v, width=0.34, color=color, label=lvl, edgecolor=SURFACE, lw=1)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [CAT_TICK[k] for k in cats], fontsize=8.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("집행률 (지출액 / 예산현액), %")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    hgrid(ax)
    fig.text(0.01, -0.09, "2025, 층위(광역·기초) 안에서 범주별 지출액 합 / 예산현액 합.", fontsize=8, color=INK2)
    save(fig, "fig4_execution.png")


# ---------- 5. 2019 → 2025 ----------

def fig5_change(ts: pd.DataFrame, ch: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (col, title) in zip(axes, [("사람중심비율_지출", "사람 중심 교통예산 비율"),
                                       ("자동차확대비율_지출", "자동차 용량 확대 비율")]):
        s = ts[ts["지표"] == col]
        for i, tp in enumerate(TYPES[::-1]):
            a = s[(s["유형"] == tp) & (s["연도"] == 2019)]["중앙값"].iloc[0] * 100
            b = s[(s["유형"] == tp) & (s["연도"] == 2025)]["중앙값"].iloc[0] * 100
            ax.plot([a, b], [i, i], color=AXIS, lw=2, zorder=1)
            ax.scatter([a], [i], s=60, color=MUTED, zorder=2, label="2019" if i == 0 else None)
            ax.scatter([b], [i], s=60, color=BLUE, zorder=3, label="2025" if i == 0 else None)
            # 값은 점의 바깥쪽(왼쪽 점은 왼쪽, 오른쪽 점은 오른쪽)에 붙여 위아래 겹침을 없앤다
            for val, color, size in ((a, INK2, 8), (b, INK, 8.5)):
                right = val >= max(a, b)
                ax.text(val + (2.2 if right else -2.2), i, f"{val:.1f}%", ha="left" if right else "right",
                        va="center", fontsize=size, color=color)
            c = ch[(ch["지표"] == col) & (ch["유형"] == tp)]
            ax.text(101, i, f"{c['증가한_자치단체_비율'].iloc[0] * 100:.0f}%", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(TYPES)), TYPES[::-1])
        ax.set_xlim(-12, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_ylim(-0.6, len(TYPES) - 0.2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=12)
        ax.text(101, len(TYPES) - 0.55, "증가한\n자치단체", fontsize=7.5, color=MUTED, va="bottom")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("유형별 중앙값, %")
    axes[0].legend(frameon=False, loc="lower right")
    fig.text(0.01, -0.04, "지출액 기준. 점은 같은 유형 자치단체 비율의 중앙값, 오른쪽 숫자는 2019보다 값이 커진 자치단체 비율. "
             "군위군은 대구 기준으로 연결.", fontsize=8, color=INK2)
    save(fig, "fig5_change.png")


# ---------- 6. 검증 ----------

def fig6_validation() -> None:
    cm_path, ex_path = VAL / "confusion_matrix.csv", VAL / "examples.csv"
    if not cm_path.exists():
        print("건너뜀: fig6_validation.png (05 검증 결과가 아직 없음)")
        return
    cm = pd.read_csv(cm_path, index_col=0)
    ex = pd.read_csv(ex_path) if ex_path.exists() else pd.DataFrame()
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_axes([0.04, 0.12, 0.38, 0.78])
    seq = ListedColormap(["#fcfcfb"] + SEQ)
    m = cm.values.astype(float)
    ax.imshow(m, cmap=seq, vmin=0, vmax=max(m.max(), 1))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8.5,
                        color="white" if m[i, j] > m.max() * 0.55 else INK)
    ax.set_xticks(range(m.shape[1]), cm.columns)
    ax.set_yticks(range(m.shape[0]), cm.index)
    ax.set_xlabel("AI 분류")
    ax.set_ylabel("분석자 분류")
    ax.set_title("혼동행렬 (검증 표본 200건)", fontsize=10.5, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    if len(ex):
        tx = fig.add_axes([0.47, 0.05, 0.52, 0.85])
        tx.set_axis_off()
        tbl = tx.table(cellText=ex.values, colLabels=ex.columns, loc="upper left", cellLoc="left", colLoc="left",
                       colWidths=[0.64, 0.12, 0.12, 0.12])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.35)
        for (r, _c), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID)
            if r == 0:
                cell.set_text_props(fontweight="bold")
        tx.set_title("분류 예시", fontsize=10.5, color=INK, loc="left")
    save(fig, "fig6_validation.png")


def main() -> None:
    font = setup_font()
    print(f"한글 폰트: {font or '없음'}")
    t = pd.read_csv(IND / "gov_indicators.csv")
    ts = pd.read_csv(IND / "type_summary.csv")
    ex = pd.read_csv(IND / "execution_rate.csv")
    ch = pd.read_csv(IND / "change_2019_2025.csv")
    fig1_flow(t)
    fig2_map(t)
    fig3_box(t)
    fig4_exec(ex)
    fig5_change(ts, ch)
    fig6_validation()


if __name__ == "__main__":
    main()
