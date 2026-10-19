"""04. LLM 분류 (기준 v1.2, prompts/classify_v1_2.txt).

분류 단위: (정규화 사업명, 분야, 부문). 정규화 = NFKC + 공백 제거, 분야는 2019 이름을 2025 이름으로 맞춘다 (D20).
같은 단위의 행은 같은 분류를 받는다.

모드
  --mode cache   (기본) API 호출 없이 캐시만으로 결과를 원본 행에 결합한다. run_all.py가 쓰는 모드
  --mode trial   사람 라벨 표본(sample_ids.csv)과 사업명이 겹치지 않는 단위 50개를 Haiku로 분류하고,
                 low 건은 Sonnet으로 재분류한다. 실제 토큰 사용량으로 전체 비용을 추정한다
  --mode full    전체 단위를 Haiku로 분류 (--yes 필요, 원칙 1: 사용자 승인 후)
  --mode rerun-low  Haiku 결과 중 confidence=low를 Sonnet으로 재분류 (--yes 필요)
  --mode consistency  검증 표본(sample_ids.csv) 200건의 단위를 같은 2단계 절차로 한 번 더 분류해
                consistency_{haiku,sonnet}_v1_2.csv에 저장 (05의 일관성 검사용, 약 0.1달러)
  --mode demo   시연용. 무작위 N개(--n, 기본 20)를 지금 실제 API로 같은 2단계 절차로 분류해
                캐시의 최종 결과와 비교한다. 보고서 수치에 쓰는 캐시는 건드리지 않는다 (약 30초, 1센트 안팎)

API 모드는 이미 캐시에 있는 단위를 건너뛰므로 중단 후 같은 명령으로 이어서 실행할 수 있다 (--use-cache 기본값 True).

출력 (outputs/llm_cache/, 커밋 대상)
  items.csv                 분류 단위 목록 (id ↔ 사업명·분야·부문, 해당 행 수)
  {trial_,}haiku_v1_2.csv   Haiku 결과 / {trial_,}sonnet_v1_2.csv  Sonnet 재분류 결과
  (v1_1 파일은 v1.1 시험 기록으로 보존)
  usage_log.csv             실행별 토큰 사용량과 비용
출력 (재생성 가능, 커밋 제외): data/processed/classified.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "processed" / "candidates.parquet"
SAMPLE_IDS = ROOT / "data" / "labels" / "sample_ids.csv"
PROMPT_VERSION = "v1.2"
VTAG = PROMPT_VERSION.replace(".", "_")
PROMPT_PATH = ROOT / "prompts" / f"classify_{VTAG}.txt"
CACHE_DIR = ROOT / "outputs" / "llm_cache"
QA_DIR = ROOT / "outputs" / "qa"
OUT_PARQUET = ROOT / "data" / "processed" / "classified.parquet"

MAIN_MODEL = "claude-haiku-5-5"
RERUN_MODEL = "claude-sonnet-5-5"
PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}  # (입력, 출력) 달러
FIELD_CANON = {"수송및교통": "교통및물류", "환경보호": "환경", "산업ㆍ중소기업": "산업ㆍ중소기업및에너지"}
CATEGORIES = set("ABCDEFGH")
CONFIDENCES = {"high", "low"}
CACHE_COLS = ["id", "사업명", "category", "confidence", "reason", "model", "prompt_version", "실행일"]
BATCH_SIZE = 50
MAX_ATTEMPTS = 3
TRIAL_N = 50
SEED = 20261017

_write_lock = threading.Lock()


# ---------- 분류 단위 ----------

def name_key(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def unit_id(k: str, field: str, sector: str) -> str:
    return "p" + hashlib.sha1(f"{k}|{field}|{sector}".encode("utf-8")).hexdigest()[:10]


def load_rows() -> pd.DataFrame:
    c = pd.read_parquet(CANDIDATES)
    c["사업명키"] = c["세부사업명"].map(name_key)
    c["분야정규"] = c["분야"].replace(FIELD_CANON)
    c["단위id"] = [unit_id(k, f, s) for k, f, s in zip(c["사업명키"], c["분야정규"], c["부문"])]
    return c


def build_items(rows: pd.DataFrame) -> pd.DataFrame:
    """단위별 대표 사업명은 가장 많이 쓰인 원문 표기."""
    g = rows.groupby("단위id")
    items = pd.DataFrame({
        "사업명": g["세부사업명"].agg(lambda s: s.value_counts().index[0]),
        "분야": g["분야정규"].first(),
        "부문": g["부문"].first(),
        "사업명키": g["사업명키"].first(),
        "행수": g.size(),
    }).reset_index().rename(columns={"단위id": "id"})
    return items.sort_values("id").reset_index(drop=True)


# ---------- 프롬프트·호출 ----------

def render_prompt(batch: pd.DataFrame) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    lines = [f'{r.id}, "{r.사업명.replace(chr(34), chr(39))}", {r.분야}, {r.부문}' for r in batch.itertuples()]
    return template.replace("<<INPUT>>", "\n".join(lines))  # 프롬프트에 중괄호가 있어 str.format() 금지


def parse_output(text: str, expected_ids: set[str]) -> dict[str, dict]:
    """JSON 배열을 찾아 검증하고, 형식이 맞는 항목만 돌려준다."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    out = {}
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        i, cat, conf, reason = d.get("id"), d.get("category"), d.get("confidence"), d.get("reason")
        if i in expected_ids and cat in CATEGORIES and conf in CONFIDENCES and isinstance(reason, str):
            out[i] = {"category": cat, "confidence": conf, "reason": reason.strip()}
    return out


def call_model(client, model: str, effort: str, batch: pd.DataFrame) -> tuple[dict, str, dict]:
    kwargs = dict(model=model, max_tokens=16000, output_config={"effort": effort},
                  messages=[{"role": "user", "content": render_prompt(batch)}])
    if model == RERUN_MODEL:
        # 안전 분류기가 거절하면 서버가 다른 모델로 이어서 처리 (실제 처리 모델은 response.model에 기록)
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    if resp.stop_reason in ("refusal", "max_tokens"):
        return {}, resp.model, usage
    text = "".join(b.text for b in resp.content if b.type == "text")
    return parse_output(text, set(batch["id"])), resp.model, usage


def append_cache(path: Path, rows: list[dict]) -> None:
    with _write_lock:
        df = pd.DataFrame(rows, columns=CACHE_COLS)
        df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLS)
    return pd.read_csv(path, dtype=str).drop_duplicates("id", keep="last")


def classify(items: pd.DataFrame, model: str, effort: str, cache_path: Path, workers: int, use_cache: bool) -> dict:
    """배치 분류. 실패한 항목은 배치를 반으로 나눠 최대 MAX_ATTEMPTS회 재시도."""
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic(max_retries=5)

    if use_cache:
        done = set(read_cache(cache_path)["id"])
        items = items[~items["id"].isin(done)]
    todo = [items.iloc[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    totals = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "items": 0, "failed": []}
    print(f"[{model}] 분류 대상 {len(items):,}개, 배치 {len(todo):,}개, 동시 {workers}")
    t0 = time.time()

    def run(batch: pd.DataFrame, attempt: int) -> list[tuple[pd.DataFrame, int]]:
        try:
            got, served_by, usage = call_model(client, model, effort, batch)
        except anthropic.APIStatusError as e:
            if e.status_code < 500 and e.status_code != 429:
                raise
            got, served_by, usage = {}, model, {"input_tokens": 0, "output_tokens": 0}
        with _write_lock:
            totals["input_tokens"] += usage["input_tokens"]
            totals["output_tokens"] += usage["output_tokens"]
            totals["calls"] += 1
            totals["items"] += len(got)
        today = date.today().isoformat()
        names = dict(zip(batch["id"], batch["사업명"]))
        append_cache(cache_path, [{"id": i, "사업명": names[i], **v, "model": served_by,
                                   "prompt_version": PROMPT_VERSION, "실행일": today} for i, v in got.items()])
        missing = batch[~batch["id"].isin(got)]
        if missing.empty:
            return []
        if attempt >= MAX_ATTEMPTS:
            with _write_lock:
                totals["failed"] += list(missing["id"])
            return []
        half = max(1, len(missing) // 2)
        return [(missing.iloc[:half], attempt + 1), (missing.iloc[half:], attempt + 1)] if len(missing) > 1 \
            else [(missing, attempt + 1)]

    queue = [(b, 1) for b in todo]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while queue:
            futures = [pool.submit(run, b, a) for b, a in queue if len(b)]
            queue = []
            for n, f in enumerate(as_completed(futures), 1):
                queue += f.result()
                if n % 20 == 0:
                    print(f"  {n}/{len(futures)} 배치 완료, 분류 {totals['items']:,}개, {time.time() - t0:,.0f}초")

    totals["seconds"] = round(time.time() - t0, 1)
    return totals


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICE_PER_MTOK[model]
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def log_usage(mode: str, model: str, effort: str, totals: dict) -> float:
    cost = cost_usd(model, totals["input_tokens"], totals["output_tokens"])
    row = {"실행시각": datetime.now().isoformat(timespec="seconds"), "mode": mode, "model": model, "effort": effort,
           "prompt_version": PROMPT_VERSION, "호출수": totals["calls"], "분류수": totals["items"],
           "실패수": len(totals["failed"]), "input_tokens": totals["input_tokens"],
           "output_tokens": totals["output_tokens"], "비용_usd": round(cost, 4), "소요초": totals["seconds"]}
    path = CACHE_DIR / "usage_log.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
    print(f"[{model}] 호출 {totals['calls']}회, 분류 {totals['items']:,}개, 실패 {len(totals['failed'])}개, "
          f"입력 {totals['input_tokens']:,} / 출력 {totals['output_tokens']:,} 토큰, ${cost:.4f}, {totals['seconds']}초")
    return cost


# ---------- 결과 결합 ----------

def combine(rows: pd.DataFrame, items: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Haiku 결과에 Sonnet 재분류를 덮어써 최종 분류를 만들고 원본 행에 붙인다."""
    haiku = read_cache(CACHE_DIR / f"{prefix}haiku_{VTAG}.csv")
    sonnet = read_cache(CACHE_DIR / f"{prefix}sonnet_{VTAG}.csv")
    res = items[["id"]].merge(
        haiku[["id", "category", "confidence", "reason", "model"]].add_suffix("_haiku").rename(columns={"id_haiku": "id"}),
        on="id", how="left")
    res = res.merge(
        sonnet[["id", "category", "confidence", "reason", "model"]].add_suffix("_sonnet").rename(columns={"id_sonnet": "id"}),
        on="id", how="left")
    use_sonnet = res["category_sonnet"].notna()
    for col in ("category", "confidence", "reason", "model"):
        res[col] = res[f"{col}_sonnet"].where(use_sonnet, res[f"{col}_haiku"])
    return rows.merge(res, left_on="단위id", right_on="id", how="left", suffixes=("", "_unit")).drop(columns="id_unit")


def md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join([" --- "] + [" ---: "] * (len(df.columns) - 1)) + "|"
    body = ["| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def report(classified: pd.DataFrame, items: pd.DataFrame) -> str:
    unit = classified.drop_duplicates("단위id")
    lines = ["# 04 LLM 분류 결과 요약", "", "자동 생성: src/04_classify_llm.py (캐시 기준)", "",
             f"- 분류 단위 {len(items):,}개 중 분류됨 {unit['category'].notna().sum():,}개, 미분류 {unit['category'].isna().sum():,}개",
             f"- Haiku low → Sonnet 재분류 {unit['category_sonnet'].notna().sum():,}개", ""]
    for y in sorted(classified["연도"].unique(), reverse=True):
        d = classified[classified["연도"] == y]
        t = d.groupby("category", dropna=False).agg(행수=("id", "size"), 지출액_억=("지출액", "sum"))
        t["지출액_억"] = (t["지출액_억"] / 1e8).round(0).astype(int)
        t["지출 비중%"] = (t["지출액_억"] / t["지출액_억"].sum() * 100).round(1)
        lines += [f"## {y} 범주별 (행 기준)", "", md(t), ""]
    conf = unit.groupby(["category", "confidence"], dropna=False).size().unstack(fill_value=0)
    lines += ["## 단위별 범주 × 확신도", "", md(conf), ""]
    return "\n".join(lines)


# ---------- 시연 ----------

def demo(items: pd.DataFrame, args) -> None:
    """무작위 단위를 지금 다시 분류해 캐시의 최종 결과와 비교한다."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed != SEED else int(time.time()) % 100000  # 기본은 매번 다른 표본
    pick = items.sample(args.n, random_state=seed)
    h_path, s_path = out_dir / f"{stamp}_haiku.csv", out_dir / f"{stamp}_sonnet.csv"

    t0 = time.time()
    th = classify(pick, MAIN_MODEL, args.effort, h_path, 1, use_cache=False)
    cost = cost_usd(MAIN_MODEL, th["input_tokens"], th["output_tokens"])
    h = read_cache(h_path)
    low = pick[pick["id"].isin(h.loc[h["confidence"] == "low", "id"])]
    if len(low):
        ts = classify(low, RERUN_MODEL, args.effort, s_path, 1, use_cache=False)
        cost += cost_usd(RERUN_MODEL, ts["input_tokens"], ts["output_tokens"])
    s = read_cache(s_path)
    now = h.set_index("id")["category"].copy()
    now.update(s.set_index("id")["category"])

    cached = combine(pick.rename(columns={"id": "단위id"}).assign(id=pick["id"]), items)
    cached = cached.set_index("단위id")["category"]
    table = pick[["id", "사업명"]].assign(저장된_결과=pick["id"].map(cached), 지금_결과=pick["id"].map(now))
    table["일치"] = (table["저장된_결과"] == table["지금_결과"]).map({True: "O", False: "X"})
    table.to_csv(out_dir / f"{stamp}_compare.csv", index=False, encoding="utf-8-sig")

    print(f"\n시연: 무작위 {len(pick)}개 사업 (시드 {seed}) 실시간 분류 vs 저장된 결과")
    for r in table.itertuples():
        print(f"  {r.일치}  {r.저장된_결과} → {r.지금_결과}  {r.사업명}")
    agree = (table["일치"] == "O").mean()
    print(f"\n일치율 {agree:.0%} ({(table['일치'] == 'O').sum()}/{len(table)}), "
          f"비용 ${cost:.4f}, {time.time() - t0:.0f}초 (Sonnet 재분류 {len(low)}개)")
    print(f"비교표: {(out_dir / f'{stamp}_compare.csv').relative_to(ROOT)}")


# ---------- 실행 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cache", "trial", "full", "rerun-low", "demo", "consistency"], default="cache")
    ap.add_argument("--yes", action="store_true", help="전체 유료 실행 승인 (full, rerun-low)")
    ap.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED, help="trial 표본 추출 시드")
    ap.add_argument("--n", type=int, default=20, help="demo 모드 사업 수")
    ap.add_argument("--trial-ids", type=Path, help="trial 대상 id 목록 CSV (이전 시험과 같은 단위로 비교할 때)")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    items = build_items(rows)
    items[["id", "사업명", "분야", "부문", "행수"]].to_csv(CACHE_DIR / "items.csv", index=False, encoding="utf-8")
    print(f"분류 단위 {len(items):,}개 (후보 {len(rows):,}행)")

    if args.mode == "trial":
        sample_keys = set(pd.read_csv(SAMPLE_IDS)["세부사업명"].map(name_key))
        pool = items[~items["사업명키"].isin(sample_keys)]  # D13: 라벨 표본과 겹치는 사업명 제외
        if args.trial_ids:
            ids = set(pd.read_csv(args.trial_ids, dtype=str)["id"])
            trial = pool[pool["id"].isin(ids)]
            if len(trial) != len(ids):
                raise SystemExit(f"지정한 id {len(ids)}개 중 {len(ids) - len(trial)}개가 라벨 표본과 겹치거나 없습니다")
        else:
            trial = pool.sample(TRIAL_N, random_state=args.seed)
        totals = classify(trial, MAIN_MODEL, args.effort, CACHE_DIR / f"trial_haiku_{VTAG}.csv", 1, args.use_cache)
        h_cost = log_usage("trial", MAIN_MODEL, args.effort, totals)
        h = read_cache(CACHE_DIR / f"trial_haiku_{VTAG}.csv")
        h = h[h["id"].isin(trial["id"])]
        low = trial[trial["id"].isin(h.loc[h["confidence"] == "low", "id"])]
        s_cost, s_items = 0.0, 0
        if len(low):
            st = classify(low, RERUN_MODEL, args.effort, CACHE_DIR / f"trial_sonnet_{VTAG}.csv", 1, args.use_cache)
            s_cost, s_items = log_usage("trial", RERUN_MODEL, args.effort, st), st["items"]
        low_rate = len(low) / max(len(h), 1)
        n = len(items)
        est_h = h_cost / max(totals["items"], 1) * n
        est_s = (s_cost / s_items if s_items else 0) * n * low_rate
        print(f"\n시험 low 비율 {low_rate:.0%} ({len(low)}/{len(h)})")
        print(f"전체 {n:,}개 추정: Haiku ${est_h:.2f} + Sonnet ${est_s:.2f} = ${est_h + est_s:.2f}")
        trial_rows = rows[rows["단위id"].isin(trial["id"])]
        out = combine(trial_rows.drop_duplicates("단위id"), items, prefix="trial_")
        cols = ["세부사업명", "분야정규", "부문", "category_haiku", "confidence_haiku", "category_sonnet",
                "confidence_sonnet", "category", "reason"]
        tag = args.trial_ids.stem if args.trial_ids else f"s{args.seed}"
        review = QA_DIR / f"04_trial_review_{VTAG}_{tag}.csv"
        out[cols].to_csv(review, index=False, encoding="utf-8-sig")
        print(f"시험 결과 검토용: {review.relative_to(ROOT)}")
        return

    if args.mode == "demo":
        demo(items, args)
        return

    if args.mode == "consistency":
        ids = pd.read_csv(SAMPLE_IDS, dtype=str)["id"]
        unit_ids = rows.loc[rows["id"].isin(ids), "단위id"].unique()
        sub = items[items["id"].isin(unit_ids)]
        th = classify(sub, MAIN_MODEL, args.effort, CACHE_DIR / f"consistency_haiku_{VTAG}.csv", 4, args.use_cache)
        log_usage("consistency", MAIN_MODEL, args.effort, th)
        h = read_cache(CACHE_DIR / f"consistency_haiku_{VTAG}.csv")
        low = sub[sub["id"].isin(h.loc[h["confidence"] == "low", "id"])]
        ts = classify(low, RERUN_MODEL, args.effort, CACHE_DIR / f"consistency_sonnet_{VTAG}.csv", 4, args.use_cache)
        log_usage("consistency", RERUN_MODEL, args.effort, ts)
        return

    if args.mode in ("full", "rerun-low"):
        if not args.yes:
            raise SystemExit("유료 전체 실행입니다. 사용자 승인 후 --yes를 붙여 실행하세요 (CLAUDE.md 원칙 1).")
        if args.mode == "full":
            totals = classify(items, MAIN_MODEL, args.effort, CACHE_DIR / f"haiku_{VTAG}.csv", args.workers, args.use_cache)
            log_usage("full", MAIN_MODEL, args.effort, totals)
        else:
            h = read_cache(CACHE_DIR / f"haiku_{VTAG}.csv")
            low = items[items["id"].isin(h.loc[h["confidence"] == "low", "id"])]
            totals = classify(low, RERUN_MODEL, args.effort, CACHE_DIR / f"sonnet_{VTAG}.csv", args.workers, args.use_cache)
            log_usage("rerun-low", RERUN_MODEL, args.effort, totals)
        if totals["failed"]:
            print(f"실패 {len(totals['failed'])}개: 같은 명령을 다시 실행하면 이어서 처리합니다")

    classified = combine(rows, items)
    classified.drop(columns=["사업명키"]).to_parquet(OUT_PARQUET, index=False)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "04_classify_report.md").write_text(report(classified, items), encoding="utf-8")
    print(f"결합 완료: {OUT_PARQUET.relative_to(ROOT)}, 미분류 단위 "
          f"{classified.drop_duplicates('단위id')['category'].isna().sum():,}개")


if __name__ == "__main__":
    main()
