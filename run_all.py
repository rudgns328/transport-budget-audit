"""캐시만으로 01~07 전체 재현 (API 키·비용 없음).

사전 준비: data/raw/에 원본을 내려받아 둔다 (README의 데이터 표 참고).
  - {2019,2024,2025}_세부사업별세출현황.xlsx (지방재정365)
  - population/101_DT_1B04006_*.xls (KOSIS 주민등록인구 2025.12)
  - boundary/국가데이터처_SGIS 행정구역 통계 및 경계_20250630.zip

LLM 분류는 outputs/llm_cache/의 커밋된 결과를 쓴다 (04 --mode cache).
03(라벨링 표본)은 data/labels/sample_ids.csv가 이미 있으면 다시 뽑지 않는다 (사람 라벨과의 연결 보존).
05(검증)는 사람 라벨 파일이 있을 때만 실행한다.

실행: python run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LABELS = ROOT / "data" / "labels"


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}".rstrip(), flush=True)
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # 캐시 재현은 키 없이 돈다
    subprocess.run([sys.executable, str(SRC / script), *args], cwd=ROOT, env=env, check=True)
    print(f"--- {script} 완료 ({time.time() - t0:,.0f}초)", flush=True)


def main() -> None:
    missing = [p for p in ("data/raw/2025_세부사업별세출현황.xlsx", "data/raw/2024_세부사업별세출현황.xlsx",
                           "data/raw/2019_세부사업별세출현황.xlsx") if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("원본 파일이 없습니다: " + ", ".join(missing))

    run("01_load_clean.py")
    run("02_extract_candidates.py")
    if (LABELS / "sample_ids.csv").exists():
        print("\n=== 03_sample_for_labeling.py 건너뜀 (sample_ids.csv 있음)")
    else:
        run("03_sample_for_labeling.py")
    run("04_classify_llm.py", "--mode", "cache")
    run("04b_aux_data.py")
    if (SRC / "05_validate.py").exists() and any(LABELS.glob("labels_*.xlsx")):
        run("05_validate.py")
    else:
        print("\n=== 05_validate.py 건너뜀 (스크립트 또는 사람 라벨 없음)")
    run("06_indicators.py")
    run("07_figures.py")
    print("\n전체 재현 완료. 결과: outputs/results_summary.md, outputs/figures/")


if __name__ == "__main__":
    main()
