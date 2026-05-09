#!/usr/bin/env python3
"""Run Cygnus full pipeline end-to-end for smoke testing (--input CSV path).

Usage:
  cd /home/david/.cursor-tutor/SEA
  python test_pipeline.py --input /path/to/all_cells.csv
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEA Cygnus pipeline test (run_full_pipeline).")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to processed object CSV (e.g. all_cells.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/test_pipeline_run",
        type=str,
        help="Directory for pipeline outputs (default: output/test_pipeline_run)",
    )
    args = parser.parse_args()

    inp = args.input.expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()

    print("=== Cygnus pipeline test ===")
    print(f"Input CSV : {inp}")
    print(f"Output dir: {out}")
    print()

    if not inp.is_file():
        print(f"[FAIL] Input file does not exist: {inp}")
        sys.exit(1)

    try:
        from run_pipeline import run_full_pipeline

        print("Starting run_full_pipeline ...")
        run_full_pipeline(filepath=str(inp), output_dir=str(out))
    except Exception:
        print()
        print("[FAIL] Exception during pipeline:")
        traceback.print_exc()
        sys.exit(1)

    report = out / "cygnus_report.html"
    print()
    if report.is_file():
        print(f"[OK] Pipeline completed. Report: {report}")
    else:
        print("[WARN] Pipeline returned without raising, but cygnus_report.html not found.")
        print(f"       Expected at: {report}")
    sys.exit(0)


if __name__ == "__main__":
    main()
