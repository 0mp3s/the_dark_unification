#!/usr/bin/env python3
"""
run_pipeline.py
===============
Pipeline runner for dark-energy-T-breaking tests.

Usage:
    python run_pipeline.py              # run ALL tests
    python run_pipeline.py --list       # show pipeline table
    python run_pipeline.py --test 19    # run Test 19 only
    python run_pipeline.py --from 6     # run from Test order 6 onwards
    python run_pipeline.py --dry-run    # show what would run, don't execute

Each run:
  - captures stdout + stderr to data/output/<script>_<date>_r<N>.txt
  - appends a row to docs/runs_log.csv
  - sends Telegram notification (silent=True for individual tests, loud for final)

Requires core/ to be on the path (handled automatically).
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).resolve().parent
DOCS     = ROOT / "docs"
PIPELINE = DOCS / "execution_pipeline.csv"
OUTPUT   = ROOT / "data" / "output"
CORE     = ROOT / "core"

# ── ensure output dir exists ──────────────────────────────────────────────────
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── add core/ to path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(CORE))
from run_logger import RunLogger, log_run
from tg_notify  import notify


# ── load pipeline ─────────────────────────────────────────────────────────────

def load_pipeline():
    steps = []
    with open(PIPELINE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(row)
    return steps


def print_pipeline(steps):
    print(f"\n{'#':>4}  {'Test':>10}  {'Script':<45}  {'Reads'}")
    print("-" * 100)
    for s in steps:
        reads = s["Reads"] if s["Reads"] != "none" else "—"
        print(f"{s['Order']:>4}  {s['Test']:>10}  {s['Script']:<45}  {reads}")
    print()


# ── run one script ────────────────────────────────────────────────────────────

def _output_path(script_name: str) -> pathlib.Path:
    stem = pathlib.Path(script_name).stem
    ts   = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return OUTPUT / f"{stem}_{ts}.txt"


def run_step(step: dict, dry_run: bool = False) -> bool:
    """Run one pipeline step. Returns True on success."""
    script = ROOT / step["Script"]
    label  = f"{step['Test']} — {step['Description']}"

    if not script.exists():
        print(f"  ⚠  SKIP  {label}  (script not found: {script.name})")
        return False

    out_path = _output_path(step["Script"])

    if dry_run:
        print(f"  DRY   {label}  → {out_path.name}")
        return True

    print(f"\n{'='*70}")
    print(f"  ▶  {label}")
    print(f"     {script.name}  →  {out_path.name}")
    print(f"{'='*70}")

    t0 = time.monotonic()
    status = "OK"
    key_result = ""

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            timeout=600,   # 10 min max per test
        )
        elapsed = round(time.monotonic() - t0, 2)

        # write combined output
        out_path.write_text(
            f"=== {label} ===\n"
            f"Script : {script.name}\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Elapsed: {elapsed}s\n"
            f"Exit   : {result.returncode}\n"
            f"{'='*60}\n\n"
            f"{result.stdout}"
            + (f"\n\n--- STDERR ---\n{result.stderr}" if result.stderr.strip() else ""),
            encoding="utf-8",
        )

        # print stdout to console too
        if result.stdout:
            print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)

        if result.returncode != 0:
            status = "FAILED"
            print(f"\n  ✗ FAILED (exit {result.returncode})")
            if result.stderr:
                print(result.stderr[-1000:])
        else:
            print(f"\n  ✓ OK  ({elapsed}s)")

        # try to extract last non-empty stdout line as key_result
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if lines:
            key_result = lines[-1][:120]

    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - t0, 2)
        status = "TIMEOUT"
        print(f"\n  ✗ TIMEOUT after {elapsed}s")
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        status = "FAILED"
        print(f"\n  ✗ ERROR: {e}")

    # log the run
    log_run(
        script   = step["Script"],
        test     = f"{step['Test']} - {step['Description']}",
        params   = {},
        output_files = [str(out_path)],
        status   = status,
        duration_sec = elapsed,
        key_result   = key_result,
        notes    = "",
    )

    # Telegram (silent for individual steps)
    icon = "✅" if status == "OK" else "❌"
    notify(
        f"{icon} {step['Test']} | {step['Description']}\n"
        f"status={status}  t={elapsed}s\n"
        f"{key_result}",
        silent=True,
    )

    return status == "OK"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="dark-energy-T-breaking pipeline runner")
    parser.add_argument("--list",    action="store_true", help="Show pipeline table and exit")
    parser.add_argument("--test",    type=str, default=None, help="Run a specific test label, e.g. '19' or 'Test 19'")
    parser.add_argument("--from",    dest="from_order", type=int, default=None,
                        help="Run from this order number onwards")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run, don't execute")
    args = parser.parse_args()

    steps = load_pipeline()

    if args.list:
        print_pipeline(steps)
        return

    # filter steps
    if args.test is not None:
        label = args.test.strip().lower()
        def _matches(s):
            # "test 19" or "19" → match Test column "Test 19"
            if label == s["Test"].lower():
                return True
            # bare number "19" → match "Test 19" (ends with " 19")
            if label.isdigit() and s["Test"].lower() in (f"test {label}", f"verify {label}", f"test {label}b"):
                return True
            # "19" at end of test label: "Test 19" ends with "19"
            if label.isdigit() and s["Test"].lower().split()[-1] == label:
                return True
            # description or script filename
            if label in s["Description"].lower():
                return True
            if label in s["Script"].lower():
                return True
            return False
        filtered = [s for s in steps if _matches(s)]
        if not filtered:
            print(f"No step found matching '{args.test}'")
            sys.exit(1)
        steps = filtered
    elif args.from_order is not None:
        steps = [s for s in steps if int(s["Order"]) >= args.from_order]

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for s in steps:
            run_step(s, dry_run=True)
        return

    print(f"\nRunning {len(steps)} test(s)  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n")

    passed = failed = skipped = 0
    t_total = time.monotonic()

    for step in steps:
        ok = run_step(step)
        if ok is True:
            passed += 1
        elif ok is False and (ROOT / step["Script"]).exists():
            failed += 1
        else:
            skipped += 1

    elapsed_total = round(time.monotonic() - t_total, 1)

    summary = (
        f"🔬 dark-energy pipeline done\n"
        f"✅ {passed} passed  ❌ {failed} failed  ⏭ {skipped} skipped\n"
        f"Total: {elapsed_total}s"
    )
    print(f"\n{'='*60}")
    print(summary)
    notify(summary)   # loud final notification


if __name__ == "__main__":
    main()
