#!/usr/bin/env python3
"""Compare run-time settings of dataframes on the machine at hand.

    scripts/tune.py [--rows 100000,10000000] [--case sum_i64,frame_filter] [--rounds 2]

Runs the dataframes benchmark driver once per setting and case, interleaved,
and prints the best median per (case, setting) as a Markdown table. The
settings are environment variables the library reads at start-up, so nothing
is rebuilt between runs. Used to choose defaults on hardware one cannot sit
in front of (the tuning workflow runs it on a GitHub-hosted runner).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SETTINGS = {
    "default": {},
    "half the threads": {"DATAFRAMES_THREADS": str(max(1, (os.cpu_count() or 2) // 2))},
    "one thread": {"DATAFRAMES_THREADS": "1"},
    "workers sleep at once": {"DATAFRAMES_SPIN_US": "0"},
    "1 MiB per task": {"DATAFRAMES_MIN_TASK_BYTES": str(1 << 20)},
    "no streaming stores": {"DATAFRAMES_STREAM_BYTES": str(1 << 60)},
    "no huge pages": {"DATAFRAMES_NO_HUGEPAGES": "1"},
    "no index lists": {"DATAFRAMES_INDEX_EMIT_PERMILLE": "0"},
    "index lists to 40%": {"DATAFRAMES_INDEX_EMIT_PERMILLE": "400"},
    "AVX2 tier": {"DATAFRAMES_SIMD": "avx2"},
}

DEFAULT_CASES = [
    "mask_i64", "filter_i64_10pct", "filter_i64_50pct", "frame_filter_10pct", "frame_filter_50pct",
    "frame_filter_expr_50pct", "frame_filter_and_25pct", "filter_then_sum", "sum_i64", "sum_f64",
    "min_max_f64", "min_max_i64", "mutate_f64_inplace", "get_random_1k", "slice_series",
]


def fmt_ns(ns):
    for unit, scale in (("ns", 1), ("µs", 1e3), ("ms", 1e6), ("s", 1e9)):
        if ns < 1000 * scale or unit == "s":
            return f"{ns / scale:.2f} {unit}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default="100000,1000000,10000000")
    p.add_argument("--case", default=",".join(DEFAULT_CASES))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--time-ms", type=int, default=200)
    p.add_argument("--settings", default=",".join(SETTINGS))
    p.add_argument("--target-dir", default=str(ROOT / "target" / "native"))
    args = p.parse_args()

    driver = Path(args.target_dir) / "release" / "bench-ours"
    if not driver.exists():
        sys.exit(f"{driver} is not built")
    settings = [s for s in args.settings.split(",") if s in SETTINGS]
    cases = args.case.split(",")
    best = defaultdict(dict)  # (rows, case) -> setting -> ns
    with tempfile.TemporaryDirectory() as tmp:
        for rows in [int(r) for r in args.rows.split(",")]:
            for _ in range(args.rounds):
                for case in cases:
                    for name in settings:
                        out = Path(tmp) / "out.jsonl"
                        out.unlink(missing_ok=True)
                        cmd = [str(driver), "--rows", str(rows), "--time-ms", str(args.time_ms), "--case", "=" + case, "--out", str(out)]
                        subprocess.run(cmd, env={**os.environ, **SETTINGS[name]}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        if not out.exists():
                            continue
                        ns = min(json.loads(line)["median_ns"] for line in out.read_text().splitlines() if line.strip())
                        slot = best[(rows, case)]
                        slot[name] = min(ns, slot.get(name, float("inf")))

    print("| rows | case | " + " | ".join(settings) + " |")
    print("|---:|---|" + "---:|" * len(settings))
    for (rows, case), slot in best.items():
        base = slot.get("default")
        cells = []
        for name in settings:
            ns = slot.get(name)
            if ns is None:
                cells.append("-")
            elif name == "default" or not base:
                cells.append(fmt_ns(ns))
            else:
                cells.append(f"{fmt_ns(ns)} ({base / ns:.2f}x)")
        print(f"| {rows} | {case} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
