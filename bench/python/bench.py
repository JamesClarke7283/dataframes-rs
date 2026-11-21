#!/usr/bin/env python3
"""Benchmark driver: pandas and the official Polars Python wheel.

Loads the dataset written by `datagen` (so the columns are bit-identical to
the ones the Rust drivers use), times the same cases, and appends JSON lines
in the common result format.

    bench.py --rows N [--time-ms T] [--out FILE] [--case SUBSTR] [--data DIR]

The Polars wheel is the build its authors ship: nightly Rust with explicit
SIMD, tuned allocator, link-time optimisation. It is the strongest
configuration of Polars there is, which is why it is in the comparison.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent


def load(rows, data_dir):
    d = Path(data_dir) / str(rows)
    if not (d / "rows.txt").exists():
        gen = ROOT / "target" / "native" / "release" / "datagen"
        subprocess.run([str(gen), "--rows", str(rows), "--out", str(d)], check=True)
    cols = {
        "id": np.fromfile(d / "id.i64", dtype="<i8"),
        "a": np.fromfile(d / "a.i64", dtype="<i8"),
        "b": np.fromfile(d / "b.f64", dtype="<f8"),
        "c": np.fromfile(d / "c.i32", dtype="<i4"),
        "d": np.fromfile(d / "d.f32", dtype="<f4"),
        "e": np.fromfile(d / "e.i64", dtype="<i8"),
    }
    vocab = np.array((d / "vocab.txt").read_text().split("\n")[:-1], dtype=object)
    s = vocab[np.fromfile(d / "s.u32", dtype="<u4")]
    lookups = np.fromfile(d / "lookups.u32", dtype="<u4").tolist()
    return cols, s, lookups


class Suite:
    def __init__(self, lib, opts):
        self.lib, self.opts, self.records = lib, opts, []

    def bench(self, case, f, check):
        if self.opts.case:
            base = case.split("@")[0]
            if not any((base == p[1:]) if p.startswith("=") else (p in case) for p in self.opts.case.split(",")):
                return
        check_value = float(check(f()))
        min_time = self.opts.time_ms / 1000.0
        warm = max(min_time / 4, 0.02)
        start, iters = time.perf_counter(), 0
        while time.perf_counter() - start < warm or iters < 3:
            f()
            iters += 1
        per_iter = (time.perf_counter() - start) / iters
        batch = max(1, min(1_000_000, int(20e-6 / max(per_iter, 1e-11)) + 1)) if per_iter < 20e-6 else 1
        samples = []
        start = time.perf_counter()
        while (time.perf_counter() - start < min_time or len(samples) < 15) and len(samples) < 20000:
            t = time.perf_counter_ns()
            for _ in range(batch):
                f()
            samples.append((time.perf_counter_ns() - t) / batch)
        samples.sort()
        rec = {
            "lib": self.lib,
            "case": case,
            "rows": self.opts.rows,
            "median_ns": samples[len(samples) // 2],
            "min_ns": samples[0],
            "mean_ns": sum(samples) / len(samples),
            "samples": len(samples),
            "check": check_value,
        }
        self.records.append(rec)
        print(f"{self.lib:<12} {case:<28} rows={self.opts.rows:<9} median={rec['median_ns'] / 1e3:>12.2f} µs", file=sys.stderr)

    def finish(self):
        text = "".join(json.dumps(r) + "\n" for r in self.records)
        if self.opts.out:
            with open(self.opts.out, "a") as f:
                f.write(text)
        else:
            sys.stdout.write(text)


def run_pandas(opts, cols, s, lookups):
    suite = Suite("pandas", opts)
    n = opts.rows
    num = pd.DataFrame(cols, copy=True)
    with_str = pd.DataFrame({"id": cols["id"], "a": cols["a"], "b": cols["b"], "s": pd.Series(s, dtype="str")})
    a, b, c, d, e = (num[k] for k in "abcde")
    e_values = e.to_numpy()
    frame_check = lambda out: len(out) + int(out["e"].sum())

    suite.bench("mask_i64", lambda: a < 500, lambda m: int(m.sum()))
    for pct, thr in [(1, 10), (10, 100), (50, 500), (90, 900)]:
        suite.bench(f"filter_i64_{pct}pct", lambda thr=thr: a[a < thr], lambda out: len(out) + int(out.sum()))
    suite.bench("filter_f64_50pct", lambda: b[b < 0.5], len)
    suite.bench("filter_i32_50pct", lambda: c[c < 50], lambda out: len(out) + int(out.sum()))
    suite.bench("filter_f32_50pct", lambda: d[d < 0.5], len)
    for pct, thr in [(10, 100), (50, 500)]:
        suite.bench(f"frame_filter_{pct}pct", lambda thr=thr: num[num["a"] < thr], frame_check)
    suite.bench("frame_filter_str_50pct", lambda: with_str[with_str["a"] < 500], lambda out: len(out) + int(out["id"].sum()))
    suite.bench("frame_filter_expr_50pct@query", lambda: num.query("a < 500"), frame_check)
    suite.bench("frame_filter_expr_50pct@mask", lambda: num[num["a"] < 500], frame_check)
    suite.bench("frame_filter_and_25pct@query", lambda: num.query("a < 500 and b < 0.5"), frame_check)
    suite.bench("frame_filter_and_25pct@mask", lambda: num[(num["a"] < 500) & (num["b"] < 0.5)], frame_check)
    suite.bench("filter_then_sum", lambda: num.loc[num["a"] < 500, "b"].sum(), float)
    suite.bench("slice_frame", lambda: num.iloc[n // 4 : n // 4 + n // 2], frame_check)
    suite.bench("head_frame", lambda: num.head(10), len)
    suite.bench("slice_series", lambda: a.iloc[n // 4 : n // 4 + n // 2], lambda out: len(out) + int(out.iloc[0]))

    def get_random():
        acc = 0
        for i in lookups:
            acc += e_values[i]
        return acc

    suite.bench("get_random_1k", get_random, int)
    # The same question asked once instead of a thousand times.
    positions = np.asarray(lookups, dtype=np.intp)
    suite.bench("get_random_1k@take", lambda: e_values.take(positions).sum(), int)
    suite.bench("sum_i64", lambda: e.sum(), int)
    suite.bench("sum_f64", lambda: b.sum(), float)
    suite.bench("mean_f64", lambda: b.mean(), float)
    suite.bench("min_max_f64", lambda: b.min() + b.max(), float)
    suite.bench("min_max_i64", lambda: e.min() + e.max(), int)
    own = cols["b"].copy()

    def mutate():
        np.multiply(own, 1.0000001, out=own)
        np.add(own, 0.5, out=own)

    suite.bench("mutate_f64_inplace", mutate, lambda _: 0.0)
    suite.finish()


def run_polars(opts, cols, s, lookups):
    suite = Suite("polars-py", opts)
    n = opts.rows
    num = pl.DataFrame(cols)
    with_str = pl.DataFrame({"id": cols["id"], "a": cols["a"], "b": cols["b"], "s": pl.Series(s, dtype=pl.String)})
    a, b, c, d, e = (num[k] for k in "abcde")
    frame_check = lambda out: out.height + int(out["e"].sum())

    suite.bench("mask_i64", lambda: a < 500, lambda m: int(m.sum()))
    for pct, thr in [(1, 10), (10, 100), (50, 500), (90, 900)]:
        suite.bench(f"filter_i64_{pct}pct", lambda thr=thr: a.filter(a < thr), lambda out: len(out) + int(out.sum()))
    suite.bench("filter_f64_50pct", lambda: b.filter(b < 0.5), len)
    suite.bench("filter_i32_50pct", lambda: c.filter(c < 50), lambda out: len(out) + int(out.sum()))
    suite.bench("filter_f32_50pct", lambda: d.filter(d < 0.5), len)
    for pct, thr in [(10, 100), (50, 500)]:
        suite.bench(f"frame_filter_{pct}pct", lambda thr=thr: num.filter(num["a"] < thr), frame_check)
    suite.bench("frame_filter_str_50pct", lambda: with_str.filter(with_str["a"] < 500), lambda out: out.height + int(out["id"].sum()))
    suite.bench("frame_filter_expr_50pct@eager", lambda: num.filter(pl.col("a") < 500), frame_check)
    suite.bench("frame_filter_expr_50pct@lazy", lambda: num.lazy().filter(pl.col("a") < 500).collect(), frame_check)
    suite.bench("frame_filter_expr_50pct@stream", lambda: num.lazy().filter(pl.col("a") < 500).collect(engine="streaming"), frame_check)
    both = (pl.col("a") < 500) & (pl.col("b") < 0.5)
    suite.bench("frame_filter_and_25pct@lazy", lambda: num.lazy().filter(both).collect(), frame_check)
    suite.bench("frame_filter_and_25pct@stream", lambda: num.lazy().filter(both).collect(engine="streaming"), frame_check)
    summed = lambda engine: num.lazy().filter(pl.col("a") < 500).select(pl.col("b").sum()).collect(engine=engine)
    suite.bench("filter_then_sum@lazy", lambda: summed("in-memory"), lambda out: float(out["b"][0]))
    suite.bench("filter_then_sum@stream", lambda: summed("streaming"), lambda out: float(out["b"][0]))
    suite.bench("slice_frame", lambda: num.slice(n // 4, n // 2), frame_check)
    suite.bench("head_frame", lambda: num.head(10), lambda out: out.height)
    suite.bench("slice_series", lambda: a.slice(n // 4, n // 2), lambda out: len(out) + int(out[0]))

    def get_random():
        acc = 0
        for i in lookups:
            acc += e[i]
        return acc

    suite.bench("get_random_1k", get_random, int)
    # The same question asked once instead of a thousand times.
    positions = pl.Series("positions", lookups, dtype=pl.UInt32)
    suite.bench("get_random_1k@take", lambda: e.gather(positions).sum(), int)
    suite.bench("sum_i64", lambda: e.sum(), int)
    suite.bench("sum_f64", lambda: b.sum(), float)
    suite.bench("mean_f64", lambda: b.mean(), float)
    suite.bench("min_max_f64", lambda: b.min() + b.max(), float)
    suite.bench("min_max_i64", lambda: e.min() + e.max(), int)
    own = [pl.Series("b", cols["b"])]

    def mutate():
        # Polars has no in-place arithmetic; rebinding is its idiom.
        own[0] = own[0] * 1.0000001 + 0.5

    suite.bench("mutate_f64_inplace", mutate, lambda _: 0.0)
    suite.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--time-ms", type=int, default=300)
    p.add_argument("--out", default=None)
    p.add_argument("--case", default=None)
    p.add_argument("--data", default=str(ROOT / "bench" / "results" / "data"))
    p.add_argument("--libs", default="pandas,polars")
    opts = p.parse_args()
    cols, s, lookups = load(opts.rows, opts.data)
    if "pandas" in opts.libs:
        run_pandas(opts, cols, s, lookups)
    if "polars" in opts.libs:
        run_polars(opts, cols, s, lookups)
    print(f"pandas {pd.__version__}, polars {pl.__version__}, numpy {np.__version__}, threads {os.cpu_count()}", file=sys.stderr)


if __name__ == "__main__":
    main()
