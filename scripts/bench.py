#!/usr/bin/env python3
"""Run the comparison benchmarks and render the results.

    scripts/bench.py run     [--rows 100000,1000000] [--rounds 3] [--libs ours,polars,arrow,python]
    scripts/bench.py report  [--gate]

`run` executes one driver per library, several rounds, round-robin, so that
frequency scaling and noisy neighbours hit every library alike, and appends
raw records to a JSON-lines file. `report` reduces them to one number per
(library, case, size) - the best median over rounds and over the variants a
library offers for that case - checks that all libraries computed the same
thing, and writes a Markdown table, a JSON summary and an SVG chart. With
`--readme FILE` it also refreshes the block between the `bench:start` and
`bench:end` markers of that file.

A comparison counts as a win when dataframes is more than `--tie` (default
5 %, about what identical code varies by between two processes on a shared
cloud machine) faster than the fastest competitor, as level when the two are
within that band of each other, and as a loss otherwise. `--gate` fails on losses
and on results that disagree between libraries.

Only the Python standard library is used.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"

DRIVERS = {
    "ours": {"bin": "bench-ours", "lib": "dataframes"},
    "polars": {"bin": "bench-polars", "lib": "polars"},
    "arrow": {"bin": "bench-arrow", "lib": "arrow-rs"},
    # One Python process reports two libraries: pandas and the Polars wheel.
    "python": {"script": ROOT / "bench" / "python" / "bench.py", "lib": None},
}

# Display order and names.
LIBS = ["dataframes", "polars", "polars-py", "arrow-rs", "pandas"]
LIB_TITLES = {
    "dataframes": "dataframes",
    "polars": "Polars (Rust crate)",
    "polars-py": "Polars (Python wheel)",
    "arrow-rs": "Arrow (arrow-rs)",
    "pandas": "pandas",
}

CASE_TITLES = {
    "mask_i64": "compare i64 column to scalar -> mask",
    "filter_i64_1pct": "filter i64 column, 1% selected",
    "filter_i64_10pct": "filter i64 column, 10% selected",
    "filter_i64_50pct": "filter i64 column, 50% selected",
    "filter_i64_90pct": "filter i64 column, 90% selected",
    "filter_f64_50pct": "filter f64 column, 50% selected",
    "filter_i32_50pct": "filter i32 column, 50% selected",
    "filter_f32_50pct": "filter f32 column, 50% selected",
    "frame_filter_10pct": "filter 6-column frame by mask, 10%",
    "frame_filter_50pct": "filter 6-column frame by mask, 50%",
    "frame_filter_str_50pct": "filter frame with a string column, 50%",
    "frame_filter_expr_50pct": "filter 6-column frame by expression, 50%",
    "frame_filter_and_25pct": "filter frame by `a < x & b < y`, 25%",
    "filter_then_sum": "sum of one column over filtered rows",
    "slice_frame": "slice 6-column frame (zero-copy)",
    "head_frame": "head(10) of 6-column frame",
    "slice_series": "slice one column (zero-copy)",
    "get_random_1k": "the values at 1000 random positions of a column",
    "cell_random_1k": "1000 random cell look-ups by column name",
    "sum_i64": "sum of i64 column",
    "sum_f64": "sum of f64 column",
    "mean_f64": "mean of f64 column",
    "min_max_f64": "min and max of f64 column",
    "min_max_i64": "min and max of i64 column",
    "mutate_f64_inplace": "in-place update x = x*c + d",
}


def fmt_ns(ns):
    if ns is None:
        return "-"
    if ns < 1e3:
        return f"{ns:.1f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.3f} s"


def fmt_rows(n):
    return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n // 1000}K" if n % 1000 == 0 else str(n)


# --------------------------------------------------------------------------- run


def run(args):
    target = Path(args.target_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.append and out.exists():
        out.unlink()
    libs = args.libs.split(",")
    rows_list = [int(r.replace("_", "")) for r in args.rows.split(",")]

    def command(name):
        driver = DRIVERS[name]
        if "bin" in driver:
            path = target / "release" / driver["bin"]
            return [str(path)] if path.exists() else None
        return [args.python, str(driver["script"])]

    def invoke(name, rows, case, label):
        cmd = command(name)
        if cmd is None:
            print(f"skip {name}: not built", file=sys.stderr)
            return
        cmd += ["--rows", str(rows), "--time-ms", str(args.time_ms), "--out", str(out)]
        if case:
            cmd += ["--case", case]
        print(f"[rows={rows} {label}] {name}", file=sys.stderr, flush=True)
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            sys.exit(f"driver {name} failed with status {proc.returncode}")
        if args.verbose:
            print(proc.stderr, file=sys.stderr, end="")

    selected = [c for c in CASE_TITLES if not args.case or any(p in c for p in args.case.split(","))]
    for rows in rows_list:
        for rnd in range(args.rounds):
            label = f"round={rnd + 1}/{args.rounds}"
            if args.per_case:
                # One case at a time across all libraries: on machines whose
                # speed drifts within seconds (thermal limits, noisy
                # neighbours) every library then meets the same conditions.
                for case in selected:
                    for name in libs:
                        invoke(name, rows, "=" + case, f"{label} {case}")
            else:
                for name in libs:
                    invoke(name, rows, args.case, label)


# ------------------------------------------------------------------------ report


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def reduce(records):
    """-> {rows: {case: {lib: {"ns": time, "variant": name, "check": value, "rounds": [...]}}}}

    Every round is a fresh process. The time of a spelling ("variant") of a
    case is the mean, over the rounds, of the median that each round
    measured; a library is represented by its fastest spelling. The mean,
    not the best round: the same binary can be half as fast again from one
    process to the next (address-space layout, where its buffers land), and
    what a user gets is the average of that, not the luckiest draw. The
    records of one (library, case, size) are in round order.
    """
    rounds = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in records:
        case, _, variant = r["case"].partition("@")
        per_variant = rounds[r["rows"]][case][r["lib"]]
        per_variant.setdefault(variant, {"times": [], "check": r["check"]})["times"].append(r["median_ns"])
    best = defaultdict(lambda: defaultdict(dict))
    for rows, cases in rounds.items():
        for case, libs in cases.items():
            for lib, variants in libs.items():
                variant, entry = min(variants.items(), key=lambda kv: sum(kv[1]["times"]) / len(kv[1]["times"]))
                times = entry["times"]
                best[rows][case][lib] = {
                    "ns": sum(times) / len(times),
                    "variant": variant,
                    "check": entry["check"],
                    "rounds": times,
                }
    return best


def checks_agree(entries):
    """All libraries must report the same check value (they did the same work)."""
    values = [e["check"] for e in entries.values()]
    lo, hi = min(values), max(values)
    return hi - lo <= 1e-6 * max(1.0, abs(hi), abs(lo))


def summarise(best):
    summary = []
    for rows in sorted(best):
        for case in CASE_TITLES:
            entries = best[rows].get(case)
            if not entries or "dataframes" not in entries:
                continue
            ours = entries["dataframes"]["ns"]
            rivals = {lib: e["ns"] for lib, e in entries.items() if lib != "dataframes"}
            fastest_rival = min(rivals, key=rivals.get) if rivals else None
            summary.append(
                {
                    "rows": rows,
                    "case": case,
                    "ns": {lib: e["ns"] for lib, e in entries.items()},
                    "variant": {lib: e["variant"] for lib, e in entries.items() if e["variant"]},
                    "fastest_rival": fastest_rival,
                    "speedup": (rivals[fastest_rival] / ours) if fastest_rival else None,
                    "rounds": {lib: e["rounds"] for lib, e in entries.items()},
                    "checks_agree": checks_agree(entries),
                }
            )
    return summary


def verdict(entry, tie):
    """'win', 'level' or 'loss' for one comparison: more than `tie` faster
    than the fastest competitor, within `tie` of it, or more than `tie` slower."""
    speedup = entry["speedup"]
    if speedup >= 1.0 + tie:
        return "win"
    return "level" if speedup > 1.0 - tie else "loss"


def headline(summary, tie):
    speedups = [s["speedup"] for s in summary if s["speedup"]]
    kinds = [verdict(s, tie) for s in summary if s["speedup"]]
    wins, level, losses = kinds.count("win"), kinds.count("level"), kinds.count("loss")
    geo = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else float("nan")
    text = f"**dataframes is the fastest library in {wins} of {len(speedups)} comparisons**"
    if level:
        text += f", level with the fastest (within {tie:.0%}) in {level}"
    text += f" and slower in {losses}" if losses else " and slower in none"
    text += (f". Geometric-mean speed-up over the fastest competitor: **{geo:.2f}x** "
             f"(smallest {min(speedups):.2f}x, largest {max(speedups):.2f}x).")
    return text


def machine_line(meta):
    line = f"Machine: {meta['cpu']} · {meta['threads']} threads · {meta['os']} · {meta['when']}"
    if meta.get("versions"):
        line += "  \nVersions: " + " · ".join(f"{k} {v}" for k, v in meta["versions"].items())
    return line


def markdown(summary, meta, tie):
    libs = [lib for lib in LIBS if any(lib in s["ns"] for s in summary)]
    lines = []
    lines.append(headline(summary, tie))
    lines.append("")
    lines.append(machine_line(meta))
    lines.append("")
    lines.append("Times are the mean over the rounds of each round's median (lower is better). "
                 "The last column is the fastest competitor's time divided by ours.")
    for rows in sorted({s["rows"] for s in summary}):
        lines.append("")
        lines.append(f"### {fmt_rows(rows)} rows")
        lines.append("")
        lines.append("| operation | " + " | ".join(LIB_TITLES[l] for l in libs) + " | vs fastest competitor |")
        lines.append("|---|" + "---:|" * (len(libs) + 1))
        for s in (s for s in summary if s["rows"] == rows):
            cells = []
            for lib in libs:
                ns = s["ns"].get(lib)
                text = fmt_ns(ns)
                if lib == "dataframes":
                    text = f"**{text}**"
                cells.append(text)
            if s["speedup"] is None:
                cell = "-"
            else:
                mark = "" if s["checks_agree"] else " ⚠ results differ"
                note = {"win": "", "level": " level", "loss": " slower"}[verdict(s, tie)]
                cell = f"**{s['speedup']:.2f}x**{note} ({LIB_TITLES[s['fastest_rival']]}){mark}"
            lines.append(f"| {CASE_TITLES[s['case']]} | " + " | ".join(cells) + f" | {cell} |")
    return "\n".join(lines) + "\n"


def svg_chart(summary, rows, tie):
    """Horizontal bars: speed-up over the fastest competitor, log scale."""
    items = [s for s in summary if s["rows"] == rows and s["speedup"]]
    if not items:
        return None
    label_w, bar_w, row_h, top = 300, 420, 22, 56
    height = top + row_h * len(items) + 34
    width = label_w + bar_w + 90
    hi = max(4.0, max(s["speedup"] for s in items))
    lo = min(0.5, min(s["speedup"] for s in items))
    scale = lambda x: (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) * bar_w
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="12">',
        "<style>.t{fill:#1f2328}.m{fill:#59636e}.g{stroke:#d1d9e0}.b{fill:#1f6feb}.l{fill:#cf222e}.e{fill:#9a6700}"
        "@media(prefers-color-scheme:dark){.t{fill:#f0f6fc}.m{fill:#9198a1}.g{stroke:#3d444d}.b{fill:#4493f8}.l{fill:#f85149}.e{fill:#d29922}}</style>",
        f'<text class="t" x="0" y="18" font-size="15" font-weight="600">Speed-up over the fastest competitor - {fmt_rows(rows)} rows</text>',
        f'<text class="m" x="0" y="36">competitor time / dataframes time, log scale; 1x = parity</text>',
    ]
    tick = 0.5
    ticks = []
    while tick <= hi * 1.001:
        if tick >= lo:
            ticks.append(tick)
        tick *= 2
    for t in ticks:
        x = label_w + scale(t)
        out.append(f'<line class="g" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 28}" stroke-width="{2 if t == 1 else 1}"/>')
        out.append(f'<text class="m" x="{x:.1f}" y="{height - 12}" text-anchor="middle">{t:g}x</text>')
    one = label_w + scale(1.0)
    for i, s in enumerate(items):
        y = top + i * row_h
        x = label_w + scale(s["speedup"])
        cls = {"win": "b", "level": "e", "loss": "l"}[verdict(s, tie)]
        out.append(f'<text class="t" x="{label_w - 8}" y="{y + 14}" text-anchor="end">{CASE_TITLES[s["case"]].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</text>')
        out.append(f'<rect class="{cls}" x="{min(one, x):.1f}" y="{y + 3}" width="{max(abs(x - one), 2.0):.1f}" height="{row_h - 8}" rx="2"/>')
        out.append(f'<text class="t" x="{max(one, x) + 6:.1f}" y="{y + 14}">{s["speedup"]:.2f}x</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def versions(libs):
    """Versions of the libraries that took part, read from Cargo.lock and the Python environment."""
    found = {}
    try:
        lock = (ROOT / "Cargo.lock").read_text()
    except OSError:
        lock = ""
    for lib, package in (("polars", "polars"), ("arrow-rs", "arrow-array")):
        marker = f'name = "{package}"\nversion = "'
        if lib in libs and marker in lock:
            found[LIB_TITLES[lib]] = lock.split(marker, 1)[1].split('"', 1)[0]
    for lib, module in (("polars-py", "polars"), ("pandas", "pandas")):
        if lib in libs:
            try:
                found[LIB_TITLES[lib]] = __import__(module).__version__
            except Exception:  # the report may run where the wheels are not installed
                pass
    return found


def readme_block(summary, meta, tie, name, results_dir):
    """The generated part of the README: headline, charts, link to the tables."""
    rel = results_dir.resolve().relative_to(ROOT).as_posix()
    lines = [headline(summary, tie), "", machine_line(meta), ""]
    for rows in sorted({s["rows"] for s in summary}):
        lines.append(f"![Speed-up over the fastest competitor, {fmt_rows(rows)} rows]({rel}/{name}-{fmt_rows(rows)}.svg)")
        lines.append("")
    lines.append(f"Every number behind the charts, per library: [{rel}/{name}.md]({rel}/{name}.md).")
    return "\n".join(lines)


def update_readme(path, block):
    start, end = "<!-- bench:start -->", "<!-- bench:end -->"
    text = Path(path).read_text()
    if start not in text or end not in text:
        sys.exit(f"{path} has no {start} ... {end} block")
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    Path(path).write_text(f"{head}{start}\n{block}\n{end}{tail}")


def report(args):
    records = load(args.input)
    summary = summarise(reduce(records))
    if not summary:
        sys.exit("no results to report")
    meta = {
        "cpu": cpu_name(),
        "threads": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "when": subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M UTC"], capture_output=True, text=True).stdout.strip(),
        "versions": versions({lib for s in summary for lib in s["ns"]}),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name
    for s in summary:
        s["verdict"] = verdict(s, args.tie) if s["speedup"] else None
    md = markdown(summary, meta, args.tie)
    (out_dir / f"{name}.md").write_text(md)
    (out_dir / f"{name}.json").write_text(json.dumps({"meta": meta, "tie": args.tie, "results": summary}, indent=1) + "\n")
    for rows in sorted({s["rows"] for s in summary}):
        chart = svg_chart(summary, rows, args.tie)
        if chart:
            (out_dir / f"{name}-{fmt_rows(rows)}.svg").write_text(chart)
    if args.readme:
        update_readme(args.readme, readme_block(summary, meta, args.tie, name, out_dir))
    print(md)

    losses = [s for s in summary if s["verdict"] == "loss"]
    level = [s for s in summary if s["verdict"] == "level"]
    mismatches = [s for s in summary if not s["checks_agree"]]
    for s in mismatches:
        print(f"CHECK MISMATCH {fmt_rows(s['rows'])} {s['case']}", file=sys.stderr)
    for s in level:
        print(f"LEVEL {fmt_rows(s['rows'])} {s['case']}: {s['speedup']:.2f}x vs {s['fastest_rival']}", file=sys.stderr)
    for s in losses:
        print(f"SLOWER {fmt_rows(s['rows'])} {s['case']}: {s['speedup']:.2f}x vs {s['fastest_rival']}", file=sys.stderr)
    if args.gate and (losses or mismatches):
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rows", default="100000,1000000,10000000")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--time-ms", type=int, default=300)
    r.add_argument("--libs", default="ours,polars,arrow")
    r.add_argument("--case", default=None)
    r.add_argument("--out", default=str(RESULTS / "raw.jsonl"))
    r.add_argument("--append", action="store_true")
    r.add_argument("--target-dir", default=str(ROOT / "target" / "native"))
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--per-case", action="store_true", help="interleave the libraries case by case (slower, fairer)")
    r.set_defaults(func=run)
    t = sub.add_parser("report")
    t.add_argument("--input", default=str(RESULTS / "raw.jsonl"))
    t.add_argument("--out-dir", default=str(RESULTS))
    t.add_argument("--name", default="latest")
    t.add_argument("--gate", action="store_true", help="exit 1 if dataframes is slower than a competitor anywhere")
    t.add_argument("--tie", type=float, default=0.05, help="relative difference below which two libraries count as level")
    t.add_argument("--readme", default=None, help="README whose bench:start/bench:end block to refresh")
    t.set_defaults(func=report)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
