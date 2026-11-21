//! Kernel-level timings (development tool): fork-join latency, compare,
//! compaction and reductions at cache-resident and larger sizes, by thread count.
//!
//!     kern [rows] [filter]
use std::hint::black_box;
use std::time::Instant;

use dataframes::kernels::{self, CmpOp, aggregate, par, team};
use dataframes::memory::AlignedVec;

/// Best-of-several nanoseconds per call.
fn time<R>(mut f: impl FnMut() -> R) -> f64 {
    let mut iters = 1u64;
    loop {
        let t = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        if t.elapsed().as_micros() >= 2_000 {
            break;
        }
        iters *= 2;
    }
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    best
}

fn report(name: &str, rows: usize, ns: f64) {
    println!("{name:<44} {:>10.2} µs  {:>7.3} ns/row", ns / 1e3, ns / rows as f64);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let filter = args.get(2).cloned().unwrap_or_default();
    let want = |name: &str| filter.is_empty() || filter.split(',').any(|p| name.contains(p));

    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let a: Vec<i64> = (0..rows).map(|_| (next() % 1000) as i64).collect();
    let b: Vec<f64> = (0..rows).map(|_| (next() % 1_000_000) as f64 / 1e6).collect();
    let c: Vec<i32> = (0..rows).map(|_| (next() % 100) as i32).collect();
    println!(
        "rows {rows}  threads {}  one thread streams {} KiB/µs from cache  L2 {} KiB",
        par::threads(),
        par::stream_rate() >> 10,
        kernels::cache::cache_info().l2 >> 10
    );

    if want("fork") {
        for helpers in [1usize, 3, 7] {
            let ns = time(|| team::run(helpers + 1, helpers, true, &|i| {
                black_box(i);
            }));
            println!("fork-join, {} empty tasks, {helpers} helpers       {:>8.0} ns   (hot helpers now: {})", helpers + 1, ns, team::hot_helpers());
        }
    }

    let thread_counts = [1usize, 2, 4, 8];
    if want("mask") {
        for &t in &thread_counts {
            report(&format!("mask i64 < 500, {t} threads"), rows, time(|| kernels::cmp_scalar_mask_with(&a, CmpOp::Lt, 500, t)));
        }
        report("mask f64 < 0.5, 1 thread", rows, time(|| kernels::cmp_scalar_mask_with(&b, CmpOp::Lt, 0.5, 1)));
        report("mask i32 < 50, 1 thread", rows, time(|| kernels::cmp_scalar_mask_with(&c, CmpOp::Lt, 50, 1)));
    }
    if want("compact") {
        for pct in [1i64, 10, 50, 90] {
            let (mask, count) = kernels::cmp_scalar_mask_with(&a, CmpOp::Lt, pct * 10, 1);
            for &t in &thread_counts {
                report(&format!("compact i64 {pct}% ({count}), {t} threads"), rows, time(|| kernels::filter_by_mask_with(&a, &mask, t, Some(false))));
            }
            let (mask, _) = kernels::cmp_scalar_mask_with(&c, CmpOp::Lt, pct as i32, 1);
            report(&format!("compact i32 {pct}%, 1 thread"), rows, time(|| kernels::filter_by_mask_with(&c, &mask, 1, Some(false))));
        }
    }
    if want("fused") {
        for pct in [1i64, 10, 50, 90] {
            for &t in &thread_counts {
                report(&format!("fused filter i64 {pct}%, {t} threads"), rows, time(|| kernels::pipeline::filter_fused(&a, CmpOp::Lt, pct * 10, t, Some(false))));
            }
        }
    }
    if want("reduce") {
        report("sum i64 (auto)", rows, time(|| aggregate::sum(&a)));
        report("sum f64 (auto)", rows, time(|| aggregate::sum(&b)));
        report("min i64 (auto)", rows, time(|| aggregate::min(&a)));
        report("min_max i64 (auto)", rows, time(|| aggregate::min_max(&a)));
        report("min f64 (auto)", rows, time(|| aggregate::min(&b)));
        report("min_max f64 (auto)", rows, time(|| aggregate::min_max(&b)));
    }
    if want("mutate") {
        let mut own: AlignedVec<f64> = dataframes::memory::zeroed_vec(rows);
        own.copy_from_slice(&b);
        report("in place x*c+d (auto)", rows, time(|| kernels::arith::unary_in_place(&mut own, |v| v * 1.0000001 + 0.5)));
    }
}
