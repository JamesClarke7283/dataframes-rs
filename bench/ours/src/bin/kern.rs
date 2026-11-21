//! Kernel-level timings (development tool): fork-join latency, compare,
//! compaction and reductions at cache-resident and larger sizes, by thread count.
//!
//!     kern [rows] [filter]
//!
//! The `duo` section is an experiment about the machine rather than about the
//! library: it splits one sum between two pinned threads with the smallest
//! possible hand-off, which shows what a second core can give for an input
//! that already sits in the first core's cache.
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
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

/// A value on cache lines of its own.
#[repr(align(128))]
struct Line<T>(T);

/// Restrict the calling thread to one CPU (`None`: to all of them).
fn pin_to(cpu: Option<usize>) {
    // SAFETY: a zeroed CPU set is valid, and the call only reads it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        match cpu {
            Some(cpu) => libc::CPU_SET(cpu, &mut set),
            None => (0..libc::CPU_SETSIZE as usize).for_each(|c| libc::CPU_SET(c, &mut set)),
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[inline(never)]
fn sum_slice(v: &[i64]) -> i64 {
    v.iter().fold(0i64, |a, &b| a.wrapping_add(b))
}

/// The data of the `duo` experiment: shared between two threads that take
/// turns, which the type system cannot see.
struct Turns(*mut i64, usize);
// SAFETY: the two threads never touch the data at the same time in a way
// that conflicts; see `duo`.
unsafe impl Sync for Turns {}

/// One sum split between the calling thread (`data[..split]`) and a helper
/// thread (`data[split..]`) that polls one cache line for work and answers
/// on another: nanoseconds per call. With `rewrite`, the calling thread first
/// updates every element, which leaves the whole input in *its* cache only —
/// the opposite of the steady state where each thread keeps its own half.
fn duo(data: &mut [i64], split: usize, caller_cpu: Option<usize>, helper_cpu: Option<usize>, pause: bool, rewrite: bool) -> f64 {
    let turns = Turns(data.as_mut_ptr(), data.len());
    let turns = &turns;
    let go = Line(AtomicU64::new(0));
    let answer = Line((AtomicU64::new(0), AtomicI64::new(0)));
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            pin_to(helper_cpu);
            let mut seen = 0u64;
            loop {
                let turn = loop {
                    let turn = go.0.load(Ordering::Acquire);
                    if turn != seen {
                        break turn;
                    }
                    if pause {
                        std::hint::spin_loop();
                    }
                };
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                seen = turn;
                // SAFETY: between `go` and the answer the caller only reads
                // its own part, and it writes nothing.
                let part = unsafe { std::slice::from_raw_parts(turns.0.add(split), turns.1 - split) };
                answer.0.1.store(sum_slice(part), Ordering::Relaxed);
                answer.0.0.store(turn, Ordering::Release);
            }
        });
        pin_to(caller_cpu);
        let ns = time(|| {
            if rewrite {
                // SAFETY: the helper is idle: the previous turn was answered.
                let all = unsafe { std::slice::from_raw_parts_mut(turns.0, turns.1) };
                all.iter_mut().for_each(|v| *v = v.wrapping_add(1));
            }
            let turn = go.0.fetch_add(1, Ordering::AcqRel) + 1;
            // SAFETY: nobody writes during a turn.
            let mine = sum_slice(unsafe { std::slice::from_raw_parts(turns.0, split) });
            while answer.0.0.load(Ordering::Acquire) != turn {}
            mine.wrapping_add(answer.0.1.load(Ordering::Relaxed))
        });
        stop.store(true, Ordering::Relaxed);
        go.0.fetch_add(1, Ordering::AcqRel);
        ns
    })
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
    if want("duo") {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        let siblings = |cpu: usize| {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            std::fs::read_to_string(path).unwrap_or_default().trim().to_string()
        };
        for cpu in 0..cpus {
            println!("cpu {cpu} shares a core with {}", siblings(cpu));
        }
        let mut d = a.clone();
        let half = rows / 2;
        report("one thread, not pinned", rows, time(|| sum_slice(&d)));
        report("team, 2 tasks, 1 helper", rows, time(|| {
            team::run(2, 1, true, &|i| {
                black_box(sum_slice(&d[i * half..(i + 1) * half]));
            })
        }));
        report("team, 4 tasks, 3 helpers", rows, time(|| {
            team::run(4, 3, true, &|i| {
                black_box(sum_slice(&d[i * (half / 2)..(i + 1) * (half / 2)]));
            })
        }));
        report("duo, nobody pinned", rows, duo(&mut d, half, None, None, true, false));
        pin_to(Some(0));
        report("one thread on cpu 0", rows, time(|| sum_slice(&d)));
        report("rewrite, then one thread on cpu 0", rows, time(|| {
            d.iter_mut().for_each(|v| *v = v.wrapping_add(1));
            sum_slice(&d)
        }));
        for cpu in 1..cpus {
            let place = if siblings(cpu) == siblings(0) { "same core" } else { "other core" };
            report(&format!("duo, cpu 0 + cpu {cpu} ({place})"), rows, duo(&mut d, half, Some(0), Some(cpu), true, false));
            report(&format!("duo without pause, cpu 0 + cpu {cpu}"), rows, duo(&mut d, half, Some(0), Some(cpu), false, false));
            report(&format!("rewrite, then duo, cpu 0 + cpu {cpu}"), rows, duo(&mut d, half, Some(0), Some(cpu), true, true));
            report(&format!("hand-off only, cpu 0 + cpu {cpu}"), rows, duo(&mut [], 0, Some(0), Some(cpu), true, false));
        }
        pin_to(None);
    }
    if want("mutate") {
        let mut own: AlignedVec<f64> = dataframes::memory::zeroed_vec(rows);
        own.copy_from_slice(&b);
        report("in place x*c+d (auto)", rows, time(|| kernels::arith::unary_in_place(&mut own, |v| v * 1.0000001 + 0.5)));
    }
}
