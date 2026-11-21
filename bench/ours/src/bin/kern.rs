//! Kernel-level timings (development tool): fork-join latency, compare,
//! compaction and reductions at cache-resident and larger sizes, by thread count.
//!
//!     kern [rows] [filter]
//!
//! The `duo` section is an experiment about the machine rather than about the
//! library: it splits one sum between two pinned threads with the smallest
//! possible hand-off, which shows what a second core can give for an input
//! that already sits in the first core's cache.
// The allocator of the benchmark drivers, so that "a plain vector" here is
// what a competitor's buffer is there.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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

/// A thousand look-ups through the column's accessor, the way the benchmark
/// driver spells them.
#[inline(never)]
fn column_gets(column: &dataframes::prelude::Int64Chunked, positions: &[u32]) -> i64 {
    let mut acc = 0i64;
    for &i in positions {
        if let Some(v) = column.get(black_box(i as usize)) {
            acc = acc.wrapping_add(v);
        }
    }
    acc
}

#[inline(never)]
fn column_gets_or(column: &dataframes::prelude::Int64Chunked, positions: &[u32]) -> i64 {
    let mut acc = 0i64;
    for &i in positions {
        acc = acc.wrapping_add(column.get(black_box(i as usize)).unwrap_or(0));
    }
    acc
}

fn report(name: &str, rows: usize, ns: f64) {
    let learned: Vec<String> = par::learned_decisions()
        .iter()
        .map(|&(bytes, split, period, kernel, other)| {
            format!("{}K:{}/{period} {kernel}|{other}", bytes >> 10, if split { "split" } else { "one" })
        })
        .collect();
    println!("{name:<44} {:>10.2} µs  {:>7.3} ns/row   {}", ns / 1e3, ns / rows as f64, learned.join(" "));
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
    // Aligned like the columns of the library, so that no vector load
    // straddles two cache lines.
    fn aligned<T: Copy + dataframes::memory::Zeroable>(v: Vec<T>) -> AlignedVec<T> {
        let mut out: AlignedVec<T> = dataframes::memory::zeroed_vec(v.len());
        out.copy_from_slice(&v);
        out
    }
    let a = aligned((0..rows).map(|_| (next() % 1000) as i64).collect());
    let b = aligned((0..rows).map(|_| (next() % 1_000_000) as f64 / 1e6).collect());
    let c = aligned((0..rows).map(|_| (next() % 100) as i32).collect());
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
        // The pattern in which a split loses: the calling thread writes the
        // whole column, then asks for a reduction. The policy has to notice.
        let mut fresh = aligned(b.to_vec());
        let mut rewrite_then_sum = || {
            fresh.iter_mut().for_each(|v| *v += 1.0);
            aggregate::sum(&fresh)
        };
        report("rewrite, then sum f64, first calls", rows, time(&mut rewrite_then_sum));
        // Long enough for the policy to have tried the other way at least once.
        (0..10_000).for_each(|_| {
            black_box(rewrite_then_sum());
        });
        report("rewrite, then sum f64, settled", rows, time(&mut rewrite_then_sum));
        report("sum f64 again, first calls", rows, time(|| aggregate::sum(&b)));
        (0..10_000).for_each(|_| {
            black_box(aggregate::sum(&b));
        });
        report("sum f64 again, settled", rows, time(|| aggregate::sum(&b)));
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
        let mut d = a.to_vec();
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
    if want("pages") {
        // Does it matter for a scan whether a column that fits in the cache
        // lies in one huge page or in a few hundred small ones? A block of
        // 2 MiB comes from the allocator's huge-page class; the first `rows`
        // elements of it are the column.
        let mut big: AlignedVec<i64> = dataframes::memory::zeroed_vec((2usize << 20) / 8);
        let n = rows.min(big.len());
        big[..n].copy_from_slice(&a[..n]);
        let huge = &big[..n];
        for round in 0..2 {
            report(&format!("mask i64, 1 thread, small pages ({round})"), rows, time(|| kernels::cmp_scalar_mask_with(&a, CmpOp::Lt, 500, 1)));
            report(&format!("mask i64, 1 thread, one huge page ({round})"), rows, time(|| kernels::cmp_scalar_mask_with(huge, CmpOp::Lt, 500, 1)));
            report(&format!("plain sum, 1 thread, small pages ({round})"), rows, time(|| sum_slice(black_box(&a))));
            report(&format!("plain sum, 1 thread, one huge page ({round})"), rows, time(|| sum_slice(black_box(huge))));
        }
    }
    if want("lookup") {
        // A thousand point look-ups at fixed random positions, the same in
        // every call: into a column from the library's allocator (huge pages
        // from 2 MiB), into the same data in a plain vector (small pages),
        // and into a plain mapping that asked for huge pages.
        let positions: Vec<u32> = (0..1000u64).map(|i| ((i * 0x9E37_79B9_7F4A_7C15) >> 11) as u32 % rows as u32).collect();
        let plain: Vec<i64> = a.to_vec();
        // One copy of the loop for all, so that where the compiler put it
        // (which alone can cost a third) is the same for all.
        #[inline(never)]
        fn look(data: &[i64], positions: &[u32]) -> i64 {
            let mut acc = 0i64;
            for &i in positions {
                acc = acc.wrapping_add(data.get(black_box(i as usize)).copied().unwrap_or(0));
            }
            acc
        }
        /// The same without the detour of `unwrap_or`: the load can be part
        /// of the addition.
        #[inline(never)]
        fn look_some(data: &[i64], positions: &[u32]) -> i64 {
            let mut acc = 0i64;
            for &i in positions {
                if let Some(&v) = data.get(black_box(i as usize)) {
                    acc = acc.wrapping_add(v);
                }
            }
            acc
        }
        /// `look_some` with the miss marked as the rare case.
        #[inline(never)]
        fn look_cold(data: &[i64], positions: &[u32]) -> i64 {
            let mut acc = 0i64;
            for &i in positions {
                match data.get(black_box(i as usize)) {
                    Some(&v) => acc = acc.wrapping_add(v),
                    None => std::hint::cold_path(),
                }
            }
            acc
        }
        /// `look` with the miss marked as the rare case.
        #[inline(never)]
        fn look_or_cold(data: &[i64], positions: &[u32]) -> i64 {
            let mut acc = 0i64;
            for &i in positions {
                let v = match data.get(black_box(i as usize)) {
                    Some(&v) => v,
                    None => {
                        std::hint::cold_path();
                        0
                    }
                };
                acc = acc.wrapping_add(v);
            }
            acc
        }
        /// Kilobytes of `data` that sit in huge pages, from the kernel's map.
        fn huge_kb(data: &[i64]) -> usize {
            let (lo, hi) = (data.as_ptr() as usize, data.as_ptr() as usize + size_of_val(data));
            let Ok(maps) = std::fs::read_to_string("/proc/self/smaps") else { return 0 };
            let (mut inside, mut total) = (false, 0);
            for line in maps.lines() {
                let mut parts = line.split_whitespace();
                let first = parts.next().unwrap_or("");
                if let Some((start, end)) = first.split_once('-')
                    && let (Ok(start), Ok(end)) = (usize::from_str_radix(start, 16), usize::from_str_radix(end, 16))
                {
                    inside = start < hi && end > lo;
                } else if inside && first == "AnonHugePages:" {
                    total += parts.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
                }
            }
            total
        }
        // A mapping of its own, aligned to 2 MiB, with the request for huge
        // pages that the library's allocator makes.
        let bytes = (rows * 8).next_multiple_of(2 << 20);
        // SAFETY: an anonymous private mapping; checked below; never unmapped.
        let advised: &mut [i64] = unsafe {
            let raw = libc::mmap(std::ptr::null_mut(), bytes + (2 << 20), libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
            assert!(raw != libc::MAP_FAILED);
            let start = (raw as usize).next_multiple_of(2 << 20) as *mut libc::c_void;
            libc::madvise(start, bytes, libc::MADV_HUGEPAGE);
            std::slice::from_raw_parts_mut(start.cast::<i64>(), rows)
        };
        advised.copy_from_slice(&a);
        println!(
            "library column at {:p} ({} KiB huge), plain vector at {:p} ({} KiB huge), advised mapping at {:p} ({} KiB huge)",
            a.as_ptr(), huge_kb(&a), plain.as_ptr(), huge_kb(&plain), advised.as_ptr(), huge_kb(advised)
        );
        // The same column again, starting somewhere in the middle of a huge
        // page instead of at its beginning.
        let mut shifted_block: AlignedVec<i64> = dataframes::memory::zeroed_vec(rows + (1 << 18));
        let shift = 123_457.min(shifted_block.len() - rows);
        shifted_block[shift..shift + rows].copy_from_slice(&a);
        let shifted = &shifted_block[shift..shift + rows];
        // A column short enough for every looked-up line to stay in the
        // first-level cache: what the loop costs when nothing misses.
        let tiny = &a[..rows.min(2048)];
        let tiny_positions: Vec<u32> = positions.iter().map(|&p| p % tiny.len() as u32).collect();
        for round in 0..2 {
            report(&format!("1000 look-ups, library column ({round})"), 1000, time(|| look(black_box(&a), &positions)));
            report(&format!("1000 look-ups, plain vector ({round})"), 1000, time(|| look(black_box(&plain), &positions)));
            report(&format!("1000 look-ups, advised mapping ({round})"), 1000, time(|| look(black_box(advised), &positions)));
            report(&format!("1000 look-ups, library column, shifted ({round})"), 1000, time(|| look(black_box(shifted), &positions)));
            report(&format!("1000 look-ups, 2048-row column ({round})"), 1000, time(|| look(black_box(tiny), &tiny_positions)));
            report(&format!("1000 look-ups, library column, if-let ({round})"), 1000, time(|| look_some(black_box(&a), &positions)));
            report(&format!("1000 look-ups, library column, if-let, cold miss ({round})"), 1000, time(|| look_cold(black_box(&a), &positions)));
            report(&format!("1000 look-ups, library column, or-zero, cold miss ({round})"), 1000, time(|| look_or_cold(black_box(&a), &positions)));
        }
    }
    if want("take") {
        // The values at a thousand positions in one call: the kernel alone,
        // and the pieces around it that a column-level `take` adds.
        use dataframes::prelude::*;
        let positions: Vec<u32> = (0..1000u64).map(|i| ((i * 0x9E37_79B9_7F4A_7C15) >> 11) as u32 % rows as u32).collect();
        let column = Int64Chunked::from_vec("e".into(), a.to_vec());
        let index = IdxCa::from_vec("positions".into(), positions.clone());
        for round in 0..2 {
            report(&format!("take kernel, 1000 of i64 ({round})"), 1000, time(|| kernels::arith::take(&a[..], black_box(&positions))));
            report(&format!("take kernel, 1000 of i32 ({round})"), 1000, time(|| kernels::arith::take(&c[..], black_box(&positions))));
            report(&format!("largest position alone ({round})"), 1000, time(|| black_box(&positions).iter().copied().fold(0u32, u32::max)));
            report(&format!("column take ({round})"), 1000, time(|| column.take(black_box(&index)).unwrap()));
            report(&format!("column take, then sum ({round})"), 1000, time(|| column.take(black_box(&index)).unwrap().sum()));
            report(&format!("1000 x column get ({round})"), 1000, time(|| column_gets(black_box(&column), &positions)));
            report(&format!("1000 x column get, or zero ({round})"), 1000, time(|| column_gets_or(black_box(&column), &positions)));
        }
    }
    if want("filter") {
        // The way a column filters itself: a mask, then a compaction, each
        // asking the splitting policy on its own.
        for pct in [1i64, 10, 50] {
            report(&format!("mask, then compact i64 {pct}% (auto)"), rows, time(|| {
                let (mask, _) = kernels::cmp_scalar_mask(&a, CmpOp::Lt, pct * 10);
                kernels::filter_by_mask(&a, &mask)
            }));
        }
        report("mask i64 (auto)", rows, time(|| kernels::cmp_scalar_mask(&a, CmpOp::Lt, 500)));
        report("sum as f64 (auto)", rows, time(|| aggregate::sum_f64(&b)));
    }
    if want("mutate") {
        let mut own: AlignedVec<f64> = dataframes::memory::zeroed_vec(rows);
        own.copy_from_slice(&b);
        report("in place x*c+d (auto)", rows, time(|| kernels::arith::unary_in_place(&mut own, |v| v * 1.0000001 + 0.5)));
    }
}
