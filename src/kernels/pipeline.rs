//! Ordered-commit pipeline: parallel, single-pass, contiguous output.
//!
//! Filtering in parallel into *one* buffer has a chicken-and-egg problem: a
//! worker cannot know where its output goes until every earlier row range has
//! been counted. The usual answers are to count everything first (which reads
//! the predicate column from memory twice) or to give every worker its own
//! output chunk (which leaves the result fragmented).
//!
//! This module does neither. The rows are cut into cache-sized blocks and each
//! worker repeats three steps:
//!
//! 1. **claim** the next block and **scan** it: evaluate the predicate into a
//!    bitmask and count the matches. The block is now in the worker's L2.
//! 2. **commit**: wait until all earlier blocks have committed, read the
//!    running output offset, add this block's count to it, and pass the turn
//!    on. This is the only serial section, and it is two atomic operations
//!    long.
//! 3. **emit**: compact the block — still cache-resident — into the output at
//!    the offset just obtained. Emits of different blocks write disjoint
//!    ranges and overlap freely in time.
//!
//! Blocks are claimed in increasing order and every scan takes about the same
//! time, so a worker rarely waits at step 2 for longer than a few hundred
//! nanoseconds. The result is what a sequential filter would produce, each
//! input byte crosses the memory bus once, and all cores share the work.

use std::mem::DropGuard;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::{CmpOp, Kernel, cache, cmp_scalar_mask, compact, filter_by_mask_with, par};
use crate::memory::{AlignedVec, ColumnAlloc};

/// Keeps a hot atomic on its own pair of cache lines (128 bytes covers the
/// adjacent-line prefetcher as well).
#[repr(align(128))]
struct Isolated<T>(T);

struct Shared {
    /// Next block to hand out.
    next: Isolated<AtomicUsize>,
    /// Index of the block whose turn it is to commit.
    turn: Isolated<AtomicUsize>,
    /// Output offset of the block whose turn it is. Written only by the turn
    /// holder, published by the `Release` store of `turn`.
    offset: AtomicUsize,
    /// A worker unwound; everyone else should stop waiting.
    abort: AtomicBool,
    /// Some block did not fit below `limit`; later blocks are counted but not emitted.
    overflow: AtomicBool,
}

/// Outcome of [`scan_then_emit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Committed {
    /// Sum of the counts returned by `scan` over all blocks.
    pub total: usize,
    /// `false` if `total` exceeded `limit`; in that case `emit` ran only for
    /// the blocks that fit entirely below `limit`.
    pub complete: bool,
}

/// Rows per block for a payload of `row_bytes` bytes per row: half of L2,
/// a multiple of 64, between 1 024 and 65 536 rows.
pub fn block_rows(row_bytes: usize) -> usize {
    let rows = cache::cache_info().l2 / 2 / row_bytes.max(1);
    (rows / 64 * 64).clamp(1024, MAX_BLOCK_ROWS)
}

/// Rows per block when several columns are compacted by one mask.
///
/// Most of those columns are not part of the predicate, so nothing is gained
/// by keeping a block cache-resident between scan and emit. What matters is
/// that each column of a block is a long sequential run for the hardware
/// prefetchers, and that every thread still gets a handful of blocks.
pub fn frame_block_rows(rows: usize, tasks: usize) -> usize {
    (rows / (4 * tasks.max(1)) / 64 * 64).clamp(1024, MAX_BLOCK_ROWS)
}

/// Upper bound of [`block_rows`]; scratch masks are sized for it.
pub const MAX_BLOCK_ROWS: usize = 65_536;

/// Run the claim / scan / commit / emit loop described in the
/// [module documentation](self) over `0..rows`.
///
/// * `scan(range, scratch)` examines a block and returns how many output slots
///   it needs. It may leave notes (typically a bitmask) in `scratch`.
/// * `emit(range, scratch, offset, count)` writes the block's output at
///   `offset..offset + count`. Offsets of successive blocks are contiguous and
///   start at zero. It is skipped for blocks with `count == 0`.
///
/// `scratch` values are created per worker by `make_scratch`. `limit` is the
/// capacity of the output: blocks that would end beyond it are not emitted and
/// the result reports `complete: false`.
///
/// # Panics
/// Propagates a panic from `scan` or `emit` after all workers have stopped.
pub fn scan_then_emit<S, Scan, Emit>(
    rows: usize,
    block_rows: usize,
    tasks: usize,
    limit: usize,
    make_scratch: impl Fn() -> S + Sync,
    scan: Scan,
    emit: Emit,
) -> Committed
where
    Scan: Fn(Range<usize>, &mut S) -> usize + Sync,
    Emit: Fn(Range<usize>, &mut S, usize, usize) + Sync,
{
    assert!(block_rows > 0, "block_rows must be positive");
    let n_blocks = rows.div_ceil(block_rows);
    let shared = Shared {
        next: Isolated(AtomicUsize::new(0)),
        turn: Isolated(AtomicUsize::new(0)),
        offset: AtomicUsize::new(0),
        abort: AtomicBool::new(false),
        overflow: AtomicBool::new(false),
    };

    let worker = || {
        // If `scan` or `emit` unwinds, release the workers waiting for our turn.
        let on_unwind = DropGuard::new((), |()| shared.abort.store(true, Ordering::Release));
        let mut scratch = make_scratch();
        loop {
            let block = shared.next.0.fetch_add(1, Ordering::Relaxed);
            if block >= n_blocks {
                break;
            }
            let range = block * block_rows..((block + 1) * block_rows).min(rows);
            let count = scan(range.clone(), &mut scratch);

            // Wait for our turn. Earlier blocks were claimed before ours and
            // take about as long to scan, so this loop is usually short.
            let mut spins = 0u32;
            while shared.turn.0.load(Ordering::Acquire) != block {
                if shared.abort.load(Ordering::Relaxed) {
                    DropGuard::dismiss(on_unwind);
                    return;
                }
                spins += 1;
                if spins < 256 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
            // We hold the turn: nobody else touches `offset` until we pass it on.
            let offset = shared.offset.load(Ordering::Relaxed);
            let end = offset + count;
            let fits = end <= limit && !shared.overflow.load(Ordering::Relaxed);
            if !fits {
                shared.overflow.store(true, Ordering::Relaxed);
            }
            shared.offset.store(end, Ordering::Relaxed);
            shared.turn.0.store(block + 1, Ordering::Release);

            if fits && count > 0 {
                emit(range, &mut scratch, offset, count);
            }
        }
        DropGuard::dismiss(on_unwind);
    };

    if tasks <= 1 || n_blocks <= 1 {
        worker();
    } else {
        // One loop per thread; a thread that never shows up claims no block.
        let workers = tasks.min(n_blocks);
        par::run(workers, workers, |_| worker());
    }
    Committed {
        total: shared.offset.load(Ordering::Relaxed),
        complete: !shared.overflow.load(Ordering::Relaxed),
    }
}

/// Make this thread's non-temporal stores visible to every other thread.
///
/// The streaming kernels leave this to their driver so that it happens once
/// per worker rather than once per block (the fence waits for the write
/// combining buffers to drain, which is expensive to do thousands of times).
#[inline]
pub fn stream_fence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SFENCE is part of SSE, which every x86-64 CPU has.
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}

/// Estimate how many of `values` satisfy the predicate, with a safety margin.
///
/// Counts the matches in evenly spaced 64-row stripes and pads the scaled
/// result by four standard errors of a proportion plus one block.
fn estimate_capacity<T: Kernel>(values: &[T], op: CmpOp, rhs: T, block_rows: usize) -> usize {
    const STRIPES: usize = 64;
    const STRIPE: usize = 64;
    let n = values.len();
    if n <= 8 * STRIPES * STRIPE {
        return n;
    }
    let step = n / STRIPES;
    let mut hits = 0usize;
    let mut word = [0u64; 1];
    for s in 0..STRIPES {
        let start = s * step;
        hits += T::cmp_scalar(&values[start..start + STRIPE], op, rhs, &mut word);
    }
    let sampled = (STRIPES * STRIPE) as f64;
    let p = hits as f64 / sampled;
    let margin = 4.0 * 0.5 / sampled.sqrt();
    (((p + margin) * n as f64) as usize + block_rows).min(n)
}

/// The elements of `values` satisfying `value op rhs`, computed by `tasks`
/// workers in a single pass over the column.
///
/// `stream: None` picks non-temporal stores when the output is larger than
/// the cache.
pub fn filter_fused<T: Kernel>(
    values: &[T],
    op: CmpOp,
    rhs: T,
    tasks: usize,
    stream: Option<bool>,
) -> AlignedVec<T> {
    let n = values.len();
    let block = block_rows(size_of::<T>());
    let capacity = estimate_capacity(values, op, rhs, block);
    let stream = stream.unwrap_or(capacity * size_of::<T>() >= cache::stream_threshold());
    let mut out: AlignedVec<T> = Vec::with_capacity_in(capacity.max(1), ColumnAlloc);
    let dst = par::SendPtr(out.as_mut_ptr());

    let done = scan_then_emit(
        n,
        block,
        tasks,
        capacity,
        || FenceOnDrop { words: vec![0u64; block / 64], stream },
        |range, scratch| T::cmp_scalar(&values[range], op, rhs, &mut scratch.words),
        |range, scratch, offset, count| {
            let rows = &values[range];
            // SAFETY: the pipeline hands out disjoint `offset..offset + count`
            // intervals that lie below `capacity`; `compact` is told to write
            // at most `count` elements, and `scratch.words` is the mask `scan`
            // just produced for exactly these rows.
            unsafe {
                let written = compact(rows.as_ptr(), &scratch.words, rows.len(), dst.get().add(offset), count, stream);
                debug_assert_eq!(written, count);
            }
        },
    );

    if !done.complete {
        // The sample underestimated the selectivity (possible for adversarial
        // layouts). Redo with an exact mask; correctness over speed here.
        drop(out);
        let (mask, _) = cmp_scalar_mask(values, op, rhs);
        return filter_by_mask_with(values, &mask, tasks, Some(stream));
    }
    // SAFETY: every block's `count` slots were initialised by `emit`, and the
    // blocks tile `0..done.total` without gaps.
    unsafe { out.set_len(done.total) };
    if out.capacity() > 2 * out.len() + block {
        out.shrink_to_fit();
    }
    out
}

/// Worker scratch for the streaming drivers: fences when the worker is done.
pub(crate) struct FenceOnDrop {
    pub(crate) words: Vec<u64>,
    pub(crate) stream: bool,
}

impl Drop for FenceOnDrop {
    fn drop(&mut self) {
        if self.stream {
            stream_fence();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;
    use std::sync::Mutex;

    #[test]
    fn offsets_are_contiguous_and_in_block_order() {
        for (rows, block, tasks) in [(0usize, 64usize, 4usize), (1, 64, 4), (1000, 64, 1), (100_000, 256, 8), (99_999, 1024, 3)] {
            let log = Mutex::new(Vec::new());
            let done = scan_then_emit(
                rows,
                block,
                tasks,
                usize::MAX,
                || (),
                |range, _| range.len() / 3,
                |range, _, offset, count| log.lock().unwrap().push((range.start, offset, count)),
            );
            let mut log = log.into_inner().unwrap();
            log.sort();
            let mut expect_offset = 0;
            let mut expect_total = 0;
            for start in (0..rows).step_by(block) {
                expect_total += (start + block).min(rows).saturating_sub(start) / 3;
            }
            for &(start, offset, count) in &log {
                assert_eq!(offset, expect_offset, "block at row {start}");
                expect_offset += count;
            }
            assert_eq!(done, Committed { total: expect_total, complete: true });
            assert_eq!(expect_offset, expect_total);
        }
    }

    #[test]
    fn blocks_beyond_the_limit_are_counted_but_not_emitted() {
        let emitted = AtomicUsize::new(0);
        let done = scan_then_emit(
            10_000,
            100,
            4,
            2_500,
            || (),
            |range, _| range.len(),
            |_, _, offset, count| {
                assert!(offset + count <= 2_500);
                emitted.fetch_add(count, Ordering::Relaxed);
            },
        );
        assert_eq!(done, Committed { total: 10_000, complete: false });
        assert_eq!(emitted.load(Ordering::Relaxed), 2_500);
    }

    #[test]
    fn a_panicking_worker_does_not_hang_the_others() {
        let result = std::panic::catch_unwind(|| {
            scan_then_emit(
                100_000,
                64,
                8,
                usize::MAX,
                || (),
                |range, _| {
                    if range.start == 64 * 500 {
                        panic!("boom");
                    }
                    1
                },
                |_, _, _, _| {},
            )
        });
        assert!(result.is_err());
    }

    #[test]
    fn fused_filter_matches_sequential_filter() {
        let mut rng = Rng::new(77);
        for &n in &[0usize, 1, 63, 5000, 40_000, 300_000] {
            let values: Vec<i64> = (0..n).map(|_| rng.below(1000) as i64).collect();
            for threshold in [0i64, 7, 500, 993, 1000] {
                let want: Vec<i64> = values.iter().copied().filter(|&v| v < threshold).collect();
                for tasks in [1, 2, 8] {
                    for stream in [Some(false), Some(true), None] {
                        let got = filter_fused(&values, CmpOp::Lt, threshold, tasks, stream);
                        assert_eq!(&got[..], &want[..], "n {n} thr {threshold} tasks {tasks} {stream:?}");
                    }
                }
            }
        }
        // Layouts that defeat the sample: all matches packed between stripes.
        let n = 400_000;
        let step = n / 64;
        let tricky: Vec<i32> = (0..n).map(|i| if i % step >= 64 { 1 } else { 0 }).collect();
        let got = filter_fused(&tricky, CmpOp::Eq, 1, 8, Some(false));
        assert_eq!(got.len(), tricky.iter().filter(|&&v| v == 1).count());
        assert!(got.iter().all(|&v| v == 1));
        let sorted: Vec<u32> = (0..500_000).collect();
        let got = filter_fused(&sorted, CmpOp::Ge, 123_457, 8, None);
        assert!(got.iter().copied().eq(123_457..500_000));
    }
}
