//! Work splitting for the data-parallel kernels.
//!
//! A kernel asks how many threads its input is worth ([`tasks_for`]), cuts
//! the input into a few chunks per thread ([`chunk_rows`]) and hands the
//! chunks to the [`team`](super::team) ([`run`], [`for_each_chunk_mut`],
//! [`map`]). Chunks are claimed dynamically, so a helper that arrives late or
//! is descheduled does not hold the others up.
//!
//! Whether an input goes parallel depends on its size and on the state of the
//! team:
//!
//! * below two chunks of [`min_task_bytes`] (32 KiB each) it never does;
//! * while team workers are polling ("hot"), anything above that does;
//! * when they are asleep, waking them costs a system call, which only inputs
//!   of [`WAKE_BYTES`] or more are worth — unless calls arrive in a burst, in
//!   which case the second call of the burst wakes the team for the rest.

use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use super::team;

/// Default for [`min_task_bytes`].
pub const DEFAULT_MIN_TASK_BYTES: usize = 32 << 10;

/// Inputs of at least this many bytes wake sleeping team workers.
pub const WAKE_BYTES: usize = 256 << 10;

/// Two parallelisable calls closer together than this count as a burst.
const BURST_NANOS: u64 = 200_000;

/// Chunks per participating thread: enough slack to absorb a slow helper.
const CHUNKS_PER_THREAD: usize = 4;

static MIN_TASK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Smallest amount of input, in bytes, worth giving a chunk of its own.
///
/// Defaults to [`DEFAULT_MIN_TASK_BYTES`]; the `DATAFRAMES_MIN_TASK_BYTES`
/// environment variable or [`set_min_task_bytes`] override it.
#[inline]
pub fn min_task_bytes() -> usize {
    match MIN_TASK_BYTES.load(Ordering::Relaxed) {
        0 => {
            let v = std::env::var("DATAFRAMES_MIN_TASK_BYTES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_MIN_TASK_BYTES)
                .max(1);
            MIN_TASK_BYTES.store(v, Ordering::Relaxed);
            v
        }
        v => v,
    }
}

/// Change [`min_task_bytes`]. `usize::MAX` disables intra-column parallelism.
pub fn set_min_task_bytes(bytes: usize) {
    MIN_TASK_BYTES.store(bytes.max(1), Ordering::Relaxed);
}

/// Threads kernels may use, the calling thread included (one per logical CPU
/// by default, `DATAFRAMES_THREADS` overrides).
#[inline]
pub fn threads() -> usize {
    team::threads()
}

/// Hardware threads per physical core (2 with SMT / Hyper-Threading, else 1).
pub fn threads_per_core() -> usize {
    static CACHED: AtomicUsize = AtomicUsize::new(0);
    match CACHED.load(Ordering::Relaxed) {
        0 => {
            let n = detect_threads_per_core().clamp(1, 8);
            CACHED.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

fn detect_threads_per_core() -> usize {
    #[cfg(target_os = "linux")]
    if let Ok(list) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") {
        // Either "0,4" or "0-1" (or just "0").
        let mut n = 0;
        for part in list.trim().split(',') {
            n += match part.split_once('-') {
                Some((a, b)) => match (a.parse::<usize>(), b.parse::<usize>()) {
                    (Ok(a), Ok(b)) if b >= a => b - a + 1,
                    _ => 1,
                },
                None => 1,
            };
        }
        if n > 0 {
            return n;
        }
    }
    1
}

/// Threads worth using for memory-bound kernels: one per physical core.
///
/// Two hardware threads on one core share its L1/L2 and its memory bandwidth,
/// so for streaming kernels the second thread adds contention, not throughput.
#[inline]
pub fn memory_threads() -> usize {
    (threads() / threads_per_core()).max(1)
}

/// Nanoseconds since the first call, for burst detection.
fn now_nanos() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// Whether sleeping workers should be woken for an input of `bytes` bytes.
#[cold]
fn worth_waking(bytes: usize) -> bool {
    static LAST: AtomicU64 = AtomicU64::new(0);
    if bytes >= WAKE_BYTES {
        return true;
    }
    // A small input: wake the team only if such calls are arriving in a
    // burst, in which case the calls that follow will find it polling.
    let now = now_nanos().max(1);
    let last = LAST.swap(now, Ordering::Relaxed);
    last != 0 && now.saturating_sub(last) < BURST_NANOS
}

/// Number of threads (at least 1) to use for `rows` rows of `row_bytes` bytes
/// of input each. See the [module documentation](self) for the policy.
#[inline]
pub fn tasks_for(rows: usize, row_bytes: usize) -> usize {
    let bytes = rows.saturating_mul(row_bytes);
    let by_size = bytes / min_task_bytes();
    if by_size <= 1 {
        return 1;
    }
    let limit = by_size.min(memory_threads());
    if limit <= 1 {
        return 1;
    }
    let hot = 1 + team::hot_helpers();
    if hot >= limit || worth_waking(bytes) { limit } else { hot.min(limit) }
}

/// Rows per chunk when `rows` rows are processed by `threads` threads: a
/// multiple of 64 (so every chunk owns whole mask words) that gives each
/// thread a few chunks without making any chunk smaller than
/// [`min_task_bytes`].
#[inline]
pub fn chunk_rows(rows: usize, row_bytes: usize, threads: usize) -> usize {
    let threads = threads.max(1);
    let min_rows = (min_task_bytes() / row_bytes.max(1)).max(64);
    let chunks = (rows / min_rows).clamp(threads, threads * CHUNKS_PER_THREAD);
    (rows.div_ceil(chunks).div_ceil(64) * 64).max(64)
}

/// Rows per task when `rows` rows are split evenly over `tasks` tasks: a
/// multiple of 64 so that every task owns whole mask words.
#[inline]
pub fn rows_per_task(rows: usize, tasks: usize) -> usize {
    (rows.div_ceil(tasks.max(1)).div_ceil(64) * 64).max(64)
}

/// Half-open row ranges of `step` rows covering `0..rows`.
pub fn ranges(rows: usize, step: usize) -> Vec<(usize, usize)> {
    let step = step.max(1);
    (0..rows).step_by(step).map(|start| (start, (start + step).min(rows))).collect()
}

/// Half-open row ranges covering `0..rows`, one per task; every start is a
/// multiple of 64.
pub fn word_ranges(rows: usize, tasks: usize) -> Vec<(usize, usize)> {
    ranges(rows, rows_per_task(rows, tasks))
}

/// Call `f(i)` for every `i` in `0..tasks` on up to `threads` threads (the
/// caller included) and return when all calls are done.
///
/// # Panics
/// Re-raises a panic of `f` after all threads have stopped.
#[inline]
pub fn run(tasks: usize, threads: usize, f: impl Fn(usize) + Sync) {
    team::run(tasks, threads.saturating_sub(1), true, &f);
}

/// Call `f(index, chunk)` for every `chunk_len`-sized piece of `slice` (the
/// last one may be shorter) on up to `threads` threads.
pub fn for_each_chunk_mut<T: Send>(
    slice: &mut [T],
    chunk_len: usize,
    threads: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    assert!(chunk_len > 0, "chunk length must be positive");
    let len = slice.len();
    let chunks = len.div_ceil(chunk_len);
    if chunks <= 1 || threads <= 1 {
        slice.chunks_mut(chunk_len).enumerate().for_each(|(i, chunk)| f(i, chunk));
        return;
    }
    let base = SendPtr(slice.as_mut_ptr());
    run(chunks, threads, |i| {
        let start = i * chunk_len;
        // SAFETY: chunk `i` covers `start..min(start + chunk_len, len)`;
        // chunks with different indices are disjoint, each index is handed out
        // once, and the exclusive borrow of `slice` outlives `run`.
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), chunk_len.min(len - start)) };
        f(i, chunk);
    });
}

/// `(0..n).map(f).collect()`, evaluated on up to `threads` threads. The
/// results are in index order whatever order they were computed in.
pub fn map<R: Send>(n: usize, threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    if n <= 1 || threads <= 1 {
        return (0..n).map(f).collect();
    }
    let mut out: Vec<MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: `MaybeUninit` needs no initialisation.
    unsafe { out.set_len(n) };
    for_each_chunk_mut(&mut out, 1, threads, |i, slot| {
        slot[0].write(f(i));
    });
    // SAFETY: every slot was written exactly once (a panic in `f` would have
    // propagated out of `for_each_chunk_mut`, leaking the written results,
    // before reaching this point). `MaybeUninit<R>` and `R` have the same layout.
    unsafe {
        let mut out = std::mem::ManuallyDrop::new(out);
        Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity())
    }
}

/// A raw pointer that may cross threads.
///
/// Used by kernels whose tasks write disjoint regions of one output buffer.
/// Every use site documents why the regions cannot overlap.
pub struct SendPtr<T>(pub *mut T);

impl<T> SendPtr<T> {
    /// The pointer. Going through a method makes a closure capture the whole
    /// wrapper (which is `Send`) rather than its raw-pointer field.
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

// SAFETY: the wrapper only moves the address between threads; dereferencing
// it is `unsafe` and each use site upholds exclusive access to its region.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_everything_on_word_boundaries() {
        for rows in [0usize, 1, 63, 64, 65, 1000, 4096, 100_003] {
            for tasks in [1usize, 2, 3, 8, 64] {
                let r = word_ranges(rows, tasks);
                let mut next = 0;
                for &(s, e) in &r {
                    assert_eq!(s, next);
                    assert_eq!(s % 64, 0);
                    assert!(e > s);
                    next = e;
                }
                assert_eq!(next, rows);
                assert!(r.len() <= tasks.max(1) || rows < 64 * tasks);
            }
        }
    }

    #[test]
    fn chunk_rows_respects_bounds() {
        for rows in [1usize, 64, 1000, 100_000, 10_000_000] {
            for row_bytes in [1usize, 4, 8, 16] {
                for threads in [1usize, 2, 4, 8] {
                    let step = chunk_rows(rows, row_bytes, threads);
                    assert!(step >= 64 && step % 64 == 0);
                    let chunks = rows.div_ceil(step);
                    assert!(chunks <= threads * CHUNKS_PER_THREAD, "rows {rows} step {step}");
                }
            }
        }
    }

    #[test]
    fn chunked_mutation_and_map_cover_every_element_once() {
        for len in [0usize, 1, 5, 1000, 4097] {
            for chunk in [1usize, 7, 64, 5000] {
                for threads in [1usize, 2, 8] {
                    let mut v = vec![0u32; len];
                    for_each_chunk_mut(&mut v, chunk, threads, |i, part| {
                        for (k, x) in part.iter_mut().enumerate() {
                            *x += (i * chunk + k) as u32 + 1;
                        }
                    });
                    assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32 + 1));
                }
            }
            let squares = map(len, 8, |i| (i * i) as u64);
            assert!(squares.iter().enumerate().all(|(i, &x)| x == (i * i) as u64));
        }
        let strings = map(100, 4, |i| i.to_string());
        assert_eq!(strings[42], "42");
    }
}
