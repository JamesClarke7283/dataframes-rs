//! Work splitting for the data-parallel kernels.
//!
//! The policy is deliberately simple: a kernel asks how many tasks its input
//! is worth ([`tasks_for`]), gets row ranges that start on mask-word
//! boundaries ([`word_ranges`]), and hands them to the shared rayon pool. An
//! input goes parallel only when every task has at least
//! [`min_task_bytes`] of data to chew on, so small columns never pay for a
//! thread wake-up.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Default for [`min_task_bytes`].
pub const DEFAULT_MIN_TASK_BYTES: usize = 1 << 20;

static MIN_TASK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Smallest amount of input, in bytes, worth giving a task of its own.
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

/// Worker threads in the pool (one per logical CPU by default).
#[inline]
pub fn threads() -> usize {
    rayon::current_num_threads()
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

/// Workers worth using for memory-bound kernels: one per physical core.
///
/// Two hardware threads on one core share its L1/L2 and its memory bandwidth,
/// so for streaming kernels the second thread adds contention, not throughput.
#[inline]
pub fn memory_threads() -> usize {
    (threads() / threads_per_core()).max(1)
}

/// Number of tasks to split `rows` rows of `row_bytes` bytes each into (at least 1).
#[inline]
pub fn tasks_for(rows: usize, row_bytes: usize) -> usize {
    let by_size = rows.saturating_mul(row_bytes) / min_task_bytes();
    if by_size <= 1 {
        return 1;
    }
    by_size.min(memory_threads()).max(1)
}

/// Rows per task: a multiple of 64 so that every task owns whole mask words.
#[inline]
pub fn rows_per_task(rows: usize, tasks: usize) -> usize {
    (rows.div_ceil(tasks.max(1)).div_ceil(64) * 64).max(64)
}

/// Half-open row ranges covering `0..rows`; every start is a multiple of 64.
pub fn word_ranges(rows: usize, tasks: usize) -> Vec<(usize, usize)> {
    let step = rows_per_task(rows, tasks);
    (0..rows).step_by(step).map(|start| (start, (start + step).min(rows))).collect()
}

/// A raw pointer that may cross threads.
///
/// Used by kernels whose tasks write disjoint regions of one output buffer.
/// Every use site documents why the regions cannot overlap.
pub struct SendPtr<T>(pub *mut T);

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
}
