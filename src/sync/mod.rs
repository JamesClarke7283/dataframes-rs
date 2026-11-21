//! Layer 7 — concurrency: how frames are mutated while other threads read them.
//!
//! The crate offers four ways to change data, ordered from cheapest to most
//! general. Each rests on a different proof of safety, and none of them puts a
//! lock on the read path.
//!
//! | tier | mechanism | who proves it safe | read path |
//! |------|-----------|--------------------|-----------|
//! | 0 | frozen sharing: `DataFrame`, `Series`, `Buffer` are `Send + Sync` and immutable while shared | the type system (`&self` API) | plain loads |
//! | 1 | exclusive mutation: `&mut` methods update in place if the buffer is uniquely owned, else copy-on-write, per column | the borrow checker + reference count | plain loads |
//! | 2 | partitioned mutation: split one `&mut [T]` or the columns of one `&mut DataFrame` across workers | the borrow checker (disjoint `&mut`) | plain loads |
//! | 3 | snapshot isolation: [`SharedFrame`] publishes immutable versions; writers build the next one privately | an atomic pointer swap | one atomic load per access |
//!
//! # Specification of tier 3
//!
//! A [`SharedFrame`] holds a current *version*: an immutable [`DataFrame`]
//! and an epoch number.
//!
//! * **Readers never block and never observe a partial write.** A reader
//!   takes a [`Snapshot`] (or reuses the one cached in its [`FrameReader`]),
//!   which pins one version for as long as it is held. Everything reachable
//!   from a snapshot is frozen (tier 0), so reads run at full speed.
//! * **Writers are serialised and atomic.** [`SharedFrame::write`] gives the
//!   closure a private frame that starts as the current version. Nothing it
//!   does is visible until the closure returns, at which point the result
//!   becomes the next version in one step. If the closure panics or returns an
//!   error, the current version stays in place.
//! * **A write costs what it touches.** The private frame shares every buffer
//!   with the current version. Replacing or updating a column copies that
//!   column (copy-on-write) and leaves the others shared; appending rows
//!   extends the shared buffers in place past the end of every published
//!   view (`Buffer::append_in_place`) and copies nothing but the new rows.
//! * **Versions are totally ordered** by epoch, and a reader's epochs never go
//!   backwards.
//! * **Reclamation is by reference count.** A version's buffers are released
//!   (to the column pool) when the last snapshot of it is dropped.
//!
//! What this deliberately does not offer is concurrent in-place updates of
//! the same column under readers: that would put an atomic or a lock on every
//! element read. Batch such updates into one `write`, or partition the column
//! under a single writer (tier 2).

use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::{PolarsError, PolarsResult};
use crate::frame::DataFrame;
use crate::kernels::par;
use crate::series::Column;

struct Version {
    frame: DataFrame,
    epoch: u64,
}

/// A frame that many threads read while one at a time replaces it.
/// See the [module documentation](self) for the guarantees.
///
/// ```
/// use dataframes::prelude::*;
/// use dataframes::sync::SharedFrame;
///
/// let shared = SharedFrame::new(df!("id" => [1i64, 2], "v" => [10.0, 20.0])?);
/// let before = shared.snapshot();
///
/// std::thread::scope(|s| {
///     // A writer appends a batch and rewrites a column, atomically.
///     s.spawn(|| {
///         shared.try_write(|df| {
///             df.vstack_mut(&df!("id" => [3i64], "v" => [30.0])?)?;
///             df.apply("v", |c| c * 2)?;
///             Ok(())
///         })
///     });
///     // Readers see either the old version or the new one, never a mix.
///     s.spawn(|| {
///         let snap = shared.snapshot();
///         let v = snap.column("v").unwrap().f64().unwrap();
///         assert!(v.cont_slice().unwrap() == [10.0, 20.0] || v.cont_slice().unwrap() == [20.0, 40.0, 60.0]);
///     });
/// });
///
/// assert_eq!(before.height(), 2, "an old snapshot is unaffected");
/// assert_eq!(shared.snapshot().height(), 3);
/// # Ok::<(), PolarsError>(())
/// ```
pub struct SharedFrame {
    /// The current version. Held only long enough to clone or replace the `Arc`.
    current: Mutex<Arc<Version>>,
    /// Epoch of the current version, readable without the lock.
    epoch: AtomicU64,
    /// Serialises writers.
    writer: Mutex<()>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A poisoned lock means some thread panicked while holding it. Neither
    // mutex guards an invariant that a panic can break (the version pointer is
    // replaced with a single assignment), so carry on.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl SharedFrame {
    /// Share `frame` as version 0.
    pub fn new(frame: DataFrame) -> Self {
        SharedFrame {
            current: Mutex::new(Arc::new(Version { frame, epoch: 0 })),
            epoch: AtomicU64::new(0),
            writer: Mutex::new(()),
        }
    }

    fn load(&self) -> Arc<Version> {
        lock(&self.current).clone()
    }

    fn publish(&self, frame: DataFrame, epoch: u64) {
        let next = Arc::new(Version { frame, epoch });
        let previous = std::mem::replace(&mut *lock(&self.current), next);
        self.epoch.store(epoch, Ordering::Release);
        // Dropped outside the lock: releasing buffers may take a while.
        drop(previous);
    }

    /// The epoch of the current version (0 for the initial frame, +1 per write).
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Pin the current version.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot(self.load())
    }

    /// A reader handle that caches a snapshot and refreshes it only when the
    /// frame has changed. Use one per thread.
    pub fn reader(&self) -> FrameReader<'_> {
        FrameReader { shared: self, cached: self.load() }
    }

    /// Change the frame. The closure works on a private copy (which shares
    /// all buffers with the current version); its result is published as the
    /// next version when it returns.
    ///
    /// Writers run one at a time. Readers are never blocked: they keep seeing
    /// the previous version until the new one is published.
    pub fn write<R>(&self, f: impl FnOnce(&mut DataFrame) -> R) -> R {
        let _writer = lock(&self.writer);
        let base = self.load();
        let mut frame = base.frame.clone();
        let out = f(&mut frame);
        self.publish(frame, base.epoch + 1);
        out
    }

    /// Like [`write`](Self::write) for fallible changes: the new version is
    /// published only if the closure returns `Ok`.
    pub fn try_write<R>(&self, f: impl FnOnce(&mut DataFrame) -> PolarsResult<R>) -> PolarsResult<R> {
        let _writer = lock(&self.writer);
        let base = self.load();
        let mut frame = base.frame.clone();
        let out = f(&mut frame)?;
        self.publish(frame, base.epoch + 1);
        Ok(out)
    }

    /// Replace the frame by a function of the current one (a functional
    /// [`try_write`](Self::try_write)).
    pub fn update(&self, f: impl FnOnce(&DataFrame) -> PolarsResult<DataFrame>) -> PolarsResult<()> {
        let _writer = lock(&self.writer);
        let base = self.load();
        let frame = f(&base.frame)?;
        self.publish(frame, base.epoch + 1);
        Ok(())
    }

    /// Optimistic concurrency: publish `frame` only if the current version is
    /// still `expected_epoch`. On conflict the frame is handed back.
    ///
    /// This lets several threads prepare a new version without holding the
    /// writer lock, at the price of retrying when they lose the race.
    pub fn compare_and_set(&self, expected_epoch: u64, frame: DataFrame) -> Result<(), DataFrame> {
        let _writer = lock(&self.writer);
        if self.epoch() != expected_epoch {
            return Err(frame);
        }
        self.publish(frame, expected_epoch + 1);
        Ok(())
    }

    /// Append the rows of `batch`. Existing snapshots keep their height; the
    /// value buffers are extended in place where possible, so the cost is
    /// proportional to the batch, not to the frame.
    pub fn append(&self, batch: &DataFrame) -> PolarsResult<()> {
        self.try_write(|df| df.vstack_mut(batch).map(|_| ()))
    }

    /// Stop sharing and return the current frame.
    pub fn into_inner(self) -> DataFrame {
        let version = self.current.into_inner().unwrap_or_else(|e| e.into_inner());
        match Arc::try_unwrap(version) {
            Ok(v) => v.frame,
            Err(shared) => shared.frame.clone(),
        }
    }
}

impl From<DataFrame> for SharedFrame {
    fn from(frame: DataFrame) -> Self {
        SharedFrame::new(frame)
    }
}

impl DataFrame {
    /// Wrap the frame for concurrent readers and writers. See [`SharedFrame`].
    pub fn into_shared(self) -> SharedFrame {
        SharedFrame::new(self)
    }

    /// Mutate the columns in parallel, one worker per column.
    ///
    /// Each closure call receives exclusive access to one column; the borrow
    /// checker guarantees the calls cannot touch each other's data. A column
    /// may be replaced by one of a different type but must keep its length
    /// and name (the frame is rolled back and an error returned otherwise).
    pub fn par_apply_columns(&mut self, f: impl Fn(&mut Column) -> PolarsResult<()> + Send + Sync) -> PolarsResult<()> {
        let height = self.height;
        let names = self.get_column_names_owned();
        let backup = self.columns().to_vec();
        let columns = self.columns_mut();
        // Each column goes to exactly one thread as `&mut`; the first error wins.
        let failed: Mutex<Option<PolarsError>> = Mutex::new(None);
        par::for_each_chunk_mut(columns, 1, par::threads(), |_, column| {
            if let Err(e) = f(&mut column[0]) {
                lock(&failed).get_or_insert(e);
            }
        });
        let result: PolarsResult<()> = match failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let consistent = columns.iter().zip(&names).all(|(c, n)| c.len() == height && c.name() == n);
        match (result, consistent) {
            (Ok(()), true) => Ok(()),
            (Err(e), _) => {
                *columns = backup;
                Err(e)
            }
            (Ok(()), false) => {
                *columns = backup;
                Err(PolarsError::ShapeMismatch(
                    "a column changed its length or name during `par_apply_columns`".into(),
                ))
            }
        }
    }
}

/// One pinned version of a [`SharedFrame`]. Dereferences to the [`DataFrame`].
///
/// Holding a snapshot keeps the buffers of its version alive; it never blocks
/// writers and is never changed by them.
#[derive(Clone)]
pub struct Snapshot(Arc<Version>);

impl Snapshot {
    /// The epoch of this version.
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.0.epoch
    }

    /// The frame of this version.
    #[inline]
    pub fn frame(&self) -> &DataFrame {
        &self.0.frame
    }
}

impl Deref for Snapshot {
    type Target = DataFrame;
    #[inline]
    fn deref(&self) -> &DataFrame {
        &self.0.frame
    }
}

/// A per-thread reader of a [`SharedFrame`].
///
/// [`get`](Self::get) costs one atomic load while the frame is unchanged: the
/// reader keeps its own snapshot and touches the shared lock only after a
/// writer has published a new version. Readers therefore do not contend with
/// each other at all (not even on a reference count).
pub struct FrameReader<'a> {
    shared: &'a SharedFrame,
    cached: Arc<Version>,
}

impl FrameReader<'_> {
    /// The latest published frame.
    #[inline]
    pub fn get(&mut self) -> &DataFrame {
        if self.shared.epoch.load(Ordering::Acquire) != self.cached.epoch {
            self.refresh();
        }
        &self.cached.frame
    }

    #[cold]
    fn refresh(&mut self) {
        let latest = self.shared.load();
        // Epochs only move forward, even if the atomic was read mid-publish.
        if latest.epoch > self.cached.epoch {
            self.cached = latest;
        }
    }

    /// The epoch of the frame last returned by [`get`](Self::get).
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.cached.epoch
    }

    /// The cached frame, without checking for a newer version.
    #[inline]
    pub fn cached(&self) -> &DataFrame {
        &self.cached.frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_array::Int64Chunked;
    use crate::df;
    use crate::series::{IntoSeries, Series};
    use std::sync::atomic::AtomicBool;

    fn pair(n: i64) -> DataFrame {
        let a: Vec<i64> = (0..n).collect();
        let b: Vec<i64> = a.iter().map(|x| x * 2).collect();
        df!("a" => &a, "b" => &b).unwrap()
    }

    #[test]
    fn writes_are_atomic_and_snapshots_are_stable() {
        let shared = SharedFrame::new(pair(4));
        let before = shared.snapshot();
        assert_eq!((before.epoch(), shared.epoch()), (0, 0));
        shared.write(|df| {
            df.apply("a", |c| c + 100).unwrap();
            df.apply("b", |c| c + 200).unwrap();
        });
        let after = shared.snapshot();
        assert_eq!((after.epoch(), shared.epoch()), (1, 1));
        assert_eq!(before["a"].i64().unwrap().cont_slice().unwrap(), &[0, 1, 2, 3]);
        assert_eq!(after["a"].i64().unwrap().cont_slice().unwrap(), &[100, 101, 102, 103]);
        assert_eq!(after.frame().height(), 4);
        // A failed write publishes nothing.
        let err = shared.try_write(|df| {
            df.apply("a", |c| c * 7)?;
            df.column("missing").map(|_| ())
        });
        assert!(err.is_err());
        assert_eq!(shared.epoch(), 1);
        assert_eq!(shared.snapshot()["a"].i64().unwrap().get(0), Some(100));
        shared.update(|df| Ok(df.head(2))).unwrap();
        assert_eq!((shared.epoch(), shared.snapshot().height()), (2, 2));
        assert!(shared.compare_and_set(1, pair(1)).is_err(), "stale epoch is rejected");
        assert!(shared.compare_and_set(2, pair(1)).is_ok());
        assert_eq!(shared.into_inner().height(), 1);
    }

    #[test]
    fn untouched_columns_are_shared_between_versions() {
        let shared = pair(1000).into_shared();
        let v0 = shared.snapshot();
        shared.write(|df| df.apply("a", |c| c + 1).map(|_| ()).unwrap());
        let v1 = shared.snapshot();
        let ptr = |s: &Snapshot, c: &str| s[c].i64().unwrap().values().as_ptr();
        assert_eq!(ptr(&v0, "b"), ptr(&v1, "b"), "column `b` was not copied");
        assert_ne!(ptr(&v0, "a"), ptr(&v1, "a"), "column `a` was copied on write");
    }

    #[test]
    fn appends_extend_shared_buffers_without_copying() {
        let shared = pair(100).into_shared();
        let batch = |from: i64| {
            df!("a" => (from..from + 10).collect::<Vec<i64>>(), "b" => (from..from + 10).map(|x| x * 2).collect::<Vec<i64>>())
                .unwrap()
        };
        // The first append has to reallocate (the initial buffer is exact)...
        shared.append(&batch(100)).unwrap();
        let grown = shared.snapshot();
        let ptr = |s: &Snapshot| s["a"].i64().unwrap().values().as_ptr();
        // ...later ones write past the end of every published view.
        let mut snapshots = vec![grown.clone()];
        for i in 0..5 {
            shared.append(&batch(110 + i * 10)).unwrap();
            snapshots.push(shared.snapshot());
        }
        for (i, s) in snapshots.iter().enumerate() {
            assert_eq!(ptr(s), ptr(&grown), "append {i} must not have copied the prefix");
            assert_eq!(s.height(), 110 + i * 10);
            let a = s["a"].i64().unwrap().cont_slice().unwrap();
            assert!(a.iter().enumerate().all(|(row, &v)| v == row as i64), "snapshot {i} is intact");
        }
        assert!(shared.append(&df!("x" => [1]).unwrap()).is_err());
    }

    #[test]
    fn readers_see_consistent_versions_under_concurrent_writes() {
        let shared = SharedFrame::new(pair(20_000));
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let writer = s.spawn(|| {
                for round in 1..=200i64 {
                    shared.write(|df| {
                        // Two columns change together; the invariant b == 2a must
                        // hold in every version a reader can see.
                        let a = df["a"].i64().unwrap().clone();
                        let new_a = a.apply_values(|v| v + round);
                        let new_b = new_a.apply_values(|v| v * 2);
                        df.replace("a", new_a.into_series()).unwrap();
                        df.replace("b", new_b.into_series()).unwrap();
                    });
                }
                stop.store(true, Ordering::Release);
            });
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        let mut reader = shared.reader();
                        let (mut checks, mut last_epoch) = (0u64, 0u64);
                        while !stop.load(Ordering::Acquire) {
                            let df = reader.get();
                            let a = df["a"].i64().unwrap().cont_slice().unwrap();
                            let b = df["b"].i64().unwrap().cont_slice().unwrap();
                            assert!(a.iter().zip(b).all(|(&x, &y)| y == 2 * x), "torn read");
                            assert!(reader.epoch() >= last_epoch, "epochs went backwards");
                            last_epoch = reader.epoch();
                            checks += 1;
                        }
                        checks
                    })
                })
                .collect();
            writer.join().unwrap();
            let total: u64 = readers.into_iter().map(|r| r.join().unwrap()).sum();
            assert!(total > 0);
        });
        assert_eq!(shared.epoch(), 200);
        let last = shared.snapshot();
        let expect: i64 = (1..=200).sum();
        assert_eq!(last["a"].i64().unwrap().get(0), Some(expect));
    }

    #[test]
    fn several_writers_serialise() {
        let shared = SharedFrame::new(df!("n" => [0i64]).unwrap());
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..250 {
                        shared.write(|df| df.apply("n", |c| c + 1).map(|_| ()).unwrap());
                    }
                });
            }
        });
        assert_eq!(shared.snapshot()["n"].i64().unwrap().get(0), Some(2000));
        assert_eq!(shared.epoch(), 2000);
        // Optimistic writers retry on conflict and still lose no update.
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..100 {
                        loop {
                            let snap = shared.snapshot();
                            let mut next = snap.frame().clone();
                            next.apply("n", |c| c + 1).unwrap();
                            if shared.compare_and_set(snap.epoch(), next).is_ok() {
                                break;
                            }
                        }
                    }
                });
            }
        });
        assert_eq!(shared.snapshot()["n"].i64().unwrap().get(0), Some(2400));
    }

    #[test]
    fn partitioned_mutation_of_one_column_and_of_many() {
        // One column, split across threads by the borrow checker.
        let mut ca = Int64Chunked::from_slice("x".into(), &(0..100_000).collect::<Vec<i64>>());
        let (left, right) = ca.values_mut().split_at_mut(50_000);
        std::thread::scope(|s| {
            s.spawn(|| left.iter_mut().for_each(|v| *v += 1));
            s.spawn(|| right.iter_mut().for_each(|v| *v -= 1));
        });
        assert_eq!((ca.get(0), ca.get(99_999)), (Some(1), Some(99_998)));

        // Many columns, one worker each.
        let mut df = pair(50_000);
        df.par_apply_columns(|c| {
            c.i64_mut()?.apply_mut(|v| v + 7);
            Ok(())
        })
        .unwrap();
        assert_eq!(df["a"].i64().unwrap().get(1), Some(8));
        assert_eq!(df["b"].i64().unwrap().get(1), Some(9));
        // A closure that breaks the frame's shape is rolled back.
        let before = df.clone();
        let err = df.par_apply_columns(|c| {
            *c = Series::new(c.name().clone(), [1i64]);
            Ok(())
        });
        assert!(err.is_err());
        assert_eq!(df, before);
        assert!(df.par_apply_columns(|c| c.f64_mut().map(|_| ())).is_err());
        assert_eq!(df, before);
    }
}
