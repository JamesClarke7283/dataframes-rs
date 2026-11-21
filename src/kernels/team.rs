//! A fork-join team whose workers stay hot.
//!
//! General-purpose thread pools put idle workers to sleep almost at once, so
//! handing work to one costs a system call and a scheduler round trip: tens
//! of microseconds, which is longer than most operations on a column that
//! fits in the cache. Such columns are therefore processed by one core, even
//! when seven others are idle.
//!
//! The team trades a little idle CPU for that latency. After a parallel
//! region its workers keep polling for the next one for [`spin_time`]
//! (100 µs by default) before they go to sleep. While they poll, starting a
//! region costs a few cache-line transfers — well under a microsecond — which
//! makes it worth splitting inputs of a few hundred kilobytes. A program that
//! issues kernel calls back to back keeps the team hot; one that issues them
//! rarely pays for a short burst of polling after each and nothing in between.
//!
//! # Protocol
//!
//! There is one job slot. [`run`] publishes a job (a closure and a task
//! count), works on it together with the helpers that show up, and returns
//! when every task has finished and every helper has left.
//!
//! * The task indices are dealt out in equal consecutive shares, one per
//!   participant, and each share has its own counter. A participant claims
//!   from its own share first (`fetch_add`), then from whatever the others
//!   have left, so each index is executed exactly once whichever threads take
//!   part. A helper that is late finds its share partly done; one that never
//!   arrives costs nothing.
//! * A job that wants `h` helpers is served by workers `0..h`, and the caller
//!   always has the first share. The same thread therefore gets the same
//!   part of the input in every job of a given shape: a column that is
//!   processed again and again stays in the caches of the cores that hold
//!   its parts, and the rest of the team sleeps.
//! * The closure lives on the caller's stack. A helper may use it only between
//!   announcing itself (`active += 1`) and leaving (`active -= 1`), and only
//!   if it saw the job `open` *after* announcing itself. The caller closes the
//!   job and then waits for `active == 0` before it returns. Whichever of
//!   "announce" and "close" comes first in the sequentially consistent order,
//!   either the helper sees the job closed and stays out, or the caller sees
//!   the helper and waits for it.
//! * Calls from several threads at once, and calls from inside a task, do not
//!   queue: they run their tasks on the calling thread.
//!
//! A panic in a task cancels the tasks not yet claimed and is re-raised on
//! the caller once all helpers have left.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::mem::DropGuard;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

/// Default for [`spin_time`].
pub const DEFAULT_SPIN: Duration = Duration::from_micros(100);

/// Keeps a hot atomic on its own pair of cache lines (128 bytes also covers
/// the adjacent-line prefetcher).
#[repr(align(128))]
struct Isolated<T>(T);

type Task<'a> = dyn Fn(usize) + Sync + 'a;

#[derive(Clone, Copy)]
struct Job {
    /// The caller's closure with its lifetime erased; see the module docs for
    /// when it may be dereferenced.
    f: *const Task<'static>,
    tasks: usize,
    /// Number of shares: the helpers the job wants plus the caller.
    parts: usize,
}

struct Shared {
    /// Written by the caller while the job is closed and `active == 0`.
    job: UnsafeCell<Job>,
    /// Bumped when a job opens. Idle workers watch it.
    epoch: Isolated<AtomicU64>,
    /// Whether helpers may still join the current job.
    open: AtomicBool,
    /// Number of helpers the current job wants: workers `0..helpers`.
    helpers: AtomicUsize,
    /// Per share: the next task index to claim from it. Share `p` covers
    /// `p * tasks / parts .. (p + 1) * tasks / parts`; the caller owns share
    /// 0 and worker `k` share `k + 1`.
    cursors: Vec<Isolated<AtomicUsize>>,
    /// Helpers that announced themselves and have not left.
    active: Isolated<AtomicUsize>,
    /// Number of workers asleep (or about to be).
    sleepers: AtomicUsize,
    /// Per worker: whether it is asleep (or about to be) and needs an `unpark`.
    sleeping: Vec<AtomicBool>,
    /// Handles for waking the workers; set once all of them have been spawned.
    handles: OnceLock<Vec<Thread>>,
    /// First panic payload of the current job.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    spin: Duration,
}

impl Shared {
    /// Number of workers that were started.
    #[inline]
    fn workers(&self) -> usize {
        self.handles.get().map_or(0, Vec::len)
    }
}

// SAFETY: `job` is the only field that is not `Sync` by itself. It is written
// by the single caller holding `Team::caller` while no helper can read it and
// read by helpers only under the protocol described in the module docs.
unsafe impl Sync for Shared {}
// SAFETY: the raw closure pointer inside `job` is only dereferenced while the
// caller that published it is blocked in `run`.
unsafe impl Send for Shared {}

struct Team {
    shared: Arc<Shared>,
    /// Held by the thread whose job occupies the slot.
    caller: Mutex<()>,
}

thread_local! {
    /// Set on team workers, and on a caller while its job is running, so that
    /// nested calls run inline instead of deadlocking on the job slot.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // None of the mutexes guards an invariant that a panic can break.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Number of threads kernels may use, the calling thread included.
///
/// One per logical CPU available to the process; the `DATAFRAMES_THREADS`
/// environment variable overrides it (`1` disables intra-column parallelism).
pub fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("DATAFRAMES_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
            .min(256)
    })
}

/// How long an idle worker polls for the next job before it goes to sleep.
///
/// Defaults to [`DEFAULT_SPIN`]; set the `DATAFRAMES_SPIN_US` environment
/// variable to change it (`0` makes workers sleep immediately, which gives up
/// parallelism for small inputs in exchange for no idle polling at all).
pub fn spin_time() -> Duration {
    static SPIN: OnceLock<Duration> = OnceLock::new();
    *SPIN.get_or_init(|| {
        std::env::var("DATAFRAMES_SPIN_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(DEFAULT_SPIN, Duration::from_micros)
    })
}

fn team() -> Option<&'static Team> {
    static TEAM: OnceLock<Option<Team>> = OnceLock::new();
    TEAM.get_or_init(|| {
        let wanted = threads().saturating_sub(1);
        if wanted == 0 || cfg!(miri) {
            return None;
        }
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(Job { f: std::ptr::null::<fn(usize)>() as *const Task<'static>, tasks: 0, parts: 1 }),
            epoch: Isolated(AtomicU64::new(0)),
            open: AtomicBool::new(false),
            helpers: AtomicUsize::new(0),
            cursors: (0..=wanted).map(|_| Isolated(AtomicUsize::new(0))).collect(),
            active: Isolated(AtomicUsize::new(0)),
            sleepers: AtomicUsize::new(0),
            sleeping: (0..wanted).map(|_| AtomicBool::new(false)).collect(),
            handles: OnceLock::new(),
            panic: Mutex::new(None),
            spin: spin_time(),
        });
        let mut handles = Vec::with_capacity(wanted);
        for i in 0..wanted {
            let shared = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("dataframes-{i}"))
                .spawn(move || worker_main(&shared, i));
            match spawned {
                Ok(handle) => handles.push(handle.thread().clone()),
                // Without this worker the later indices would never be woken
                // in order; make do with the ones that started.
                Err(_) => break,
            }
        }
        let started = handles.len();
        shared.handles.set(handles).ok();
        (started > 0).then_some(Team { shared, caller: Mutex::new(()) })
    })
    .as_ref()
}

/// Helpers that are awake right now (polling for work, or working).
///
/// A job handed to the team while this is zero only gets help if the caller
/// asks for sleepers to be woken, which costs a system call.
pub fn hot_helpers() -> usize {
    match team() {
        Some(team) => team.shared.workers().saturating_sub(team.shared.sleepers.load(Ordering::Relaxed)),
        None => 0,
    }
}

/// Call `f(i)` once for every `i` in `0..tasks`, on the calling thread and on
/// up to `helpers` team workers, and return when all calls have returned.
///
/// With `wake`, sleeping workers are woken to help; without it only workers
/// that are still polling take part, so the call never makes a system call.
/// Which thread runs which index, and in what order, is unspecified.
///
/// # Panics
/// Re-raises the first panic of `f` after every thread has left the job.
/// Tasks not yet started at that point are skipped.
pub fn run(tasks: usize, helpers: usize, wake: bool, f: &Task<'_>) {
    let inline = || (0..tasks).for_each(f);
    if tasks <= 1 || helpers == 0 || BUSY.get() {
        return inline();
    }
    let Some(team) = team() else { return inline() };
    let Ok(_slot) = team.caller.try_lock() else { return inline() };
    let s = &*team.shared;

    // SAFETY: we hold the job slot, the previous job was closed and its
    // helpers have left (`run` does not return before), so nobody reads `job`.
    // The lifetime of `f` is erased; the guard below keeps this frame alive
    // until no helper can use the pointer any more.
    let helpers = helpers.min(s.workers());
    let parts = helpers + 1;
    unsafe {
        let f: *const Task<'static> = std::mem::transmute::<*const Task<'_>, *const Task<'static>>(f);
        *s.job.get() = Job { f, tasks, parts };
    }
    *lock(&s.panic) = None;
    for (share, cursor) in s.cursors[..parts].iter().enumerate() {
        cursor.0.store(share * tasks / parts, Ordering::Relaxed);
    }
    s.helpers.store(helpers, Ordering::Relaxed);
    s.open.store(true, Ordering::SeqCst);
    s.epoch.0.fetch_add(1, Ordering::SeqCst);
    if wake && s.sleepers.load(Ordering::SeqCst) > 0 {
        // Wake exactly the designated helpers that are asleep. The epoch was
        // bumped first and a worker flags itself asleep before its last look
        // at the epoch, so one side always notices the other.
        if let Some(handles) = s.handles.get() {
            for (flag, handle) in s.sleeping.iter().zip(handles).take(helpers) {
                if flag.load(Ordering::SeqCst) {
                    handle.unpark();
                }
            }
        }
    }

    BUSY.set(true);
    // Runs on return and on unwind: stop new claims, close the job, and wait
    // until no helper is inside it. Only then may `f` go out of scope.
    let close = DropGuard::new((), |()| {
        cancel(s, tasks, parts);
        s.open.store(false, Ordering::SeqCst);
        let mut spins = 0u32;
        while s.active.0.load(Ordering::SeqCst) != 0 {
            spins += 1;
            if spins < 1 << 12 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        BUSY.set(false);
    });
    drain(s, tasks, parts, 0, f);
    drop(close);
    let payload = lock(&s.panic).take();
    if let Some(payload) = payload {
        resume_unwind(payload);
    }
}

/// Run tasks of the current job as participant `me`: the ones of its own
/// share first, then the ones the other participants have not claimed yet.
#[inline]
fn drain(s: &Shared, tasks: usize, parts: usize, me: usize, call: impl Fn(usize)) {
    for step in 0..parts {
        let share = (me + step) % parts;
        let end = (share + 1) * tasks / parts;
        loop {
            let i = s.cursors[share].0.fetch_add(1, Ordering::Relaxed);
            if i >= end {
                break;
            }
            call(i);
        }
    }
}

/// Make every further claim on the current job fail.
fn cancel(s: &Shared, tasks: usize, parts: usize) {
    for cursor in &s.cursors[..parts] {
        cursor.0.fetch_max(tasks, Ordering::Relaxed);
    }
}

fn worker_main(s: &Shared, index: usize) {
    BUSY.set(true);
    let mut seen = 0u64;
    // The polling budget runs from the last time this worker did real work:
    // a worker that the jobs do not ask for goes to sleep instead of polling
    // forever next to a team that does not need it.
    let mut idle_since = Instant::now();
    loop {
        seen = wait_for_job(s, index, seen, idle_since);
        s.active.0.fetch_add(1, Ordering::SeqCst);
        if s.open.load(Ordering::SeqCst) && index < s.helpers.load(Ordering::Relaxed) {
            // SAFETY: we announced ourselves and then saw the job open, so
            // the caller that published it cannot return (and cannot publish
            // another) before we leave; see the module docs. `helpers` was
            // written before `open` was set, like the job itself.
            let job = unsafe { *s.job.get() };
            drain(s, job.tasks, job.parts, index + 1, |i| {
                // SAFETY: as above; the closure is `Sync`.
                let result = catch_unwind(AssertUnwindSafe(|| unsafe { (*job.f)(i) }));
                if let Err(payload) = result {
                    cancel(s, job.tasks, job.parts);
                    lock(&s.panic).get_or_insert(payload);
                }
            });
            s.active.0.fetch_sub(1, Ordering::SeqCst);
            idle_since = Instant::now();
        } else {
            s.active.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Block until the epoch differs from `seen`; returns the new epoch. Polls
/// until `idle_since + spin`, then sleeps until a caller unparks this worker.
fn wait_for_job(s: &Shared, index: usize, seen: u64, idle_since: Instant) -> u64 {
    while idle_since.elapsed() < s.spin {
        for _ in 0..64 {
            let epoch = s.epoch.0.load(Ordering::Acquire);
            if epoch != seen {
                return epoch;
            }
            std::hint::spin_loop();
        }
    }
    // Announce the intention to sleep *before* the final look at the epoch:
    // a caller bumps the epoch first and reads the flags second, so one of
    // the two sides always notices the other. An `unpark` that arrives before
    // the `park` makes the `park` return at once.
    s.sleeping[index].store(true, Ordering::SeqCst);
    s.sleepers.fetch_add(1, Ordering::SeqCst);
    let epoch = loop {
        let epoch = s.epoch.0.load(Ordering::SeqCst);
        if epoch != seen {
            break epoch;
        }
        std::thread::park();
    };
    s.sleeping[index].store(false, Ordering::SeqCst);
    s.sleepers.fetch_sub(1, Ordering::SeqCst);
    epoch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_exactly_once() {
        for tasks in [0usize, 1, 2, 7, 64, 1000] {
            for helpers in [0usize, 1, 3, 64] {
                for wake in [false, true] {
                    let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                    run(tasks, helpers, wake, &|i| {
                        hits[i].fetch_add(1, Ordering::Relaxed);
                    });
                    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "tasks {tasks} helpers {helpers}");
                }
            }
        }
    }

    #[test]
    fn borrows_from_the_caller_and_can_be_repeated() {
        let data: Vec<u64> = (0..100_000).collect();
        for round in 0..200 {
            let sum = AtomicU64::new(0);
            let chunks = 1 + round % 17;
            let step = data.len().div_ceil(chunks);
            run(chunks, 8, round % 3 == 0, &|i| {
                let part = &data[(i * step).min(data.len())..((i + 1) * step).min(data.len())];
                sum.fetch_add(part.iter().sum::<u64>(), Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), 99_999 * 100_000 / 2);
        }
    }

    #[test]
    fn nested_and_concurrent_calls_run_inline() {
        let total = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..50 {
                        run(8, 8, true, &|_| {
                            run(4, 4, true, &|_| {
                                total.fetch_add(1, Ordering::Relaxed);
                            });
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 4 * 50 * 8 * 4);
    }

    #[test]
    fn a_panic_is_propagated_and_the_team_survives() {
        for wake in [true, false] {
            let result = catch_unwind(|| {
                run(64, 8, wake, &|i| {
                    if i == 13 {
                        panic!("boom");
                    }
                })
            });
            assert!(result.is_err());
        }
        let count = AtomicUsize::new(0);
        run(32, 8, true, &|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 32);
    }
}
