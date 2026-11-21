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
//! * [`warm`] wakes workers without giving them a job, for a caller that
//!   wants its next job to find them polling.
//! * On Linux the first workers — one for every physical core but the
//!   caller's — are kept to one core each, so that "one helper" means "a
//!   second core" and not "the other half of mine"; when the caller turns up
//!   on another core, they move. `DATAFRAMES_HOME_CORES=0` leaves the
//!   placement to the scheduler.
//!
//! A panic in a task cancels the tasks not yet claimed and is re-raised on
//! the caller once all helpers have left.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::mem::DropGuard;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
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
    /// Whether `panic` holds a payload: lets a job that went well skip the lock.
    panicked: AtomicBool,
    /// Where the first workers live, if that is managed.
    homes: Option<Homes>,
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

/// Physical cores the team can use: at most [`threads`], and fewer where
/// several hardware threads share a core.
///
/// Two threads of one core share its caches and its path to them, so a
/// kernel that is limited by how fast data arrives gains nothing from the
/// second one; the splitting policy uses this count for inputs that sit in
/// the cache. Where the operating system does not say which threads share a
/// core, every thread counts as one.
pub fn cores() -> usize {
    static CORES: OnceLock<usize> = OnceLock::new();
    *CORES.get_or_init(|| count_cores().unwrap_or(usize::MAX).clamp(1, threads()))
}

fn count_cores() -> Option<usize> {
    core_sets().map(|cores| cores.len())
}

/// The CPUs this process may run on, grouped by physical core, in the order
/// in which the cores first appear.
#[cfg(target_os = "linux")]
fn core_sets() -> Option<Vec<Vec<usize>>> {
    // SAFETY: a zeroed CPU set is valid and the call fills at most its size.
    let allowed = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        set
    };
    let mut cores: Vec<(String, Vec<usize>)> = Vec::new();
    for cpu in 0..libc::CPU_SETSIZE as usize {
        // SAFETY: `cpu` is below the size of the set.
        if !unsafe { libc::CPU_ISSET(cpu, &allowed) } {
            continue;
        }
        // The same text for every thread of one core, e.g. "0,2" or "4-5".
        let siblings = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")).ok()?;
        match cores.iter_mut().find(|(key, _)| *key == siblings) {
            Some((_, cpus)) => cpus.push(cpu),
            None => cores.push((siblings, vec![cpu])),
        }
    }
    (!cores.is_empty()).then(|| cores.into_iter().map(|(_, cpus)| cpus).collect())
}

#[cfg(not(target_os = "linux"))]
fn core_sets() -> Option<Vec<Vec<usize>>> {
    None
}

/// The least time between two rearrangements of the workers' homes: a caller
/// that hops between cores, or several that take turns, is not chased.
const REHOME_NANOS: u64 = 1_000_000;

/// Where the first workers live: one physical core each, none of them the
/// calling thread's.
///
/// Left alone, the scheduler places a woken worker wherever is convenient at
/// that moment, quite often on the second hardware thread of the caller's own
/// core, where it shares the caller's cache and adds nothing; measured on the
/// benchmark runners, that alone decided whether splitting a cache-resident
/// column halved its time or changed nothing. So worker `k`, for as many
/// workers as there are *other* cores, is kept to the hardware threads of
/// the `k + 1`-th core after the caller's: a job that wants one helper per
/// core finds its helpers on cores other than the caller's and other than
/// each other's.
///
/// The calling thread is never restricted, and it may move. When a job
/// finds it on another core than the homes were arranged around, the homes
/// are rearranged ([`Homes::follow`]). The remaining workers — the ones that
/// make up the second hardware threads — have no home: wherever the caller
/// is, no core then has more threads tied to it than it can run at once.
struct Homes {
    /// The CPUs of every physical core this process may use.
    cores: Vec<Vec<usize>>,
    /// CPU number to index into `cores` (`u16::MAX`: not ours).
    core_of: Vec<u16>,
    /// The core the homes are arranged around.
    base: AtomicUsize,
    /// Kernel thread ids of the workers with a home; 0 until they run.
    tids: Vec<AtomicI32>,
    /// When the homes were last rearranged, in nanoseconds since `since`.
    moved: AtomicU64,
    since: Instant,
}

impl Homes {
    /// Homes for the first of `workers` workers, arranged around the calling
    /// thread. `None` if workers are left to the scheduler
    /// (`DATAFRAMES_HOME_CORES=0`, a single core, or a system that does not
    /// tell which CPUs share one).
    fn new(workers: usize) -> Option<Self> {
        if std::env::var("DATAFRAMES_HOME_CORES").is_ok_and(|v| v.trim() == "0") {
            return None;
        }
        let cores = core_sets().filter(|cores| (2..usize::from(u16::MAX)).contains(&cores.len()))?;
        let mut core_of = vec![u16::MAX; cores.iter().flatten().max()? + 1];
        for (core, cpus) in cores.iter().enumerate() {
            for &cpu in cpus {
                core_of[cpu] = core as u16;
            }
        }
        let homed = workers.min(cores.len() - 1);
        let homes = Self {
            cores,
            core_of,
            base: AtomicUsize::new(0),
            tids: (0..homed).map(|_| AtomicI32::new(0)).collect(),
            moved: AtomicU64::new(0),
            since: Instant::now(),
        };
        homes.base.store(homes.core_here().unwrap_or(0), Ordering::Relaxed);
        Some(homes)
    }

    /// The core the calling thread is running on right now.
    fn core_here(&self) -> Option<usize> {
        let core = *self.core_of.get(current_cpu()?)?;
        (core != u16::MAX).then_some(usize::from(core))
    }

    /// The CPUs of worker `worker`'s home when the caller is on core `base`.
    fn home(&self, worker: usize, base: usize) -> &[usize] {
        &self.cores[(base + 1 + worker) % self.cores.len()]
    }

    /// Called by worker `worker` when it starts.
    fn move_in(&self, worker: usize) {
        if let Some(tid) = self.tids.get(worker) {
            tid.store(current_tid(), Ordering::Relaxed);
            stay_on(0, self.home(worker, self.base.load(Ordering::Relaxed)));
        }
    }

    /// Rearrange the homes if the calling thread is not on the core they
    /// were arranged around. Called with the job slot held, so by one thread
    /// at a time; costs a few nanoseconds when there is nothing to do.
    fn follow(&self) {
        let Some(here) = self.core_here() else { return };
        if here == self.base.load(Ordering::Relaxed) {
            return;
        }
        let now = (self.since.elapsed().as_nanos() as u64).max(1);
        let moved = self.moved.load(Ordering::Relaxed);
        if moved != 0 && now.saturating_sub(moved) < REHOME_NANOS {
            return;
        }
        self.moved.store(now, Ordering::Relaxed);
        self.base.store(here, Ordering::Relaxed);
        for (worker, tid) in self.tids.iter().enumerate() {
            let tid = tid.load(Ordering::Relaxed);
            if tid != 0 {
                stay_on(tid, self.home(worker, here));
            }
        }
    }
}

/// The CPU the calling thread is running on.
#[cfg(target_os = "linux")]
fn current_cpu() -> Option<usize> {
    // SAFETY: no arguments, no preconditions.
    usize::try_from(unsafe { libc::sched_getcpu() }).ok()
}

#[cfg(not(target_os = "linux"))]
fn current_cpu() -> Option<usize> {
    None
}

/// The kernel's id of the calling thread.
#[cfg(target_os = "linux")]
fn current_tid() -> i32 {
    // SAFETY: the call takes no arguments and cannot fail.
    unsafe { libc::syscall(libc::SYS_gettid) as i32 }
}

#[cfg(not(target_os = "linux"))]
fn current_tid() -> i32 {
    0
}

/// Restrict thread `tid` (0: the calling thread) to `cpus`. Failure is
/// harmless: the thread then runs wherever the scheduler puts it.
#[cfg(target_os = "linux")]
fn stay_on(tid: i32, cpus: &[usize]) {
    // SAFETY: a zeroed CPU set is valid, every index is checked against its
    // size, and the call only reads it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cpus.iter().filter(|&&cpu| cpu < libc::CPU_SETSIZE as usize) {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(tid, size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn stay_on(_tid: i32, _cpus: &[usize]) {}

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
            panicked: AtomicBool::new(false),
            homes: Homes::new(wanted),
            spin: spin_time(),
        });
        let mut handles = Vec::with_capacity(wanted);
        for i in 0..wanted {
            let shared = Arc::clone(&shared);
            let spawned = std::thread::Builder::new().name(format!("dataframes-{i}")).spawn(move || {
                if let Some(homes) = &shared.homes {
                    homes.move_in(i);
                }
                worker_main(&shared, i)
            });
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

/// Whether the workers a job with `helpers` helpers would use are all awake,
/// so that the job would start without a system call.
pub fn helpers_awake(helpers: usize) -> bool {
    match team() {
        Some(team) => {
            let s = &*team.shared;
            helpers <= s.workers()
                && (s.sleepers.load(Ordering::Relaxed) == 0 || s.sleeping[..helpers].iter().all(|f| !f.load(Ordering::Relaxed)))
        }
        None => helpers == 0,
    }
}

/// Wake the workers a job with `helpers` helpers would use, without giving
/// them anything to do: they poll for [`spin_time`], and a job that follows
/// within that time starts without a system call. Does nothing while another
/// thread's job is running.
pub fn warm(helpers: usize) {
    if helpers == 0 || BUSY.get() {
        return;
    }
    let Some(team) = team() else { return };
    let Ok(_slot) = team.caller.try_lock() else { return };
    let s = &*team.shared;
    if let Some(homes) = &s.homes {
        homes.follow();
    }
    if s.sleepers.load(Ordering::SeqCst) == 0 {
        return;
    }
    // A new epoch with no job open: a worker that wakes up for it finds
    // nothing to join and starts polling.
    s.epoch.0.fetch_add(1, Ordering::SeqCst);
    wake(s, helpers.min(s.workers()));
}

/// Wake those of the first `helpers` workers that are asleep. The epoch must
/// have been bumped first: a worker flags itself asleep before its last look
/// at the epoch, so one side always notices the other.
fn wake(s: &Shared, helpers: usize) {
    if let Some(handles) = s.handles.get() {
        for (flag, handle) in s.sleeping.iter().zip(handles).take(helpers) {
            if flag.load(Ordering::SeqCst) {
                handle.unpark();
            }
        }
    }
}

/// Call `f(i)` once for every `i` in `0..tasks`, on the calling thread and on
/// up to `helpers` team workers, and return when all calls have returned.
///
/// With `wake_sleepers`, sleeping workers are woken to help; without it only
/// workers that are still polling take part, so the call never makes a
/// system call.
/// Which thread runs which index, and in what order, is unspecified.
///
/// # Panics
/// Re-raises the first panic of `f` after every thread has left the job.
/// Tasks not yet started at that point are skipped.
pub fn run(tasks: usize, helpers: usize, wake_sleepers: bool, f: &Task<'_>) {
    let inline = || (0..tasks).for_each(f);
    if tasks <= 1 || helpers == 0 || BUSY.get() {
        return inline();
    }
    let Some(team) = team() else { return inline() };
    let Ok(_slot) = team.caller.try_lock() else { return inline() };
    let s = &*team.shared;
    if let Some(homes) = &s.homes {
        homes.follow();
    }

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
    for (share, cursor) in s.cursors[..parts].iter().enumerate() {
        cursor.0.store(share * tasks / parts, Ordering::Relaxed);
    }
    s.helpers.store(helpers, Ordering::Relaxed);
    s.open.store(true, Ordering::SeqCst);
    s.epoch.0.fetch_add(1, Ordering::SeqCst);
    if wake_sleepers && s.sleepers.load(Ordering::SeqCst) > 0 {
        wake(s, helpers);
    }

    BUSY.set(true);
    // Runs on return and on unwind: stop new claims, close the job, and wait
    // until no helper is inside it. Only then may `f` go out of scope.
    let close = DropGuard::new((), |()| {
        // After a normal `drain` every share is used up; only an unwinding
        // caller has to stop the helpers from claiming more.
        if std::thread::panicking() {
            cancel(s, tasks, parts);
        }
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
        // The caller's own panic wins; a helper's must not surface in the
        // next job.
        if std::thread::panicking() && s.panicked.swap(false, Ordering::Relaxed) {
            *lock(&s.panic) = None;
        }
        BUSY.set(false);
    });
    drain(s, tasks, parts, 0, f);
    drop(close);
    if s.panicked.swap(false, Ordering::Relaxed)
        && let Some(payload) = lock(&s.panic).take()
    {
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
        // Look before claiming from somebody else's share: a claim takes the
        // counter's cache line away from its owner, a look at a share that
        // is used up (the usual case) does not.
        if step > 0 && s.cursors[share].0.load(Ordering::Relaxed) >= end {
            continue;
        }
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
        let (epoch, slept) = wait_for_job(s, index, seen, idle_since);
        seen = epoch;
        if slept {
            // Somebody woke this worker on purpose: poll for a full budget
            // even if the job that follows is not this one.
            idle_since = Instant::now();
        }
        // A job that does not want this worker is none of its business; this
        // look is only a shortcut, the real check follows the announcement.
        if index >= s.helpers.load(Ordering::Relaxed) {
            continue;
        }
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
                    s.panicked.store(true, Ordering::Relaxed);
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
/// The next epoch after `seen`, and whether the worker slept while waiting.
fn wait_for_job(s: &Shared, index: usize, seen: u64, idle_since: Instant) -> (u64, bool) {
    while idle_since.elapsed() < s.spin {
        for _ in 0..64 {
            let epoch = s.epoch.0.load(Ordering::Acquire);
            if epoch != seen {
                return (epoch, false);
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
    (epoch, true)
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

    #[cfg(target_os = "linux")]
    #[test]
    fn the_first_workers_live_on_the_cores_the_caller_is_not_on() {
        let Some(cores) = core_sets() else { return };
        assert!(cores.iter().all(|cpus| !cpus.is_empty()));
        assert_eq!(count_cores(), Some(cores.len()));
        let Some(homes) = Homes::new(2 * cores.len()) else {
            assert!(cores.len() < 2 || std::env::var("DATAFRAMES_HOME_CORES").is_ok_and(|v| v.trim() == "0"));
            return;
        };
        // One worker per other core; the rest are left to the scheduler, so
        // that no core has more threads tied to it than it can run.
        assert_eq!(homes.tids.len(), cores.len() - 1);
        for base in 0..cores.len() {
            let taken: Vec<&[usize]> = (0..homes.tids.len()).map(|k| homes.home(k, base)).collect();
            assert!(taken.iter().all(|home| *home != cores[base].as_slice()), "a worker at home on the caller's core");
            assert!(taken.iter().enumerate().all(|(i, home)| !taken[..i].contains(home)), "two workers on one core");
        }
        assert!(homes.core_here().is_some_and(|core| core < cores.len()));
        // A worker moves in; then the caller turns up elsewhere and the homes
        // follow (at most once per millisecond).
        std::thread::scope(|scope| {
            scope.spawn(|| homes.move_in(0)).join().unwrap();
        });
        assert_ne!(homes.tids[0].load(Ordering::Relaxed), 0);
        // That thread is gone; do not let the test move whoever has its id now.
        homes.tids[0].store(0, Ordering::Relaxed);
        let elsewhere = (homes.base.load(Ordering::Relaxed) + 1) % cores.len();
        homes.base.store(elsewhere, Ordering::Relaxed);
        homes.follow();
        assert_eq!(Some(homes.base.load(Ordering::Relaxed)), homes.core_here());
        homes.base.store(elsewhere, Ordering::Relaxed);
        homes.follow();
        assert_eq!(homes.base.load(Ordering::Relaxed), elsewhere, "rearranged twice within a millisecond");
    }

    #[test]
    fn warming_wakes_helpers_without_a_job() {
        // Whatever the other tests of this process are doing with the team,
        // these calls must neither block nor disturb a job.
        for helpers in [0usize, 1, 3, 64] {
            warm(helpers);
            let _ = helpers_awake(helpers);
        }
        assert!(helpers_awake(0));
        assert!(!helpers_awake(threads() + 64), "more helpers than the team has are never all awake");
        let count = AtomicUsize::new(0);
        run(16, 3, true, &|_| {
            warm(3);
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 16);
        assert!((1..=threads()).contains(&cores()));
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
        // Neither that panic nor one that a helper may have hit at the same
        // time surfaces in a later job.
        for _ in 0..3 {
            let count = AtomicUsize::new(0);
            run(32, 8, true, &|_| {
                count.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(count.load(Ordering::Relaxed), 32);
        }
    }
}
