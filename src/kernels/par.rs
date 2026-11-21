//! Work splitting for the data-parallel kernels.
//!
//! A kernel asks how many threads its input is worth ([`plan`]), cuts the
//! input into chunks ([`chunk_rows`]) and hands the chunks to the [`team`]
//! ([`run`], [`for_each_chunk_mut`], [`map`]). Every thread has its own share
//! of the chunks — the same part of the input each time — and takes over
//! what is left of the others' shares, so a helper that arrives late or is
//! descheduled does not hold the rest up.
//!
//! # Inputs far larger than a core's cache
//!
//! They are split over every hardware thread, a few chunks each: while team
//! workers are polling ("hot") always, and when they are asleep if the input
//! has [`WAKE_BYTES`] or more — waking them costs a system call — or if calls
//! arrive in a burst, in which case the second call of the burst wakes the
//! team for the rest.
//!
//! # Inputs that fit in a core's cache, or nearly
//!
//! Below two chunks of [`min_task_bytes`] (32 KiB each), or below what one
//! core streams through in a microsecond ([`stream_rate`], measured once),
//! an input is never split.
//!
//! Above that and up to a few times the private (L2) cache, where the data
//! *is* decides, and no rule on sizes can know that. Measured on the CPUs of
//! the benchmark runners: if each core already holds its part of a column —
//! because the column is read again and again, or was last written by the
//! same split — two cores finish in half the time of one (a 100 000-row sum:
//! 1.9 µs instead of 3.2 µs on a Zen 5, 3.8 instead of 7.7 on a Zen 4). If
//! the calling thread has just written the column, the second core must pull
//! every cache line across, at a tenth of the speed, and the calling thread
//! must get them back before it can write again: the same split then takes
//! two to three times *longer* than no split.
//!
//! So the policy measures instead of guessing. For each size class it keeps
//! one decision — split or not. It starts as "not" for an input that fits in
//! the private cache (it may well be there) and as "split" for a larger one.
//! Every so often the policy times six calls as they are, then runs the
//! other way for a moment untimed (two calls to move the data and, if the
//! helpers had been asleep, 200 µs for their cores to come back up to
//! speed), times six calls that way (see [`Plan`]), and
//! changes its mind if the typical kernel call took at least a ninth less
//! time that way (a thirty-second is enough for the way back to where it
//! started) *and* the caller's whole cycle, from one call to the next, did
//! not get longer — a split that makes the kernel faster by making the
//! caller's own writes slower is no gain. A trial that changes nothing makes
//! the next one wait four times as long, up to 4096 calls, so a workload
//! that is settled pays next to nothing, and one that changes character is
//! followed. The decision is shared by all kernels, so that one writing a
//! column and one reading it agree on which core holds what.
//!
//! A split of an input that fits in the private cache uses one thread per
//! *physical* core (two threads of one core share its cache and the path to
//! it), at most [`RESIDENT_PARTS`], with one equal chunk each: nothing is
//! left to be taken over, so nothing changes cores.

use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use super::{aggregate, cache, team};

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

/// Most threads a kernel spreads its input over, the calling thread included.
///
/// Every hardware thread by default, which is what an input beyond the cache
/// gets: measured on the benchmark runners (2 cores, 4 threads) the second
/// thread of a core never hurt there and bought up to 30 % on multi-column
/// filters. An input that fits in the cache gets one thread per physical
/// core at most ([`team::cores`]). The `DATAFRAMES_THREADS` environment
/// variable overrides the count; `1` disables intra-column parallelism.
#[inline]
pub fn threads() -> usize {
    team::threads()
}

/// Nanoseconds since the first call, for burst detection.
fn now_nanos() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// Whether this call follows another one that found the team asleep within
/// [`BURST_NANOS`]: the calls after a wake-up will then find it polling.
#[cold]
fn burst() -> bool {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = now_nanos().max(1);
    let last = LAST.swap(now, Ordering::Relaxed);
    last != 0 && now.saturating_sub(last) < BURST_NANOS
}

/// Whether sleeping workers should be woken for an input of `bytes` bytes.
#[cold]
fn worth_waking(bytes: usize) -> bool {
    bytes >= WAKE_BYTES || burst()
}

/// A serial pass shorter than this is not worth splitting when its input
/// already sits in the calling core's L2 and nothing is known about where
/// the helpers' parts are ([`tasks_for`]).
const MIN_PARALLEL_NANOS: usize = 4_000;

/// An input that one core streams through faster than this is never split:
/// the hand-off alone takes a few hundred nanoseconds.
const MIN_SPLIT_NANOS: usize = 1_000;

/// Most threads that share an input which fits in one core's cache.
pub const RESIDENT_PARTS: usize = 4;

/// Bytes per microsecond that one thread of this machine streams through a
/// reduction when the data is in its own cache. Measured once, on first use
/// (a few tens of microseconds).
pub fn stream_rate() -> usize {
    static RATE: OnceLock<usize> = OnceLock::new();
    *RATE.get_or_init(|| {
        const WORDS: usize = 16 << 10; // 128 KiB: in L2 on every CPU of interest
        let data: Vec<u64> = (0..WORDS as u64).collect();
        let mut best = u128::MAX;
        for _ in 0..16 {
            let start = Instant::now();
            std::hint::black_box(aggregate::calibration_pass(std::hint::black_box(&data)));
            best = best.min(start.elapsed().as_nanos());
        }
        ((WORDS * 8) as u128 * 1000 / best.max(1)).clamp(1 << 10, 1 << 30) as usize
    })
}

/// Number of threads (at least 1) to use for `rows` rows of `row_bytes` bytes
/// of input each, from the size of the input and the state of the team alone.
///
/// For callers that split by something other than row ranges (one column per
/// thread, say), where the learned decision of [`plan`] does not apply. An
/// input that fits in the calling core's cache is split only if that core
/// would need more than a few microseconds for it.
#[inline]
pub fn tasks_for(rows: usize, row_bytes: usize) -> usize {
    let bytes = rows.saturating_mul(row_bytes);
    let by_size = bytes / min_task_bytes();
    if by_size <= 1 {
        return 1;
    }
    // Twice the cost of a plain sum per byte: a comparison or a compaction.
    if bytes <= cache::cache_info().l2 && bytes.saturating_mul(2) < stream_rate() * (MIN_PARALLEL_NANOS / 1000) {
        return 1;
    }
    tasks_beyond_cache(bytes, by_size)
}

/// Threads for an input that no single core holds in its cache.
#[inline]
fn tasks_beyond_cache(bytes: usize, by_size: usize) -> usize {
    let limit = by_size.min(threads());
    if limit <= 1 {
        return 1;
    }
    let hot = 1 + team::hot_helpers();
    if hot >= limit || worth_waking(bytes) { limit } else { hot.min(limit) }
}

/// Calls before a size class first tries the other way.
const FIRST_TRIAL: u32 = 8;
/// Calls between a change of mind and the next trial. Every trial that
/// changes nothing multiplies the wait by four, up to [`MAX_TRIAL_PERIOD`]:
/// that bounds both what a settled workload pays for trials (fourteen calls
/// in four thousand) and how long a workload that changed character runs the
/// wrong way.
const TRIAL_PERIOD: u32 = 128;
const MAX_TRIAL_PERIOD: u32 = 1 << 12;
/// Timed calls on either side of a trial.
const WINDOW: u8 = 6;
/// Calls at the start of a trial that are not timed: they move the data to
/// where the mode under trial wants it.
const TRIAL_SKIP: u8 = 2;
/// Calls a trial of splitting waits for the helpers to wake up (they need
/// tens of microseconds, and a call can take less than one).
const WARM_CALLS: u32 = 512;
/// How long a trial of splitting runs before it is timed. A helper that was
/// asleep sits on a core that has slowed down, or lost its cache, and takes a
/// moment to be worth what it will be worth: measured too early, a split
/// that halves the time looked slower than no split.
const WARM_NANOS: u64 = 200_000;
/// "Not measured" in [`Lane::kernel`] and [`Lane::cycle`].
const UNKNOWN: u32 = u32::MAX;

/// Nothing is timed; the decision is simply applied.
const STEADY: u8 = 0;
/// The last [`WINDOW`] calls before a trial, timed under the decision.
const BASELINE: u8 = 1;
/// A trial of splitting waits for the helpers to wake up.
const WARMING: u8 = 2;
/// Calls run the other way: [`TRIAL_SKIP`] untimed, then [`WINDOW`] timed.
const TRIAL: u8 = 3;

/// [`Lane::decision`]: not made yet.
const UNSET: u8 = 0;
/// [`Lane::decision`]: one thread.
const ALONE: u8 = 1;
/// [`Lane::decision`]: split.
const SPLIT: u8 = 2;

/// What the policy knows about inputs of one size class that fit in a
/// core's cache, or nearly.
///
/// Plain atomics, read and written without coordination: the fields are
/// statistics and a decision that is re-examined all the time, and callers
/// on several threads at worst lose an update. (Only one of them can have
/// the team anyway.) Times are nanoseconds.
#[repr(align(128))]
struct Lane {
    /// Calls since the last decision.
    calls: AtomicU32,
    /// Calls between a decision and the next trial.
    period: AtomicU32,
    /// [`UNSET`], [`ALONE`] or [`SPLIT`].
    decision: AtomicU8,
    /// [`STEADY`], [`BASELINE`], [`WARMING`] or [`TRIAL`].
    phase: AtomicU8,
    /// Calls left in the phase.
    left: AtomicU8,
    /// How long the kernel calls of the baseline window took (`[0]`, under
    /// the decision) and those of the trial (`[1]`, the other way).
    kernel: [[AtomicU32; WINDOW as usize]; 2],
    /// The time from the start of one call to the start of the next, in the
    /// same two windows: the kernel plus whatever the caller does in
    /// between, which is where a split can cost what it saved.
    cycle: [[AtomicU32; WINDOW as usize]; 2],
    /// When the previous call of the window started; 0 at its beginning.
    last_start: AtomicU64,
    /// When the trial began.
    trial_start: AtomicU64,
}

/// What a call asks of its lane.
#[derive(Clone, Copy)]
struct Request {
    /// Threads a split would use.
    parts: usize,
    /// Whether the input fits in one core's cache (else: nearly).
    resident: bool,
    bytes: usize,
}

impl Lane {
    const fn new() -> Self {
        Self {
            calls: AtomicU32::new(0),
            period: AtomicU32::new(FIRST_TRIAL),
            decision: AtomicU8::new(UNSET),
            phase: AtomicU8::new(STEADY),
            left: AtomicU8::new(0),
            kernel: [const { [const { AtomicU32::new(UNKNOWN) }; WINDOW as usize] }; 2],
            cycle: [const { [const { AtomicU32::new(UNKNOWN) }; WINDOW as usize] }; 2],
            last_start: AtomicU64::new(0),
            trial_start: AtomicU64::new(0),
        }
    }

    /// Decide about one call.
    fn plan(&'static self, request: Request) -> Plan {
        const R: Ordering = Ordering::Relaxed;
        let Request { parts, resident, bytes } = request;
        let calls = self.calls.load(R).wrapping_add(1);
        self.calls.store(calls, R);
        let mut split = match self.decision.load(R) {
            UNSET => {
                // Until something has been measured: an input that is in the
                // calling core's cache stays there; one that is not is as
                // near to every other core.
                self.decision.store(if resident { ALONE } else { SPLIT }, R);
                !resident
            }
            decision => decision == SPLIT,
        };
        let (mut phase, mut left) = (self.phase.load(R), self.left.load(R));

        // Phase changes. The windows on both sides of a trial are adjacent
        // in time, so they see the same workload; a call is only looked at
        // after the previous one has reported, hence "left == 0" is checked
        // on the call *after* a window.
        if phase == STEADY && calls >= self.period.load(R) {
            for sample in self.kernel.iter().chain(&self.cycle).flatten() {
                sample.store(UNKNOWN, R);
            }
            self.last_start.store(0, R);
            (phase, left) = (BASELINE, WINDOW);
        } else if phase == BASELINE && left == 0 {
            self.last_start.store(0, R);
            if split {
                (phase, left) = (TRIAL, TRIAL_SKIP + WINDOW);
            } else {
                // The helpers of a class that does not split are asleep.
                team::warm(parts - 1);
                phase = WARMING;
            }
        }
        if phase == WARMING {
            if team::helpers_awake(parts - 1) {
                (phase, left) = (TRIAL, TRIAL_SKIP + WINDOW);
                self.trial_start.store(now_nanos(), R);
            } else if calls >= self.period.load(R).saturating_add(WARM_CALLS) {
                // Calls too far apart for the helpers to stay awake between
                // them: there is nothing to try.
                self.settle(false);
                phase = STEADY;
            }
        } else if phase == TRIAL && left == 0 {
            // `split == resident`: the decision in force is not the one the
            // lane started with.
            let better = self.other_way_is_better(split == resident);
            if better {
                split = !split;
                self.decision.store(if split { SPLIT } else { ALONE }, R);
            }
            self.settle(better);
            phase = STEADY;
        }

        // Inside a window every call notes when it started.
        let start = if phase == BASELINE || phase == TRIAL { now_nanos().max(1) } else { 0 };
        let (want_split, timed, trial) = match phase {
            BASELINE => {
                left = left.saturating_sub(1);
                (split, true, false)
            }
            TRIAL => {
                // The last untimed call of a trial of splitting repeats
                // until the helpers have been at it for a while.
                let warming = !split
                    && left == WINDOW + 1
                    && start.saturating_sub(self.trial_start.load(R)) < WARM_NANOS
                    && calls < self.period.load(R).saturating_add(2 * WARM_CALLS);
                if !warming {
                    left = left.saturating_sub(1);
                }
                (!split, left < WINDOW, true)
            }
            _ => (split, false, false),
        };
        if phase != STEADY {
            self.phase.store(phase, R);
            self.left.store(left, R);
        }

        let tasks = if !want_split {
            1
        } else if team::helpers_awake(parts - 1) {
            parts
        } else if !resident {
            // An input this large is worth a wake-up, as it is beyond the
            // cache-resident sizes.
            let hot = 1 + team::hot_helpers();
            if worth_waking(bytes) { parts } else { hot.min(parts) }
        } else {
            // The class splits, but the team went to sleep since: this call
            // runs alone, and if more are coming the team is woken for them.
            if phase == STEADY && burst() {
                team::warm(parts - 1);
            }
            1
        };
        if phase != BASELINE && phase != TRIAL {
            return Plan::fixed(tasks);
        }
        // The timed calls of a window report. A call that could not run the
        // way it was meant to says nothing.
        let previous = self.last_start.swap(start, R);
        let timed = timed && (tasks > 1) == want_split;
        let slot = usize::from(left).min(WINDOW as usize - 1);
        if timed && previous != 0 {
            self.cycle[usize::from(trial)][slot].store(clamp_nanos(start.saturating_sub(previous)), R);
        }
        Plan { tasks, sample: timed.then_some(Sample { lane: self, trial, slot, start }) }
    }

    /// The verdict of a trial. `back` says that the other way is the one the
    /// lane started with.
    fn other_way_is_better(&self, back: bool) -> bool {
        let [Some(kernel), Some(other_kernel)] = [0, 1].map(|i| median(&self.kernel[i])) else {
            return false;
        };
        // Leaving the starting assumption takes a clear gain, a ninth less
        // time in the kernel; going back to it takes only a visible one, a
        // thirty-second. Otherwise a lane that was talked into a split by
        // one good trial could keep it while it is a tenth slower.
        let kernel_faster = if back { other_kernel * 32 < kernel * 31 } else { other_kernel * 9 < kernel * 8 };
        let [Some(cycle), Some(other_cycle)] = [0, 1].map(|i| median(&self.cycle[i])) else {
            return kernel_faster;
        };
        if cycle >= 16 * kernel {
            // The caller does so much between two calls that its cycle says
            // nothing about the kernel.
            return kernel_faster;
        }
        // Otherwise the caller's whole cycle has the last word: a kernel
        // that got faster by making its caller slower (the caller writes the
        // column, the helpers have to fetch it, the caller has to get it
        // back) is no gain, and one that got slower may be.
        (kernel_faster && other_cycle <= cycle) || other_cycle + kernel / 4 < cycle
    }

    /// Back to [`STEADY`] after a trial. One that changed nothing makes the
    /// next one wait four times as long.
    fn settle(&self, changed: bool) {
        const R: Ordering = Ordering::Relaxed;
        let period = if changed { TRIAL_PERIOD } else { self.period.load(R).saturating_mul(4).min(MAX_TRIAL_PERIOD) };
        self.period.store(period, R);
        self.calls.store(0, R);
        self.phase.store(STEADY, R);
        self.left.store(0, R);
    }
}

fn clamp_nanos(nanos: u64) -> u32 {
    nanos.min(u64::from(UNKNOWN - 1)) as u32
}

/// The typical value of a window: the median of what was measured, if at
/// least half of the window was. (The best value would favour a split, whose
/// best case is good and whose hand-off sometimes is not.)
fn median(window: &[AtomicU32; WINDOW as usize]) -> Option<u64> {
    let mut known = [0u32; WINDOW as usize];
    let mut n = 0;
    for sample in window {
        let v = sample.load(Ordering::Relaxed);
        if v != UNKNOWN {
            known[n] = v;
            n += 1;
        }
    }
    if n < WINDOW as usize / 2 {
        return None;
    }
    known[..n].sort_unstable();
    Some(u64::from(known[n / 2]))
}

/// One lane per power of two of input size, 64 KiB to 8 MiB.
static LANES: [Lane; 8] = [const { Lane::new() }; 8];

fn lane_for(bytes: usize) -> &'static Lane {
    &LANES[(bytes.max(1).ilog2() as usize).clamp(16, 23) - 16]
}

/// What the policy currently does with inputs that fit in a core's cache or
/// nearly: for every size class that has been used, the smallest input size
/// of the class in bytes, whether such inputs are split, how many calls lie
/// between two trials of the alternative, and the typical kernel call
/// (nanoseconds, 0 if unknown) of the last trial under the decision and the
/// other way. For diagnostics and tuning.
pub fn learned_decisions() -> Vec<(usize, bool, u32, u32, u32)> {
    const R: Ordering = Ordering::Relaxed;
    let typical = |window| median(window).unwrap_or(0) as u32;
    LANES
        .iter()
        .enumerate()
        .filter(|(_, lane)| lane.decision.load(R) != UNSET)
        .map(|(i, lane)| {
            (1usize << (16 + i), lane.decision.load(R) == SPLIT, lane.period.load(R), typical(&lane.kernel[0]), typical(&lane.kernel[1]))
        })
        .collect()
}

struct Sample {
    lane: &'static Lane,
    trial: bool,
    slot: usize,
    start: u64,
}

/// How one kernel call is split, from [`plan`].
///
/// Keep it alive until the work is done: for inputs that fit in a core's
/// cache, or nearly, some calls are timed, from the creation of the plan to
/// its drop, and the policy learns from the result (see the
/// [module documentation](self)).
#[must_use = "a plan times the work it was made for: keep it until the work is done"]
pub struct Plan {
    tasks: usize,
    sample: Option<Sample>,
}

impl Plan {
    /// Number of threads (at least 1) to use, the calling thread included.
    #[inline]
    pub fn tasks(&self) -> usize {
        self.tasks
    }

    #[inline]
    fn fixed(tasks: usize) -> Self {
        Self { tasks, sample: None }
    }
}

impl Drop for Plan {
    #[inline]
    fn drop(&mut self) {
        if let Some(sample) = self.sample.take() {
            let nanos = clamp_nanos(now_nanos().saturating_sub(sample.start));
            sample.lane.kernel[usize::from(sample.trial)][sample.slot].store(nanos, Ordering::Relaxed);
        }
    }
}

/// How far beyond the private cache the policy still asks where the data is:
/// an input of up to this many times its size is small enough for the part
/// that a core has just written, and that others would have to fetch from
/// it, to matter.
const NEAR_CACHE: usize = 4;

/// How to split a kernel call over `rows` rows of `row_bytes` bytes of input
/// each. See the [module documentation](self) for the policy.
#[inline]
pub fn plan(rows: usize, row_bytes: usize) -> Plan {
    let bytes = rows.saturating_mul(row_bytes);
    let by_size = bytes / min_task_bytes();
    if by_size <= 1 {
        return Plan::fixed(1);
    }
    let l2 = cache::cache_info().l2;
    if bytes > l2.saturating_mul(NEAR_CACHE) {
        return Plan::fixed(tasks_beyond_cache(bytes, by_size));
    }
    learned_plan(bytes, by_size, bytes <= l2)
}

/// [`plan`] for an input that fits in the calling core's cache, or nearly.
fn learned_plan(bytes: usize, by_size: usize, resident: bool) -> Plan {
    let parts = if resident { by_size.min(RESIDENT_PARTS).min(team::cores()) } else { by_size.min(threads()) };
    if parts <= 1 || bytes < stream_rate() * (MIN_SPLIT_NANOS / 1000) {
        return Plan::fixed(1);
    }
    lane_for(bytes).plan(Request { parts, resident, bytes })
}

/// Rows per chunk when `rows` rows are processed by `threads` threads: a
/// multiple of 64, so that every chunk owns whole mask words.
///
/// An input larger than a core's cache gets a few chunks per thread (none
/// smaller than [`min_task_bytes`]), which lets a thread that finishes early
/// take over from one that is late. An input that fits in the cache gets
/// exactly one chunk per thread: each thread then touches the same part
/// every time, and that part stays in its core's cache.
#[inline]
pub fn chunk_rows(rows: usize, row_bytes: usize, threads: usize) -> usize {
    let threads = threads.max(1);
    let chunks = if rows.saturating_mul(row_bytes) <= cache::cache_info().l2 {
        threads
    } else {
        let min_rows = (min_task_bytes() / row_bytes.max(1)).max(64);
        (rows / min_rows).clamp(threads, threads * CHUNKS_PER_THREAD)
    };
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

    fn lane_with(kernel: [u32; 2], cycle: [u32; 2]) -> Lane {
        let lane = Lane::new();
        for i in 0..2 {
            for slot in 0..WINDOW as usize {
                // One outlier per window: an interrupt must not decide.
                let spike = if slot == 2 && kernel[i] != UNKNOWN { 50_000 } else { 0 };
                lane.kernel[i][slot].store(kernel[i].saturating_add(spike), Ordering::Relaxed);
                lane.cycle[i][slot].store(cycle[i], Ordering::Relaxed);
            }
        }
        lane
    }

    #[test]
    fn a_trial_is_judged_by_the_kernel_and_by_the_callers_cycle() {
        // (kernel now, kernel the other way), (cycle now, cycle the other way)
        let cases = [
            // Nothing measured on one side: no change.
            (([UNKNOWN, 1_000], [UNKNOWN, UNKNOWN]), false),
            (([3_000, UNKNOWN], [3_100, UNKNOWN]), false),
            // A tight loop over a column that is only read: both get faster.
            (([3_000, 1_700], [3_100, 1_800]), true),
            // A gain below a ninth is not worth changing for.
            (([3_000, 2_800], [3_100, 2_900]), false),
            // The kernel got faster but the caller's loop got slower: the
            // helpers took the column away from the thread that writes it.
            (([2_500, 1_500], [8_800, 10_900]), false),
            // The same from the other side: the kernel alone is slower
            // without helpers, the loop as a whole clearly faster.
            (([1_500, 2_500], [10_900, 8_800]), true),
            // The kernel got slower and the loop did not gain: stay.
            (([1_700, 3_000], [1_800, 3_100]), false),
            // The caller does so much between calls that its cycle says
            // nothing: the kernel decides alone.
            (([3_000, 1_700], [900_000, 1_100_000]), true),
            (([3_000, 2_900], [900_000, 500_000]), false),
            // No cycle was measured (calls did not follow each other).
            (([3_000, 1_700], [UNKNOWN, UNKNOWN]), true),
        ];
        for ((kernel, cycle), want) in cases {
            assert_eq!(lane_with(kernel, cycle).other_way_is_better(false), want, "kernel {kernel:?} cycle {cycle:?}");
        }
        // Going back to the starting assumption takes less: a split that was
        // adopted and is now 8 % slower than one thread is dropped, though an
        // 8 % gain would never have had it adopted.
        let slightly = lane_with([3_240, 3_000], [3_300, 3_060]);
        assert!(slightly.other_way_is_better(true) && !slightly.other_way_is_better(false));
        let equal = lane_with([3_000, 3_000], [3_060, 3_060]);
        assert!(!equal.other_way_is_better(true) && !equal.other_way_is_better(false));
    }

    #[test]
    fn a_lane_starts_alone_and_times_a_window_before_it_tries_anything() {
        let lane: &'static Lane = Box::leak(Box::new(Lane::new()));
        let request = Request { parts: 2, resident: true, bytes: 512 << 10 };
        for call in 1..FIRST_TRIAL {
            let plan = lane.plan(request);
            assert!(plan.tasks() == 1 && plan.sample.is_none(), "call {call}");
        }
        for call in 0..WINDOW {
            let plan = lane.plan(request);
            assert_eq!(plan.tasks(), 1, "baseline call {call}");
            assert!(plan.sample.as_ref().is_some_and(|s| !s.trial), "baseline call {call}");
        }
        assert!(median(&lane.kernel[0]).is_some());
        assert!(median(&lane.cycle[0]).is_some());
        // Whatever the team does from here on, the lane keeps answering and
        // comes back to a decision.
        for _ in 0..20_000 {
            let plan = lane.plan(request);
            assert!((1..=2).contains(&plan.tasks()));
        }
        assert!(lane.period.load(Ordering::Relaxed) > FIRST_TRIAL);
        assert!(lane.period.load(Ordering::Relaxed) <= MAX_TRIAL_PERIOD);

        // An input that is not in the caller's cache starts out split.
        let lane: &'static Lane = Box::leak(Box::new(Lane::new()));
        let _ = lane.plan(Request { parts: 2, resident: false, bytes: 4 << 20 });
        assert_eq!(lane.decision.load(Ordering::Relaxed), SPLIT);
    }

    #[test]
    fn the_typical_call_of_a_window_is_its_median() {
        let window = |values: [u32; WINDOW as usize]| values.map(AtomicU32::new);
        assert_eq!(median(&window([UNKNOWN; WINDOW as usize])), None);
        assert_eq!(median(&window([5, 7, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN])), None);
        assert_eq!(median(&window([5, 7, 6, UNKNOWN, UNKNOWN, UNKNOWN])), Some(6));
        assert_eq!(median(&window([9, 1, 8, 2, 7, 900_000])), Some(8));
    }

    #[test]
    fn plans_are_sized_by_the_input() {
        assert_eq!(plan(0, 8).tasks(), 1);
        assert_eq!(plan(100, 8).tasks(), 1);
        assert_eq!(plan(min_task_bytes() / 8, 8).tasks(), 1);
        for rows in [10_000usize, 100_000, 1_000_000, 50_000_000] {
            let tasks = plan(rows, 8).tasks();
            assert!((1..=threads()).contains(&tasks), "rows {rows}: {tasks}");
            // One chunk per thread where the input fits in the cache, a few
            // where it does not.
            let chunks = rows.div_ceil(chunk_rows(rows, 8, tasks.max(2)));
            if rows * 8 <= cache::cache_info().l2 {
                assert!(chunks <= tasks.max(2), "rows {rows}: {chunks} chunks");
            }
        }
        assert!(team::cores() >= 1 && team::cores() <= threads());
    }

    #[test]
    fn chunk_rows_respects_bounds() {
        for rows in [1usize, 64, 1000, 100_000, 10_000_000] {
            for row_bytes in [1usize, 4, 8, 16] {
                for threads in [1usize, 2, 4, 8] {
                    let step = chunk_rows(rows, row_bytes, threads);
                    assert!(step >= 64 && step.is_multiple_of(64));
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
