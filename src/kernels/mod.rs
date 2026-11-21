//! Layer 1 — kernels.
//!
//! Tight loops over native slices and 64-bit mask words. Nothing here knows
//! about columns, names, nulls-as-a-concept or frames; the layers above
//! translate those into slices and bitmasks and call down.
//!
//! # Dispatch
//!
//! The crate is compiled for the baseline of its target and selects wider
//! instruction sets at run time, once, in [`level`]. Hand-written AVX2 kernels
//! live in [`x86`]; loops that LLVM vectorises well on its own are written
//! once and re-instantiated inside a `#[target_feature]` wrapper by the
//! `simd_dispatch!` macro, so they too run with 256-bit vectors without the
//! binary being built for a specific CPU.
//!
//! # Filtering
//!
//! A filter is two kernels joined by a bitmask: *compare* turns 64 rows into
//! one word, *compact* turns a word and 64 rows into the surviving rows. The
//! drivers in this module combine them three ways:
//!
//! * [`filter_by_mask`] — the mask already exists (a boolean column).
//! * [`filter_by_cmp_scalar`] — predicate and payload are the same column:
//!   the two kernels alternate over 4096-row blocks so the block is still in
//!   L1/L2 when it is compacted, and the column is read from memory once.
//! * the parallel form of both — the mask is computed by all cores, per-range
//!   population counts give every range its exact output offset, and the cores
//!   compact disjoint ranges straight into one contiguous allocation.

pub mod aggregate;
pub mod arith;
pub mod cache;
pub mod par;
pub mod pipeline;
pub mod scalar;
pub mod tables;
#[cfg(target_arch = "x86_64")]
pub mod x86;

use std::cmp::Ordering as CmpOrdering;
use std::sync::atomic::{AtomicU8, Ordering};

use rayon::prelude::*;

use crate::memory::bitmap::count_ones_range;
use crate::memory::{APRON, AlignedVec, ColumnAlloc, Pod, words_for, zeroed_vec};

/// Instruction-set tier selected at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    /// Portable code only.
    Scalar = 1,
    /// x86-64-v3: AVX2, BMI1/2, POPCNT, LZCNT, FMA.
    Avx2 = 2,
}

static LEVEL: AtomicU8 = AtomicU8::new(0);

/// The instruction-set tier in use. Detected on first call.
///
/// Set `DATAFRAMES_SIMD=scalar` in the environment to force the portable
/// kernels (useful for testing and for measuring what the vector paths buy).
#[inline]
pub fn level() -> Level {
    match LEVEL.load(Ordering::Relaxed) {
        1 => Level::Scalar,
        2 => Level::Avx2,
        _ => detect(),
    }
}

#[cold]
fn detect() -> Level {
    let mut found = Level::Scalar;
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("popcnt")
            && std::is_x86_feature_detected!("lzcnt")
            && std::is_x86_feature_detected!("fma")
        {
            found = Level::Avx2;
        }
    }
    if matches!(std::env::var("DATAFRAMES_SIMD").as_deref(), Ok("scalar")) {
        found = Level::Scalar;
    }
    LEVEL.store(found as u8, Ordering::Relaxed);
    found
}

/// Define `fn name(args)` that runs the `#[inline(always)]` implementation
/// `imp` re-compiled for x86-64-v3 when the CPU has it, and as-is otherwise.
macro_rules! simd_dispatch {
    ($(#[$m:meta])* $vis:vis fn $name:ident $(<$($g:ident : $b:path),*>)? ($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? = $imp:path) => {
        $(#[$m])*
        #[inline]
        $vis fn $name $(<$($g: $b),*>)? ($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,bmi1,bmi2,popcnt,lzcnt,fma")]
                fn v3 $(<$($g: $b),*>)? ($($arg: $ty),*) $(-> $ret)? {
                    $imp($($arg),*)
                }
                if $crate::kernels::level() >= $crate::kernels::Level::Avx2 {
                    // SAFETY: `level()` verified the CPU features at run time.
                    return unsafe { v3($($arg),*) };
                }
            }
            $imp($($arg),*)
        }
    };
}
pub(crate) use simd_dispatch;

/// A comparison operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl CmpOp {
    /// Evaluate the operator on two values in the total order.
    #[inline]
    pub fn eval<T: Kernel>(self, a: T, b: T) -> bool {
        match self {
            CmpOp::Eq => a.tot_eq(b),
            CmpOp::Ne => !a.tot_eq(b),
            CmpOp::Lt => a.tot_lt(b),
            CmpOp::Le => !b.tot_lt(a),
            CmpOp::Gt => b.tot_lt(a),
            CmpOp::Ge => !a.tot_lt(b),
        }
    }

    /// The operator `op'` with `a op b == b op' a`.
    #[inline]
    pub fn swap_operands(self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Ge => CmpOp::Le,
            other => other,
        }
    }

    /// The operator `op'` with `a op' b == !(a op b)`.
    #[inline]
    pub fn negate(self) -> Self {
        match self {
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Le => CmpOp::Gt,
            CmpOp::Gt => CmpOp::Le,
            CmpOp::Ge => CmpOp::Lt,
        }
    }
}

/// Primitive element types the kernels operate on.
///
/// Comparisons use a *total order*: for floats, NaN equals NaN and is greater
/// than every other value (the convention Polars and Arrow use); for integers
/// it is the usual order.
pub trait Kernel: Pod + PartialOrd + PartialEq + Default + std::fmt::Debug {
    /// Equality in the total order.
    fn tot_eq(self, other: Self) -> bool;
    /// Strict less-than in the total order.
    fn tot_lt(self, other: Self) -> bool;

    /// Three-way comparison in the total order.
    #[inline]
    fn tot_cmp(self, other: Self) -> CmpOrdering {
        if self.tot_lt(other) {
            CmpOrdering::Less
        } else if other.tot_lt(self) {
            CmpOrdering::Greater
        } else {
            CmpOrdering::Equal
        }
    }

    /// Set bit `i` of `out` to `values[i] op rhs`; bits past `values.len()` in
    /// the last word are cleared. Returns the number of set bits.
    ///
    /// `out` must hold at least `words_for(values.len())` words.
    fn cmp_scalar(values: &[Self], op: CmpOp, rhs: Self, out: &mut [u64]) -> usize;

    /// Set bit `i` of `out` to `lhs[i] op rhs[i]`. Same contract as
    /// [`cmp_scalar`](Self::cmp_scalar). Panics if the lengths differ.
    #[inline]
    fn cmp_values(lhs: &[Self], rhs: &[Self], op: CmpOp, out: &mut [u64]) -> usize {
        scalar::cmp_values(lhs, rhs, op, out)
    }
}

macro_rules! impl_kernel_int {
    ($($t:ty => $signed:ty, $unsigned:expr, $avx2:ident;)*) => {$(
        impl Kernel for $t {
            #[inline(always)]
            fn tot_eq(self, other: Self) -> bool { self == other }
            #[inline(always)]
            fn tot_lt(self, other: Self) -> bool { self < other }

            fn cmp_scalar(values: &[Self], op: CmpOp, rhs: Self, out: &mut [u64]) -> usize {
                #[cfg(target_arch = "x86_64")]
                if level() >= Level::Avx2 {
                    let full = values.len() / 64 * 64;
                    let (head, tail) = values.split_at(full);
                    let (out_head, out_tail) = out.split_at_mut(full / 64);
                    // SAFETY: same-width integer reinterpretation of an initialised slice.
                    let head: &[$signed] =
                        unsafe { std::slice::from_raw_parts(head.as_ptr().cast(), head.len()) };
                    // SAFETY: AVX2 was detected; `head` is a whole number of words.
                    let n = unsafe { x86::$avx2::<{ $unsigned }>(head, rhs as $signed, op, out_head) };
                    return n + if tail.is_empty() { 0 } else { scalar::cmp_scalar(tail, op, rhs, out_tail) };
                }
                scalar::cmp_scalar(values, op, rhs, out)
            }
        }
    )*};
}

impl_kernel_int! {
    i8  => i8,  false, cmp_scalar_i8;
    u8  => i8,  true,  cmp_scalar_i8;
    i16 => i16, false, cmp_scalar_i16;
    u16 => i16, true,  cmp_scalar_i16;
    i32 => i32, false, cmp_scalar_i32;
    u32 => i32, true,  cmp_scalar_i32;
    i64 => i64, false, cmp_scalar_i64;
    u64 => i64, true,  cmp_scalar_i64;
}

macro_rules! impl_kernel_float {
    ($($t:ty => $avx2:ident;)*) => {$(
        impl Kernel for $t {
            #[inline(always)]
            fn tot_eq(self, other: Self) -> bool {
                if self.is_nan() { other.is_nan() } else { self == other }
            }
            #[inline(always)]
            fn tot_lt(self, other: Self) -> bool {
                // `!(self >= other)` is true for NaN on either side; NaN is the
                // maximum, so the result is true only when `self` is not NaN.
                !(self >= other) && !self.is_nan()
            }

            fn cmp_scalar(values: &[Self], op: CmpOp, rhs: Self, out: &mut [u64]) -> usize {
                #[cfg(target_arch = "x86_64")]
                if level() >= Level::Avx2 && !rhs.is_nan() {
                    let full = values.len() / 64 * 64;
                    let (head, tail) = values.split_at(full);
                    let (out_head, out_tail) = out.split_at_mut(full / 64);
                    // SAFETY: AVX2 was detected; `head` is a whole number of
                    // words; `rhs` is not NaN.
                    let n = unsafe { x86::$avx2::<false>(head, rhs, op, out_head) };
                    return n + if tail.is_empty() { 0 } else { scalar::cmp_scalar(tail, op, rhs, out_tail) };
                }
                scalar::cmp_scalar(values, op, rhs, out)
            }
        }
    )*};
}

impl_kernel_float! {
    f32 => cmp_scalar_f32;
    f64 => cmp_scalar_f64;
}

// ---------------------------------------------------------------------------
// compaction
// ---------------------------------------------------------------------------

/// Copy the elements of `values[..n]` whose bit is set in `mask` to `out`, in
/// order, and return how many were written.
///
/// With `stream` the output is written with non-temporal stores (worth it only
/// for outputs larger than the cache, see [`cache::stream_threshold`]).
///
/// # Safety
/// * `values` must be valid for `n` reads and `mask` must hold at least
///   `words_for(n)` words.
/// * `out` must be aligned for `T`, valid for `out_cap` writes and must not
///   overlap `values`. Any of those `out_cap` slots may be overwritten,
///   including ones past the returned count; nothing beyond them is touched.
///   If more than `out_cap` bits are set, only the first `out_cap` selected
///   elements are written.
/// * With `stream`, the calling thread must call [`pipeline::stream_fence`]
///   before the output is read.
#[inline]
pub unsafe fn compact<T: Pod>(
    values: *const T,
    mask: &[u64],
    n: usize,
    out: *mut T,
    out_cap: usize,
    stream: bool,
) -> usize {
    debug_assert!(mask.len() >= words_for(n));
    #[cfg(target_arch = "x86_64")]
    if level() >= Level::Avx2 {
        // SAFETY: the caller's contract is forwarded unchanged; the pointer
        // casts keep the element width; AVX2 was detected.
        unsafe {
            match (size_of::<T>(), stream) {
                (8, true) => return x86::compact8_stream(values.cast(), mask, n, out.cast(), out_cap),
                (4, true) => return x86::compact4_stream(values.cast(), mask, n, out.cast(), out_cap),
                (8, false) => return x86::compact8(values.cast(), mask, n, out.cast(), out_cap),
                (4, false) => return x86::compact4(values.cast(), mask, n, out.cast(), out_cap),
                (2, _) => return x86::compact2(values.cast(), mask, n, out.cast(), out_cap),
                (1, _) => return x86::compact1(values.cast(), mask, n, out.cast(), out_cap),
                (16, _) if values as usize % 8 == 0 && out as usize % 8 == 0 => {
                    return compact16(values.cast(), mask, n, out.cast(), out_cap, stream);
                }
                _ => {}
            }
        }
    }
    let _ = stream;
    // SAFETY: forwarded.
    unsafe { scalar::compact(values, mask, n, out, out_cap) }
}

/// Copy `src` to `dst`, bypassing the cache when `stream` is set.
///
/// # Safety
/// `dst` must be valid for `src.len()` writes and must not overlap `src`.
#[inline]
pub unsafe fn copy_to<T: Pod>(src: &[T], dst: *mut T, stream: bool) {
    #[cfg(target_arch = "x86_64")]
    if stream && level() >= Level::Avx2 {
        // SAFETY: forwarded; AVX2 was detected.
        unsafe { x86::copy_stream(src.as_ptr().cast(), dst.cast(), size_of_val(src)) };
        return;
    }
    let _ = stream;
    // SAFETY: forwarded.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
}

/// Spread the low 32 bits of `x` so that bit `i` lands on bits `2i` and `2i + 1`.
#[inline]
fn duplicate_bits(x: u32) -> u64 {
    let mut v = x as u64;
    v = (v | v << 16) & 0x0000_FFFF_0000_FFFF;
    v = (v | v << 8) & 0x00FF_00FF_00FF_00FF;
    v = (v | v << 4) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | v << 2) & 0x3333_3333_3333_3333;
    v = (v | v << 1) & 0x5555_5555_5555_5555;
    v | v << 1
}

/// [`compact`] for 16-byte elements whose storage is 8-byte aligned: each
/// element is treated as two 64-bit lanes and each mask bit is doubled, which
/// lets the 64-bit vector kernels (plain and streaming) do the work.
///
/// # Safety
/// As [`compact`], with `values` and `out` aligned to 8 bytes and AVX2 available.
#[cfg(target_arch = "x86_64")]
unsafe fn compact16(values: *const u64, mask: &[u64], n: usize, out: *mut u64, out_cap: usize, stream: bool) -> usize {
    const BLOCK_WORDS: usize = 32;
    let mut doubled = [0u64; 2 * BLOCK_WORDS];
    let n_words = words_for(n);
    let (mut written, mut wi) = (0usize, 0usize);
    while wi < n_words {
        let words = BLOCK_WORDS.min(n_words - wi);
        let rows = (words * 64).min(n - wi * 64);
        for j in 0..words {
            let w = mask[wi + j];
            doubled[2 * j] = duplicate_bits(w as u32);
            doubled[2 * j + 1] = duplicate_bits((w >> 32) as u32);
        }
        // SAFETY: the block covers `rows` elements = `2 * rows` lanes of
        // `values`; `out` has `out_cap - written` elements left, i.e. twice
        // as many lanes, and the lane kernels never exceed their cap.
        let lanes = unsafe {
            let (src, dst, cap) = (values.add(2 * wi * 64), out.add(2 * written), 2 * (out_cap - written));
            if stream {
                x86::compact8_stream(src, &doubled, 2 * rows, dst, cap)
            } else {
                x86::compact8(src, &doubled, 2 * rows, dst, cap)
            }
        };
        written += lanes / 2;
        wi += words;
    }
    written
}

/// The positions of the set bits among the first `n` bits of `mask`, each
/// offset by `base`, in ascending order.
pub fn mask_indices(mask: &[u64], n: usize, base: u32) -> AlignedVec<u32> {
    assert!(mask.len() >= words_for(n), "mask too short for {n} rows");
    assert!(n <= u32::MAX as usize - base as usize, "row indices must fit in 32 bits");
    let count = count_ones_range(mask, 0, n);
    // The allocator's apron provides the eight slots of vector overhang.
    let mut out: AlignedVec<u32> = Vec::with_capacity_in(count, ColumnAlloc);
    #[cfg(target_arch = "x86_64")]
    if level() >= Level::Avx2 {
        // SAFETY: capacity `count` plus a 64-byte apron (16 slots); AVX2 detected.
        unsafe {
            let written = x86::mask_indices(mask, n, base, out.as_mut_ptr());
            debug_assert_eq!(written, count);
            out.set_len(count);
        }
        return out;
    }
    for (wi, &word) in mask[..words_for(n)].iter().enumerate() {
        let mut w = if wi + 1 == words_for(n) { word & crate::memory::tail_mask(n) } else { word };
        while w != 0 {
            out.push(base + (wi * 64) as u32 + w.trailing_zeros());
            w &= w - 1;
        }
    }
    out
}

/// Selections sparser than this are compacted through an index list
/// ([`mask_indices`] + gather) instead of by the permute kernels.
pub const SPARSE_SELECTIVITY: f64 = 0.125;

/// Keep the bits of `values` selected by `mask` (both `n` bits long, aligned),
/// packed. Returns the packed words; their bit length is the number of set
/// bits among the first `n` bits of `mask`.
pub fn compact_bits(values: &[u64], mask: &[u64], n: usize, count: usize) -> AlignedVec<u64> {
    let mut out: AlignedVec<u64> = zeroed_vec(words_for(count) + 1);
    let len = {
        #[cfg(target_arch = "x86_64")]
        if level() >= Level::Avx2 {
            // SAFETY: BMI2 and POPCNT are part of the detected level.
            unsafe { x86::compact_bits(values, mask, n, &mut out, 0) }
        } else {
            scalar::compact_bits(values, mask, n, &mut out, 0)
        }
        #[cfg(not(target_arch = "x86_64"))]
        scalar::compact_bits(values, mask, n, &mut out, 0)
    };
    debug_assert_eq!(len, count);
    out.truncate(words_for(count));
    out
}

// ---------------------------------------------------------------------------
// drivers
// ---------------------------------------------------------------------------

/// Rows per block in the fused filter: 64 mask words, 4 KiB to 32 KiB of
/// payload, sized to stay cache-resident between compare and compact.
pub const BLOCK_ROWS: usize = 4096;

/// `values[i] op rhs` as mask words plus the number of matches.
/// Uses every core for large inputs.
pub fn cmp_scalar_mask<T: Kernel>(values: &[T], op: CmpOp, rhs: T) -> (AlignedVec<u64>, usize) {
    cmp_scalar_mask_with(values, op, rhs, par::tasks_for(values.len(), size_of::<T>()))
}

/// [`cmp_scalar_mask`] with an explicit degree of parallelism.
pub fn cmp_scalar_mask_with<T: Kernel>(
    values: &[T],
    op: CmpOp,
    rhs: T,
    tasks: usize,
) -> (AlignedVec<u64>, usize) {
    let n = values.len();
    let mut words: AlignedVec<u64> = zeroed_vec(words_for(n));
    let count = if tasks <= 1 || n < 128 {
        T::cmp_scalar(values, op, rhs, &mut words)
    } else {
        let rows = par::rows_per_task(n, tasks);
        values
            .par_chunks(rows)
            .zip(words.par_chunks_mut(rows / 64))
            .map(|(v, w)| T::cmp_scalar(v, op, rhs, w))
            .sum()
    };
    (words, count)
}

/// `lhs[i] op rhs[i]` as mask words plus the number of matches.
///
/// # Panics
/// If the slices differ in length.
pub fn cmp_values_mask<T: Kernel>(lhs: &[T], rhs: &[T], op: CmpOp) -> (AlignedVec<u64>, usize) {
    assert_eq!(lhs.len(), rhs.len(), "comparison operands must have equal length");
    let n = lhs.len();
    let mut words: AlignedVec<u64> = zeroed_vec(words_for(n));
    let tasks = par::tasks_for(n, 2 * size_of::<T>());
    let count = if tasks <= 1 || n < 128 {
        T::cmp_values(lhs, rhs, op, &mut words)
    } else {
        let rows = par::rows_per_task(n, tasks);
        lhs.par_chunks(rows)
            .zip(rhs.par_chunks(rows))
            .zip(words.par_chunks_mut(rows / 64))
            .map(|((a, b), w)| T::cmp_values(a, b, op, w))
            .sum()
    };
    (words, count)
}

/// The elements of `values` whose bit is set in `mask` (aligned words covering
/// `values.len()` bits). Uses every core for large inputs.
pub fn filter_by_mask<T: Pod>(values: &[T], mask: &[u64]) -> AlignedVec<T> {
    filter_by_mask_with(values, mask, par::tasks_for(values.len(), size_of::<T>()), None)
}

/// [`filter_by_mask`] with the degree of parallelism and the store policy
/// chosen by the caller (`stream: None` decides from the output size).
pub fn filter_by_mask_with<T: Pod>(
    values: &[T],
    mask: &[u64],
    tasks: usize,
    stream: Option<bool>,
) -> AlignedVec<T> {
    let n = values.len();
    assert!(mask.len() >= words_for(n), "mask too short for {n} rows");
    let apron = APRON / size_of::<T>();
    let streaming = |count: usize| stream.unwrap_or(count * size_of::<T>() >= cache::stream_threshold());
    if tasks <= 1 || n < 128 {
        let count = count_ones_range(mask, 0, n);
        if (count as f64) < SPARSE_SELECTIVITY * n as f64 && n <= u32::MAX as usize {
            // Sparse: list the selected rows, then gather only those.
            let idx = mask_indices(mask, n, 0);
            // SAFETY: every index is the position of a set bit below `n`.
            return unsafe { arith::take_unchecked(values, &idx) };
        }
        let mut out: AlignedVec<T> = Vec::with_capacity_in(count, ColumnAlloc);
        // SAFETY: `out` has room for `count` elements plus the allocator's
        // apron; `compact` writes at most `count + apron` slots and `count`
        // is the exact population count, so `count` slots are initialised.
        unsafe {
            if count == n {
                copy_to(values, out.as_mut_ptr(), streaming(count));
            } else {
                let stream = streaming(count);
                let written = compact(values.as_ptr(), mask, n, out.as_mut_ptr(), count + apron, stream);
                debug_assert_eq!(written, count);
                if stream {
                    pipeline::stream_fence();
                }
            }
            out.set_len(count);
        }
        return out;
    }

    let ranges = par::word_ranges(n, tasks);
    let mut offsets = Vec::with_capacity(ranges.len() + 1);
    let mut count = 0usize;
    for &(start, end) in &ranges {
        offsets.push(count);
        count += count_ones_range(mask, start, end - start);
    }
    offsets.push(count);
    let stream = streaming(count);
    let mut out: AlignedVec<T> = Vec::with_capacity_in(count, ColumnAlloc);
    let dst = par::SendPtr(out.as_mut_ptr());
    let last = ranges.len() - 1;
    ranges.par_iter().enumerate().for_each(|(i, &(start, end))| {
        let dst = dst;
        let exact = offsets[i + 1] - offsets[i];
        // Only the final range may spill into the apron; the others must stay
        // inside their own slot or they would race with their neighbour.
        let cap = if i == last { exact + apron } else { exact };
        // SAFETY: ranges start on word boundaries, so `mask[start / 64..]`
        // lines up with `values[start..end]`. Each range writes only
        // `offsets[i] .. offsets[i] + cap`, and those intervals are disjoint
        // and inside the allocation (the last one extends into the apron).
        unsafe {
            let written = compact(
                values.as_ptr().add(start),
                &mask[start / 64..],
                end - start,
                dst.0.add(offsets[i]),
                cap,
                stream,
            );
            debug_assert_eq!(written, exact);
            if stream {
                pipeline::stream_fence();
            }
        }
    });
    // SAFETY: every range initialised exactly its `exact` elements.
    unsafe { out.set_len(count) };
    out
}

/// The elements of `values` that satisfy `value op rhs`.
///
/// Compare and compact alternate block by block, so the column is read from
/// memory once and no full-length mask is materialised.
pub fn filter_by_cmp_scalar<T: Kernel>(values: &[T], op: CmpOp, rhs: T) -> AlignedVec<T> {
    filter_by_cmp_scalar_with(values, op, rhs, 1, None)
}

/// [`filter_by_cmp_scalar`] with explicit parallelism and store policy.
/// With `tasks > 1` the mask is materialised and both phases run in parallel.
pub fn filter_by_cmp_scalar_with<T: Kernel>(
    values: &[T],
    op: CmpOp,
    rhs: T,
    tasks: usize,
    stream: Option<bool>,
) -> AlignedVec<T> {
    if tasks > 1 {
        let (mask, _) = cmp_scalar_mask_with(values, op, rhs, tasks);
        return filter_by_mask_with(values, &mask, tasks, stream);
    }
    let apron = APRON / size_of::<T>();
    let estimate = estimate_matches(values, op, rhs);
    let stream = stream.unwrap_or(estimate * size_of::<T>() >= cache::stream_threshold());
    let mut out: AlignedVec<T> = Vec::with_capacity_in(estimate, ColumnAlloc);
    let mut words = [0u64; BLOCK_ROWS / 64];
    for block in values.chunks(BLOCK_ROWS) {
        let count = T::cmp_scalar(block, op, rhs, &mut words);
        if count == 0 {
            continue;
        }
        out.reserve(count);
        let len = out.len();
        // SAFETY: after `reserve` there are at least `count` free slots, and
        // the allocator guarantees an apron past the capacity, so `compact`
        // may clobber `count + apron` slots; exactly `count` are initialised.
        unsafe {
            let dst = out.as_mut_ptr().add(len);
            let written = compact(block.as_ptr(), &words, block.len(), dst, count + apron, stream);
            debug_assert_eq!(written, count);
            out.set_len(len + count);
        }
    }
    if stream {
        pipeline::stream_fence();
    }
    if out.capacity() > 2 * out.len() + BLOCK_ROWS {
        out.shrink_to_fit();
    }
    out
}

/// Number of elements satisfying `value op rhs`.
pub fn count_cmp_scalar<T: Kernel>(values: &[T], op: CmpOp, rhs: T) -> usize {
    let serial = |v: &[T]| -> usize {
        let mut words = [0u64; BLOCK_ROWS / 64];
        v.chunks(BLOCK_ROWS).map(|b| T::cmp_scalar(b, op, rhs, &mut words)).sum()
    };
    let tasks = par::tasks_for(values.len(), size_of::<T>());
    if tasks <= 1 {
        return serial(values);
    }
    values.par_chunks(par::rows_per_task(values.len(), tasks)).map(serial).sum()
}

/// Guess the number of matches from 16 evenly spaced 64-row stripes, padded so
/// that a typical column never needs to grow its output.
fn estimate_matches<T: Kernel>(values: &[T], op: CmpOp, rhs: T) -> usize {
    const STRIPES: usize = 16;
    const STRIPE: usize = 64;
    let n = values.len();
    if n <= 4 * STRIPES * STRIPE {
        return n;
    }
    let step = n / STRIPES;
    let mut hits = 0usize;
    let mut w = [0u64; 1];
    for s in 0..STRIPES {
        let start = s * step;
        hits += T::cmp_scalar(&values[start..start + STRIPE], op, rhs, &mut w);
    }
    let sampled = STRIPES * STRIPE;
    // Pad by three standard errors of the estimate (worst case p = 1/2) plus a block.
    let est = (n as f64) * (hits as f64 / sampled as f64 + 1.5 / (sampled as f64).sqrt());
    (est as usize + BLOCK_ROWS).min(n)
}

#[cfg(test)]
mod tests;
