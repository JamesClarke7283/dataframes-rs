//! Reductions over native slices.
//!
//! Two properties hold for every reduction here:
//!
//! * **Vector code from a portable build.** The loops are instantiated for
//!   AVX2 at run time. Integer sums, minima and maxima are plain folds, which
//!   is the shape LLVM vectorises best. A floating-point sum may not be
//!   reordered by the compiler, so it keeps a fixed number of accumulator
//!   *lanes* (element `i` goes to lane `i % LANES`); floating-point minima
//!   and maxima use explicit vector kernels.
//! * **One answer.** The association order is spelled out — lanes within a
//!   piece of [`PIECE_ROWS`] rows, pieces left to right — so a floating-point
//!   sum is bit-for-bit identical on every CPU, at every vector width and for
//!   any number of threads. As a bonus the lanes bound rounding error much
//!   like pairwise summation does.

use super::{Kernel, par, simd_dispatch};
use crate::memory::tail_mask;

/// Accumulator lanes: eight 256-bit vectors of `f64`/`i64`.
const LANES: usize = 32;

/// Arithmetic the reductions need from an element type.
pub trait Num: Kernel {
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Smallest value in the total order (`-inf` for floats).
    const MIN_VALUE: Self;
    /// Largest non-NaN value (`+inf` for floats).
    const MAX_VALUE: Self;
    /// `true` for `f32` and `f64`.
    const IS_FLOAT: bool;
    /// `true` for signed integers and floats.
    const IS_SIGNED: bool;

    /// `self + rhs`; wraps on integer overflow.
    fn add(self, rhs: Self) -> Self;
    /// `self - rhs`; wraps on integer overflow.
    fn sub(self, rhs: Self) -> Self;
    /// `self * rhs`; wraps on integer overflow.
    fn mul(self, rhs: Self) -> Self;
    /// `self / rhs`; `None` for integer division by zero or overflow.
    fn checked_div(self, rhs: Self) -> Option<Self>;
    /// `self % rhs`; `None` for integer remainder by zero or overflow.
    fn checked_rem(self, rhs: Self) -> Option<Self>;
    /// Lossy conversion to `f64`.
    fn to_f64(self) -> f64;
    /// Lossy conversion to `i64` (saturating for floats, wrapping for `u64`).
    fn to_i64(self) -> i64;
    /// Conversion from `f64` with Rust `as` semantics (saturating, NaN → 0).
    fn from_f64(v: f64) -> Self;
    /// Conversion from `i64` with Rust `as` semantics (wrapping for narrower integers).
    fn from_i64(v: i64) -> Self;
    /// `true` only for a floating-point NaN.
    fn is_nan(self) -> bool;

    /// `(min, max)` over the non-NaN values of one piece; `(MAX_VALUE,
    /// MIN_VALUE)` if there are none. Overridden where a hand-written vector
    /// kernel beats the generic loop.
    #[doc(hidden)]
    #[inline]
    fn min_max_piece(values: &[Self]) -> (Self, Self) {
        min_max_serial(values)
    }

    /// Smallest non-NaN value of one piece (`MAX_VALUE` if there is none).
    #[doc(hidden)]
    #[inline]
    fn min_piece(values: &[Self]) -> Self {
        min_serial(values)
    }

    /// Largest non-NaN value of one piece (`MIN_VALUE` if there is none).
    #[doc(hidden)]
    #[inline]
    fn max_piece(values: &[Self]) -> Self {
        max_serial(values)
    }
}

macro_rules! impl_num_int {
    ($($t:ty: $signed:expr),*) => {$(
        impl Num for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MIN_VALUE: Self = <$t>::MIN;
            const MAX_VALUE: Self = <$t>::MAX;
            const IS_FLOAT: bool = false;
            const IS_SIGNED: bool = $signed;
            #[inline(always)] fn add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
            #[inline(always)] fn sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
            #[inline(always)] fn mul(self, rhs: Self) -> Self { self.wrapping_mul(rhs) }
            #[inline(always)] fn checked_div(self, rhs: Self) -> Option<Self> { <$t>::checked_div(self, rhs) }
            #[inline(always)] fn checked_rem(self, rhs: Self) -> Option<Self> { <$t>::checked_rem(self, rhs) }
            #[inline(always)] fn to_f64(self) -> f64 { self as f64 }
            #[inline(always)] fn to_i64(self) -> i64 { self as i64 }
            #[inline(always)] fn from_f64(v: f64) -> Self { v as $t }
            #[inline(always)] fn from_i64(v: i64) -> Self { v as $t }
            #[inline(always)] fn is_nan(self) -> bool { false }
        }
    )*};
}
impl_num_int!(i8: true, i16: true, i32: true, i64: true, u8: false, u16: false, u32: false, u64: false);

macro_rules! impl_num_float {
    ($($t:ty => $kernel:ident, $min:ident, $max:ident);*) => {$(
        impl Num for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const MIN_VALUE: Self = <$t>::NEG_INFINITY;
            const MAX_VALUE: Self = <$t>::INFINITY;
            const IS_FLOAT: bool = true;
            const IS_SIGNED: bool = true;
            #[inline(always)] fn add(self, rhs: Self) -> Self { self + rhs }
            #[inline(always)] fn sub(self, rhs: Self) -> Self { self - rhs }
            #[inline(always)] fn mul(self, rhs: Self) -> Self { self * rhs }
            #[inline(always)] fn checked_div(self, rhs: Self) -> Option<Self> { Some(self / rhs) }
            #[inline(always)] fn checked_rem(self, rhs: Self) -> Option<Self> { Some(self % rhs) }
            #[inline(always)] fn to_f64(self) -> f64 { self as f64 }
            #[inline(always)] fn to_i64(self) -> i64 { self as i64 }
            #[inline(always)] fn from_f64(v: f64) -> Self { v as $t }
            #[inline(always)] fn from_i64(v: i64) -> Self { v as $t }
            #[inline(always)] fn is_nan(self) -> bool { <$t>::is_nan(self) }

            #[inline]
            fn min_max_piece(values: &[Self]) -> (Self, Self) {
                // LLVM keeps the select-based lanes scalar for floats, so use
                // `vminp*`/`vmaxp*` directly where they exist.
                #[cfg(target_arch = "x86_64")]
                if super::level() >= super::Level::Avx512 {
                    // SAFETY: AVX-512 was detected.
                    return unsafe { super::x86_512::$kernel(values) };
                }
                #[cfg(target_arch = "x86_64")]
                if super::level() >= super::Level::Avx2 {
                    // SAFETY: AVX2 was detected.
                    return unsafe { super::x86::$kernel(values) };
                }
                min_max_serial(values)
            }

            #[inline]
            fn min_piece(values: &[Self]) -> Self {
                #[cfg(target_arch = "x86_64")]
                if super::level() >= super::Level::Avx512 {
                    // SAFETY: AVX-512 was detected.
                    return unsafe { super::x86_512::$min(values) };
                }
                #[cfg(target_arch = "x86_64")]
                if super::level() >= super::Level::Avx2 {
                    // SAFETY: AVX2 was detected.
                    return unsafe { super::x86::$min(values) };
                }
                min_serial(values)
            }

            #[inline]
            fn max_piece(values: &[Self]) -> Self {
                #[cfg(target_arch = "x86_64")]
                if super::level() >= super::Level::Avx512 {
                    // SAFETY: AVX-512 was detected.
                    return unsafe { super::x86_512::$max(values) };
                }
                #[cfg(target_arch = "x86_64")]
                if super::level() >= super::Level::Avx2 {
                    // SAFETY: AVX2 was detected.
                    return unsafe { super::x86::$max(values) };
                }
                max_serial(values)
            }
        }
    )*};
}
impl_num_float!(f32 => min_max_f32, min_f32, max_f32; f64 => min_max_f64, min_f64, max_f64);

/// Fold `LANES` accumulators into one, pairwise, in a fixed order.
#[inline(always)]
fn fold_lanes<A: Copy>(mut acc: [A; LANES], f: impl Fn(A, A) -> A) -> A {
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        let mut i = 0;
        while i < width {
            acc[i] = f(acc[i], acc[i + width]);
            i += 1;
        }
    }
    acc[0]
}

#[inline(always)]
fn sum_impl<T: Num>(values: &[T]) -> T {
    if !T::IS_FLOAT {
        // Integer addition is associative, so the plain reduction is what LLVM
        // vectorises best. (Given explicit lanes it "improves" them into
        // strided gathers across iterations, which is far slower.)
        return values.iter().fold(T::ZERO, |a, &b| a.add(b));
    }
    let mut acc = [T::ZERO; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for c in &mut chunks {
        let mut i = 0;
        while i < LANES {
            acc[i] = acc[i].add(c[i]);
            i += 1;
        }
    }
    let mut tail = T::ZERO;
    for &v in chunks.remainder() {
        tail = tail.add(v);
    }
    fold_lanes(acc, T::add).add(tail)
}

#[inline(always)]
fn sum_f64_impl<T: Num>(values: &[T]) -> f64 {
    let mut acc = [0.0f64; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for c in &mut chunks {
        let mut i = 0;
        while i < LANES {
            acc[i] += c[i].to_f64();
            i += 1;
        }
    }
    let mut tail = 0.0f64;
    for &v in chunks.remainder() {
        tail += v.to_f64();
    }
    fold_lanes(acc, |a, b| a + b) + tail
}

#[inline(always)]
fn sum_i64_impl<T: Num>(values: &[T]) -> i64 {
    values.iter().fold(0i64, |a, &b| a.wrapping_add(b.to_i64()))
}

/// For a 4-bit mask: four 64-bit lanes, all ones where the bit is set.
#[repr(C, align(64))]
struct Select4([[u64; 4]; 16]);

static SELECT4: Select4 = Select4({
    let mut t = [[0u64; 4]; 16];
    let mut m = 0;
    while m < 16 {
        let mut i = 0;
        while i < 4 {
            if m >> i & 1 == 1 {
                t[m][i] = u64::MAX;
            }
            i += 1;
        }
        m += 1;
    }
    t
});

/// Sum, in `f64`, of the elements whose bit is set in `mask`.
///
/// Every row takes part: four rows at a time are ANDed with a lane mask
/// looked up from their four mask bits, so a row that is not selected
/// contributes `+0.0`. That keeps the loop free of branches and of any
/// compaction step, and the fixed lanes keep the result independent of the
/// vector width.
#[inline(always)]
fn sum_f64_selected_impl<T: Num>(values: &[T], mask: &[u64]) -> f64 {
    let mut acc = [0.0f64; LANES];
    let full = values.len() / 64;
    for (&w, rows) in mask[..full].iter().zip(values.chunks_exact(64)) {
        let mut k = 0;
        while k < 16 {
            let select = &SELECT4.0[((w >> (4 * k)) & 0xF) as usize];
            let lane = (k % (LANES / 4)) * 4;
            let mut i = 0;
            while i < 4 {
                acc[lane + i] += f64::from_bits(rows[4 * k + i].to_f64().to_bits() & select[i]);
                i += 1;
            }
            k += 1;
        }
    }
    let mut tail = 0.0f64;
    for (j, &v) in values[full * 64..].iter().enumerate() {
        if (mask[full] >> j) & 1 != 0 {
            tail += v.to_f64();
        }
    }
    fold_lanes(acc, |a, b| a + b) + tail
}

/// Wrapping sum, in `i64`, of the elements whose bit is set in `mask`.
/// Same scheme as [`sum_f64_selected_impl`].
#[inline(always)]
fn sum_i64_selected_impl<T: Num>(values: &[T], mask: &[u64]) -> i64 {
    let mut acc = [0i64; 16];
    let full = values.len() / 64;
    for (&w, rows) in mask[..full].iter().zip(values.chunks_exact(64)) {
        let mut k = 0;
        while k < 16 {
            let select = &SELECT4.0[((w >> (4 * k)) & 0xF) as usize];
            let lane = (k % 4) * 4;
            let mut i = 0;
            while i < 4 {
                acc[lane + i] = acc[lane + i].wrapping_add(rows[4 * k + i].to_i64() & select[i] as i64);
                i += 1;
            }
            k += 1;
        }
    }
    let mut total = acc.iter().fold(0i64, |a, &b| a.wrapping_add(b));
    for (j, &v) in values[full * 64..].iter().enumerate() {
        if (mask[full] >> j) & 1 != 0 {
            total = total.wrapping_add(v.to_i64());
        }
    }
    total
}

/// Sum of squared deviations from `mean`, in `f64`.
#[inline(always)]
fn sum_sq_dev_impl<T: Num>(values: &[T], mean: f64) -> f64 {
    let mut acc = [0.0f64; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for c in &mut chunks {
        let mut i = 0;
        while i < LANES {
            let d = c[i].to_f64() - mean;
            acc[i] += d * d;
            i += 1;
        }
    }
    let mut tail = 0.0f64;
    for &v in chunks.remainder() {
        let d = v.to_f64() - mean;
        tail += d * d;
    }
    fold_lanes(acc, |a, b| a + b) + tail
}

/// `(min, max)` in the total order restricted to non-NaN values.
/// For an empty slice (or all NaN) returns `(MAX_VALUE, MIN_VALUE)`.
#[inline(always)]
fn min_max_impl<T: Num>(values: &[T]) -> (T, T) {
    if !T::IS_FLOAT {
        let (mut min, mut max) = (T::MAX_VALUE, T::MIN_VALUE);
        for &v in values {
            min = if v < min { v } else { min };
            max = if v > max { v } else { max };
        }
        return (min, max);
    }
    const W: usize = 16;
    let mut lo = [T::MAX_VALUE; W];
    let mut hi = [T::MIN_VALUE; W];
    let mut chunks = values.chunks_exact(W);
    for c in &mut chunks {
        let mut i = 0;
        while i < W {
            // Both comparisons are false for NaN, so NaN never enters a lane.
            lo[i] = if c[i] < lo[i] { c[i] } else { lo[i] };
            hi[i] = if c[i] > hi[i] { c[i] } else { hi[i] };
            i += 1;
        }
    }
    let (mut min, mut max) = (T::MAX_VALUE, T::MIN_VALUE);
    for i in 0..W {
        min = if lo[i] < min { lo[i] } else { min };
        max = if hi[i] > max { hi[i] } else { max };
    }
    for &v in chunks.remainder() {
        min = if v < min { v } else { min };
        max = if v > max { v } else { max };
    }
    (min, max)
}

/// Smallest non-NaN value (`MAX_VALUE` if there is none).
#[inline(always)]
fn min_impl<T: Num>(values: &[T]) -> T {
    // The comparison is false for NaN, so NaN never replaces the running minimum.
    values.iter().fold(T::MAX_VALUE, |m, &v| if v < m { v } else { m })
}

/// Largest non-NaN value (`MIN_VALUE` if there is none).
#[inline(always)]
fn max_impl<T: Num>(values: &[T]) -> T {
    values.iter().fold(T::MIN_VALUE, |m, &v| if v > m { v } else { m })
}

#[inline(always)]
fn count_nan_impl<T: Num>(values: &[T]) -> usize {
    values.iter().map(|v| v.is_nan() as usize).sum()
}

simd_dispatch!(fn sum_serial<T: Num>(values: &[T]) -> T = sum_impl);
simd_dispatch!(fn sum_f64_serial<T: Num>(values: &[T]) -> f64 = sum_f64_impl);
simd_dispatch!(fn sum_i64_serial<T: Num>(values: &[T]) -> i64 = sum_i64_impl);
simd_dispatch!(fn sum_f64_selected_serial<T: Num>(values: &[T], mask: &[u64]) -> f64 = sum_f64_selected_impl);
simd_dispatch!(fn sum_i64_selected_serial<T: Num>(values: &[T], mask: &[u64]) -> i64 = sum_i64_selected_impl);
simd_dispatch!(fn sum_sq_dev_serial<T: Num>(values: &[T], mean: f64) -> f64 = sum_sq_dev_impl);
simd_dispatch!(fn min_max_serial<T: Num>(values: &[T]) -> (T, T) = min_max_impl);
simd_dispatch!(fn count_nan_serial<T: Num>(values: &[T]) -> usize = count_nan_impl);
simd_dispatch!(fn min_serial<T: Num>(values: &[T]) -> T = min_impl);
simd_dispatch!(fn max_serial<T: Num>(values: &[T]) -> T = max_impl);

/// Most rows in a reduction piece.
///
/// A column longer than this is reduced piece by piece and the per-piece
/// results are folded left to right. Where the pieces begin depends on the
/// length of the column alone ([`piece_rows`]), not on the thread count, so
/// a floating-point sum has the same value whether the pieces were computed
/// by one core or by sixteen.
pub const PIECE_ROWS: usize = 1 << 14;

/// Up to this many pieces, a column is cut into a power of two of them.
const BALANCED_PIECES: usize = 64;

/// Rows per reduction piece for a column of `n` rows (the last piece may be
/// shorter): a multiple of 64, at most [`PIECE_ROWS`].
///
/// A column of up to 64 pieces is cut into a power of two of equal pieces
/// rather than into full ones and a remainder, so that two or four threads
/// can share it evenly — which matters exactly for the columns small enough
/// to sit in the cache, where a thread should always get the same part.
#[inline]
pub fn piece_rows(n: usize) -> usize {
    let count = n.div_ceil(PIECE_ROWS);
    if count > BALANCED_PIECES {
        return PIECE_ROWS;
    }
    (n.div_ceil(count.next_power_of_two().max(1)).div_ceil(64) * 64).max(64)
}

#[inline]
fn reduce_pieces<T: Num, R: Send>(
    values: &[T],
    map: impl Fn(&[T]) -> R + Sync,
    init: R,
    fold: impl Fn(R, R) -> R,
) -> R {
    let n = values.len();
    if n <= PIECE_ROWS {
        return fold(init, map(values));
    }
    let piece = piece_rows(n);
    let plan = par::plan(n, size_of::<T>());
    let tasks = plan.tasks();
    if tasks <= 1 {
        return values.chunks(piece).map(&map).fold(init, fold);
    }
    // The threads take contiguous runs of whole pieces - long sequential
    // streams, which is what the hardware prefetchers and a shared memory bus
    // want - and store one result per piece. The fold below runs over the
    // pieces in order, so it does not depend on who computed what.
    let pieces = n.div_ceil(piece);
    let run = par::chunk_rows(n, size_of::<T>(), tasks).div_ceil(piece).max(1);
    let work = |chunk: usize, slots: &mut [std::mem::MaybeUninit<R>]| {
        for (k, slot) in slots.iter_mut().enumerate() {
            let start = (chunk * run + k) * piece;
            slot.write(map(&values[start..(start + piece).min(n)]));
        }
    };
    if pieces <= BALANCED_PIECES {
        // Few pieces: their results fit on the stack.
        let mut results = [const { std::mem::MaybeUninit::<R>::uninit() }; BALANCED_PIECES];
        par::for_each_chunk_mut(&mut results[..pieces], run, tasks, work);
        // SAFETY: each of the first `pieces` slots was written exactly once
        // above; reading it out moves the value, and the array itself never
        // drops its (uninitialised-typed) slots.
        return results[..pieces].iter().map(|r| unsafe { r.assume_init_read() }).fold(init, fold);
    }
    let mut results: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(pieces);
    // SAFETY: `MaybeUninit` needs no initialisation.
    unsafe { results.set_len(pieces) };
    par::for_each_chunk_mut(&mut results, run, tasks, work);
    // SAFETY: every slot was written exactly once above.
    results.into_iter().map(|r| unsafe { r.assume_init() }).fold(init, fold)
}

/// One serial, vectorised pass over `values`: the yardstick the work-splitting
/// policy times to learn how fast a core of this machine streams.
pub(super) fn calibration_pass(values: &[u64]) -> u64 {
    sum_serial(values)
}

/// Sum of all elements. Integers wrap on overflow.
pub fn sum<T: Num>(values: &[T]) -> T {
    reduce_pieces(values, sum_serial, T::ZERO, T::add)
}

/// Sum of all elements converted to `f64`.
pub fn sum_f64<T: Num>(values: &[T]) -> f64 {
    reduce_pieces(values, sum_f64_serial, 0.0, |a, b| a + b)
}

/// Sum of all elements converted to `i64`, wrapping on overflow.
pub fn sum_i64<T: Num>(values: &[T]) -> i64 {
    reduce_pieces(values, sum_i64_serial, 0, i64::wrapping_add)
}

/// Sum, in `f64`, of the elements of `values` whose bit is set in `mask`
/// (bit `i % 64` of word `i / 64`), computed in one pass without compacting.
/// Meant for cache-sized blocks; it does not split its input.
///
/// # Panics
/// If `mask` holds fewer than `values.len()` bits.
pub fn sum_f64_selected<T: Num>(values: &[T], mask: &[u64]) -> f64 {
    assert!(mask.len() * 64 >= values.len(), "mask too short for {} rows", values.len());
    sum_f64_selected_serial(values, mask)
}

/// Wrapping sum, in `i64`, of the elements of `values` whose bit is set in
/// `mask`. See [`sum_f64_selected`].
///
/// # Panics
/// If `mask` holds fewer than `values.len()` bits.
pub fn sum_i64_selected<T: Num>(values: &[T], mask: &[u64]) -> i64 {
    assert!(mask.len() * 64 >= values.len(), "mask too short for {} rows", values.len());
    sum_i64_selected_serial(values, mask)
}

/// Sum of squared deviations from `mean`.
pub fn sum_sq_dev<T: Num>(values: &[T], mean: f64) -> f64 {
    reduce_pieces(values, |c| sum_sq_dev_serial(c, mean), 0.0, |a, b| a + b)
}

/// Number of NaN elements (always 0 for integers).
pub fn count_nan<T: Num>(values: &[T]) -> usize {
    if !T::IS_FLOAT {
        return 0;
    }
    reduce_pieces(values, count_nan_serial, 0, |a, b| a + b)
}

/// Smallest and largest element, ignoring NaN. `None` if the slice is empty.
/// If every element is NaN, both are NaN.
pub fn min_max<T: Num>(values: &[T]) -> Option<(T, T)> {
    if values.is_empty() {
        return None;
    }
    let (min, max) = reduce_pieces(values, T::min_max_piece, (T::MAX_VALUE, T::MIN_VALUE), |(lo, hi), (a, b)| {
        (if a < lo { a } else { lo }, if b > hi { b } else { hi })
    });
    if T::IS_FLOAT && min > max {
        // No non-NaN value was seen: the lanes still hold their sentinels.
        return Some((values[0], values[0]));
    }
    Some((min, max))
}

/// Smallest element, ignoring NaN. `None` if the slice is empty; NaN if every
/// element is NaN.
pub fn min<T: Num>(values: &[T]) -> Option<T> {
    let first = *values.first()?;
    let m = reduce_pieces(values, T::min_piece, T::MAX_VALUE, |a, b| if b < a { b } else { a });
    // The sentinel survives only if no element is below it: either the
    // column is all NaN, or its minimum really is the largest value.
    if T::IS_FLOAT && m == T::MAX_VALUE && count_nan(values) == values.len() {
        return Some(first);
    }
    Some(m)
}

/// Largest element, ignoring NaN. `None` if the slice is empty; NaN if every
/// element is NaN.
pub fn max<T: Num>(values: &[T]) -> Option<T> {
    let first = *values.first()?;
    let m = reduce_pieces(values, T::max_piece, T::MIN_VALUE, |a, b| if b > a { b } else { a });
    if T::IS_FLOAT && m == T::MIN_VALUE && count_nan(values) == values.len() {
        return Some(first);
    }
    Some(m)
}

/// Visit the valid elements of `values` (those whose bit is set in the aligned
/// `validity` words), a whole word at a time when the word is all-valid.
#[inline(always)]
fn for_each_valid_run<T: Copy>(
    values: &[T],
    validity: &[u64],
    mut dense: impl FnMut(&[T]),
    mut single: impl FnMut(T),
) {
    let n = values.len();
    for (wi, chunk) in values.chunks(64).enumerate() {
        let mut w = validity[wi];
        if wi == n / 64 {
            w &= tail_mask(n);
        }
        if chunk.len() == 64 && w == u64::MAX {
            dense(chunk);
        } else {
            while w != 0 {
                single(chunk[w.trailing_zeros() as usize]);
                w &= w - 1;
            }
        }
    }
}

/// Sum of the valid elements. Integers wrap on overflow.
pub fn sum_valid<T: Num>(values: &[T], validity: &[u64]) -> T {
    let (mut dense, mut sparse) = (T::ZERO, T::ZERO);
    for_each_valid_run(values, validity, |c| dense = dense.add(sum_serial(c)), |v| sparse = sparse.add(v));
    dense.add(sparse)
}

/// Sum of the valid elements converted to `f64`.
pub fn sum_f64_valid<T: Num>(values: &[T], validity: &[u64]) -> f64 {
    let (mut dense, mut sparse) = (0.0f64, 0.0f64);
    for_each_valid_run(values, validity, |c| dense += sum_f64_serial(c), |v| sparse += v.to_f64());
    dense + sparse
}

/// Sum of the valid elements converted to `i64`, wrapping on overflow.
pub fn sum_i64_valid<T: Num>(values: &[T], validity: &[u64]) -> i64 {
    let (mut dense, mut sparse) = (0i64, 0i64);
    for_each_valid_run(
        values,
        validity,
        |c| dense = dense.wrapping_add(sum_i64_serial(c)),
        |v| sparse = sparse.wrapping_add(v.to_i64()),
    );
    dense.wrapping_add(sparse)
}

/// Sum of squared deviations from `mean` over the valid elements.
pub fn sum_sq_dev_valid<T: Num>(values: &[T], validity: &[u64], mean: f64) -> f64 {
    let (mut dense, mut sparse) = (0.0f64, 0.0f64);
    for_each_valid_run(
        values,
        validity,
        |c| dense += sum_sq_dev_serial(c, mean),
        |v| {
            let d = v.to_f64() - mean;
            sparse += d * d;
        },
    );
    dense + sparse
}

/// Smallest and largest valid element, ignoring NaN. `None` if nothing is valid.
/// If every valid element is NaN, both are NaN.
pub fn min_max_valid<T: Num>(values: &[T], validity: &[u64]) -> Option<(T, T)> {
    let n = values.len();
    let first = (0..n).find(|&i| validity[i / 64] >> (i % 64) & 1 != 0)?;
    let (mut dense_lo, mut dense_hi) = (T::MAX_VALUE, T::MIN_VALUE);
    let (mut sparse_lo, mut sparse_hi) = (T::MAX_VALUE, T::MIN_VALUE);
    for_each_valid_run(
        values,
        validity,
        |c| {
            let (lo, hi) = T::min_max_piece(c);
            dense_lo = if lo < dense_lo { lo } else { dense_lo };
            dense_hi = if hi > dense_hi { hi } else { dense_hi };
        },
        |v| {
            sparse_lo = if v < sparse_lo { v } else { sparse_lo };
            sparse_hi = if v > sparse_hi { v } else { sparse_hi };
        },
    );
    let min = if sparse_lo < dense_lo { sparse_lo } else { dense_lo };
    let max = if sparse_hi > dense_hi { sparse_hi } else { dense_hi };
    if T::IS_FLOAT && min > max {
        return Some((values[first], values[first]));
    }
    Some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_sum_f64(v: &[f64]) -> f64 {
        v.iter().sum()
    }

    #[test]
    fn integer_sums_match_naive_and_wrap() {
        for n in [0usize, 1, 31, 32, 33, 100, 1000, 4097] {
            let v: Vec<i64> = (0..n as i64).map(|i| i * 37 - 500).collect();
            assert_eq!(sum(&v), v.iter().sum::<i64>());
            let v32: Vec<i32> = (0..n as i32).map(|i| i.wrapping_mul(1_000_003)).collect();
            assert_eq!(sum(&v32), v32.iter().fold(0i32, |a, &b| a.wrapping_add(b)));
            assert_eq!(sum_i64(&v32), v32.iter().map(|&x| x as i64).sum::<i64>());
            let v8: Vec<u8> = (0..n).map(|i| (i * 7) as u8).collect();
            assert_eq!(sum_i64(&v8), v8.iter().map(|&x| x as i64).sum::<i64>());
        }
        assert_eq!(sum(&[i64::MAX, 1]), i64::MIN);
    }

    #[test]
    fn float_sums_are_close_to_naive_and_deterministic() {
        for n in [0usize, 1, 31, 32, 33, 1000, 100_000] {
            let v: Vec<f64> = (0..n).map(|i| ((i * 2654435761) % 10007) as f64 / 7.0 - 300.0).collect();
            let got = sum(&v);
            let want = naive_sum_f64(&v);
            assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "n {n}: {got} vs {want}");
            // The association order is fixed by the length alone: the same
            // pieces, summed by the portable build of the kernel.
            let pieces = v.chunks(piece_rows(n)).map(sum_impl).fold(0.0, |a, b| a + b);
            assert_eq!(got.to_bits(), pieces.to_bits(), "vector and scalar builds must agree");
            assert_eq!(sum_f64(&v).to_bits(), got.to_bits());
        }
    }

    #[test]
    fn pieces_are_even_and_a_sum_does_not_depend_on_who_computes_it() {
        for n in [1usize, 64, PIECE_ROWS, PIECE_ROWS + 1, 30_000, 100_000, 131_072, 131_073, 1_000_000, 5_000_000] {
            let piece = piece_rows(n);
            assert!(piece % 64 == 0 && piece <= PIECE_ROWS.max(64), "n {n}: {piece}");
            let count = n.div_ceil(piece);
            if n > PIECE_ROWS && count <= BALANCED_PIECES {
                assert!(count.is_power_of_two(), "n {n}: {count} pieces");
                assert!(n - (count - 1) * piece > piece / 2, "n {n}: a stub of a last piece");
            }
        }
        // Small enough to sit in a core's cache on most machines, where the
        // policy tries both one thread and several: every call must agree.
        let v: Vec<f64> = (0..30_000).map(|i| ((i * 2654435761usize) % 10007) as f64 / 7.0 - 300.0).collect();
        let first = sum(&v).to_bits();
        for _ in 0..400 {
            assert_eq!(sum(&v).to_bits(), first);
        }
        assert_eq!(first, v.chunks(piece_rows(v.len())).map(sum_impl).fold(0.0, |a, b| a + b).to_bits());
    }

    #[test]
    fn min_max_ignores_nan() {
        assert_eq!(min_max::<i32>(&[]), None);
        assert_eq!(min_max(&[3i32, -7, 12, 0]), Some((-7, 12)));
        let v: Vec<f64> = (0..1000).map(|i| if i % 10 == 3 { f64::NAN } else { (i as f64 - 400.0) * 0.5 }).collect();
        assert_eq!(min_max(&v), Some((-200.0, 299.5)));
        let (lo, hi) = min_max(&[f64::NAN, f64::NAN]).unwrap();
        assert!(lo.is_nan() && hi.is_nan());
        assert_eq!(min_max(&[f64::INFINITY]), Some((f64::INFINITY, f64::INFINITY)));
        assert_eq!((min(&v), max(&v)), (Some(-200.0), Some(299.5)));
        assert!(min(&[f32::NAN, f32::NAN]).unwrap().is_nan() && max(&[f64::NAN]).unwrap().is_nan());
        assert_eq!((min(&[f64::INFINITY, f64::NAN]), max(&[f64::NEG_INFINITY])), (Some(f64::INFINITY), Some(f64::NEG_INFINITY)));
        assert_eq!((min::<i64>(&[]), max::<u8>(&[])), (None, None));
        let big: Vec<i64> = (0..300_000).map(|i| (i * 7919) % 100_003 - 50_000).collect();
        assert_eq!((min(&big), max(&big)), (big.iter().copied().min(), big.iter().copied().max()));
        let bigf: Vec<f32> = big.iter().map(|&x| x as f32).collect();
        assert_eq!((min(&bigf), max(&bigf)), (Some(-50_000.0), Some(50_002.0)));
        assert_eq!(count_nan(&v), 100);
        let u: Vec<u64> = (0..500).map(|i| i * i).collect();
        assert_eq!(min_max(&u), Some((0, 499 * 499)));
    }

    #[test]
    fn selected_sums_match_a_filter_then_sum() {
        let mut rng = crate::testutil::Rng::new(91);
        for n in [0usize, 1, 63, 64, 65, 200, 4096, 10_001] {
            let ints: Vec<i32> = (0..n).map(|_| rng.below(2001) as i32 - 1000).collect();
            let floats: Vec<f64> = ints.iter().map(|&v| v as f64 * 0.125).collect();
            for density in [0u64, 3, 50, 97, 100] {
                let keep: Vec<bool> = (0..n).map(|_| rng.below(100) < density).collect();
                let bm = crate::memory::Bitmap::from_bools(&keep);
                let words = bm.words();
                let want_i: i64 = ints.iter().zip(&keep).filter(|(_, k)| **k).map(|(&v, _)| v as i64).sum();
                assert_eq!(sum_i64_selected(&ints, &words), want_i, "n {n} density {density}");
                // Multiples of 1/8 of moderate size: every order of addition is exact.
                assert_eq!(sum_f64_selected(&floats, &words), want_i as f64 * 0.125, "n {n} density {density}");
                assert_eq!(sum_f64_selected(&ints, &words), want_i as f64);
            }
        }
        // Unselected NaN and infinity must not leak into the sum.
        let values = [f64::NAN, 1.5, f64::INFINITY, 2.5];
        assert_eq!(sum_f64_selected(&values, &[0b1010]), 4.0);
        assert!(sum_f64_selected(&values, &[0b0011]).is_nan());
    }

    #[test]
    fn masked_reductions_skip_invalid_rows() {
        let n = 300;
        let v: Vec<i64> = (0..n as i64).collect();
        let valid: Vec<bool> = (0..n).map(|i| i % 3 != 0 || (64..128).contains(&i)).collect();
        let bm = crate::memory::Bitmap::from_bools(&valid);
        let words = bm.words();
        let want: i64 = (0..n).filter(|&i| valid[i]).map(|i| v[i]).sum();
        assert_eq!(sum_valid(&v, &words), want);
        assert_eq!(sum_i64_valid(&v, &words), want);
        assert_eq!(sum_f64_valid(&v, &words), want as f64);
        assert_eq!(min_max_valid(&v, &words), Some((1, 299)));
        let none = crate::memory::Bitmap::new_zeroed(n);
        assert_eq!(min_max_valid(&v, &none.words()), None);
    }
}
