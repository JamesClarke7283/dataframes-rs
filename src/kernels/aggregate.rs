//! Reductions over native slices.
//!
//! Every reduction keeps a fixed number of independent accumulator *lanes*
//! (element `i` always goes to lane `i % LANES`) and folds the lanes together
//! in a fixed order at the end. LLVM turns the lane loop into straight vector
//! code, and because the association order is spelled out rather than left to
//! the optimiser, a floating-point sum is bit-for-bit identical on every CPU
//! and at every vector width. As a bonus the lanes bound rounding error much
//! like pairwise summation does.

use rayon::prelude::*;

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
                if super::level() >= super::Level::Avx2 {
                    // SAFETY: AVX2 was detected.
                    return unsafe { super::x86::$kernel(values) };
                }
                min_max_serial(values)
            }

            #[inline]
            fn min_piece(values: &[Self]) -> Self {
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
simd_dispatch!(fn sum_sq_dev_serial<T: Num>(values: &[T], mean: f64) -> f64 = sum_sq_dev_impl);
simd_dispatch!(fn min_max_serial<T: Num>(values: &[T]) -> (T, T) = min_max_impl);
simd_dispatch!(fn count_nan_serial<T: Num>(values: &[T]) -> usize = count_nan_impl);
simd_dispatch!(fn min_serial<T: Num>(values: &[T]) -> T = min_impl);
simd_dispatch!(fn max_serial<T: Num>(values: &[T]) -> T = max_impl);

/// Rows per reduction piece.
///
/// A column longer than this is reduced piece by piece and the per-piece
/// results are folded left to right. The piece size is a constant (not a
/// function of the thread count), so a floating-point sum has the same value
/// whether the pieces were computed by one core or by sixteen.
pub const PIECE_ROWS: usize = 1 << 16;

#[inline]
fn reduce_pieces<T: Num, R: Send>(
    values: &[T],
    map: impl Fn(&[T]) -> R + Sync,
    init: R,
    fold: impl Fn(R, R) -> R,
) -> R {
    if values.len() <= PIECE_ROWS {
        return fold(init, map(values));
    }
    let tasks = par::tasks_for(values.len(), size_of::<T>());
    if tasks <= 1 {
        return values.chunks(PIECE_ROWS).map(&map).fold(init, fold);
    }
    // Every worker gets one contiguous run of whole pieces - a single
    // sequential stream per physical core, which is what the hardware
    // prefetchers and a shared memory bus want - and returns its per-piece
    // results in order, so the fold below does not depend on the worker count.
    let pieces = values.len().div_ceil(PIECE_ROWS);
    let run = pieces.div_ceil(tasks) * PIECE_ROWS;
    let parts: Vec<Vec<R>> =
        values.par_chunks(run).map(|run| run.chunks(PIECE_ROWS).map(&map).collect()).collect();
    parts.into_iter().flatten().fold(init, fold)
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
            assert_eq!(got.to_bits(), sum_impl(&v).to_bits(), "vector and scalar builds must agree");
            assert_eq!(sum_f64(&v).to_bits(), got.to_bits());
        }
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
