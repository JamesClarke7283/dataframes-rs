//! Element-wise maps.
//!
//! The closures passed in are ordinary Rust; [`with_simd`] re-instantiates the
//! loop around them inside a `#[target_feature]` frame so LLVM vectorises them
//! with 256-bit registers when the CPU has them. Inputs of a few hundred
//! kilobytes and more are split across the team.

use std::mem::MaybeUninit;

use super::{Level, level, par};
use crate::memory::{AlignedVec, ColumnAlloc, Pod};

/// Run `f`, compiled for the best instruction set the CPU supports.
///
/// `f` must be marked `#[inline(always)]` (closures accept the attribute) for
/// the wider instruction set to reach its body.
#[inline]
pub fn with_simd<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,bmi1,bmi2,popcnt,lzcnt,fma")]
        fn v3<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        if level() >= Level::Avx2 {
            // SAFETY: `level()` verified the CPU features at run time.
            return unsafe { v3(f) };
        }
    }
    let _ = (level, Level::Scalar);
    f()
}

#[inline(always)]
fn map_slice<T: Copy, U>(src: &[T], dst: &mut [MaybeUninit<U>], f: impl Fn(T) -> U) {
    for (d, &s) in dst.iter_mut().zip(src) {
        d.write(f(s));
    }
}

#[inline(always)]
fn map_slice2<A: Copy, B: Copy, U>(a: &[A], b: &[B], dst: &mut [MaybeUninit<U>], f: impl Fn(A, B) -> U) {
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        d.write(f(x, y));
    }
}

/// `out[i] = f(values[i])`.
pub fn unary<T: Pod, U: Pod>(values: &[T], f: impl Fn(T) -> U + Sync) -> AlignedVec<U> {
    let n = values.len();
    let mut out: AlignedVec<U> = Vec::with_capacity_in(n, ColumnAlloc);
    let dst = &mut out.spare_capacity_mut()[..n];
    let tasks = par::tasks_for_light(n, size_of::<T>().max(size_of::<U>()));
    if tasks <= 1 {
        with_simd(
            #[inline(always)]
            || map_slice(values, dst, &f),
        );
    } else {
        let rows = par::chunk_rows(n, size_of::<T>().max(size_of::<U>()), tasks);
        par::for_each_chunk_mut(dst, rows, tasks, |i, d| {
            let s = &values[i * rows..i * rows + d.len()];
            with_simd(
                #[inline(always)]
                || map_slice(s, d, &f),
            )
        });
    }
    // SAFETY: every one of the first `n` slots was written above.
    unsafe { out.set_len(n) };
    out
}

/// `out[i] = f(a[i], b[i])`.
///
/// # Panics
/// If the slices differ in length.
pub fn binary<A: Pod, B: Pod, U: Pod>(a: &[A], b: &[B], f: impl Fn(A, B) -> U + Sync) -> AlignedVec<U> {
    assert_eq!(a.len(), b.len(), "operands must have equal length");
    let n = a.len();
    let mut out: AlignedVec<U> = Vec::with_capacity_in(n, ColumnAlloc);
    let dst = &mut out.spare_capacity_mut()[..n];
    let tasks = par::tasks_for_light(n, size_of::<A>() + size_of::<B>());
    if tasks <= 1 {
        with_simd(
            #[inline(always)]
            || map_slice2(a, b, dst, &f),
        );
    } else {
        let rows = par::chunk_rows(n, size_of::<A>() + size_of::<B>(), tasks);
        par::for_each_chunk_mut(dst, rows, tasks, |i, d| {
            let range = i * rows..i * rows + d.len();
            let (x, y) = (&a[range.clone()], &b[range]);
            with_simd(
                #[inline(always)]
                || map_slice2(x, y, d, &f),
            )
        });
    }
    // SAFETY: every one of the first `n` slots was written above.
    unsafe { out.set_len(n) };
    out
}

/// `values[i] = f(values[i])`, in place.
///
/// The workers receive disjoint `&mut` sub-slices, so the borrow checker —
/// not a lock — is what makes the concurrent mutation sound.
pub fn unary_in_place<T: Pod>(values: &mut [T], f: impl Fn(T) -> T + Sync) {
    #[inline(always)]
    fn run<T: Copy>(chunk: &mut [T], f: impl Fn(T) -> T) {
        for v in chunk {
            *v = f(*v);
        }
    }
    let n = values.len();
    let tasks = par::tasks_for_light(n, size_of::<T>());
    if tasks <= 1 {
        with_simd(
            #[inline(always)]
            || run(values, &f),
        );
    } else {
        let rows = par::chunk_rows(n, size_of::<T>(), tasks);
        par::for_each_chunk_mut(values, rows, tasks, |_, chunk| {
            with_simd(
                #[inline(always)]
                || run(chunk, &f),
            )
        });
    }
}

/// `out[i] = values[indices[i]]`.
///
/// # Panics
/// If an index is out of bounds.
pub fn take<T: Pod>(values: &[T], indices: &[u32]) -> AlignedVec<T> {
    let max = with_simd(
        #[inline(always)]
        || indices.iter().copied().fold(0u32, u32::max),
    );
    assert!(
        indices.is_empty() || (max as usize) < values.len(),
        "index {max} out of bounds for column of length {}",
        values.len()
    );
    // SAFETY: every index was just checked against `values.len()`.
    unary(indices, |i| unsafe { *values.get_unchecked(i as usize) })
}

/// `out[i] = values[indices[i]]` without validating the indices.
///
/// # Safety
/// Every index must be below `values.len()`.
pub unsafe fn take_unchecked<T: Pod>(values: &[T], indices: &[u32]) -> AlignedVec<T> {
    // SAFETY: guaranteed by the caller.
    unary(indices, |i| unsafe { *values.get_unchecked(i as usize) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_match_iterators() {
        for n in [0usize, 1, 100, 5000, 300_000] {
            let a: Vec<i64> = (0..n as i64).collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
            assert_eq!(&unary(&a, |x| x * 3 + 1)[..], &a.iter().map(|x| x * 3 + 1).collect::<Vec<_>>()[..]);
            assert_eq!(&unary(&a, |x| x as f32)[..], &a.iter().map(|&x| x as f32).collect::<Vec<_>>()[..]);
            let want: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x as f64 - y).collect();
            assert_eq!(&binary(&a, &b, |x, y| x as f64 - y)[..], &want[..]);
            let mut c = b.clone();
            unary_in_place(&mut c, |x| x * 2.0 + 1.0);
            assert!(c.iter().zip(&b).all(|(&got, &x)| got == x * 2.0 + 1.0));
            let idx: Vec<u32> = (0..n as u32).rev().collect();
            let taken = take(&a, &idx);
            assert!(taken.iter().enumerate().all(|(i, &v)| v == (n - 1 - i) as i64));
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn take_checks_bounds() {
        take(&[1u8, 2, 3], &[0, 3]);
    }
}
