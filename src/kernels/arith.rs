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
    let plan = par::plan(n, size_of::<T>().max(size_of::<U>()));
    let tasks = plan.tasks();
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
    let plan = par::plan(n, size_of::<A>() + size_of::<B>());
    let tasks = plan.tasks();
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
    let plan = par::plan(n, size_of::<T>());
    let tasks = plan.tasks();
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

/// The largest of `indices`; 0 if there are none.
fn max_index(indices: &[u32]) -> u32 {
    with_simd(
        #[inline(always)]
        || indices.iter().copied().fold(0u32, u32::max),
    )
}

/// `out[i] = values[indices[i]]`, or `None` if an index is out of bounds.
pub fn try_take<T: Pod>(values: &[T], indices: &[u32]) -> Option<AlignedVec<T>> {
    if !indices.is_empty() && max_index(indices) as usize >= values.len() {
        return None;
    }
    // SAFETY: every index was just checked against `values.len()`.
    Some(unsafe { take_unchecked(values, indices) })
}

/// `out[i] = values[indices[i]]`.
///
/// # Panics
/// If an index is out of bounds.
pub fn take<T: Pod>(values: &[T], indices: &[u32]) -> AlignedVec<T> {
    try_take(values, indices).unwrap_or_else(|| {
        panic!("index {} out of bounds for column of length {}", max_index(indices), values.len())
    })
}

/// `out[i] = values[indices[i]]` without validating the indices.
///
/// # Safety
/// Every index must be below `values.len()`.
pub unsafe fn take_unchecked<T: Pod>(values: &[T], indices: &[u32]) -> AlignedVec<T> {
    let n = indices.len();
    let mut out: AlignedVec<T> = Vec::with_capacity_in(n, ColumnAlloc);
    let dst = &mut out.spare_capacity_mut()[..n];
    let plan = par::plan(n, size_of::<u32>() + size_of::<T>());
    let tasks = plan.tasks();
    if tasks <= 1 {
        // SAFETY: forwarded; `dst` has one slot per index.
        unsafe { gather_into(values, indices, dst) };
    } else {
        let rows = par::chunk_rows(n, size_of::<u32>() + size_of::<T>(), tasks);
        par::for_each_chunk_mut(dst, rows, tasks, |i, d| {
            // SAFETY: forwarded; chunk `i` of the output belongs to chunk `i`
            // of the indices.
            unsafe { gather_into(values, &indices[i * rows..i * rows + d.len()], d) }
        });
    }
    // SAFETY: every one of the first `n` slots was written above.
    unsafe { out.set_len(n) };
    out
}

/// `dst[j] = values[indices[j]]`.
///
/// # Safety
/// Every index must be below `values.len()`; `dst.len() == indices.len()`.
#[inline]
unsafe fn gather_into<T: Pod>(values: &[T], indices: &[u32], dst: &mut [MaybeUninit<T>]) {
    debug_assert_eq!(dst.len(), indices.len());
    #[cfg(target_arch = "x86_64")]
    if level() >= Level::Avx512 && indices.len() >= 16 && vector_gather_pays() {
        // SAFETY: AVX-512 was detected; `Pod` types of these sizes are plain
        // bits, moved here as integers of the same width and at least the
        // same alignment requirement as bytes; the rest is forwarded.
        unsafe {
            match size_of::<T>() {
                8 => return super::x86_512::gather8(values.as_ptr().cast(), indices, dst.as_mut_ptr().cast()),
                4 => return super::x86_512::gather4(values.as_ptr().cast(), indices, dst.as_mut_ptr().cast()),
                _ => {}
            }
        }
    }
    with_simd(
        #[inline(always)]
        || {
            for (d, &i) in dst.iter_mut().zip(indices) {
                // SAFETY: guaranteed by the caller.
                d.write(unsafe { *values.get_unchecked(i as usize) });
            }
        },
    );
}

/// Whether the vector gather instruction beats one load per element on this
/// CPU. It does by two to one on current Intel cores and by nothing, or less,
/// on some AMD ones, so it is measured once (a few tens of microseconds) on a
/// table that fits in the first-level cache.
#[cfg(target_arch = "x86_64")]
fn vector_gather_pays() -> bool {
    use std::sync::OnceLock;
    use std::time::Instant;
    static PAYS: OnceLock<bool> = OnceLock::new();
    *PAYS.get_or_init(|| {
        const N: usize = 1024;
        const PASSES: usize = 8;
        let table: Vec<u64> = (0..N as u64).collect();
        let indices: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2_654_435_761) >> 22).collect();
        let mut out = vec![0u64; N];
        let mut best = [u128::MAX; 2];
        for _ in 0..8 {
            for (way, best) in best.iter_mut().enumerate() {
                let start = Instant::now();
                for _ in 0..PASSES {
                    let (src, dst) = (std::hint::black_box(table.as_ptr()), out.as_mut_ptr());
                    // SAFETY: the indices are below `N` (a 32-bit product
                    // shifted down to ten bits) and both buffers hold `N`
                    // elements; this is only reached with AVX-512 detected.
                    unsafe {
                        if way == 0 {
                            with_simd(
                                #[inline(always)]
                                || {
                                    for (j, &i) in indices.iter().enumerate() {
                                        dst.add(j).write(src.add(i as usize).read());
                                    }
                                },
                            );
                        } else {
                            super::x86_512::gather8(src, &indices, dst);
                        }
                    }
                    std::hint::black_box(&mut out);
                }
                *best = (*best).min(start.elapsed().as_nanos());
            }
        }
        best[1] * 10 <= best[0] * 9
    })
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
    fn take_gathers_every_width_and_refuses_what_is_out_of_bounds() {
        // Lengths on both sides of the vector step and of the parallel split;
        // positions in no particular order, with repeats.
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 1000, 70_001, 400_000] {
            let rows = n.max(1) * 3 + 5;
            let idx: Vec<u32> = (0..n as u64).map(|i| ((i * 2_654_435_761) % rows as u64) as u32).collect();
            let wide: Vec<i64> = (0..rows as i64).map(|i| i * 7 - 3).collect();
            let floats: Vec<f64> = wide.iter().map(|&v| v as f64 * 0.5).collect();
            let narrow: Vec<i32> = wide.iter().map(|&v| v as i32).collect();
            let singles: Vec<f32> = narrow.iter().map(|&v| v as f32).collect();
            let small: Vec<u16> = wide.iter().map(|&v| v as u16).collect();
            let bytes: Vec<u8> = wide.iter().map(|&v| v as u8).collect();
            let pairs: Vec<u128> = wide.iter().map(|&v| (v as u128).wrapping_mul(3)).collect();
            assert!(take(&wide, &idx).iter().zip(&idx).all(|(&v, &i)| v == wide[i as usize]), "i64, n {n}");
            assert!(take(&floats, &idx).iter().zip(&idx).all(|(&v, &i)| v == floats[i as usize]), "f64, n {n}");
            assert!(take(&narrow, &idx).iter().zip(&idx).all(|(&v, &i)| v == narrow[i as usize]), "i32, n {n}");
            assert!(take(&singles, &idx).iter().zip(&idx).all(|(&v, &i)| v == singles[i as usize]), "f32, n {n}");
            assert!(take(&small, &idx).iter().zip(&idx).all(|(&v, &i)| v == small[i as usize]), "u16, n {n}");
            assert!(take(&bytes, &idx).iter().zip(&idx).all(|(&v, &i)| v == bytes[i as usize]), "u8, n {n}");
            assert!(take(&pairs, &idx).iter().zip(&idx).all(|(&v, &i)| v == pairs[i as usize]), "u128, n {n}");
            assert_eq!(take(&wide, &idx).len(), n);
            // One index too far anywhere in the list is caught before anything is read.
            if n > 0 {
                for at in [0, n / 2, n - 1] {
                    let mut bad = idx.clone();
                    bad[at] = rows as u32;
                    assert!(try_take(&wide, &bad).is_none(), "n {n}, bad index at {at}");
                }
                assert!(try_take::<i64>(&[], &idx).is_none());
            }
        }
        assert_eq!(try_take::<i64>(&[], &[]).map(|v| v.len()), Some(0));
    }

    #[test]
    #[should_panic(expected = "index 5 out of bounds for column of length 5")]
    fn take_panics_on_an_index_out_of_bounds() {
        take(&[1i64, 2, 3, 4, 5], &[0, 5, 2]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn take_checks_bounds() {
        take(&[1u8, 2, 3], &[0, 3]);
    }
}
