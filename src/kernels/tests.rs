//! Cross-checks: every dispatching kernel against a naive model, and every
//! vector kernel against its portable counterpart, over lengths that exercise
//! empty input, partial words, word boundaries and block boundaries.

use super::*;
use crate::testutil::Rng;

const LENGTHS: &[usize] = &[0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 1000, 4095, 4096, 4097, 9000];
const OPS: [CmpOp; 6] = [CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge];

fn mask_from(bools: &[bool]) -> Vec<u64> {
    let mut words = vec![0u64; words_for(bools.len())];
    for (i, &b) in bools.iter().enumerate() {
        words[i / 64] |= (b as u64) << (i % 64);
    }
    words
}

trait Sample: Kernel {
    fn sample(rng: &mut Rng) -> Self;
}
macro_rules! sample_int {
    ($($t:ty),*) => {$(
        impl Sample for $t {
            fn sample(rng: &mut Rng) -> Self {
                // A narrow range so that equality hits often, plus the extremes.
                match rng.below(20) {
                    0 => <$t>::MIN,
                    1 => <$t>::MAX,
                    _ => (rng.below(11) as i64 - 5) as $t,
                }
            }
        }
    )*};
}
sample_int!(i8, i16, i32, i64, u8, u16, u32, u64);
macro_rules! sample_float {
    ($($t:ty),*) => {$(
        impl Sample for $t {
            fn sample(rng: &mut Rng) -> Self {
                match rng.below(24) {
                    0 => <$t>::NAN,
                    1 => <$t>::INFINITY,
                    2 => <$t>::NEG_INFINITY,
                    3 => -0.0,
                    _ => (rng.below(11) as $t - 5.0) * 0.5,
                }
            }
        }
    )*};
}
sample_float!(f32, f64);

fn check_cmp<T: Sample>(seed: u64) {
    let mut rng = Rng::new(seed);
    for &n in LENGTHS {
        let values: Vec<T> = (0..n).map(|_| T::sample(&mut rng)).collect();
        let others: Vec<T> = (0..n).map(|_| T::sample(&mut rng)).collect();
        let mut rhs_candidates: Vec<T> = (0..4).map(|_| T::sample(&mut rng)).collect();
        rhs_candidates.push(T::default());
        for &rhs in &rhs_candidates {
            for op in OPS {
                let want: Vec<bool> = values.iter().map(|&v| op.eval(v, rhs)).collect();
                let want_words = mask_from(&want);
                let want_count = want.iter().filter(|&&b| b).count();

                // Pre-fill with garbage: the kernel must overwrite every word.
                let mut got = vec![0xDEAD_BEEF_DEAD_BEEFu64; words_for(n)];
                let count = T::cmp_scalar(&values, op, rhs, &mut got);
                assert_eq!(got, want_words, "{} {op:?} rhs {rhs:?} n {n}", std::any::type_name::<T>());
                assert_eq!(count, want_count);

                let mut portable = vec![0xDEAD_BEEF_DEAD_BEEFu64; words_for(n)];
                assert_eq!(scalar::cmp_scalar(&values, op, rhs, &mut portable), want_count);
                assert_eq!(portable, want_words);

                for tasks in [1, 2, 5] {
                    let (words, c) = cmp_scalar_mask_with(&values, op, rhs, tasks);
                    assert_eq!(&words[..], &want_words[..]);
                    assert_eq!(c, want_count);
                }
                assert_eq!(count_cmp_scalar(&values, op, rhs), want_count);
            }
        }
        for op in OPS {
            let want: Vec<bool> = values.iter().zip(&others).map(|(&a, &b)| op.eval(a, b)).collect();
            let (words, c) = cmp_values_mask(&values, &others, op);
            assert_eq!(&words[..], &mask_from(&want)[..], "{} {op:?} n {n}", std::any::type_name::<T>());
            assert_eq!(c, want.iter().filter(|&&b| b).count());
        }
    }
}

#[test]
fn compare_matches_model_for_every_type() {
    check_cmp::<i8>(1);
    check_cmp::<u8>(2);
    check_cmp::<i16>(3);
    check_cmp::<u16>(4);
    check_cmp::<i32>(5);
    check_cmp::<u32>(6);
    check_cmp::<i64>(7);
    check_cmp::<u64>(8);
    check_cmp::<f32>(9);
    check_cmp::<f64>(10);
}

#[test]
fn float_total_order_semantics() {
    let nan = f64::NAN;
    assert!(nan.tot_eq(nan));
    assert!(!nan.tot_lt(nan));
    assert!(1.0f64.tot_lt(nan));
    assert!(!nan.tot_lt(1.0));
    assert!(f64::INFINITY.tot_lt(nan));
    assert!(CmpOp::Gt.eval(nan, f64::INFINITY));
    assert!(CmpOp::Ge.eval(nan, nan));
    assert!(CmpOp::Le.eval(nan, nan));
    assert!((-0.0f64).tot_eq(0.0));
    assert_eq!(nan.tot_cmp(1.0), std::cmp::Ordering::Greater);
    assert_eq!(1.0f64.tot_cmp(nan), std::cmp::Ordering::Less);
    assert_eq!(nan.tot_cmp(nan), std::cmp::Ordering::Equal);
}

/// A mask with the given density, or with long runs when `runs` is set.
fn random_mask(rng: &mut Rng, n: usize, density_pct: u64, runs: bool) -> Vec<bool> {
    if !runs {
        return (0..n).map(|_| rng.below(100) < density_pct).collect();
    }
    let mut out = Vec::with_capacity(n);
    let mut state = rng.below(2) == 0;
    while out.len() < n {
        let len = 1 + rng.below(300) as usize;
        out.extend(std::iter::repeat_n(state, len.min(n - out.len())));
        state = !state;
    }
    out
}

fn check_compact<T: Pod + PartialEq + std::fmt::Debug>(make: impl Fn(u64) -> T, seed: u64) {
    const GUARD: usize = 80;
    let mut rng = Rng::new(seed);
    let sentinel = make(u64::MAX / 3);
    for &n in LENGTHS {
        let values: Vec<T> = (0..n as u64).map(|i| make(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed)).collect();
        for (density, runs) in [(0, false), (1, false), (10, false), (50, false), (90, false), (100, false), (50, true)] {
            let bools = random_mask(&mut rng, n, density, runs);
            let mask = mask_from(&bools);
            let want: Vec<T> = values.iter().zip(&bools).filter(|(_, b)| **b).map(|(v, _)| *v).collect();
            let apron = APRON / size_of::<T>();

            // `slack` output slots beyond the exact count are writable; the
            // slots after those are guarded and must stay untouched.
            for slack in [0, 1, 3, apron] {
                let cap = want.len() + slack;
                for name in ["dispatch", "stream", "scalar"] {
                    let mut out = vec![sentinel; cap + GUARD];
                    // SAFETY: `out` holds `cap + GUARD` initialised slots.
                    let written = unsafe {
                        match name {
                            "dispatch" => compact(values.as_ptr(), &mask, n, out.as_mut_ptr(), cap, false),
                            "stream" => {
                                let w = compact(values.as_ptr(), &mask, n, out.as_mut_ptr(), cap, true);
                                pipeline::stream_fence();
                                w
                            }
                            _ => scalar::compact(values.as_ptr(), &mask, n, out.as_mut_ptr(), cap),
                        }
                    };
                    assert_eq!(written, want.len(), "{name} n {n} density {density} slack {slack}");
                    assert_eq!(&out[..written], &want[..], "{name} n {n} density {density} slack {slack}");
                    assert!(
                        out[cap..].iter().all(|v| *v == sentinel),
                        "{name} wrote past out_cap (n {n} density {density} slack {slack})"
                    );
                }
            }

            // A too-small capacity truncates instead of overflowing.
            if want.len() > 2 {
                let cap = want.len() - 2;
                for stream in [false, true] {
                    let mut out = vec![sentinel; cap + GUARD];
                    // SAFETY: as above.
                    let written = unsafe { compact(values.as_ptr(), &mask, n, out.as_mut_ptr(), cap, stream) };
                    pipeline::stream_fence();
                    assert_eq!(written, cap, "stream {stream}");
                    assert_eq!(&out[..cap], &want[..cap]);
                    assert!(out[cap..].iter().all(|v| *v == sentinel));
                }
            }

            for tasks in [1, 2, 3, 7] {
                for stream in [Some(false), Some(true), None] {
                    let got = filter_by_mask_with(&values, &mask, tasks, stream);
                    assert_eq!(&got[..], &want[..], "filter n {n} density {density} tasks {tasks} {stream:?}");
                }
            }

            // Unaligned destinations exercise the streaming head/tail logic.
            for shift in [1usize, 3, 7] {
                let mut out = vec![sentinel; want.len() + shift + GUARD];
                // SAFETY: `shift + want.len()` slots are in bounds.
                let written = unsafe { compact(values.as_ptr(), &mask, n, out.as_mut_ptr().add(shift), want.len(), true) };
                pipeline::stream_fence();
                assert_eq!(written, want.len());
                assert_eq!(&out[shift..shift + written], &want[..]);
                assert!(out[..shift].iter().all(|v| *v == sentinel), "wrote before the output");
                assert!(out[shift + written..].iter().all(|v| *v == sentinel), "wrote after the output");
            }
        }
    }
}

#[test]
fn compaction_matches_model_for_every_width() {
    check_compact::<u8>(|x| x as u8, 11);
    check_compact::<u16>(|x| x as u16, 12);
    check_compact::<u32>(|x| x as u32, 13);
    check_compact::<u64>(|x| x, 14);
    check_compact::<u128>(|x| (x as u128) << 64 | (!x) as u128, 15);
    check_compact::<f64>(|x| x as f64, 16);
    check_compact::<i8>(|x| x as i8, 17);
}

#[test]
fn fused_filter_matches_model() {
    let mut rng = Rng::new(21);
    for &n in &[0usize, 1, 64, 4096, 4097, 20_000, 70_001] {
        let values: Vec<i64> = (0..n).map(|_| rng.below(1000) as i64).collect();
        let fvalues: Vec<f32> = values.iter().map(|&v| v as f32 * 0.25).collect();
        for threshold in [-1i64, 0, 10, 500, 990, 1000] {
            for op in OPS {
                let want: Vec<i64> = values.iter().copied().filter(|&v| op.eval(v, threshold)).collect();
                for tasks in [1, 3] {
                    for stream in [Some(false), Some(true)] {
                        let got = filter_by_cmp_scalar_with(&values, op, threshold, tasks, stream);
                        assert_eq!(&got[..], &want[..], "n {n} {op:?} {threshold} tasks {tasks} {stream:?}");
                    }
                }
                let t = threshold as f32 * 0.25;
                let fwant: Vec<f32> = fvalues.iter().copied().filter(|&v| op.eval(v, t)).collect();
                assert_eq!(&filter_by_cmp_scalar(&fvalues, op, t)[..], &fwant[..]);
            }
        }
    }
    // Sorted input: the sampling estimate is as wrong as it can be.
    let sorted: Vec<i32> = (0..100_000).collect();
    let got = filter_by_cmp_scalar_with(&sorted, CmpOp::Ge, 90_000, 1, None);
    assert_eq!(got.len(), 10_000);
    assert!(got.iter().enumerate().all(|(i, &v)| v == 90_000 + i as i32));
    let got = filter_by_cmp_scalar_with(&sorted, CmpOp::Lt, 99_999, 1, Some(true));
    assert_eq!(got.len(), 99_999);
}

#[test]
fn bit_compaction_matches_model() {
    let mut rng = Rng::new(31);
    for &n in LENGTHS {
        for density in [0, 5, 50, 95, 100] {
            let vals = random_mask(&mut rng, n, 50, false);
            let sel = random_mask(&mut rng, n, density, false);
            let want: Vec<bool> = vals.iter().zip(&sel).filter(|(_, s)| **s).map(|(v, _)| *v).collect();
            let got = compact_bits(&mask_from(&vals), &mask_from(&sel), n, want.len());
            assert_eq!(&got[..], &mask_from(&want)[..], "n {n} density {density}");
            let mut portable = vec![0u64; words_for(want.len()) + 1];
            let len = scalar::compact_bits(&mask_from(&vals), &mask_from(&sel), n, &mut portable, 0);
            assert_eq!(len, want.len());
            assert_eq!(&portable[..words_for(want.len())], &mask_from(&want)[..]);
        }
    }
}

#[test]
fn portable_pext_matches_definition() {
    let mut rng = Rng::new(41);
    for _ in 0..2000 {
        let (v, m) = (rng.next_u64(), rng.next_u64() & rng.next_u64());
        let mut want = 0u64;
        let mut k = 0;
        for bit in 0..64 {
            if m >> bit & 1 == 1 {
                want |= (v >> bit & 1) << k;
                k += 1;
            }
        }
        assert_eq!(scalar::pext(v, m), want);
    }
}

#[test]
fn level_is_detected_once_and_stable() {
    let first = level();
    assert_eq!(level(), first);
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::is_x86_feature_detected!("avx2") && std::env::var("DATAFRAMES_SIMD").is_err() {
        assert_eq!(first, Level::Avx2);
    }
}

#[test]
fn mask_indices_match_model() {
    let mut rng = Rng::new(51);
    for &n in LENGTHS {
        for density in [0, 1, 3, 10, 50, 100] {
            let bools = random_mask(&mut rng, n, density, false);
            let mask = mask_from(&bools);
            let want: Vec<u32> = (0..n as u32).filter(|&i| bools[i as usize]).map(|i| i + 7).collect();
            assert_eq!(&mask_indices(&mask, n, 7)[..], &want[..], "n {n} density {density}");
            #[cfg(target_arch = "x86_64")]
            if level() >= Level::Avx2 {
                const GUARD: u32 = 0xDEAD_BEEF;
                let mut out = vec![GUARD; want.len() + 8 + 16];
                // SAFETY: `want.len() + 8` slots are writable; AVX2 detected.
                let written = unsafe { x86::mask_indices(&mask, n, 7, out.as_mut_ptr()) };
                assert_eq!(written, want.len());
                assert_eq!(&out[..written], &want[..]);
                assert!(out[want.len() + 8..].iter().all(|&g| g == GUARD), "wrote past the allowed overhang");
            }
        }
    }
}

#[test]
fn index_lists_and_gathers_match_model() {
    let mut rng = Rng::new(61);
    let mut reused = Vec::new();
    for &n in LENGTHS {
        let values: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let wide: Vec<u128> = values.iter().map(|&v| (v as u128) << 64 | !v as u128).collect();
        let narrow: Vec<u32> = values.iter().map(|&v| v as u32).collect();
        for density in [0, 1, 10, 40, 100] {
            let bools = random_mask(&mut rng, n, density, false);
            let mask = mask_from(&bools);
            let want_idx: Vec<u32> = (0..n as u32).filter(|&i| bools[i as usize]).collect();
            mask_indices_into(&mask, n, 0, &mut reused);
            assert_eq!(reused, want_idx, "n {n} density {density}");
            mask_indices_into(&mask, n, 1000, &mut reused);
            assert!(reused.iter().zip(&want_idx).all(|(&got, &i)| got == i + 1000));

            fn check<T: Pod + PartialEq + std::fmt::Debug>(src: &[T], idx: &[u32], sentinel: T) {
                let want: Vec<T> = idx.iter().map(|&i| src[i as usize]).collect();
                for stream in [false, true] {
                    // Unaligned starts exercise the head and tail of the staged writer.
                    for shift in [0usize, 1, 5] {
                        let mut out = vec![sentinel; shift + idx.len() + 9];
                        // SAFETY: indices are in bounds; `idx.len()` slots follow `shift`.
                        unsafe { gather(src.as_ptr(), idx, out.as_mut_ptr().add(shift), stream) };
                        pipeline::stream_fence();
                        assert_eq!(&out[shift..shift + idx.len()], &want[..], "stream {stream} shift {shift}");
                        assert!(out[..shift].iter().all(|v| *v == sentinel), "wrote before the output");
                        assert!(out[shift + idx.len()..].iter().all(|v| *v == sentinel), "wrote after the output");
                    }
                }
            }
            check(&values, &want_idx, 0xDEAD_BEEF_u64);
            check(&narrow, &want_idx, 0xDEAD_BEEF_u32);
            check(&wide, &want_idx, 0xDEAD_BEEF_u128);
        }
    }
}

#[test]
fn sparse_selection_goes_through_indices_and_agrees() {
    let n = 200_000;
    let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let bools: Vec<bool> = (0..n).map(|i| i % 97 == 3).collect();
    let mask = mask_from(&bools);
    let want: Vec<f64> = values.iter().zip(&bools).filter(|(_, b)| **b).map(|(v, _)| *v).collect();
    for tasks in [1, 4] {
        assert_eq!(&filter_by_mask_with(&values, &mask, tasks, None)[..], &want[..]);
    }
}
