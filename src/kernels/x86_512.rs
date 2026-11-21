//! AVX-512 kernels (F, BW, DQ, VL; VBMI2 for 8- and 16-bit compaction).
//!
//! The same two families as the AVX2 kernels in [`super::x86`], with the
//! instructions AVX-512 has for exactly these jobs:
//!
//! * **compare → bitmask**: a comparison writes its result to a mask register,
//!   so 64 rows become one word in one to eight instructions with no
//!   sign-bit extraction, and all six operators exist for signed, unsigned
//!   and floating-point lanes.
//! * **compaction**: `vpcompress*` packs the selected lanes of a register to
//!   its front. A mask word of 64 rows is one to eight "load, compress, store"
//!   steps whose output offsets are known up front. The compressed register is
//!   always stored whole — the form that writes only the selected lanes to
//!   memory is very slow on some CPUs — so, as in the AVX2 kernels, a store
//!   may run past the selected elements by up to one vector.
//!
//! Sparse stretches of a mask, the streaming (non-temporal) variants and the
//! driver loops are shared with the AVX2 module.
//!
//! Every function is `#[target_feature]`-gated and reached only through the
//! run-time dispatch in [`super`], after [`super::level`] has confirmed the
//! features.

use std::arch::x86_64::*;

use super::CmpOp;
use super::scalar;
use super::x86::{
    GROUP_WORDS, M01, M0F, M33, M55, STAGE_BYTES, StreamOut, compact_kernel, compact_stream_kernel,
    min_max_float_kernel, min_or_max_float_kernel, sparse_steps, sparse_word,
};

// ---------------------------------------------------------------------------
// compare → bitmask
// ---------------------------------------------------------------------------

macro_rules! cmp_scalar_int {
    ($(#[$doc:meta])* $name:ident, $t:ty, set1 = $set1:ident, signed = $cmp_s:ident, unsigned = $cmp_u:ident, lanes = $lanes:expr) => {
        $(#[$doc])*
        ///
        /// Processes the whole 64-element words of `values`; `out` receives one
        /// word each. Returns the number of set bits.
        ///
        /// # Safety
        /// The CPU must support AVX-512 F and BW and POPCNT, `values.len()`
        /// must be a multiple of 64 and `out.len()` at least `values.len() / 64`.
        #[target_feature(enable = "avx512f,avx512bw,popcnt")]
        pub unsafe fn $name<const U: bool>(values: &[$t], rhs: $t, op: CmpOp, out: &mut [u64]) -> usize {
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,popcnt")]
            unsafe fn run<const P: i32, const U: bool>(values: &[$t], r: __m512i, out: &mut [u64]) -> usize {
                const LANES: usize = $lanes;
                let words = values.len() / 64;
                let mut count = 0usize;
                let mut i = 0;
                while i < words {
                    // SAFETY: word `i` is fully inside `values`; `i < out.len()`.
                    unsafe {
                        let p = values.as_ptr().add(i * 64);
                        // The masks are chained from the last vector to the
                        // first (`w = w << LANES | m`). Written as independent
                        // `m << shift` terms, the compiler vectorises the
                        // combination itself and the detour through vector
                        // registers costs more than the comparisons.
                        let mut w = 0u64;
                        let mut k = 64 / LANES;
                        while k > 0 {
                            k -= 1;
                            let v = _mm512_loadu_si512(p.add(k * LANES).cast());
                            let m = if U { $cmp_u::<P>(v, r) as u64 } else { $cmp_s::<P>(v, r) as u64 };
                            w = if LANES == 64 { m } else { (w << (LANES % 64)) | m };
                        }
                        *out.get_unchecked_mut(i) = w;
                        count += w.count_ones() as usize;
                    }
                    i += 1;
                }
                count
            }
            debug_assert!(values.len() % 64 == 0 && out.len() >= values.len() / 64);
            let r = $set1(rhs);
            // SAFETY: forwarded preconditions.
            unsafe {
                match op {
                    CmpOp::Eq => run::<{ _MM_CMPINT_EQ }, U>(values, r, out),
                    CmpOp::Ne => run::<{ _MM_CMPINT_NE }, U>(values, r, out),
                    CmpOp::Lt => run::<{ _MM_CMPINT_LT }, U>(values, r, out),
                    CmpOp::Le => run::<{ _MM_CMPINT_LE }, U>(values, r, out),
                    CmpOp::Gt => run::<{ _MM_CMPINT_NLE }, U>(values, r, out),
                    CmpOp::Ge => run::<{ _MM_CMPINT_NLT }, U>(values, r, out),
                }
            }
        }
    };
}

cmp_scalar_int!(
    /// `values[i] op rhs` for 8-bit integers (`U` selects unsigned order).
    cmp_scalar_i8, i8, set1 = _mm512_set1_epi8, signed = _mm512_cmp_epi8_mask, unsigned = _mm512_cmp_epu8_mask, lanes = 64
);
cmp_scalar_int!(
    /// `values[i] op rhs` for 16-bit integers (`U` selects unsigned order).
    cmp_scalar_i16, i16, set1 = _mm512_set1_epi16, signed = _mm512_cmp_epi16_mask, unsigned = _mm512_cmp_epu16_mask, lanes = 32
);
cmp_scalar_int!(
    /// `values[i] op rhs` for 32-bit integers (`U` selects unsigned order).
    cmp_scalar_i32, i32, set1 = _mm512_set1_epi32, signed = _mm512_cmp_epi32_mask, unsigned = _mm512_cmp_epu32_mask, lanes = 16
);
cmp_scalar_int!(
    /// `values[i] op rhs` for 64-bit integers (`U` selects unsigned order).
    cmp_scalar_i64, i64, set1 = _mm512_set1_epi64, signed = _mm512_cmp_epi64_mask, unsigned = _mm512_cmp_epu64_mask, lanes = 8
);

macro_rules! cmp_scalar_float {
    ($(#[$doc:meta])* $name:ident, $t:ty, set1 = $set1:ident, load = $load:ident, cmp = $cmp:ident, vector = $vec:ty, lanes = $lanes:expr) => {
        $(#[$doc])*
        ///
        /// The order is total, with NaN greatest; `rhs` must not be NaN.
        /// Processes the whole 64-element words of `values`; `out` receives one
        /// word each. Returns the number of set bits. (`U` is ignored; it keeps
        /// the signature in line with the integer kernels.)
        ///
        /// # Safety
        /// The CPU must support AVX-512 F and BW and POPCNT, `values.len()`
        /// must be a multiple of 64 and `out.len()` at least `values.len() / 64`.
        #[target_feature(enable = "avx512f,avx512bw,popcnt")]
        pub unsafe fn $name<const U: bool>(values: &[$t], rhs: $t, op: CmpOp, out: &mut [u64]) -> usize {
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,popcnt")]
            unsafe fn run<const P: i32>(values: &[$t], r: $vec, out: &mut [u64]) -> usize {
                const LANES: usize = $lanes;
                let words = values.len() / 64;
                let mut count = 0usize;
                let mut i = 0;
                while i < words {
                    // SAFETY: word `i` is fully inside `values`; `i < out.len()`.
                    unsafe {
                        let p = values.as_ptr().add(i * 64);
                        // Chained from the last vector to the first; see the
                        // integer kernels.
                        let mut w = 0u64;
                        let mut k = 64 / LANES;
                        while k > 0 {
                            k -= 1;
                            let v = $load(p.add(k * LANES));
                            w = (w << LANES) | $cmp::<P>(v, r) as u64;
                        }
                        *out.get_unchecked_mut(i) = w;
                        count += w.count_ones() as usize;
                    }
                    i += 1;
                }
                count
            }
            debug_assert!(values.len() % 64 == 0 && out.len() >= values.len() / 64 && !rhs.is_nan());
            let r = $set1(rhs);
            // SAFETY: forwarded preconditions. The ordered predicates are false
            // for a NaN value and the unordered ones true, which is the total
            // order with NaN above every number.
            unsafe {
                match op {
                    CmpOp::Eq => run::<{ _CMP_EQ_OQ }>(values, r, out),
                    CmpOp::Ne => run::<{ _CMP_NEQ_UQ }>(values, r, out),
                    CmpOp::Lt => run::<{ _CMP_LT_OQ }>(values, r, out),
                    CmpOp::Le => run::<{ _CMP_LE_OQ }>(values, r, out),
                    CmpOp::Gt => run::<{ _CMP_NLE_UQ }>(values, r, out),
                    CmpOp::Ge => run::<{ _CMP_NLT_UQ }>(values, r, out),
                }
            }
        }
    };
}

cmp_scalar_float!(
    /// `values[i] op rhs` for `f32`.
    cmp_scalar_f32, f32, set1 = _mm512_set1_ps, load = _mm512_loadu_ps, cmp = _mm512_cmp_ps_mask, vector = __m512, lanes = 16
);
cmp_scalar_float!(
    /// `values[i] op rhs` for `f64`.
    cmp_scalar_f64, f64, set1 = _mm512_set1_pd, load = _mm512_loadu_pd, cmp = _mm512_cmp_pd_mask, vector = __m512d, lanes = 8
);

// ---------------------------------------------------------------------------
// compaction
// ---------------------------------------------------------------------------

/// Compact the 64 eight-byte elements at `base` selected by `w` to `dst`.
///
/// # Safety
/// `base` must be valid for 64 reads and `dst` for `w.count_ones() + 8`
/// writes; AVX-512 F must be available.
#[inline(always)]
unsafe fn dense_word8(w: u64, base: *const u64, dst: *mut u64) {
    // Where each mask byte's output begins: running byte counts.
    let n1 = w - ((w >> 1) & M55);
    let n2 = (n1 & M33) + ((n1 >> 2) & M33);
    let starts = ((n2 + (n2 >> 4)) & M0F).wrapping_mul(M01) << 8;
    // SAFETY: step `k` reads elements `8k..8k + 8` and writes eight elements
    // at an offset below the population count.
    unsafe {
        let mut k = 0;
        while k < 8 {
            let m = (w >> (8 * k)) as u8;
            let at = ((starts >> (8 * k)) & 0xFF) as usize;
            let v = _mm512_loadu_si512(base.add(8 * k).cast());
            _mm512_storeu_si512(dst.add(at).cast(), _mm512_maskz_compress_epi64(m, v));
            k += 1;
        }
    }
}

/// Compact the 64 four-byte elements at `base` selected by `w` to `dst`.
///
/// # Safety
/// `base` must be valid for 64 reads and `dst` for `w.count_ones() + 16`
/// writes; AVX-512 F must be available.
#[inline(always)]
unsafe fn dense_word4(w: u64, base: *const u32, dst: *mut u32) {
    // SAFETY: step `k` reads elements `16k..16k + 16` and writes sixteen
    // elements at an offset below the population count.
    unsafe {
        let mut at = 0usize;
        let mut k = 0;
        while k < 4 {
            let m = (w >> (16 * k)) as u16;
            let v = _mm512_loadu_si512(base.add(16 * k).cast());
            _mm512_storeu_si512(dst.add(at).cast(), _mm512_maskz_compress_epi32(m, v));
            at += m.count_ones() as usize;
            k += 1;
        }
    }
}

/// Compact the 64 two-byte elements at `base` selected by `w` to `dst`.
///
/// # Safety
/// `base` must be valid for 64 reads and `dst` for `w.count_ones() + 32`
/// writes; AVX-512 F, BW and VBMI2 must be available.
#[inline(always)]
unsafe fn dense_word2(w: u64, base: *const u16, dst: *mut u16) {
    // SAFETY: two steps of 32 elements; the second begins after the first
    // step's selected elements.
    unsafe {
        let (lo, hi) = (w as u32, (w >> 32) as u32);
        let a = _mm512_loadu_si512(base.cast());
        _mm512_storeu_si512(dst.cast(), _mm512_maskz_compress_epi16(lo, a));
        let b = _mm512_loadu_si512(base.add(32).cast());
        _mm512_storeu_si512(dst.add(lo.count_ones() as usize).cast(), _mm512_maskz_compress_epi16(hi, b));
    }
}

/// Compact the 64 one-byte elements at `base` selected by `w` to `dst`.
///
/// # Safety
/// `base` must be valid for 64 reads and `dst` for 64 writes; AVX-512 F, BW
/// and VBMI2 must be available.
#[inline(always)]
unsafe fn dense_word1(w: u64, base: *const u8, dst: *mut u8) {
    // SAFETY: one step of 64 elements.
    unsafe {
        let v = _mm512_loadu_si512(base.cast());
        _mm512_storeu_si512(dst.cast(), _mm512_maskz_compress_epi8(w, v));
    }
}

compact_kernel!(
    /// Compact 8-byte elements.
    compact8, u64, features = "avx512f,avx512bw,bmi1,popcnt", lanes = 8, dense = dense_word8, dense_cost = 10
);
compact_kernel!(
    /// Compact 4-byte elements.
    compact4, u32, features = "avx512f,avx512bw,bmi1,popcnt", lanes = 16, dense = dense_word4, dense_cost = 6
);
compact_kernel!(
    /// Compact 2-byte elements (needs VBMI2).
    compact2, u16, features = "avx512f,avx512bw,avx512vbmi2,bmi1,popcnt", lanes = 32, dense = dense_word2, dense_cost = 3
);
compact_kernel!(
    /// Compact 1-byte elements (needs VBMI2).
    compact1, u8, features = "avx512f,avx512bw,avx512vbmi2,bmi1,popcnt", lanes = 64, dense = dense_word1, dense_cost = 2
);

compact_stream_kernel!(
    /// Streaming compaction of 8-byte elements.
    compact8_stream, u64, features = "avx512f,avx512bw,avx2,bmi1,popcnt", dense = dense_word8, dense_cost = 10
);
compact_stream_kernel!(
    /// Streaming compaction of 4-byte elements.
    compact4_stream, u32, features = "avx512f,avx512bw,avx2,bmi1,popcnt", dense = dense_word4, dense_cost = 6
);

// ---------------------------------------------------------------------------
// mask → indices
// ---------------------------------------------------------------------------

/// Write the positions of the set bits among the first `n` bits of `mask`
/// (plus `base`) to `out`, ascending. Returns how many were written.
///
/// # Safety
/// * `out` must be valid for `count + 16` writes, where `count` is the number
///   of set bits among the first `n` (whole vectors are stored).
/// * The CPU must support AVX-512 F and BW, BMI1 and POPCNT.
#[target_feature(enable = "avx512f,avx512bw,bmi1,popcnt")]
pub unsafe fn mask_indices(mask: &[u64], n: usize, base: u32, out: *mut u32) -> usize {
    let n_words = n.div_ceil(64);
    let iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let sixteen = _mm512_set1_epi32(16);
    let mut o = 0usize;
    let mut wi = 0;
    // Bits past `n` in the last word do not count.
    let word = |wi: usize| -> u64 {
        // SAFETY: `wi < n_words <= mask.len()`.
        let w = unsafe { *mask.get_unchecked(wi) };
        if wi + 1 == n_words && !n.is_multiple_of(64) { w & ((1u64 << (n % 64)) - 1) } else { w }
    };
    // SAFETY: a sparse word writes four slots at `o`, a dense word writes
    // sixteen slots at offsets below its population count; in both cases the
    // writes end at most sixteen slots past the running count, inside the
    // `count + 16` slots the caller provides.
    unsafe {
        while wi < n_words {
            let end = (wi + GROUP_WORDS).min(n_words);
            let ones: usize = (wi..end).map(|i| word(i).count_ones() as usize).sum();
            if 2 * ones <= 3 * (end - wi) {
                while wi < end {
                    let mut w = word(wi);
                    let pc = w.count_ones() as usize;
                    if pc > 4 {
                        break; // denser than its group: the vector steps take over
                    }
                    let row = base.wrapping_add((wi * 64) as u32);
                    let mut j = 0;
                    while j < 4 {
                        out.add(o + j).write(row.wrapping_add(_tzcnt_u64(w) as u32));
                        w = _blsr_u64(w);
                        j += 1;
                    }
                    o += pc;
                    wi += 1;
                }
            }
            while wi < end {
                let w = word(wi);
                if w != 0 {
                    let mut rows = _mm512_add_epi32(iota, _mm512_set1_epi32(base.wrapping_add((wi * 64) as u32) as i32));
                    let mut k = 0;
                    while k < 4 {
                        let m = (w >> (16 * k)) as u16;
                        _mm512_storeu_si512(out.add(o).cast(), _mm512_maskz_compress_epi32(m, rows));
                        o += m.count_ones() as usize;
                        rows = _mm512_add_epi32(rows, sixteen);
                        k += 1;
                    }
                }
                wi += 1;
            }
        }
    }
    o
}

// ---------------------------------------------------------------------------
// floating-point minimum and maximum
// ---------------------------------------------------------------------------

min_max_float_kernel!(
    /// Smallest and largest `f64`.
    min_max_f64, f64, __m512d, features = "avx512f", lanes = 8,
    load = _mm512_loadu_pd, set1 = _mm512_set1_pd, min = _mm512_min_pd, max = _mm512_max_pd, store = _mm512_storeu_pd
);
min_max_float_kernel!(
    /// Smallest and largest `f32`.
    min_max_f32, f32, __m512, features = "avx512f", lanes = 16,
    load = _mm512_loadu_ps, set1 = _mm512_set1_ps, min = _mm512_min_ps, max = _mm512_max_ps, store = _mm512_storeu_ps
);
min_or_max_float_kernel!(
    /// Smallest `f64`.
    min_f64, f64, features = "avx512f", lanes = 8, init = f64::INFINITY,
    load = _mm512_loadu_pd, set1 = _mm512_set1_pd, op = _mm512_min_pd, store = _mm512_storeu_pd, better = <
);
min_or_max_float_kernel!(
    /// Largest `f64`.
    max_f64, f64, features = "avx512f", lanes = 8, init = f64::NEG_INFINITY,
    load = _mm512_loadu_pd, set1 = _mm512_set1_pd, op = _mm512_max_pd, store = _mm512_storeu_pd, better = >
);
min_or_max_float_kernel!(
    /// Smallest `f32`.
    min_f32, f32, features = "avx512f", lanes = 16, init = f32::INFINITY,
    load = _mm512_loadu_ps, set1 = _mm512_set1_ps, op = _mm512_min_ps, store = _mm512_storeu_ps, better = <
);
min_or_max_float_kernel!(
    /// Largest `f32`.
    max_f32, f32, features = "avx512f", lanes = 16, init = f32::NEG_INFINITY,
    load = _mm512_loadu_ps, set1 = _mm512_set1_ps, op = _mm512_max_ps, store = _mm512_storeu_ps, better = >
);
