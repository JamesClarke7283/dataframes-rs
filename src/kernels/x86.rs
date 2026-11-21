//! AVX2 kernels (x86-64-v3: AVX2 + BMI1/2 + POPCNT + LZCNT + FMA).
//!
//! Every function here is `#[target_feature]`-gated and reached only through
//! the run-time dispatch in [`super`], after [`super::level`] has confirmed
//! the features are present.
//!
//! Two families:
//!
//! * **compare → bitmask**: one 64-bit word per 64 rows, built from
//!   `vpcmp*` + `vmovmsk*`. Six operators are covered by three vector
//!   predicates and an optional complement of the finished word.
//! * **compaction**: `vpermd` (32/64-bit lanes) or `pshufb` (8/16-bit lanes)
//!   with a table lookup per mask chunk. The store is a full vector and the
//!   output cursor advances by the population count, so the loop body has no
//!   data-dependent branch; the only loop-carried dependency is one add.

use std::arch::x86_64::*;

use super::CmpOp;
use super::scalar;
use super::tables::{PERM32, PERM64, SHUF8, SHUF16};

/// Base predicate selectors (meaning depends on the element kind, see below).
const P0: u8 = 0;
const P1: u8 = 1;
const P2: u8 = 2;

// ---------------------------------------------------------------------------
// compare → bitmask
// ---------------------------------------------------------------------------
//
// Integers: P0 = `v < r`, P1 = `v > r`, P2 = `v == r`.
// Floats (total order, `r` not NaN): P0 = `v < r`, P1 = `v <= r`, P2 = `v == r`.
// The complement turns these into `>=`, `<=`/`>`, `!=`.
//
// Unsigned integers are compared as signed after flipping the sign bit of
// both operands (`U = true`); the scalar operand arrives already flipped.

/// # Safety
/// `p` must be valid for reading 64 elements.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn word_i8<const P: u8, const U: bool>(p: *const i8, r: __m256i) -> u64 {
    let bias = _mm256_set1_epi8(i8::MIN);
    let mut w = 0u64;
    let mut k = 0;
    while k < 2 {
        // SAFETY: `k * 32 + 32 <= 64` elements from `p`.
        let mut v = unsafe { _mm256_loadu_si256(p.add(k * 32).cast()) };
        if U {
            v = _mm256_xor_si256(v, bias);
        }
        let c = match P {
            P0 => _mm256_cmpgt_epi8(r, v),
            P1 => _mm256_cmpgt_epi8(v, r),
            _ => _mm256_cmpeq_epi8(v, r),
        };
        w |= (_mm256_movemask_epi8(c) as u32 as u64) << (k * 32);
        k += 1;
    }
    w
}

/// # Safety
/// `p` must be valid for reading 64 elements.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn word_i16<const P: u8, const U: bool>(p: *const i16, r: __m256i) -> u64 {
    let bias = _mm256_set1_epi16(i16::MIN);
    let mut w = 0u64;
    let mut k = 0;
    while k < 4 {
        // SAFETY: `k * 16 + 16 <= 64` elements from `p`.
        let mut v = unsafe { _mm256_loadu_si256(p.add(k * 16).cast()) };
        if U {
            v = _mm256_xor_si256(v, bias);
        }
        let c = match P {
            P0 => _mm256_cmpgt_epi16(r, v),
            P1 => _mm256_cmpgt_epi16(v, r),
            _ => _mm256_cmpeq_epi16(v, r),
        };
        // Narrow each 16-bit lane to a byte (per 128-bit half), then take the
        // sign bits: bits 0..8 are lanes 0..8, bits 16..24 are lanes 8..16.
        let m = _mm256_movemask_epi8(_mm256_packs_epi16(c, c)) as u32;
        let m16 = (m & 0xFF) | (m >> 8 & 0xFF00);
        w |= (m16 as u64) << (k * 16);
        k += 1;
    }
    w
}

/// # Safety
/// `p` must be valid for reading 64 elements.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn word_i32<const P: u8, const U: bool>(p: *const i32, r: __m256i) -> u64 {
    let bias = _mm256_set1_epi32(i32::MIN);
    let mut w = 0u64;
    let mut k = 0;
    while k < 8 {
        // SAFETY: `k * 8 + 8 <= 64` elements from `p`.
        let mut v = unsafe { _mm256_loadu_si256(p.add(k * 8).cast()) };
        if U {
            v = _mm256_xor_si256(v, bias);
        }
        let c = match P {
            P0 => _mm256_cmpgt_epi32(r, v),
            P1 => _mm256_cmpgt_epi32(v, r),
            _ => _mm256_cmpeq_epi32(v, r),
        };
        w |= (_mm256_movemask_ps(_mm256_castsi256_ps(c)) as u32 as u64) << (k * 8);
        k += 1;
    }
    w
}

/// # Safety
/// `p` must be valid for reading 64 elements.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn word_i64<const P: u8, const U: bool>(p: *const i64, r: __m256i) -> u64 {
    let bias = _mm256_set1_epi64x(i64::MIN);
    let mut w = 0u64;
    let mut k = 0;
    while k < 16 {
        // SAFETY: `k * 4 + 4 <= 64` elements from `p`.
        let mut v = unsafe { _mm256_loadu_si256(p.add(k * 4).cast()) };
        if U {
            v = _mm256_xor_si256(v, bias);
        }
        let c = match P {
            P0 => _mm256_cmpgt_epi64(r, v),
            P1 => _mm256_cmpgt_epi64(v, r),
            _ => _mm256_cmpeq_epi64(v, r),
        };
        w |= (_mm256_movemask_pd(_mm256_castsi256_pd(c)) as u32 as u64) << (k * 4);
        k += 1;
    }
    w
}

/// # Safety
/// `p` must be valid for reading 64 elements.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn word_f32<const P: u8, const U: bool>(p: *const f32, r: __m256) -> u64 {
    let mut w = 0u64;
    let mut k = 0;
    while k < 8 {
        // SAFETY: `k * 8 + 8 <= 64` elements from `p`.
        let v = unsafe { _mm256_loadu_ps(p.add(k * 8)) };
        let c = match P {
            P0 => _mm256_cmp_ps::<_CMP_LT_OQ>(v, r),
            P1 => _mm256_cmp_ps::<_CMP_LE_OQ>(v, r),
            _ => _mm256_cmp_ps::<_CMP_EQ_OQ>(v, r),
        };
        w |= (_mm256_movemask_ps(c) as u32 as u64) << (k * 8);
        k += 1;
    }
    w
}

/// # Safety
/// `p` must be valid for reading 64 elements.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn word_f64<const P: u8, const U: bool>(p: *const f64, r: __m256d) -> u64 {
    let mut w = 0u64;
    let mut k = 0;
    while k < 16 {
        // SAFETY: `k * 4 + 4 <= 64` elements from `p`.
        let v = unsafe { _mm256_loadu_pd(p.add(k * 4)) };
        let c = match P {
            P0 => _mm256_cmp_pd::<_CMP_LT_OQ>(v, r),
            P1 => _mm256_cmp_pd::<_CMP_LE_OQ>(v, r),
            _ => _mm256_cmp_pd::<_CMP_EQ_OQ>(v, r),
        };
        w |= (_mm256_movemask_pd(c) as u32 as u64) << (k * 4);
        k += 1;
    }
    w
}

macro_rules! cmp_scalar_kernel {
    (
        $(#[$doc:meta])*
        $name:ident, $t:ty, $word:ident, $vec:ty,
        splat = |$rhs:ident, $unsigned:ident| $splat:expr,
        ops = { $($op:ident => ($p:ident, $flip:expr)),* $(,)? }
    ) => {
        $(#[$doc])*
        ///
        /// Processes the whole 64-element words of `values`; `out` receives one
        /// word each. Returns the number of set bits.
        ///
        /// # Safety
        /// The CPU must support AVX2 and POPCNT, `values.len()` must be a
        /// multiple of 64 and `out.len()` at least `values.len() / 64`.
        #[target_feature(enable = "avx2,popcnt")]
        pub unsafe fn $name<const U: bool>(values: &[$t], $rhs: $t, op: CmpOp, out: &mut [u64]) -> usize {
            #[inline]
            #[target_feature(enable = "avx2,popcnt")]
            unsafe fn run<const P: u8, const U: bool>(
                values: &[$t],
                r: $vec,
                flip: u64,
                out: &mut [u64],
            ) -> usize {
                let words = values.len() / 64;
                let mut count = 0usize;
                let mut i = 0;
                while i < words {
                    // SAFETY: word `i` is fully inside `values`; `i < out.len()`.
                    unsafe {
                        let w = $word::<P, U>(values.as_ptr().add(i * 64), r) ^ flip;
                        *out.get_unchecked_mut(i) = w;
                        count += w.count_ones() as usize;
                    }
                    i += 1;
                }
                count
            }
            debug_assert!(values.len() % 64 == 0 && out.len() >= values.len() / 64);
            let r: $vec = {
                let $unsigned = U;
                $splat
            };
            // SAFETY: forwarded preconditions.
            unsafe {
                match op {
                    $(CmpOp::$op => run::<{ $p }, U>(values, r, if $flip { u64::MAX } else { 0 }, out),)*
                }
            }
        }
    };
}

cmp_scalar_kernel!(
    /// `values[i] op rhs` for 8-bit integers (`U` selects unsigned order).
    cmp_scalar_i8, i8, word_i8, __m256i,
    splat = |rhs, unsigned| _mm256_set1_epi8(if unsigned { rhs ^ i8::MIN } else { rhs }),
    ops = { Lt => (P0, false), Ge => (P0, true), Gt => (P1, false), Le => (P1, true), Eq => (P2, false), Ne => (P2, true) }
);
cmp_scalar_kernel!(
    /// `values[i] op rhs` for 16-bit integers (`U` selects unsigned order).
    cmp_scalar_i16, i16, word_i16, __m256i,
    splat = |rhs, unsigned| _mm256_set1_epi16(if unsigned { rhs ^ i16::MIN } else { rhs }),
    ops = { Lt => (P0, false), Ge => (P0, true), Gt => (P1, false), Le => (P1, true), Eq => (P2, false), Ne => (P2, true) }
);
cmp_scalar_kernel!(
    /// `values[i] op rhs` for 32-bit integers (`U` selects unsigned order).
    cmp_scalar_i32, i32, word_i32, __m256i,
    splat = |rhs, unsigned| _mm256_set1_epi32(if unsigned { rhs ^ i32::MIN } else { rhs }),
    ops = { Lt => (P0, false), Ge => (P0, true), Gt => (P1, false), Le => (P1, true), Eq => (P2, false), Ne => (P2, true) }
);
cmp_scalar_kernel!(
    /// `values[i] op rhs` for 64-bit integers (`U` selects unsigned order).
    cmp_scalar_i64, i64, word_i64, __m256i,
    splat = |rhs, unsigned| _mm256_set1_epi64x(if unsigned { rhs ^ i64::MIN } else { rhs }),
    ops = { Lt => (P0, false), Ge => (P0, true), Gt => (P1, false), Le => (P1, true), Eq => (P2, false), Ne => (P2, true) }
);
cmp_scalar_kernel!(
    /// `values[i] op rhs` for `f32` in the total order; `rhs` must not be NaN.
    cmp_scalar_f32, f32, word_f32, __m256,
    splat = |rhs, unsigned| { let _ = unsigned; _mm256_set1_ps(rhs) },
    ops = { Lt => (P0, false), Ge => (P0, true), Le => (P1, false), Gt => (P1, true), Eq => (P2, false), Ne => (P2, true) }
);
cmp_scalar_kernel!(
    /// `values[i] op rhs` for `f64` in the total order; `rhs` must not be NaN.
    cmp_scalar_f64, f64, word_f64, __m256d,
    splat = |rhs, unsigned| { let _ = unsigned; _mm256_set1_pd(rhs) },
    ops = { Lt => (P0, false), Ge => (P0, true), Le => (P1, false), Gt => (P1, true), Eq => (P2, false), Ne => (P2, true) }
);

// ---------------------------------------------------------------------------
// compaction
// ---------------------------------------------------------------------------

/// Compact 8-byte elements.
///
/// # Safety
/// As [`scalar::compact`], plus the CPU must support AVX2/POPCNT.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn compact8(values: *const u64, mask: &[u64], n: usize, out: *mut u64, out_cap: usize) -> usize {
    const LANES: usize = 4;
    let full = n / 64;
    let table = PERM64.0.as_ptr();
    let mut o = 0usize;
    let mut wi = 0;
    // SAFETY: each loop iteration reads the 64 in-bounds elements of a full
    // word. A vector store at cursor `c` touches `c..c+LANES`; the largest
    // cursor used for word `wi` is below `o + pc`, and the loop only proceeds
    // while `o + pc + LANES <= out_cap`.
    unsafe {
        while wi < full {
            let w = *mask.get_unchecked(wi);
            if w != 0 {
                let pc = w.count_ones() as usize;
                if o + pc + LANES > out_cap {
                    break;
                }
                let base = values.add(wi * 64);
                if w == u64::MAX {
                    std::ptr::copy_nonoverlapping(base, out.add(o), 64);
                    o += 64;
                } else {
                    let mut m = w;
                    let mut k = 0;
                    while k < 16 {
                        let mb = (m & 0xF) as usize;
                        let v = _mm256_loadu_si256(base.add(k * LANES).cast());
                        let p = _mm256_load_si256(table.add(mb).cast());
                        _mm256_storeu_si256(out.add(o).cast(), _mm256_permutevar8x32_epi32(v, p));
                        o += (mb as u32).count_ones() as usize;
                        m >>= 4;
                        k += 1;
                    }
                }
            }
            wi += 1;
        }
        o + scalar::compact_exact(values, mask, wi, n, out.add(o), out_cap - o)
    }
}

/// Compact 4-byte elements.
///
/// # Safety
/// As [`scalar::compact`], plus the CPU must support AVX2/POPCNT.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn compact4(values: *const u32, mask: &[u64], n: usize, out: *mut u32, out_cap: usize) -> usize {
    const LANES: usize = 8;
    let full = n / 64;
    let table = PERM32.0.as_ptr();
    let mut o = 0usize;
    let mut wi = 0;
    // SAFETY: see `compact8`.
    unsafe {
        while wi < full {
            let w = *mask.get_unchecked(wi);
            if w != 0 {
                let pc = w.count_ones() as usize;
                if o + pc + LANES > out_cap {
                    break;
                }
                let base = values.add(wi * 64);
                if w == u64::MAX {
                    std::ptr::copy_nonoverlapping(base, out.add(o), 64);
                    o += 64;
                } else {
                    let mut m = w;
                    let mut k = 0;
                    while k < 8 {
                        let mb = (m & 0xFF) as usize;
                        let v = _mm256_loadu_si256(base.add(k * LANES).cast());
                        let p = _mm256_load_si256(table.add(mb).cast());
                        _mm256_storeu_si256(out.add(o).cast(), _mm256_permutevar8x32_epi32(v, p));
                        o += (mb as u32).count_ones() as usize;
                        m >>= 8;
                        k += 1;
                    }
                }
            }
            wi += 1;
        }
        o + scalar::compact_exact(values, mask, wi, n, out.add(o), out_cap - o)
    }
}

/// Compact 2-byte elements.
///
/// # Safety
/// As [`scalar::compact`], plus the CPU must support AVX2/POPCNT.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn compact2(values: *const u16, mask: &[u64], n: usize, out: *mut u16, out_cap: usize) -> usize {
    const LANES: usize = 8;
    let full = n / 64;
    let table = SHUF16.0.as_ptr();
    let mut o = 0usize;
    let mut wi = 0;
    // SAFETY: see `compact8`; each step loads and stores 8 lanes (16 bytes).
    unsafe {
        while wi < full {
            let w = *mask.get_unchecked(wi);
            if w != 0 {
                let pc = w.count_ones() as usize;
                if o + pc + LANES > out_cap {
                    break;
                }
                let base = values.add(wi * 64);
                if w == u64::MAX {
                    std::ptr::copy_nonoverlapping(base, out.add(o), 64);
                    o += 64;
                } else {
                    let mut m = w;
                    let mut k = 0;
                    while k < 8 {
                        let mb = (m & 0xFF) as usize;
                        let v = _mm_loadu_si128(base.add(k * LANES).cast());
                        let s = _mm_load_si128(table.add(mb).cast());
                        _mm_storeu_si128(out.add(o).cast(), _mm_shuffle_epi8(v, s));
                        o += (mb as u32).count_ones() as usize;
                        m >>= 8;
                        k += 1;
                    }
                }
            }
            wi += 1;
        }
        o + scalar::compact_exact(values, mask, wi, n, out.add(o), out_cap - o)
    }
}

/// Compact 1-byte elements.
///
/// # Safety
/// As [`scalar::compact`], plus the CPU must support AVX2/POPCNT.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn compact1(values: *const u8, mask: &[u64], n: usize, out: *mut u8, out_cap: usize) -> usize {
    const LANES: usize = 8;
    let full = n / 64;
    let table = SHUF8.0.as_ptr();
    let mut o = 0usize;
    let mut wi = 0;
    // SAFETY: see `compact8`; each step loads and stores 8 lanes (8 bytes).
    unsafe {
        while wi < full {
            let w = *mask.get_unchecked(wi);
            if w != 0 {
                let pc = w.count_ones() as usize;
                if o + pc + LANES > out_cap {
                    break;
                }
                let base = values.add(wi * 64);
                if w == u64::MAX {
                    std::ptr::copy_nonoverlapping(base, out.add(o), 64);
                    o += 64;
                } else {
                    let mut m = w;
                    let mut k = 0;
                    while k < 8 {
                        let mb = (m & 0xFF) as usize;
                        let v = _mm_loadl_epi64(base.add(k * LANES).cast());
                        let s = _mm_load_si128(table.add(mb).cast());
                        _mm_storel_epi64(out.add(o).cast(), _mm_shuffle_epi8(v, s));
                        o += (mb as u32).count_ones() as usize;
                        m >>= 8;
                        k += 1;
                    }
                }
            }
            wi += 1;
        }
        o + scalar::compact_exact(values, mask, wi, n, out.add(o), out_cap - o)
    }
}

/// Bit compaction with the `pext` instruction. See [`scalar::compact_bits`].
///
/// # Safety
/// The CPU must support BMI2 and POPCNT.
#[target_feature(enable = "bmi2,popcnt")]
pub unsafe fn compact_bits(values: &[u64], mask: &[u64], n: usize, out_words: &mut [u64], mut out_len: usize) -> usize {
    use crate::memory::{tail_mask, words_for};
    let n_words = words_for(n);
    for wi in 0..n_words {
        let mut m = mask[wi];
        if wi + 1 == n_words {
            m &= tail_mask(n);
        }
        if m == 0 {
            continue;
        }
        let packed = _pext_u64(values[wi], m);
        let pc = m.count_ones() as usize;
        let (word, bit) = (out_len / 64, out_len % 64);
        out_words[word] |= packed << bit;
        if bit != 0 && bit + pc > 64 {
            out_words[word + 1] |= packed >> (64 - bit);
        }
        out_len += pc;
    }
    out_len
}

// ---------------------------------------------------------------------------
// streaming compaction (non-temporal stores)
// ---------------------------------------------------------------------------
//
// A plain store to a cache line that is not in the cache first *reads* the
// line from memory (a "read for ownership"), so writing a large output costs
// twice its size in memory traffic. Non-temporal stores skip the read, but
// they want whole, aligned cache lines, while compaction emits a ragged
// stream. The kernels below reconcile the two: they compact into a small
// stack buffer that stays in L1, and whenever it fills they hand its whole
// lines to `vmovntdq`. The destination is touched exactly once per line.
//
// Unlike the plain kernels these never write past the last selected element,
// so they need no apron and no slack between parallel tasks.

/// Bytes accumulated in the staging buffer before it is flushed.
const STAGE_BYTES: usize = 4096;
/// Staging buffer size: the flush threshold, plus one full mask word of the
/// widest element (64 x 8 bytes), plus one vector of overhang.
const STAGE_CAP: usize = STAGE_BYTES + 64 * 8 + 32;

#[repr(C, align(64))]
struct Stage([std::mem::MaybeUninit<u8>; STAGE_CAP]);

/// Destination cursor of a streaming kernel.
struct StreamOut {
    stage: Stage,
    /// Fill level of the stage, in bytes.
    fill: usize,
    /// Destination address of `stage[0]`; always 64-byte aligned. It may lie
    /// up to 63 bytes before the output, so it is only ever combined with
    /// offsets through wrapping arithmetic and never dereferenced below `out`.
    line: *mut u8,
    /// Bytes of the first destination line that precede the output and must
    /// not be written (non-zero only until the first flush).
    skip: usize,
}

impl StreamOut {
    #[inline]
    fn new(out: *mut u8) -> Self {
        let skip = out as usize % 64;
        StreamOut {
            stage: Stage([std::mem::MaybeUninit::uninit(); STAGE_CAP]),
            fill: skip,
            line: out.wrapping_sub(skip),
            skip,
        }
    }

    #[inline(always)]
    fn cursor(&mut self) -> *mut u8 {
        // SAFETY: `fill < STAGE_BYTES + 64 * 8` is maintained by the callers,
        // which flush after every mask word.
        unsafe { self.stage.0.as_mut_ptr().cast::<u8>().add(self.fill) }
    }

    /// Write the whole lines of the stage to the destination and keep the
    /// remainder. Returns the number of bytes moved out of the stage.
    ///
    /// # Safety
    /// AVX2 must be available and the destination valid for the staged bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn flush_lines(&mut self) -> usize {
        let base = self.stage.0.as_mut_ptr().cast::<u8>();
        let lines = self.fill / 64;
        if lines == 0 {
            return 0;
        }
        // SAFETY: lines `0..lines` of the stage are initialised except for the
        // first `skip` bytes, which are not read. Every destination byte
        // written was staged by the caller and is therefore in bounds.
        unsafe {
            let mut first = 0;
            if self.skip != 0 {
                std::ptr::copy_nonoverlapping(
                    base.add(self.skip),
                    self.line.wrapping_add(self.skip),
                    64 - self.skip,
                );
                self.skip = 0;
                first = 1;
            }
            let mut i = first;
            while i < lines {
                let a = _mm256_load_si256(base.add(i * 64).cast());
                let b = _mm256_load_si256(base.add(i * 64 + 32).cast());
                _mm256_stream_si256(self.line.wrapping_add(i * 64).cast(), a);
                _mm256_stream_si256(self.line.wrapping_add(i * 64 + 32).cast(), b);
                i += 1;
            }
            let moved = lines * 64;
            let rest = self.fill - moved;
            std::ptr::copy_nonoverlapping(base.add(moved), base, rest);
            self.line = self.line.wrapping_add(moved);
            self.fill = rest;
            moved
        }
    }

    /// Flush everything, including the final partial line.
    ///
    /// # Safety
    /// As [`flush_lines`](Self::flush_lines).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn finish(&mut self) {
        // SAFETY: as in `flush_lines`; the tail copies only staged bytes.
        unsafe {
            self.flush_lines();
            let base = self.stage.0.as_mut_ptr().cast::<u8>();
            if self.fill > self.skip {
                std::ptr::copy_nonoverlapping(
                    base.add(self.skip),
                    self.line.wrapping_add(self.skip),
                    self.fill - self.skip,
                );
            }
        }
    }
}

macro_rules! compact_stream_kernel {
    ($(#[$doc:meta])* $name:ident, $elem:ty, lanes = $lanes:expr, steps = $steps:expr, bits = $bits:expr, table = $table:ident) => {
        $(#[$doc])*
        ///
        /// Writes exactly one element per selected row and nothing else, and
        /// at most `out_cap` elements.
        ///
        /// # Safety
        /// * `values` must be valid for `n` reads and `mask` must cover `n` bits.
        /// * `out` must be valid for `out_cap` writes, aligned for the element
        ///   type, and must not overlap `values`.
        /// * The CPU must support AVX2 and POPCNT.
        /// * Non-temporal stores are weakly ordered: the calling thread must
        ///   execute `SFENCE` (see `pipeline::stream_fence`) before the output
        ///   is read by anyone. Drivers do this once per worker.
        #[target_feature(enable = "avx2,popcnt")]
        pub unsafe fn $name(values: *const $elem, mask: &[u64], n: usize, out: *mut $elem, out_cap: usize) -> usize {
            const SIZE: usize = size_of::<$elem>();
            let full = n / 64;
            let n_words = n.div_ceil(64);
            let table = $table.0.as_ptr();
            // SAFETY: see the per-step comments.
            unsafe {
                let mut dst = StreamOut::new(out.cast());
                let mut count = 0usize;
                let mut wi = 0;
                while wi < n_words {
                    let mut w = *mask.get_unchecked(wi);
                    if wi >= full {
                        // The partial last word: ignore bits past `n`.
                        w &= (1u64 << (n % 64)) - 1;
                    }
                    if w != 0 {
                        let pc = w.count_ones() as usize;
                        if count + pc > out_cap {
                            // Contract violation (more matches than room):
                            // emit only what fits, one element at a time.
                            while w != 0 && count < out_cap {
                                let bit = w.trailing_zeros() as usize;
                                dst.cursor().cast::<$elem>().write_unaligned(values.add(wi * 64 + bit).read());
                                dst.fill += SIZE;
                                count += 1;
                                w &= w - 1;
                            }
                            break;
                        }
                        let base = values.add(wi * 64);
                        if wi >= full {
                            // Reading whole vectors here could run past `n`.
                            while w != 0 {
                                let bit = w.trailing_zeros() as usize;
                                dst.cursor().cast::<$elem>().write_unaligned(base.add(bit).read());
                                dst.fill += SIZE;
                                w &= w - 1;
                            }
                        } else if w == u64::MAX {
                            std::ptr::copy_nonoverlapping(base.cast::<u8>(), dst.cursor(), 64 * SIZE);
                            dst.fill += 64 * SIZE;
                        } else {
                            // The stage has room for a full word plus one
                            // vector of overhang beyond the flush threshold.
                            let mut m = w;
                            let mut k = 0;
                            while k < $steps {
                                let mb = (m & ((1u64 << $bits) - 1)) as usize;
                                let v = _mm256_loadu_si256(base.add(k * $lanes).cast());
                                let p = _mm256_load_si256(table.add(mb).cast());
                                _mm256_storeu_si256(dst.cursor().cast(), _mm256_permutevar8x32_epi32(v, p));
                                dst.fill += (mb as u32).count_ones() as usize * SIZE;
                                m >>= $bits;
                                k += 1;
                            }
                        }
                        count += pc;
                        if dst.fill >= STAGE_BYTES {
                            dst.flush_lines();
                        }
                    }
                    wi += 1;
                }
                dst.finish();
                count
            }
        }
    };
}

compact_stream_kernel!(
    /// Streaming compaction of 8-byte elements.
    compact8_stream, u64, lanes = 4, steps = 16, bits = 4, table = PERM64
);
compact_stream_kernel!(
    /// Streaming compaction of 4-byte elements.
    compact4_stream, u32, lanes = 8, steps = 8, bits = 8, table = PERM32
);

/// Copy `n` bytes with non-temporal stores (for outputs that will not be
/// read back soon and are larger than the cache).
///
/// # Safety
/// `src`/`dst` must be valid for `n` bytes and must not overlap; AVX2 required.
#[target_feature(enable = "avx2")]
pub unsafe fn copy_stream(src: *const u8, dst: *mut u8, n: usize) {
    // SAFETY: all accesses stay within the `n` bytes the caller vouched for.
    unsafe {
        let head = (64 - dst as usize % 64) % 64;
        if n < head + 64 {
            std::ptr::copy_nonoverlapping(src, dst, n);
            return;
        }
        std::ptr::copy_nonoverlapping(src, dst, head);
        let lines = (n - head) / 64;
        let (mut s, mut d) = (src.add(head), dst.add(head));
        for _ in 0..lines {
            let a = _mm256_loadu_si256(s.cast());
            let b = _mm256_loadu_si256(s.add(32).cast());
            _mm256_stream_si256(d.cast(), a);
            _mm256_stream_si256(d.add(32).cast(), b);
            s = s.add(64);
            d = d.add(64);
        }
        let done = head + lines * 64;
        std::ptr::copy_nonoverlapping(src.add(done), dst.add(done), n - done);
        _mm_sfence();
    }
}

// ---------------------------------------------------------------------------
// float min / max
// ---------------------------------------------------------------------------

macro_rules! min_max_float_kernel {
    ($(#[$doc:meta])* $name:ident, $t:ty, $vec:ty, lanes = $lanes:expr,
     load = $load:ident, set1 = $set1:ident, min = $min:ident, max = $max:ident, store = $store:ident) => {
        $(#[$doc])*
        ///
        /// NaN is ignored. Returns `(+inf, -inf)` if there is no non-NaN value.
        ///
        /// # Safety
        /// The CPU must support AVX2.
        #[target_feature(enable = "avx2")]
        pub unsafe fn $name(values: &[$t]) -> ($t, $t) {
            const L: usize = $lanes;
            let mut lo = [$set1(<$t>::INFINITY); 4];
            let mut hi = [$set1(<$t>::NEG_INFINITY); 4];
            let mut chunks = values.chunks_exact(4 * L);
            for c in &mut chunks {
                let p = c.as_ptr();
                let mut k = 0;
                while k < 4 {
                    // SAFETY: `k * L + L <= 4 * L` elements of the chunk.
                    let v = unsafe { $load(p.add(k * L)) };
                    // `min(v, acc)` yields `acc` when `v` is NaN, so NaN never
                    // enters an accumulator.
                    lo[k] = $min(v, lo[k]);
                    hi[k] = $max(v, hi[k]);
                    k += 1;
                }
            }
            let lo_v = $min($min(lo[0], lo[1]), $min(lo[2], lo[3]));
            let hi_v = $max($max(hi[0], hi[1]), $max(hi[2], hi[3]));
            let (mut lo_a, mut hi_a) = ([0.0 as $t; L], [0.0 as $t; L]);
            // SAFETY: the arrays hold exactly one vector each.
            unsafe {
                $store(lo_a.as_mut_ptr(), lo_v);
                $store(hi_a.as_mut_ptr(), hi_v);
            }
            let (mut min, mut max) = (<$t>::INFINITY, <$t>::NEG_INFINITY);
            for i in 0..L {
                min = if lo_a[i] < min { lo_a[i] } else { min };
                max = if hi_a[i] > max { hi_a[i] } else { max };
            }
            for &v in chunks.remainder() {
                min = if v < min { v } else { min };
                max = if v > max { v } else { max };
            }
            (min, max)
        }
    };
}

min_max_float_kernel!(
    /// Smallest and largest `f64`.
    min_max_f64, f64, __m256d, lanes = 4,
    load = _mm256_loadu_pd, set1 = _mm256_set1_pd, min = _mm256_min_pd, max = _mm256_max_pd, store = _mm256_storeu_pd
);
min_max_float_kernel!(
    /// Smallest and largest `f32`.
    min_max_f32, f32, __m256, lanes = 8,
    load = _mm256_loadu_ps, set1 = _mm256_set1_ps, min = _mm256_min_ps, max = _mm256_max_ps, store = _mm256_storeu_ps
);

macro_rules! min_or_max_float_kernel {
    ($(#[$doc:meta])* $name:ident, $t:ty, lanes = $lanes:expr, init = $init:expr,
     load = $load:ident, set1 = $set1:ident, op = $op:ident, store = $store:ident, better = $better:tt) => {
        $(#[$doc])*
        ///
        /// NaN is ignored. Returns the identity (`+inf` for a minimum, `-inf`
        /// for a maximum) if there is no non-NaN value.
        ///
        /// # Safety
        /// The CPU must support AVX2.
        #[target_feature(enable = "avx2")]
        pub unsafe fn $name(values: &[$t]) -> $t {
            const L: usize = $lanes;
            // Eight accumulators: the operation has a latency of four cycles
            // and two ports, so eight independent chains keep both busy.
            let mut acc = [$set1($init); 8];
            let mut chunks = values.chunks_exact(8 * L);
            for c in &mut chunks {
                let p = c.as_ptr();
                let mut k = 0;
                while k < 8 {
                    // SAFETY: `k * L + L <= 8 * L` elements of the chunk.
                    // The operand order makes a NaN in `values` yield `acc`.
                    acc[k] = $op(unsafe { $load(p.add(k * L)) }, acc[k]);
                    k += 1;
                }
            }
            let folded = $op($op($op(acc[0], acc[1]), $op(acc[2], acc[3])), $op($op(acc[4], acc[5]), $op(acc[6], acc[7])));
            let mut lanes = [$init; L];
            // SAFETY: the array holds exactly one vector.
            unsafe { $store(lanes.as_mut_ptr(), folded) };
            let mut best: $t = $init;
            for &v in lanes.iter().chain(chunks.remainder()) {
                best = if v $better best { v } else { best };
            }
            best
        }
    };
}

min_or_max_float_kernel!(
    /// Smallest `f64`.
    min_f64, f64, lanes = 4, init = f64::INFINITY,
    load = _mm256_loadu_pd, set1 = _mm256_set1_pd, op = _mm256_min_pd, store = _mm256_storeu_pd, better = <
);
min_or_max_float_kernel!(
    /// Largest `f64`.
    max_f64, f64, lanes = 4, init = f64::NEG_INFINITY,
    load = _mm256_loadu_pd, set1 = _mm256_set1_pd, op = _mm256_max_pd, store = _mm256_storeu_pd, better = >
);
min_or_max_float_kernel!(
    /// Smallest `f32`.
    min_f32, f32, lanes = 8, init = f32::INFINITY,
    load = _mm256_loadu_ps, set1 = _mm256_set1_ps, op = _mm256_min_ps, store = _mm256_storeu_ps, better = <
);
min_or_max_float_kernel!(
    /// Largest `f32`.
    max_f32, f32, lanes = 8, init = f32::NEG_INFINITY,
    load = _mm256_loadu_ps, set1 = _mm256_set1_ps, op = _mm256_max_ps, store = _mm256_storeu_ps, better = >
);

// ---------------------------------------------------------------------------
// mask -> row indices
// ---------------------------------------------------------------------------

/// Write the positions of the set bits among the first `n` bits of `mask`
/// (plus `base`) to `out`, ascending. Returns how many were written.
///
/// One table look-up per eight rows whatever the element type that is later
/// gathered, which is what makes "indices, then gather" cheaper than
/// compaction for sparse selections and for wide elements.
///
/// # Safety
/// * `out` must be valid for `count + 8` writes, where `count` is the number
///   of set bits among the first `n` (whole vectors are stored).
/// * The CPU must support AVX2 and POPCNT.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn mask_indices(mask: &[u64], n: usize, base: u32, out: *mut u32) -> usize {
    let table = PERM32.0.as_ptr();
    let n_words = n.div_ceil(64);
    let mut o = 0usize;
    // SAFETY: every store begins at `o <= count` and writes 8 lanes, within
    // the `count + 8` slots the caller provides; table entries are 32 bytes.
    unsafe {
        for wi in 0..n_words {
            let mut w = *mask.get_unchecked(wi);
            if wi + 1 == n_words && n % 64 != 0 {
                w &= (1u64 << (n % 64)) - 1;
            }
            if w == 0 {
                continue;
            }
            let row = base.wrapping_add((wi * 64) as u32);
            if w.count_ones() <= 2 {
                // Sparse word: two cheap extractions beat eight table steps.
                while w != 0 {
                    out.add(o).write(row + w.trailing_zeros());
                    o += 1;
                    w &= w - 1;
                }
                continue;
            }
            let mut k = 0u32;
            while k < 8 {
                let mb = (w >> (8 * k) & 0xFF) as usize;
                let lanes = _mm256_load_si256(table.add(mb).cast());
                let rows = _mm256_add_epi32(lanes, _mm256_set1_epi32((row + 8 * k) as i32));
                _mm256_storeu_si256(out.add(o).cast(), rows);
                o += (mb as u32).count_ones() as usize;
                k += 1;
            }
        }
    }
    o
}
