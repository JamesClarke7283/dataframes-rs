//! Portable kernels.
//!
//! These are the reference implementations: every vectorised kernel is tested
//! against them, they finish the last partial word after a vector loop, and
//! they are what runs on targets without a hand-written path. They are written
//! so that LLVM can still vectorise them for whatever the compile-time target
//! offers.

use super::{CmpOp, Kernel};
use crate::memory::{tail_mask, words_for};

/// Evaluate `f` on eight values and pack the results into a byte.
#[inline(always)]
fn pack8<T: Copy>(c: &[T], f: &impl Fn(T) -> bool) -> u64 {
    let mut m = 0u8;
    let mut i = 0;
    while i < 8 {
        m |= (f(c[i]) as u8) << i;
        i += 1;
    }
    m as u64
}

/// `out` bit `i` = `f(values[i])`. Returns the number of set bits.
#[inline(always)]
pub(crate) fn pack_words<T: Copy>(values: &[T], out: &mut [u64], f: impl Fn(T) -> bool) -> usize {
    let n_words = words_for(values.len());
    let out = &mut out[..n_words];
    let mut count = 0usize;
    let mut chunks = values.chunks_exact(64);
    let mut wi = 0;
    for chunk in &mut chunks {
        let mut w = 0u64;
        let mut b = 0;
        while b < 8 {
            w |= pack8(&chunk[b * 8..b * 8 + 8], &f) << (b * 8);
            b += 1;
        }
        out[wi] = w;
        count += w.count_ones() as usize;
        wi += 1;
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut w = 0u64;
        for (i, &v) in rem.iter().enumerate() {
            w |= (f(v) as u64) << i;
        }
        out[wi] = w;
        count += w.count_ones() as usize;
    }
    count
}

/// `out` bit `i` = `f(lhs[i], rhs[i])`. Returns the number of set bits.
#[inline(always)]
pub(crate) fn pack_words2<T: Copy>(
    lhs: &[T],
    rhs: &[T],
    out: &mut [u64],
    f: impl Fn(T, T) -> bool,
) -> usize {
    assert_eq!(lhs.len(), rhs.len());
    let n_words = words_for(lhs.len());
    let out = &mut out[..n_words];
    let mut count = 0usize;
    for (wi, (a, b)) in lhs.chunks(64).zip(rhs.chunks(64)).enumerate() {
        let mut w = 0u64;
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            w |= (f(x, y) as u64) << i;
        }
        out[wi] = w;
        count += w.count_ones() as usize;
    }
    count
}

/// Compare every value against a scalar. See [`Kernel::cmp_scalar`].
pub fn cmp_scalar<T: Kernel>(values: &[T], op: CmpOp, rhs: T, out: &mut [u64]) -> usize {
    match op {
        CmpOp::Eq => pack_words(values, out, |v| v.tot_eq(rhs)),
        CmpOp::Ne => pack_words(values, out, |v| !v.tot_eq(rhs)),
        CmpOp::Lt => pack_words(values, out, |v| v.tot_lt(rhs)),
        CmpOp::Le => pack_words(values, out, |v| !rhs.tot_lt(v)),
        CmpOp::Gt => pack_words(values, out, |v| rhs.tot_lt(v)),
        CmpOp::Ge => pack_words(values, out, |v| !v.tot_lt(rhs)),
    }
}

/// Compare two slices element-wise. See [`Kernel::cmp_values`].
pub fn cmp_values<T: Kernel>(lhs: &[T], rhs: &[T], op: CmpOp, out: &mut [u64]) -> usize {
    match op {
        CmpOp::Eq => pack_words2(lhs, rhs, out, |a, b| a.tot_eq(b)),
        CmpOp::Ne => pack_words2(lhs, rhs, out, |a, b| !a.tot_eq(b)),
        CmpOp::Lt => pack_words2(lhs, rhs, out, |a, b| a.tot_lt(b)),
        CmpOp::Le => pack_words2(lhs, rhs, out, |a, b| !b.tot_lt(a)),
        CmpOp::Gt => pack_words2(lhs, rhs, out, |a, b| b.tot_lt(a)),
        CmpOp::Ge => pack_words2(lhs, rhs, out, |a, b| !a.tot_lt(b)),
    }
}

/// Copy the selected elements of words `first_word..` to `out`, writing
/// exactly one element per set bit and nothing else. Stops after `cap` elements.
///
/// # Safety
/// * `values` is valid for `n` reads and `mask` covers `n` bits.
/// * `out` is valid for `cap` writes.
#[inline]
pub(crate) unsafe fn compact_exact<T: Copy>(
    values: *const T,
    mask: &[u64],
    first_word: usize,
    n: usize,
    out: *mut T,
    cap: usize,
) -> usize {
    let n_words = words_for(n);
    let mut o = 0usize;
    let mut wi = first_word;
    while wi < n_words {
        // SAFETY: `wi < words_for(n) <= mask.len()`.
        let mut w = unsafe { *mask.get_unchecked(wi) };
        if wi + 1 == n_words {
            w &= tail_mask(n);
        }
        while w != 0 {
            if o == cap {
                return o;
            }
            let bit = w.trailing_zeros() as usize;
            // SAFETY: `wi * 64 + bit < n` because bits past `n` were masked
            // off, and `o < cap`.
            unsafe { out.add(o).write(values.add(wi * 64 + bit).read()) };
            o += 1;
            w &= w - 1;
        }
        wi += 1;
    }
    o
}

/// Copy the elements of `values[..n]` whose mask bit is set to `out`.
/// Returns the number of elements written, which is the number of set bits
/// unless that exceeds `out_cap`.
///
/// # Safety
/// * `values` is valid for `n` reads and `mask` covers `n` bits.
/// * `out` is valid for `out_cap` writes; any of those slots may be clobbered.
pub unsafe fn compact<T: Copy>(
    values: *const T,
    mask: &[u64],
    n: usize,
    out: *mut T,
    out_cap: usize,
) -> usize {
    let full = n / 64;
    let mut o = 0usize;
    let mut wi = 0;
    while wi < full {
        // SAFETY: `wi < n / 64 <= mask.len()`.
        let mut w = unsafe { *mask.get_unchecked(wi) };
        if w == 0 {
            wi += 1;
            continue;
        }
        let pc = w.count_ones() as usize;
        if o + pc > out_cap {
            break;
        }
        // SAFETY: the 64 elements of word `wi` are in bounds because the word is full.
        let base = unsafe { values.add(wi * 64) };
        if w == u64::MAX {
            // SAFETY: 64 in-bounds reads; `o + 64 <= out_cap` was checked above.
            unsafe { std::ptr::copy_nonoverlapping(base, out.add(o), 64) };
            o += 64;
        } else if pc <= 16 || o + pc == out_cap {
            // Sparse word, or no slack for the unconditional stores below.
            while w != 0 {
                // SAFETY: bit index < 64 within a full word; `o + pc <= out_cap`.
                unsafe { out.add(o).write(base.add(w.trailing_zeros() as usize).read()) };
                o += 1;
                w &= w - 1;
            }
        } else {
            // Dense word: store unconditionally, advance by the mask bit.
            // The furthest store lands at `o_start + pc`, which is `< out_cap`.
            let mut i = 0;
            while i < 64 {
                // SAFETY: see the bound above; reads are within the full word.
                unsafe { out.add(o).write(base.add(i).read()) };
                o += (w >> i & 1) as usize;
                i += 1;
            }
        }
        wi += 1;
    }
    // SAFETY: forwarded; the remaining words are handled one element at a time.
    o + unsafe { compact_exact(values, mask, wi, n, out.add(o), out_cap - o) }
}

/// Bit-compaction: keep the bits of `values` whose `mask` bit is set, packed
/// from bit 0 upward, appending them to `out_words` at bit position `out_len`.
/// Returns the new bit length.
///
/// `out_words` must already be long enough and zero past `out_len`.
pub fn compact_bits(values: &[u64], mask: &[u64], n: usize, out_words: &mut [u64], mut out_len: usize) -> usize {
    let n_words = words_for(n);
    for wi in 0..n_words {
        let mut m = mask[wi];
        if wi + 1 == n_words {
            m &= tail_mask(n);
        }
        if m == 0 {
            continue;
        }
        let v = values[wi];
        let (packed, pc) = if m == u64::MAX { (v, 64) } else { (pext(v, m), m.count_ones() as usize) };
        let (word, bit) = (out_len / 64, out_len % 64);
        out_words[word] |= packed << bit;
        if bit != 0 && bit + pc > 64 {
            out_words[word + 1] |= packed >> (64 - bit);
        }
        out_len += pc;
    }
    out_len
}

/// Portable parallel bit extract (the `pext` instruction).
#[inline]
pub fn pext(value: u64, mask: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // SAFETY: the `bmi2` target feature is enabled at compile time.
        return unsafe { std::arch::x86_64::_pext_u64(value, mask) };
    }
    #[allow(unreachable_code)]
    {
        let mut mask = mask;
        let mut out = 0u64;
        let mut k = 0;
        while mask != 0 {
            let bit = mask.trailing_zeros();
            out |= (value >> bit & 1) << k;
            k += 1;
            mask &= mask - 1;
        }
        out
    }
}
