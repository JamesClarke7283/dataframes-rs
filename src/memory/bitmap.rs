//! [`Bitmap`] and [`MutableBitmap`]: one bit per row.
//!
//! Bitmaps serve as boolean column values, as validity (null) masks and as
//! filter predicates. Bit `i` lives in word `i / 64` at position `i % 64`
//! (least significant bit first), which on little-endian targets is the Arrow
//! layout viewed eight bytes at a time. The 64-bit word is the unit every
//! kernel consumes: one word selects 64 rows, tests for "all" or "none" with a
//! single compare, and counts with a single `popcnt`.
//!
//! Like [`Buffer`], a `Bitmap` slices in O(1) by sharing its words. A slice
//! that does not start on a word boundary keeps a bit offset below 64; kernels
//! ask for [`Bitmap::words`], which borrows when the bits are already aligned
//! (always the case for bitmaps this crate computes) and realigns into a small
//! temporary otherwise.

use std::borrow::Cow;
use std::fmt;

use super::alloc::{AlignedVec, ColumnAlloc};
use super::buffer::{Buffer, zeroed_vec};

const UNKNOWN: usize = usize::MAX;

/// Number of 64-bit words needed for `bits` bits.
#[inline]
pub const fn words_for(bits: usize) -> usize {
    bits.div_ceil(64)
}

/// Mask selecting the low `bits % 64` bits, or everything if that is zero.
#[inline]
pub const fn tail_mask(bits: usize) -> u64 {
    match bits % 64 {
        0 => u64::MAX,
        r => (1u64 << r) - 1,
    }
}

/// Count the set bits in `len` bits starting `offset` bits into `words`.
pub fn count_ones_range(words: &[u64], offset: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let first = offset / 64;
    let last = (offset + len - 1) / 64;
    let head = u64::MAX << (offset % 64);
    let tail = tail_mask(offset + len);
    if first == last {
        return (words[first] & head & tail).count_ones() as usize;
    }
    let mut n = (words[first] & head).count_ones() as usize;
    n += words[first + 1..last].iter().map(|w| w.count_ones() as usize).sum::<usize>();
    n + (words[last] & tail).count_ones() as usize
}

/// An immutable, shareable run of bits. See the [module documentation](self).
pub struct Bitmap {
    words: Buffer<u64>,
    /// Bit offset of the first bit inside `words[0]`; always below 64.
    offset: usize,
    len: usize,
    /// Number of set bits, or `UNKNOWN` (only for a large partial slice,
    /// which counts when asked). A plain field on purpose: with no interior
    /// mutability anywhere in a column, the compiler may treat a shared
    /// reference to one as read-only and keep its pointer and length in
    /// registers across calls it cannot see into.
    ones: usize,
}

/// Slices of at most this many bits count their set bits right away (at most
/// 64 `popcnt`s); longer ones leave the count to the first caller that wants it.
const EAGER_COUNT_BITS: usize = 4096;

impl Bitmap {
    /// An empty bitmap.
    #[inline]
    pub const fn new() -> Self {
        Bitmap { words: Buffer::new(), offset: 0, len: 0, ones: 0 }
    }

    /// `len` bits, all clear.
    pub fn new_zeroed(len: usize) -> Self {
        Bitmap {
            words: Buffer::from_aligned(zeroed_vec(words_for(len))),
            offset: 0,
            len,
            ones: 0,
        }
    }

    /// `len` bits, all set.
    pub fn new_ones(len: usize) -> Self {
        let mut words: AlignedVec<u64> = Vec::with_capacity_in(words_for(len), ColumnAlloc);
        words.resize(words_for(len), u64::MAX);
        if let Some(last) = words.last_mut() {
            *last &= tail_mask(len);
        }
        Bitmap { words: words.into(), offset: 0, len, ones: len }
    }

    /// `len` copies of `value`.
    #[inline]
    pub fn new_constant(value: bool, len: usize) -> Self {
        if value { Self::new_ones(len) } else { Self::new_zeroed(len) }
    }

    /// Wrap aligned words. Bits past `len` in the last word are cleared.
    ///
    /// # Panics
    /// If `words` is too short for `len` bits.
    pub fn from_words(mut words: AlignedVec<u64>, len: usize) -> Self {
        let n = words_for(len);
        assert!(words.len() >= n, "{} words cannot hold {len} bits", words.len());
        words.truncate(n);
        if let Some(last) = words.last_mut() {
            *last &= tail_mask(len);
        }
        let ones = words.iter().map(|w| w.count_ones() as usize).sum();
        Bitmap { words: words.into(), offset: 0, len, ones }
    }

    /// Like [`from_words`](Self::from_words) with the set-bit count already
    /// known, which saves counting.
    ///
    /// The count is trusted; passing a wrong one makes later answers wrong
    /// (never unsafe).
    pub fn from_words_with_count(mut words: AlignedVec<u64>, len: usize, ones: usize) -> Self {
        let n = words_for(len);
        assert!(words.len() >= n, "{} words cannot hold {len} bits", words.len());
        words.truncate(n);
        if let Some(last) = words.last_mut() {
            *last &= tail_mask(len);
        }
        debug_assert_eq!(ones, words.iter().map(|w| w.count_ones() as usize).sum::<usize>());
        Bitmap { words: words.into(), offset: 0, len, ones }
    }

    /// Pack a slice of `bool`.
    pub fn from_bools(values: &[bool]) -> Self {
        let mut words: AlignedVec<u64> = Vec::with_capacity_in(words_for(values.len()), ColumnAlloc);
        let mut ones = 0usize;
        for chunk in values.chunks(64) {
            let mut w = 0u64;
            for (i, &b) in chunk.iter().enumerate() {
                w |= (b as u64) << i;
            }
            ones += w.count_ones() as usize;
            words.push(w);
        }
        Bitmap { words: words.into(), offset: 0, len: values.len(), ones }
    }

    /// Number of bits.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` if there are no bits.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bit `i`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.len, "bit index {i} out of bounds for bitmap of length {}", self.len);
        // SAFETY: bounds checked above.
        unsafe { self.get_unchecked(i) }
    }

    /// Bit `i` without a bounds check.
    ///
    /// # Safety
    /// `i < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> bool {
        let j = self.offset + i;
        // SAFETY: `words` covers `offset + len` bits by construction.
        (unsafe { *self.words.get_unchecked(j >> 6) } >> (j & 63)) & 1 != 0
    }

    /// Number of set bits.
    ///
    /// Known in advance for every bitmap except a partial slice longer than
    /// 4096 bits, for which this counts (one `popcnt` per 64 bits) on each call.
    #[inline]
    pub fn count_ones(&self) -> usize {
        if self.ones != UNKNOWN {
            return self.ones;
        }
        count_ones_range(&self.words, self.offset, self.len)
    }

    /// `true` if [`count_ones`](Self::count_ones) answers without counting.
    #[inline]
    pub fn count_is_known(&self) -> bool {
        self.ones != UNKNOWN
    }

    /// Number of clear bits.
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Polars/Arrow name for [`count_zeros`](Self::count_zeros).
    #[inline]
    pub fn unset_bits(&self) -> usize {
        self.count_zeros()
    }

    /// `true` if every bit is set (vacuously true when empty).
    #[inline]
    pub fn all(&self) -> bool {
        self.count_ones() == self.len
    }

    /// `true` if at least one bit is set.
    #[inline]
    pub fn any(&self) -> bool {
        self.count_ones() != 0
    }

    /// A view of `len` bits starting at `offset`. O(1), no copy.
    ///
    /// # Panics
    /// If `offset + len` exceeds the length.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "slice {offset}..{offset}+{len} out of bounds for bitmap of length {}",
            self.len
        );
        let start = self.offset + offset;
        let first = start / 64;
        let n_words = words_for(start % 64 + len);
        let ones = if len == self.len {
            self.ones
        } else if len == 0 {
            0
        } else {
            match self.ones {
                0 => 0,
                n if n == self.len => len,
                _ if len <= EAGER_COUNT_BITS => count_ones_range(&self.words, start, len),
                _ => UNKNOWN,
            }
        };
        Bitmap { words: self.words.slice(first, n_words), offset: start % 64, len, ones }
    }

    /// The bits as aligned words: bit `i` is bit `i % 64` of word `i / 64`, and
    /// bits past `len` in the last word are zero.
    ///
    /// Borrows when the bitmap already has that shape; otherwise builds a
    /// realigned copy (one word per 64 rows).
    pub fn words(&self) -> Cow<'_, [u64]> {
        let n = words_for(self.len);
        if self.offset == 0 {
            let w = &self.words[..n];
            let clean = match w.last() {
                Some(&last) => last & !tail_mask(self.len) == 0,
                None => true,
            };
            if clean {
                return Cow::Borrowed(w);
            }
            let mut owned = w.to_vec();
            *owned.last_mut().expect("non-empty") &= tail_mask(self.len);
            return Cow::Owned(owned);
        }
        let src = &self.words[..];
        let (lo, hi) = (self.offset, 64 - self.offset);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = src[i] >> lo;
            let b = src.get(i + 1).map_or(0, |&w| w << hi);
            out.push(a | b);
        }
        if let Some(last) = out.last_mut() {
            *last &= tail_mask(self.len);
        }
        Cow::Owned(out)
    }

    /// Iterate over the bits.
    #[inline]
    pub fn iter(&self) -> BitmapIter<'_> {
        BitmapIter { bitmap: self, pos: 0, end: self.len }
    }

    /// Iterate over the positions of the set bits, ascending.
    pub fn true_indices(&self) -> TrueIdxIter<'_> {
        let words = self.words();
        let current = words.first().copied().unwrap_or(0);
        TrueIdxIter { words, word: 0, current }
    }

    /// Unpack into a `Vec<bool>`.
    pub fn to_bools(&self) -> Vec<bool> {
        self.iter().collect()
    }

    fn zip_with(&self, other: &Bitmap, f: impl Fn(u64, u64) -> u64) -> Bitmap {
        assert_eq!(self.len, other.len, "bitmaps must have equal length");
        let (a, b) = (self.words(), other.words());
        let mut out: AlignedVec<u64> = Vec::with_capacity_in(a.len(), ColumnAlloc);
        out.extend(a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)));
        Bitmap::from_words(out, self.len)
    }

    /// Bitwise AND. Panics if the lengths differ.
    pub fn and(&self, other: &Bitmap) -> Bitmap {
        self.zip_with(other, |a, b| a & b)
    }

    /// Bitwise OR. Panics if the lengths differ.
    pub fn or(&self, other: &Bitmap) -> Bitmap {
        self.zip_with(other, |a, b| a | b)
    }

    /// Bitwise XOR. Panics if the lengths differ.
    pub fn xor(&self, other: &Bitmap) -> Bitmap {
        self.zip_with(other, |a, b| a ^ b)
    }

    /// `self & !other`. Panics if the lengths differ.
    pub fn and_not(&self, other: &Bitmap) -> Bitmap {
        self.zip_with(other, |a, b| a & !b)
    }

    /// Bitwise NOT.
    pub fn not(&self) -> Bitmap {
        let a = self.words();
        let mut out: AlignedVec<u64> = Vec::with_capacity_in(a.len(), ColumnAlloc);
        out.extend(a.iter().map(|&x| !x));
        // `from_words` clears the tail bits and counts.
        Bitmap::from_words(out, self.len)
    }

    /// Turn into a mutable bitmap, reusing the storage when it is not shared.
    pub fn into_mut(self) -> MutableBitmap {
        if self.offset == 0 {
            let len = self.len;
            let mut words = self.words.sliced(0, words_for(len)).into_aligned();
            if let Some(last) = words.last_mut() {
                *last &= tail_mask(len);
            }
            return MutableBitmap { words, len };
        }
        let w = self.words();
        let mut words: AlignedVec<u64> = Vec::with_capacity_in(w.len(), ColumnAlloc);
        words.extend_from_slice(&w);
        MutableBitmap { words, len: self.len }
    }
}

impl Clone for Bitmap {
    #[inline]
    fn clone(&self) -> Self {
        Bitmap { words: self.words.clone(), offset: self.offset, len: self.len, ones: self.ones }
    }
}

impl Default for Bitmap {
    fn default() -> Self {
        Bitmap::new()
    }
}

impl PartialEq for Bitmap {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.words() == other.words()
    }
}

impl fmt::Debug for Bitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Bitmap[")?;
        for (i, b) in self.iter().enumerate() {
            if i == 64 {
                f.write_str("…")?;
                break;
            }
            f.write_str(if b { "1" } else { "0" })?;
        }
        write!(f, "; len={}]", self.len)
    }
}

impl FromIterator<bool> for Bitmap {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        iter.into_iter().collect::<MutableBitmap>().freeze()
    }
}

impl From<MutableBitmap> for Bitmap {
    fn from(m: MutableBitmap) -> Self {
        m.freeze()
    }
}

/// Iterator over the bits of a [`Bitmap`].
#[derive(Clone)]
pub struct BitmapIter<'a> {
    bitmap: &'a Bitmap,
    pos: usize,
    end: usize,
}

impl Iterator for BitmapIter<'_> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        if self.pos == self.end {
            return None;
        }
        // SAFETY: `pos < end <= len`.
        let b = unsafe { self.bitmap.get_unchecked(self.pos) };
        self.pos += 1;
        Some(b)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.pos;
        (n, Some(n))
    }
}

impl DoubleEndedIterator for BitmapIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        if self.pos == self.end {
            return None;
        }
        self.end -= 1;
        // SAFETY: `end < len`.
        Some(unsafe { self.bitmap.get_unchecked(self.end) })
    }
}

impl ExactSizeIterator for BitmapIter<'_> {}

/// Iterator over the positions of set bits. See [`Bitmap::true_indices`].
pub struct TrueIdxIter<'a> {
    words: Cow<'a, [u64]>,
    word: usize,
    current: u64,
}

impl Iterator for TrueIdxIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.current == 0 {
            self.word += 1;
            self.current = *self.words.get(self.word)?;
        }
        let bit = self.current.trailing_zeros() as usize;
        self.current &= self.current - 1;
        Some(self.word * 64 + bit)
    }
}

/// A growable bitmap. Bits past `len` in the last word are always zero.
#[derive(Clone)]
pub struct MutableBitmap {
    words: AlignedVec<u64>,
    len: usize,
}

impl MutableBitmap {
    /// An empty bitmap.
    #[inline]
    pub fn new() -> Self {
        MutableBitmap { words: Vec::new_in(ColumnAlloc), len: 0 }
    }

    /// An empty bitmap with room for `bits` bits.
    #[inline]
    pub fn with_capacity(bits: usize) -> Self {
        MutableBitmap { words: Vec::with_capacity_in(words_for(bits), ColumnAlloc), len: 0 }
    }

    /// `len` clear bits.
    pub fn from_len_zeroed(len: usize) -> Self {
        MutableBitmap { words: zeroed_vec(words_for(len)), len }
    }

    /// `len` set bits.
    pub fn from_len_set(len: usize) -> Self {
        let mut m = MutableBitmap::with_capacity(len);
        m.extend_constant(len, true);
        m
    }

    /// Number of bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if there are no bits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one bit.
    #[inline]
    pub fn push(&mut self, value: bool) {
        let bit = self.len % 64;
        if bit == 0 {
            self.words.push(value as u64);
        } else {
            // SAFETY: a partially filled last word exists whenever `len % 64 != 0`.
            *unsafe { self.words.last_mut().unwrap_unchecked() } |= (value as u64) << bit;
        }
        self.len += 1;
    }

    /// Append `n` copies of `value`.
    pub fn extend_constant(&mut self, n: usize, value: bool) {
        if n == 0 {
            return;
        }
        let new_len = self.len + n;
        let bit = self.len % 64;
        if value && bit != 0 {
            let fill = u64::MAX << bit;
            *self.words.last_mut().expect("partial word exists") |= fill;
        }
        self.words.resize(words_for(new_len), if value { u64::MAX } else { 0 });
        if value {
            *self.words.last_mut().expect("n > 0") &= tail_mask(new_len);
        }
        self.len = new_len;
    }

    /// Append `bits` bits taken from aligned `words` (bit `i` in word `i / 64`).
    pub fn extend_from_words(&mut self, words: &[u64], bits: usize) {
        if bits == 0 {
            return;
        }
        let n = words_for(bits);
        let shift = self.len % 64;
        let new_len = self.len + bits;
        if shift == 0 {
            self.words.extend_from_slice(&words[..n]);
        } else {
            for &w in &words[..n] {
                *self.words.last_mut().expect("partial word exists") |= w << shift;
                self.words.push(w >> (64 - shift));
            }
        }
        self.words.truncate(words_for(new_len));
        *self.words.last_mut().expect("bits > 0") &= tail_mask(new_len);
        self.len = new_len;
    }

    /// Append all bits of `other`.
    pub fn extend_from_bitmap(&mut self, other: &Bitmap) {
        self.extend_from_words(&other.words(), other.len());
    }

    /// Bit `i`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.len, "bit index {i} out of bounds for bitmap of length {}", self.len);
        (self.words[i >> 6] >> (i & 63)) & 1 != 0
    }

    /// Overwrite bit `i`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[inline]
    pub fn set(&mut self, i: usize, value: bool) {
        assert!(i < self.len, "bit index {i} out of bounds for bitmap of length {}", self.len);
        let w = &mut self.words[i >> 6];
        *w = (*w & !(1u64 << (i & 63))) | ((value as u64) << (i & 63));
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// The aligned words backing the bitmap.
    #[inline]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Mutable access to the words.
    ///
    /// Callers must leave the bits past `len` in the last word clear; setting
    /// them makes later answers wrong (never unsafe).
    #[inline]
    pub fn as_words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    /// Freeze into an immutable [`Bitmap`] without copying. Counts the set
    /// bits once (one `popcnt` per 64 bits).
    #[inline]
    pub fn freeze(self) -> Bitmap {
        let ones = self.count_ones();
        Bitmap { words: Buffer::from_aligned(self.words), offset: 0, len: self.len, ones }
    }
}

impl Default for MutableBitmap {
    fn default() -> Self {
        MutableBitmap::new()
    }
}

impl FromIterator<bool> for MutableBitmap {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut m = MutableBitmap::with_capacity(iter.size_hint().0);
        let (mut word, mut bit) = (0u64, 0u32);
        for b in iter {
            word |= (b as u64) << bit;
            bit += 1;
            if bit == 64 {
                m.words.push(word);
                m.len += 64;
                (word, bit) = (0, 0);
            }
        }
        if bit != 0 {
            m.words.push(word);
            m.len += bit as usize;
        }
        m
    }
}

impl Extend<bool> for MutableBitmap {
    fn extend<I: IntoIterator<Item = bool>>(&mut self, iter: I) {
        for b in iter {
            self.push(b);
        }
    }
}

impl fmt::Debug for MutableBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MutableBitmap[len={}]", self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(n: usize) -> Vec<bool> {
        (0..n).map(|i| (i * 7 + i / 3) % 5 < 2).collect()
    }

    #[test]
    fn round_trips_and_counts() {
        for n in [0usize, 1, 63, 64, 65, 127, 128, 129, 1000] {
            let bools = pattern(n);
            let bm = Bitmap::from_bools(&bools);
            assert_eq!(bm.len(), n);
            assert_eq!(bm.to_bools(), bools);
            let ones = bools.iter().filter(|&&b| b).count();
            assert_eq!(bm.count_ones(), ones);
            assert_eq!(bm.count_zeros(), n - ones);
            let collected: Bitmap = bools.iter().copied().collect();
            assert_eq!(collected, bm);
            let idx: Vec<usize> = bm.true_indices().collect();
            let want: Vec<usize> = (0..n).filter(|&i| bools[i]).collect();
            assert_eq!(idx, want);
        }
    }

    #[test]
    fn slices_at_every_offset_agree_with_bools() {
        let bools = pattern(300);
        let bm = Bitmap::from_bools(&bools);
        for off in [0usize, 1, 5, 63, 64, 65, 100, 191, 192, 250] {
            for len in [0usize, 1, 7, 50, 64, 65] {
                if off + len > 300 {
                    continue;
                }
                let s = bm.slice(off, len);
                assert_eq!(s.to_bools(), &bools[off..off + len], "off {off} len {len}");
                assert_eq!(s.count_ones(), bools[off..off + len].iter().filter(|&&b| b).count());
                let w = s.words();
                assert_eq!(w.len(), words_for(len));
                if let Some(&last) = w.last() {
                    assert_eq!(last & !tail_mask(len), 0, "tail bits must be clear");
                }
                // A slice of a slice composes.
                if len >= 10 {
                    assert_eq!(s.slice(3, 5).to_bools(), &bools[off + 3..off + 8]);
                }
            }
        }
    }

    #[test]
    fn boolean_algebra() {
        let a = pattern(200);
        let b: Vec<bool> = (0..200).map(|i| i % 3 == 0).collect();
        let (ba, bb) = (Bitmap::from_bools(&a), Bitmap::from_bools(&b));
        let zip = |f: fn(bool, bool) -> bool| -> Vec<bool> {
            a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect()
        };
        assert_eq!(ba.and(&bb).to_bools(), zip(|x, y| x & y));
        assert_eq!(ba.or(&bb).to_bools(), zip(|x, y| x | y));
        assert_eq!(ba.xor(&bb).to_bools(), zip(|x, y| x ^ y));
        assert_eq!(ba.and_not(&bb).to_bools(), zip(|x, y| x & !y));
        let not = ba.not();
        assert_eq!(not.to_bools(), a.iter().map(|&x| !x).collect::<Vec<_>>());
        assert_eq!(not.count_ones(), 200 - ba.count_ones());
        // Offsets on both sides.
        let (sa, sb) = (ba.slice(3, 150), bb.slice(40, 150));
        let want: Vec<bool> = (0..150).map(|i| a[3 + i] & b[40 + i]).collect();
        assert_eq!(sa.and(&sb).to_bools(), want);
    }

    #[test]
    fn mutable_bitmap_push_extend_set() {
        let mut m = MutableBitmap::new();
        let mut model = Vec::new();
        for i in 0..150 {
            m.push(i % 4 == 1);
            model.push(i % 4 == 1);
        }
        m.extend_constant(70, true);
        model.extend(std::iter::repeat_n(true, 70));
        m.extend_constant(5, false);
        model.extend(std::iter::repeat_n(false, 5));
        let other = pattern(131);
        m.extend_from_bitmap(&Bitmap::from_bools(&other).slice(1, 130));
        model.extend_from_slice(&other[1..131]);
        m.set(0, true);
        model[0] = true;
        m.set(200, false);
        model[200] = false;
        assert_eq!(m.len(), model.len());
        assert_eq!(m.count_ones(), model.iter().filter(|&&b| b).count());
        let frozen = m.freeze();
        assert_eq!(frozen.to_bools(), model);
        let back = frozen.into_mut();
        assert_eq!(back.freeze().to_bools(), model);
    }

    #[test]
    fn constants() {
        assert!(Bitmap::new_ones(70).all());
        assert!(!Bitmap::new_zeroed(70).any());
        assert_eq!(Bitmap::new_ones(70).slice(3, 60).count_ones(), 60);
        assert!(Bitmap::new().all());
    }
}
