//! [`StringChunked`]: a UTF-8 column stored as *views*.
//!
//! Each row is a 16-byte [`View`]. A string of up to 12 bytes lives entirely
//! inside its view; a longer one stores its length, its first four bytes and
//! the position of its bytes in one of the column's shared data buffers.
//!
//! This is what makes strings as cheap to slice and filter as numbers: both
//! operations move fixed-width views and share the data buffers, so no string
//! byte is copied. Comparisons profit too — most are decided by the length
//! and prefix stored in the view without touching the data buffer at all.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use super::{BooleanChunked, IdxCa, Selection, SortOptions, fmt_array, idx_slice, normalize_validity, slice_offsets};
use crate::datatypes::{DataType, IdxSize, PlSmallStr};
use crate::error::PolarsResult;
use crate::kernels::{self, CmpOp, par};
use crate::memory::{AlignedVec, Bitmap, Buffer, ColumnAlloc, MutableBitmap, Pod};
use crate::polars_ensure;

/// Longest string stored inside a view.
pub const INLINE_LEN: usize = 12;

/// One row of a string column.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct View {
    /// Length of the string in bytes.
    len: u32,
    /// Inline: bytes 0..4. Otherwise: the first four bytes (the prefix).
    prefix: [u8; 4],
    /// Inline: bytes 4..8. Otherwise: index of the data buffer.
    buffer: u32,
    /// Inline: bytes 8..12. Otherwise: byte offset inside the data buffer.
    offset: u32,
}

// SAFETY: `View` is `repr(C)` with four 4-byte fields and no padding; every
// bit pattern is a valid value of the struct (whether it is a *consistent*
// view is a `StringChunked` invariant, not a validity requirement).
unsafe impl Pod for View {}

impl View {
    /// The view of a string of at most [`INLINE_LEN`] bytes.
    #[inline]
    fn inline(bytes: &[u8]) -> View {
        debug_assert!(bytes.len() <= INLINE_LEN);
        let mut raw = [0u8; 16];
        raw[..4].copy_from_slice(&(bytes.len() as u32).to_ne_bytes());
        raw[4..4 + bytes.len()].copy_from_slice(bytes);
        // SAFETY: `View` is 16 plain bytes (see the `Pod` impl).
        unsafe { std::mem::transmute(raw) }
    }

    #[inline]
    fn reference(bytes: &[u8], buffer: u32, offset: u32) -> View {
        debug_assert!(bytes.len() > INLINE_LEN);
        let mut prefix = [0u8; 4];
        prefix.copy_from_slice(&bytes[..4]);
        View { len: bytes.len() as u32, prefix, buffer, offset }
    }

    /// Length of the string in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` for the empty string.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn is_inline(&self) -> bool {
        self.len as usize <= INLINE_LEN
    }

    #[inline]
    fn raw(&self) -> u128 {
        // SAFETY: `View` is 16 plain bytes.
        u128::from_ne_bytes(unsafe { std::mem::transmute::<View, [u8; 16]>(*self) })
    }

    /// The bytes of an inline view.
    #[inline]
    fn inline_bytes(&self) -> &[u8] {
        debug_assert!(self.is_inline());
        // SAFETY: an inline view stores its `len <= 12` bytes right after the
        // length field, inside the 16 bytes of `self`.
        unsafe { std::slice::from_raw_parts((self as *const View as *const u8).add(4), self.len as usize) }
    }
}

/// A column of UTF-8 strings, stored as one 16-byte [`View`] per row over
/// shared, immutable data buffers.
#[derive(Clone)]
// `repr(C)` with the name first: every column type then keeps its name at the
// same offset, so `Series::name()` is one load instead of a dispatch on the type.
#[repr(C)]
pub struct StringChunked {
    pub(crate) name: PlSmallStr,
    pub(crate) views: Buffer<View>,
    /// Data buffers referenced by the non-inline views.
    pub(crate) buffers: Arc<[Buffer<u8>]>,
    pub(crate) validity: Option<Bitmap>,
}

impl Default for StringChunked {
    fn default() -> Self {
        StringChunked::new_empty(PlSmallStr::EMPTY)
    }
}

/// Incrementally builds a [`StringChunked`].
pub struct StringChunkedBuilder {
    name: PlSmallStr,
    views: AlignedVec<View>,
    data: AlignedVec<u8>,
    done: Vec<Buffer<u8>>,
    validity: MutableBitmap,
    nulls: usize,
}

impl StringChunkedBuilder {
    /// A builder expecting about `capacity` rows.
    pub fn new(name: PlSmallStr, capacity: usize) -> Self {
        StringChunkedBuilder {
            name,
            views: Vec::with_capacity_in(capacity, ColumnAlloc),
            data: Vec::new_in(ColumnAlloc),
            done: Vec::new(),
            validity: MutableBitmap::with_capacity(capacity),
            nulls: 0,
        }
    }

    /// Append a string.
    #[inline]
    pub fn append_value(&mut self, value: impl AsRef<str>) {
        let bytes = value.as_ref().as_bytes();
        let view = if bytes.len() <= INLINE_LEN {
            View::inline(bytes)
        } else {
            // Offsets are 32-bit: start a new data buffer before one overflows.
            if self.data.len() + bytes.len() > u32::MAX as usize {
                self.done.push(std::mem::replace(&mut self.data, Vec::new_in(ColumnAlloc)).into());
            }
            let offset = self.data.len() as u32;
            self.data.extend_from_slice(bytes);
            View::reference(bytes, self.done.len() as u32, offset)
        };
        self.views.push(view);
        self.validity.push(true);
    }

    /// Append a null.
    #[inline]
    pub fn append_null(&mut self) {
        self.views.push(View::inline(&[]));
        self.validity.push(false);
        self.nulls += 1;
    }

    /// Append a string or a null.
    #[inline]
    pub fn append_option(&mut self, value: Option<impl AsRef<str>>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    /// Finish the column.
    pub fn finish(mut self) -> StringChunked {
        if !self.data.is_empty() || self.done.is_empty() {
            self.done.push(self.data.into());
        }
        StringChunked {
            name: self.name,
            views: self.views.into(),
            buffers: self.done.into(),
            validity: (self.nulls > 0).then(|| self.validity.freeze()),
        }
    }
}

impl StringChunked {
    /// Build from string slices.
    pub fn from_slice<S: AsRef<str>>(name: PlSmallStr, values: &[S]) -> Self {
        Self::from_iter_values(name, values.iter())
    }

    /// Build from optional strings; `None` becomes null.
    pub fn from_slice_options<S: AsRef<str>>(name: PlSmallStr, values: &[Option<S>]) -> Self {
        Self::from_iter_options(name, values.iter().map(|v| v.as_ref()))
    }

    /// Collect an iterator of strings (no nulls).
    pub fn from_iter_values<S: AsRef<str>>(name: PlSmallStr, iter: impl IntoIterator<Item = S>) -> Self {
        let iter = iter.into_iter();
        let mut b = StringChunkedBuilder::new(name, iter.size_hint().0);
        for s in iter {
            b.append_value(s);
        }
        b.finish()
    }

    /// Collect an iterator of optional strings.
    pub fn from_iter_options<S: AsRef<str>>(
        name: PlSmallStr,
        iter: impl IntoIterator<Item = Option<S>>,
    ) -> Self {
        let iter = iter.into_iter();
        let mut b = StringChunkedBuilder::new(name, iter.size_hint().0);
        for s in iter {
            b.append_option(s);
        }
        b.finish()
    }

    /// `len` copies of `value`.
    pub fn full(name: PlSmallStr, value: &str, len: usize) -> Self {
        let one = Self::from_slice(PlSmallStr::EMPTY, &[value]);
        let mut views: AlignedVec<View> = Vec::with_capacity_in(len, ColumnAlloc);
        views.resize(len, one.views[0]);
        StringChunked { name, views: views.into(), buffers: one.buffers, validity: None }
    }

    /// `len` nulls.
    pub fn full_null(name: PlSmallStr, len: usize) -> Self {
        Self::from_iter_options(name, std::iter::repeat_n(None::<&str>, len))
    }

    /// An empty column.
    pub fn new_empty(name: PlSmallStr) -> Self {
        StringChunked { name, views: Buffer::new(), buffers: Arc::from(Vec::new()), validity: None }
    }

    /// Column name.
    #[inline]
    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    /// Rename in place.
    #[inline]
    pub fn rename(&mut self, name: PlSmallStr) {
        self.name = name;
    }

    /// Rename, by value.
    #[inline]
    pub fn with_name(mut self, name: PlSmallStr) -> Self {
        self.name = name;
        self
    }

    /// Logical type (always `String`).
    #[inline]
    pub fn dtype(&self) -> &'static DataType {
        &DataType::String
    }

    /// Number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// `true` if there are no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Number of nulls.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::count_zeros)
    }

    /// `true` if at least one row is null.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// The validity bitmap, if the column has nulls.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Always 1: columns are contiguous.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        1
    }

    /// The string a view denotes.
    #[inline]
    fn resolve<'a>(&'a self, view: &'a View) -> &'a str {
        let bytes = if view.is_inline() {
            view.inline_bytes()
        } else {
            // SAFETY: every non-inline view of this column was created
            // together with (or re-indexed against) `self.buffers` and points
            // at `len` bytes inside the buffer it names.
            unsafe {
                let buf = self.buffers.get_unchecked(view.buffer as usize);
                buf.get_unchecked(view.offset as usize..view.offset as usize + view.len as usize)
            }
        };
        // SAFETY: the bytes were copied from a `&str`.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// The string at `index`, ignoring validity.
    ///
    /// # Panics
    /// If `index` is out of bounds.
    #[inline]
    pub fn value(&self, index: usize) -> &str {
        self.resolve(&self.views[index])
    }

    /// The string at `index`, or `None` if it is null or out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&str> {
        let view = self.views.get(index)?;
        match &self.validity {
            // SAFETY: `index < views.len() == validity.len()`.
            Some(v) if !unsafe { v.get_unchecked(index) } => None,
            _ => Some(self.resolve(view)),
        }
    }

    /// Iterate over the rows as `Option<&str>`.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<&str>> + DoubleEndedIterator + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Iterate over the strings, ignoring validity.
    pub fn into_no_null_iter(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        self.views.iter().map(move |v| self.resolve(v))
    }

    /// Copy the rows out.
    pub fn to_vec(&self) -> Vec<Option<&str>> {
        self.iter().collect()
    }

    /// A window of the column. O(1): shares views and data buffers.
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let (start, len) = slice_offsets(offset, length, self.len());
        StringChunked {
            name: self.name.clone(),
            views: self.views.slice(start, len),
            buffers: self.buffers.clone(),
            validity: self.validity.as_ref().map(|v| v.slice(start, len)).and_then(|v| normalize_validity(Some(v))),
        }
    }

    /// The first `length` rows (10 if `None`).
    pub fn head(&self, length: Option<usize>) -> Self {
        self.slice(0, length.unwrap_or(10))
    }

    /// The last `length` rows (10 if `None`).
    pub fn tail(&self, length: Option<usize>) -> Self {
        let n = length.unwrap_or(10).min(self.len());
        self.slice(-(n as i64), n)
    }

    /// Keep the rows where `mask` is `true` (null mask rows count as `false`).
    /// Only the 16-byte views move; the string bytes are shared.
    pub fn filter(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        if mask.len() == 1 && self.len() != 1 {
            return Ok(if mask.get(0) == Some(true) { self.clone() } else { Self::new_empty(self.name.clone()) });
        }
        polars_ensure!(
            mask.len() == self.len(),
            ShapeMismatch: "filter's length: {} differs from that of the series: {}", mask.len(), self.len()
        );
        Ok(self.filter_selection(&Selection::from_mask(mask), None))
    }

    pub(crate) fn filter_selection(&self, sel: &Selection<'_>, tasks: Option<usize>) -> Self {
        if sel.is_all() {
            return self.clone();
        }
        if sel.is_none() {
            return Self::new_empty(self.name.clone());
        }
        let tasks = tasks.unwrap_or_else(|| par::tasks_for(self.len(), size_of::<View>()));
        let views = kernels::filter_by_mask_with(&self.views, &sel.words, tasks, Some(false));
        let validity = self.validity.as_ref().map(|v| {
            Bitmap::from_words(kernels::compact_bits(&v.words(), &sel.words, sel.len, sel.count), sel.count)
        });
        StringChunked {
            name: self.name.clone(),
            views: views.into(),
            buffers: self.buffers.clone(),
            validity: normalize_validity(validity),
        }
    }

    /// The rows at `indices`, in that order. Shares the data buffers.
    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        self.take_slice(idx_slice(indices)?)
    }

    /// The rows at `indices`, in that order. Shares the data buffers.
    pub fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Self> {
        if let Some(&max) = indices.iter().max() {
            polars_ensure!((max as usize) < self.len(), OutOfBounds: "index {} is out of bounds for column of length {}", max, self.len());
        }
        Ok(StringChunked {
            name: self.name.clone(),
            views: kernels::arith::take(&self.views, indices).into(),
            buffers: self.buffers.clone(),
            validity: normalize_validity(
                self.validity.as_ref().map(|v| indices.iter().map(|&i| v.get(i as usize)).collect::<Bitmap>()),
            ),
        })
    }

    /// Gather rows at indices already known to be in bounds.
    ///
    /// # Safety
    /// Every index must be below `self.len()`.
    pub(crate) unsafe fn take_trusted(&self, indices: &[IdxSize]) -> Self {
        StringChunked {
            name: self.name.clone(),
            // SAFETY: guaranteed by the caller.
            views: unsafe { kernels::arith::take_unchecked(&self.views, indices) }.into(),
            buffers: self.buffers.clone(),
            validity: normalize_validity(
                // SAFETY: as above.
                self.validity.as_ref().map(|v| indices.iter().map(|&i| unsafe { v.get_unchecked(i as usize) }).collect::<Bitmap>()),
            ),
        }
    }

    /// Rows in reverse order.
    pub fn reverse(&self) -> Self {
        let idx: Vec<IdxSize> = (0..self.len() as IdxSize).rev().collect();
        self.take_slice(&idx).expect("indices are in bounds")
    }

    /// This column followed by `other`. The data buffers of both are shared.
    pub fn append(&mut self, other: &Self) {
        let shift = self.buffers.len() as u32;
        let (n, m) = (self.len(), other.len());
        let shifted: Vec<View> =
            other.views.iter().map(|v| if v.is_inline() { *v } else { View { buffer: v.buffer + shift, ..*v } }).collect();
        if !self.views.append_in_place(&shifted) {
            let mut views = std::mem::take(&mut self.views).into_aligned();
            views.extend_from_slice(&shifted);
            self.views = views.into();
        }
        self.buffers = self.buffers.iter().chain(other.buffers.iter()).cloned().collect();
        if self.validity.is_some() || other.validity.is_some() {
            let mut bits = match self.validity.take() {
                Some(v) => v.into_mut(),
                None => MutableBitmap::from_len_set(n),
            };
            match &other.validity {
                Some(v) => bits.extend_from_bitmap(v),
                None => bits.extend_constant(m, true),
            }
            self.validity = normalize_validity(Some(bits.freeze()));
        }
    }

    fn mask(&self, f: impl Fn(&View) -> bool) -> BooleanChunked {
        let mut words: AlignedVec<u64> = crate::memory::zeroed_vec(self.len().div_ceil(64));
        let count = kernels::scalar::pack_words(&self.views, &mut words, |v| f(&v));
        BooleanChunked {
            name: self.name.clone(),
            values: Bitmap::from_words_with_count(words, self.len(), count),
            validity: self.validity.clone(),
        }
    }

    /// Compare the rows `rows` against a string, writing mask words to `out`
    /// (validity is not applied). Returns the number of matches.
    pub(crate) fn compare_scalar_words(&self, op: CmpOp, rhs: &str, rows: std::ops::Range<usize>, out: &mut [u64]) -> usize {
        let views = &self.views[rows];
        let bytes = rhs.as_bytes();
        match op {
            CmpOp::Eq | CmpOp::Ne => {
                let negate = op == CmpOp::Ne;
                if bytes.len() <= INLINE_LEN {
                    // Inline views are canonical: equal strings, equal 16 bytes.
                    let needle = View::inline(bytes).raw();
                    kernels::scalar::pack_words(views, out, |v| (v.raw() == needle) != negate)
                } else {
                    let mut prefix = [0u8; 4];
                    prefix.copy_from_slice(&bytes[..4]);
                    let len = bytes.len() as u32;
                    kernels::scalar::pack_words(views, out, |v| {
                        (v.len == len && v.prefix == prefix && self.resolve(&v).as_bytes() == bytes) != negate
                    })
                }
            }
            _ => kernels::scalar::pack_words(views, out, |v| {
                let ord = self.resolve(&v).cmp(rhs);
                match op {
                    CmpOp::Lt => ord == Ordering::Less,
                    CmpOp::Le => ord != Ordering::Greater,
                    CmpOp::Gt => ord == Ordering::Greater,
                    _ => ord != Ordering::Less,
                }
            }),
        }
    }

    /// Row-wise comparison with a string.
    pub fn compare_scalar(&self, op: CmpOp, rhs: &str) -> BooleanChunked {
        let n = self.len();
        let mut words: AlignedVec<u64> = crate::memory::zeroed_vec(n.div_ceil(64));
        let count = self.compare_scalar_words(op, rhs, 0..n, &mut words);
        BooleanChunked {
            name: self.name.clone(),
            values: Bitmap::from_words_with_count(words, n, count),
            validity: self.validity.clone(),
        }
    }

    /// Row-wise comparison with another string column.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn compare(&self, op: CmpOp, rhs: &StringChunked) -> BooleanChunked {
        if rhs.len() == 1 && self.len() != 1 {
            return match rhs.get(0) {
                Some(s) => self.compare_scalar(op, s),
                None => BooleanChunked::full_null(self.name.clone(), self.len()),
            };
        }
        assert_eq!(self.len(), rhs.len(), "comparison operands must have equal length");
        let values: Bitmap = self
            .into_no_null_iter()
            .zip(rhs.into_no_null_iter())
            .map(|(a, b)| {
                let ord = a.cmp(b);
                match op {
                    CmpOp::Eq => ord == Ordering::Equal,
                    CmpOp::Ne => ord != Ordering::Equal,
                    CmpOp::Lt => ord == Ordering::Less,
                    CmpOp::Le => ord != Ordering::Greater,
                    CmpOp::Gt => ord == Ordering::Greater,
                    CmpOp::Ge => ord != Ordering::Less,
                }
            })
            .collect();
        BooleanChunked {
            name: self.name.clone(),
            values,
            validity: super::combine_validity(self.validity.as_ref(), rhs.validity.as_ref()),
        }
    }

    /// `==` against a string.
    pub fn equal(&self, rhs: &str) -> BooleanChunked {
        self.compare_scalar(CmpOp::Eq, rhs)
    }

    /// `!=` against a string.
    pub fn not_equal(&self, rhs: &str) -> BooleanChunked {
        self.compare_scalar(CmpOp::Ne, rhs)
    }

    /// `true` where the string contains `pat` (a literal, not a regex).
    pub fn contains_literal(&self, pat: &str) -> BooleanChunked {
        self.mask(|v| self.resolve(v).contains(pat))
    }

    /// `true` where the string starts with `prefix`.
    pub fn starts_with(&self, prefix: &str) -> BooleanChunked {
        self.mask(|v| self.resolve(v).starts_with(prefix))
    }

    /// `true` where the string ends with `suffix`.
    pub fn ends_with(&self, suffix: &str) -> BooleanChunked {
        self.mask(|v| self.resolve(v).ends_with(suffix))
    }

    /// Length of every string in bytes.
    pub fn str_len_bytes(&self) -> super::UInt32Chunked {
        super::UInt32Chunked {
            name: self.name.clone(),
            values: kernels::arith::unary(&self.views, |v| v.len).into(),
            validity: self.validity.clone(),
        }
    }

    /// Length of every string in characters.
    pub fn str_len_chars(&self) -> super::UInt32Chunked {
        super::UInt32Chunked {
            name: self.name.clone(),
            values: self.into_no_null_iter().map(|s| s.chars().count() as u32).collect(),
            validity: self.validity.clone(),
        }
    }

    /// `f` applied to every string; nulls stay null.
    pub fn apply_values<'a>(&'a self, f: impl Fn(&'a str) -> Cow<'a, str>) -> Self {
        Self::from_iter_options(self.name.clone(), self.iter().map(|s| s.map(&f)))
    }

    /// Upper-case every string.
    pub fn to_uppercase(&self) -> Self {
        self.apply_values(|s| Cow::Owned(s.to_uppercase()))
    }

    /// Lower-case every string.
    pub fn to_lowercase(&self) -> Self {
        self.apply_values(|s| Cow::Owned(s.to_lowercase()))
    }

    /// `true` where the row is null.
    pub fn is_null(&self) -> BooleanChunked {
        let values = match &self.validity {
            None => Bitmap::new_zeroed(self.len()),
            Some(v) => v.not(),
        };
        BooleanChunked { name: self.name.clone(), values, validity: None }
    }

    /// `true` where the row is not null.
    pub fn is_not_null(&self) -> BooleanChunked {
        let values = match &self.validity {
            None => Bitmap::new_ones(self.len()),
            Some(v) => v.clone(),
        };
        BooleanChunked { name: self.name.clone(), values, validity: None }
    }

    /// Replace nulls with `value`.
    pub fn fill_null_with_values(&self, value: &str) -> Self {
        if !self.has_nulls() {
            return self.clone();
        }
        Self::from_iter_values(self.name.clone(), self.iter().map(|s| s.unwrap_or(value)))
    }

    /// The permutation that sorts the column lexicographically.
    pub fn arg_sort(&self, options: SortOptions) -> IdxCa {
        let mut rows: Vec<IdxSize> =
            (0..self.len() as IdxSize).filter(|&i| self.validity.as_ref().is_none_or(|v| v.get(i as usize))).collect();
        let cmp = |a: &IdxSize, b: &IdxSize| {
            let ord = self.value(*a as usize).cmp(self.value(*b as usize));
            (if options.descending { ord.reverse() } else { ord }).then(a.cmp(b))
        };
        rows.sort_unstable_by(cmp);
        let nulls: Vec<IdxSize> = match &self.validity {
            None => Vec::new(),
            Some(v) => (0..self.len() as IdxSize).filter(|&i| !v.get(i as usize)).collect(),
        };
        let values: Buffer<IdxSize> =
            if options.nulls_last { rows.into_iter().chain(nulls).collect() } else { nulls.into_iter().chain(rows).collect() };
        IdxCa { name: self.name.clone(), values, validity: None }
    }

    /// Sort lexicographically (nulls first).
    pub fn sort(&self, descending: bool) -> Self {
        self.sort_with(SortOptions::default().with_order_descending(descending))
    }

    /// Sort lexicographically.
    pub fn sort_with(&self, options: SortOptions) -> Self {
        self.take(&self.arg_sort(options)).expect("arg_sort yields valid indices")
    }

    /// Number of distinct values; null counts as one value if present.
    pub fn n_unique(&self) -> usize {
        let set: std::collections::HashSet<&str> = self.iter().flatten().collect();
        set.len() + self.has_nulls() as usize
    }

    /// The distinct values in order of first appearance.
    pub fn unique(&self) -> Self {
        let mut seen = std::collections::HashSet::new();
        let mut seen_null = false;
        let rows = self.iter().filter(|s| match s {
            Some(s) => seen.insert(*s),
            None => !std::mem::replace(&mut seen_null, true),
        });
        Self::from_iter_options(self.name.clone(), rows)
    }

    /// Smallest string.
    pub fn min_str(&self) -> Option<&str> {
        self.iter().flatten().min()
    }

    /// Largest string.
    pub fn max_str(&self) -> Option<&str> {
        self.iter().flatten().max()
    }

    /// Bytes of heap memory viewed by this column.
    pub fn estimated_size(&self) -> usize {
        self.views.byte_len()
            + self.buffers.iter().map(|b| b.len()).sum::<usize>()
            + self.validity.as_ref().map_or(0, |v| v.len().div_ceil(8))
    }
}

impl PartialEq for StringChunked {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<S: AsRef<str>> FromIterator<Option<S>> for StringChunked {
    fn from_iter<I: IntoIterator<Item = Option<S>>>(iter: I) -> Self {
        Self::from_iter_options(PlSmallStr::EMPTY, iter)
    }
}

impl fmt::Debug for StringChunked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_array(f, "ChunkedArray", &self.name, &DataType::String, self.len(), |i| match self.get(i) {
            Some(v) => format!("\"{v}\""),
            None => "null".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StringChunked {
        StringChunked::from_slice_options("s".into(),
            &[Some("apple"), None, Some("a string longer than twelve bytes"), Some(""), Some("twelve bytes"), Some("thirteen byte")],
        )
    }

    #[test]
    fn views_are_sixteen_bytes_and_round_trip() {
        assert_eq!(size_of::<View>(), 16);
        let s = sample();
        assert_eq!(s.len(), 6);
        assert_eq!(s.null_count(), 1);
        assert_eq!(
            s.to_vec(),
            vec![Some("apple"), None, Some("a string longer than twelve bytes"), Some(""), Some("twelve bytes"), Some("thirteen byte")]
        );
        assert_eq!(s.get(9), None);
        assert_eq!(s.value(4), "twelve bytes");
        // Only the two long strings occupy the data buffer.
        assert_eq!(s.buffers.iter().map(|b| b.len()).sum::<usize>(), 33 + 13);
        let unicode = StringChunked::from_slice("u".into(), &["héllo wörld ünïcode", "日本語"]);
        assert_eq!(unicode.value(0), "héllo wörld ünïcode");
        assert_eq!(unicode.str_len_chars().cont_slice().unwrap(), &[19, 3]);
        assert_eq!(unicode.str_len_bytes().cont_slice().unwrap(), &[23, 9]);
    }

    #[test]
    fn slice_filter_and_take_share_the_data_buffers() {
        let s = sample();
        let sl = s.slice(2, 3);
        assert_eq!(sl.to_vec(), vec![Some("a string longer than twelve bytes"), Some(""), Some("twelve bytes")]);
        assert!(Arc::ptr_eq(&sl.buffers, &s.buffers));
        assert!(sl.validity().is_none());

        let mask = BooleanChunked::from_slice("m".into(), &[true, true, true, false, false, true]);
        let f = s.filter(&mask).unwrap();
        assert_eq!(f.to_vec(), vec![Some("apple"), None, Some("a string longer than twelve bytes"), Some("thirteen byte")]);
        assert!(Arc::ptr_eq(&f.buffers, &s.buffers));

        let t = s.take_slice(&[5, 0, 1]).unwrap();
        assert_eq!(t.to_vec(), vec![Some("thirteen byte"), Some("apple"), None]);
        assert!(s.take_slice(&[6]).is_err());
        assert_eq!(s.reverse().get(0), Some("thirteen byte"));
    }

    #[test]
    fn comparisons() {
        let s = sample();
        let t = Some(true);
        let f = Some(false);
        assert_eq!(s.equal("apple").to_vec(), vec![t, None, f, f, f, f]);
        assert_eq!(s.equal("thirteen byte").to_vec(), vec![f, None, f, f, f, t]);
        assert_eq!(s.equal("thirteen bytf").to_vec(), vec![f, None, f, f, f, f]);
        assert_eq!(s.not_equal("").to_vec(), vec![t, None, t, f, t, t]);
        assert_eq!(s.compare_scalar(CmpOp::Lt, "b").to_vec(), vec![t, None, t, t, f, f]);
        assert_eq!(s.compare_scalar(CmpOp::Ge, "twelve bytes").to_vec(), vec![f, None, f, f, t, f]);
        assert_eq!(s.starts_with("a").to_vec(), vec![t, None, t, f, f, f]);
        assert_eq!(s.ends_with("bytes").to_vec(), vec![f, None, t, f, t, f]);
        assert_eq!(s.contains_literal("tw").to_vec(), vec![f, None, t, f, t, f]);
        assert_eq!(s.compare(CmpOp::Eq, &s).to_vec(), vec![t, None, t, t, t, t]);
        // An inline string with the same leading bytes as a prefix must not match.
        let tricky = StringChunked::from_slice("t".into(), &["abcd", "abcdefghijklmnop"]);
        assert_eq!(tricky.equal("abcd").to_vec(), vec![t, f]);
    }

    #[test]
    fn append_sort_unique() {
        let mut a = StringChunked::from_slice("a".into(), &["pear", "a fairly long string value"]);
        let b = sample();
        a.append(&b);
        assert_eq!(a.len(), 8);
        assert_eq!(a.get(1), Some("a fairly long string value"));
        assert_eq!(a.get(4), Some("a string longer than twelve bytes"));
        assert_eq!(a.get(7), Some("thirteen byte"));
        assert_eq!(a.get(3), None);

        let sorted = b.sort(false);
        assert_eq!(
            sorted.to_vec(),
            vec![None, Some(""), Some("a string longer than twelve bytes"), Some("apple"), Some("thirteen byte"), Some("twelve bytes")]
        );
        assert_eq!(b.sort(true).get(1), Some("twelve bytes"));
        assert_eq!(b.sort_with(SortOptions::default().with_nulls_last(true)).get(5), None);
        let dup = StringChunked::from_slice_options("d".into(), &[Some("x"), None, Some("y"), Some("x"), None]);
        assert_eq!(dup.n_unique(), 3);
        assert_eq!(dup.unique().to_vec(), vec![Some("x"), None, Some("y")]);
        assert_eq!((dup.min_str(), dup.max_str()), (Some("x"), Some("y")));
        assert_eq!(dup.fill_null_with_values("z").to_vec()[1], Some("z"));
        assert_eq!(b.to_uppercase().get(0), Some("APPLE"));
        assert_eq!(StringChunked::full("f".into(), "a value that is not inline", 3).get(2), Some("a value that is not inline"));
        assert_eq!(StringChunked::full_null("n".into(), 2).null_count(), 2);
    }
}
