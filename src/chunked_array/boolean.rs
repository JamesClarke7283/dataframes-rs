//! [`BooleanChunked`]: a boolean column, and [`Selection`], the form in which
//! a boolean column drives a filter.

use std::borrow::Cow;
use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, Not};

use super::{combine_validity, fmt_array, normalize_validity, slice_offsets, slice_validity};
use crate::datatypes::{DataType, IdxSize, PlSmallStr};
use crate::error::PolarsResult;
use crate::kernels;
use crate::memory::{Bitmap, MutableBitmap};
use crate::polars_ensure;

/// A column of booleans: one bit per value plus an optional validity bitmap.
#[derive(Clone, Default)]
// `repr(C)` with the name first: every column type then keeps its name at the
// same offset, so `Series::name()` is one load instead of a dispatch on the type.
#[repr(C)]
pub struct BooleanChunked {
    pub(crate) name: PlSmallStr,
    pub(crate) values: Bitmap,
    pub(crate) validity: Option<Bitmap>,
}

impl BooleanChunked {
    /// Wrap existing bitmaps without copying.
    ///
    /// # Panics
    /// If `validity` is present and its length differs from `values.len()`.
    pub fn from_bitmap(name: PlSmallStr, values: Bitmap, validity: Option<Bitmap>) -> Self {
        if let Some(v) = &validity {
            assert_eq!(v.len(), values.len(), "validity and values must have equal length");
        }
        BooleanChunked { name, values, validity: normalize_validity(validity) }
    }

    /// Build from a slice of `bool`.
    pub fn from_slice(name: PlSmallStr, values: &[bool]) -> Self {
        BooleanChunked { name, values: Bitmap::from_bools(values), validity: None }
    }

    /// Build from optional values; `None` becomes null.
    pub fn from_slice_options(name: PlSmallStr, values: &[Option<bool>]) -> Self {
        Self::from_iter_options(name, values.iter().copied())
    }

    /// Collect an iterator of values (no nulls).
    pub fn from_iter_values(name: PlSmallStr, iter: impl IntoIterator<Item = bool>) -> Self {
        BooleanChunked { name, values: iter.into_iter().collect(), validity: None }
    }

    /// Collect an iterator of optional values.
    pub fn from_iter_options(name: PlSmallStr, iter: impl IntoIterator<Item = Option<bool>>) -> Self {
        let (mut values, mut validity) = (MutableBitmap::new(), MutableBitmap::new());
        for v in iter {
            values.push(v.unwrap_or(false));
            validity.push(v.is_some());
        }
        Self::from_bitmap(name, values.freeze(), Some(validity.freeze()))
    }

    /// `len` copies of `value`.
    pub fn full(name: PlSmallStr, value: bool, len: usize) -> Self {
        BooleanChunked { name, values: Bitmap::new_constant(value, len), validity: None }
    }

    /// `len` nulls.
    pub fn full_null(name: PlSmallStr, len: usize) -> Self {
        BooleanChunked {
            name,
            values: Bitmap::new_zeroed(len),
            validity: (len > 0).then(|| Bitmap::new_zeroed(len)),
        }
    }

    /// An empty column.
    pub fn new_empty(name: PlSmallStr) -> Self {
        BooleanChunked { name, values: Bitmap::new(), validity: None }
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

    /// Logical type (always `Boolean`).
    #[inline]
    pub fn dtype(&self) -> &'static DataType {
        &DataType::Boolean
    }

    /// Number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if there are no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
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

    /// The value bitmap (null slots hold an unspecified bit).
    #[inline]
    pub fn values(&self) -> &Bitmap {
        &self.values
    }

    /// Always 1: columns are contiguous.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        1
    }

    /// The value at `index`, or `None` if it is null or out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len() {
            return None;
        }
        // SAFETY: bounds checked above; validity has the same length.
        unsafe {
            match &self.validity {
                Some(v) if !v.get_unchecked(index) => None,
                _ => Some(self.values.get_unchecked(index)),
            }
        }
    }

    /// Iterate over the rows as `Option<bool>`.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<bool>> + DoubleEndedIterator + '_ {
        let mut validity = self.validity.as_ref().map(Bitmap::iter);
        self.values.iter().map(move |v| match &mut validity {
            None => Some(v),
            Some(bits) => bits.next().unwrap_or(false).then_some(v),
        })
    }

    /// Copy the rows out.
    pub fn to_vec(&self) -> Vec<Option<bool>> {
        self.iter().collect()
    }

    /// A window of the column. O(1): shares the bitmaps.
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let (start, len) = slice_offsets(offset, length, self.len());
        BooleanChunked {
            name: self.name.clone(),
            values: self.values.slice(start, len),
            validity: slice_validity(self.validity.as_ref(), start, len),
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

    /// Rows in reverse order.
    pub fn reverse(&self) -> Self {
        Self::from_iter_options(self.name.clone(), self.to_vec().into_iter().rev())
    }

    /// The rows at `indices`, in that order.
    pub fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Self> {
        if let Some(&max) = indices.iter().max() {
            polars_ensure!((max as usize) < self.len(), OutOfBounds: "index {} is out of bounds for column of length {}", max, self.len());
        }
        Ok(Self::from_iter_options(self.name.clone(), indices.iter().map(|&i| self.get(i as usize))))
    }

    /// This column followed by `other`.
    pub fn append(&mut self, other: &Self) {
        let (n, m) = (self.len(), other.len());
        let mut values = std::mem::take(&mut self.values).into_mut();
        values.extend_from_bitmap(&other.values);
        self.values = values.freeze();
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

    /// The rows that are both valid and `true`, as a bitmap.
    pub fn true_mask(&self) -> Bitmap {
        match &self.validity {
            None => self.values.clone(),
            Some(v) => self.values.and(v),
        }
    }

    /// Number of `true` rows (nulls do not count).
    pub fn sum(&self) -> Option<IdxSize> {
        Some(self.true_mask().count_ones() as IdxSize)
    }

    /// Number of `true` rows as `usize`.
    pub fn num_trues(&self) -> usize {
        self.true_mask().count_ones()
    }

    /// `true` if any valid row is `true`.
    pub fn any(&self) -> bool {
        self.true_mask().any()
    }

    /// `true` if every valid row is `true`.
    pub fn all(&self) -> bool {
        self.num_trues() == self.len() - self.null_count()
    }

    /// Smallest valid value.
    pub fn min(&self) -> Option<bool> {
        let valid = self.len() - self.null_count();
        (valid > 0).then(|| self.num_trues() == valid)
    }

    /// Largest valid value.
    pub fn max(&self) -> Option<bool> {
        let valid = self.len() - self.null_count();
        (valid > 0).then(|| self.num_trues() > 0)
    }

    /// Fraction of valid rows that are `true`.
    pub fn mean(&self) -> Option<f64> {
        let valid = self.len() - self.null_count();
        (valid > 0).then(|| self.num_trues() as f64 / valid as f64)
    }

    /// Keep the rows where `mask` is `true` (null mask rows count as `false`).
    pub fn filter(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        polars_ensure!(
            mask.len() == self.len(),
            ShapeMismatch: "filter's length: {} differs from that of the series: {}", mask.len(), self.len()
        );
        Ok(self.filter_selection(&Selection::from_mask(mask)))
    }

    pub(crate) fn filter_selection(&self, sel: &Selection<'_>) -> Self {
        if sel.is_all() {
            return self.clone();
        }
        let pick = |bits: &Bitmap| -> Bitmap {
            Bitmap::from_words(kernels::compact_bits(&bits.words(), &sel.words, sel.len, sel.count), sel.count)
        };
        BooleanChunked {
            name: self.name.clone(),
            values: pick(&self.values),
            validity: normalize_validity(self.validity.as_ref().map(pick)),
        }
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

    /// Row-wise equality with another boolean column (null if either is null).
    pub fn equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        BooleanChunked {
            name: self.name.clone(),
            values: self.values.xor(&rhs.values).not(),
            validity: combine_validity(self.validity.as_ref(), rhs.validity.as_ref()),
        }
    }

    /// Row-wise inequality with another boolean column (null if either is null).
    pub fn not_equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        BooleanChunked {
            name: self.name.clone(),
            values: self.values.xor(&rhs.values),
            validity: combine_validity(self.validity.as_ref(), rhs.validity.as_ref()),
        }
    }

    /// Replace nulls with `value`.
    pub fn fill_null_with_values(&self, value: bool) -> Self {
        let values = match &self.validity {
            None => self.values.clone(),
            Some(v) if value => self.values.or(&v.not()),
            Some(v) => self.values.and(v),
        };
        BooleanChunked { name: self.name.clone(), values, validity: None }
    }

    /// Bytes of heap memory viewed by this column.
    pub fn estimated_size(&self) -> usize {
        self.len().div_ceil(8) * (1 + self.validity.is_some() as usize)
    }
}

impl PartialEq for BooleanChunked {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl FromIterator<bool> for BooleanChunked {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Self::from_iter_values(PlSmallStr::EMPTY, iter)
    }
}

impl FromIterator<Option<bool>> for BooleanChunked {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        Self::from_iter_options(PlSmallStr::EMPTY, iter)
    }
}

impl fmt::Debug for BooleanChunked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_array(f, "ChunkedArray", &self.name, &DataType::Boolean, self.len(), |i| match self.get(i) {
            Some(v) => v.to_string(),
            None => "null".to_string(),
        })
    }
}

/// Kleene AND: `false & null = false`, `true & null = null`.
fn kleene_and(a: &BooleanChunked, b: &BooleanChunked) -> BooleanChunked {
    assert_eq!(a.len(), b.len(), "boolean operands must have equal length");
    let (values, validity) = match (&a.validity, &b.validity) {
        (None, None) => (a.values.and(&b.values), None),
        _ => {
            let n = a.len();
            let va = a.validity.clone().unwrap_or_else(|| Bitmap::new_ones(n));
            let vb = b.validity.clone().unwrap_or_else(|| Bitmap::new_ones(n));
            let (ta, tb) = (a.values.and(&va), b.values.and(&vb));
            // Valid when both are valid, or either side is a valid `false`.
            let validity = va.and(&vb).or(&va.and_not(&ta)).or(&vb.and_not(&tb));
            (ta.and(&tb), Some(validity))
        }
    };
    BooleanChunked { name: a.name.clone(), values, validity: normalize_validity(validity) }
}

/// Kleene OR: `true | null = true`, `false | null = null`.
fn kleene_or(a: &BooleanChunked, b: &BooleanChunked) -> BooleanChunked {
    assert_eq!(a.len(), b.len(), "boolean operands must have equal length");
    let (values, validity) = match (&a.validity, &b.validity) {
        (None, None) => (a.values.or(&b.values), None),
        _ => {
            let n = a.len();
            let va = a.validity.clone().unwrap_or_else(|| Bitmap::new_ones(n));
            let vb = b.validity.clone().unwrap_or_else(|| Bitmap::new_ones(n));
            let (ta, tb) = (a.values.and(&va), b.values.and(&vb));
            // Valid when both are valid, or either side is a valid `true`.
            let validity = va.and(&vb).or(&ta).or(&tb);
            (ta.or(&tb), Some(validity))
        }
    };
    BooleanChunked { name: a.name.clone(), values, validity: normalize_validity(validity) }
}

impl BitAnd for &BooleanChunked {
    type Output = BooleanChunked;
    fn bitand(self, rhs: Self) -> BooleanChunked {
        kleene_and(self, rhs)
    }
}

impl BitOr for &BooleanChunked {
    type Output = BooleanChunked;
    fn bitor(self, rhs: Self) -> BooleanChunked {
        kleene_or(self, rhs)
    }
}

impl BitXor for &BooleanChunked {
    type Output = BooleanChunked;
    fn bitxor(self, rhs: Self) -> BooleanChunked {
        self.not_equal(rhs)
    }
}

impl Not for &BooleanChunked {
    type Output = BooleanChunked;
    fn not(self) -> BooleanChunked {
        BooleanChunked { name: self.name.clone(), values: self.values.not(), validity: self.validity.clone() }
    }
}

macro_rules! owned_bool_ops {
    ($($trait:ident, $method:ident;)*) => {$(
        impl $trait for BooleanChunked {
            type Output = BooleanChunked;
            fn $method(self, rhs: Self) -> BooleanChunked {
                (&self).$method(&rhs)
            }
        }
    )*};
}
owned_bool_ops! { BitAnd, bitand; BitOr, bitor; BitXor, bitxor; }

impl Not for BooleanChunked {
    type Output = BooleanChunked;
    fn not(self) -> BooleanChunked {
        !&self
    }
}

/// A filter predicate reduced to what the compaction kernels consume:
/// aligned mask words (null rows already cleared) and the number of set bits.
///
/// Building it once and applying it to every column is how a frame filter
/// avoids re-deriving the mask per column.
pub struct Selection<'a> {
    /// Bit `i` is set when row `i` is kept.
    pub words: Cow<'a, [u64]>,
    /// Number of rows the mask covers.
    pub len: usize,
    /// Number of rows kept.
    pub count: usize,
}

impl<'a> Selection<'a> {
    /// Derive the selection from a boolean column: `true` keeps the row,
    /// `false` and null drop it.
    pub fn from_mask(mask: &'a BooleanChunked) -> Self {
        match &mask.validity {
            None => Selection { words: mask.values.words(), len: mask.len(), count: mask.values.count_ones() },
            Some(_) => {
                let kept = mask.true_mask();
                Selection { count: kept.count_ones(), len: kept.len(), words: Cow::Owned(kept.words().into_owned()) }
            }
        }
    }

    /// Wrap mask words directly. The count is recomputed from the words.
    pub fn from_words(words: Cow<'a, [u64]>, len: usize) -> Self {
        let count = crate::memory::bitmap::count_ones_range(&words, 0, len);
        Selection { words, len, count }
    }

    /// Every row is kept.
    #[inline]
    pub fn is_all(&self) -> bool {
        self.count == self.len
    }

    /// No row is kept.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(v: &[Option<bool>]) -> BooleanChunked {
        BooleanChunked::from_slice_options("m".into(), v)
    }

    #[test]
    fn aggregates() {
        let b = opt(&[Some(true), None, Some(false), Some(true)]);
        assert_eq!(b.sum(), Some(2));
        assert_eq!((b.any(), b.all()), (true, false));
        assert_eq!((b.min(), b.max()), (Some(false), Some(true)));
        assert_eq!(b.mean(), Some(2.0 / 3.0));
        assert_eq!(b.null_count(), 1);
        let empty = BooleanChunked::new_empty("e".into());
        assert_eq!((empty.any(), empty.all(), empty.min()), (false, true, None));
    }

    #[test]
    fn kleene_logic() {
        let t = Some(true);
        let f = Some(false);
        let n = None;
        let a = opt(&[t, t, t, f, f, f, n, n, n]);
        let b = opt(&[t, f, n, t, f, n, t, f, n]);
        assert_eq!((&a & &b).to_vec(), vec![t, f, n, f, f, f, n, f, n]);
        assert_eq!((&a | &b).to_vec(), vec![t, t, t, t, f, n, t, n, n]);
        assert_eq!((!&a).to_vec(), vec![f, f, f, t, t, t, n, n, n]);
        assert_eq!((&a ^ &b).to_vec(), vec![f, t, n, t, f, n, n, n, n]);
        let plain = BooleanChunked::from_slice("p".into(), &[true, false, true]);
        assert_eq!((&plain & &!&plain).to_vec(), vec![f, f, f]);
    }

    #[test]
    fn filter_slice_append() {
        let b = opt(&[Some(true), None, Some(false), Some(true), Some(false)]);
        let mask = opt(&[Some(true), Some(true), None, Some(false), Some(true)]);
        assert_eq!(b.filter(&mask).unwrap().to_vec(), vec![Some(true), None, Some(false)]);
        assert!(b.filter(&opt(&[Some(true)])).is_err());
        assert_eq!(b.slice(1, 3).to_vec(), vec![None, Some(false), Some(true)]);
        assert_eq!(b.slice(-2, 2).null_count(), 0);
        let mut c = BooleanChunked::from_slice("c".into(), &[true]);
        c.append(&b);
        assert_eq!(c.len(), 6);
        assert_eq!(c.get(2), None);
        assert_eq!(b.is_null().to_vec(), vec![Some(false), Some(true), Some(false), Some(false), Some(false)]);
        assert_eq!(b.fill_null_with_values(true).to_vec()[1], Some(true));
        assert_eq!(b.reverse().to_vec()[3], None);
    }

    #[test]
    fn selection_drops_null_rows() {
        let mask = opt(&[Some(true), None, Some(false), Some(true)]);
        let sel = Selection::from_mask(&mask);
        assert_eq!((sel.len, sel.count), (4, 2));
        assert_eq!(sel.words[0], 0b1001);
        assert!(!sel.is_all() && !sel.is_none());
    }
}
