//! Layer 3 — typed columns.
//!
//! A [`ChunkedArray<T>`] is one contiguous [`Buffer`] of native values plus an
//! optional validity [`Bitmap`]. The name is kept for source compatibility
//! with Polars; unlike a Polars `ChunkedArray` it never holds more than one
//! chunk, which is what makes element access a bounds check and a load, and
//! slicing a pointer bump.
//!
//! [`BooleanChunked`] stores its values as a bitmap and [`StringChunked`] as
//! 16-byte views into shared byte buffers; both follow the same conventions.

mod boolean;
mod ops;
mod string;

use std::fmt;

pub use boolean::{BooleanChunked, Selection};
pub use ops::{ChunkCompareEq, ChunkCompareIneq, FillNullStrategy, NumericScalar, QuantileMethod, SortOptions, Wide};
pub(crate) use ops::{Resolved, resolve_cmp};
pub use string::{StringChunked, StringChunkedBuilder, View};

use crate::datatypes::{
    DataType, Float32Type, Float64Type, IdxSize, Int8Type, Int16Type, Int32Type, Int64Type, NativeType, PlSmallStr,
    PolarsDataType, PolarsNumericType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use crate::error::{PolarsError, PolarsResult};
use crate::memory::{AlignedVec, Bitmap, BitmapIter, Buffer, ColumnAlloc, MutableBitmap};

/// A column of `i8`.
pub type Int8Chunked = ChunkedArray<Int8Type>;
/// A column of `i16`.
pub type Int16Chunked = ChunkedArray<Int16Type>;
/// A column of `i32`.
pub type Int32Chunked = ChunkedArray<Int32Type>;
/// A column of `i64`.
pub type Int64Chunked = ChunkedArray<Int64Type>;
/// A column of `u8`.
pub type UInt8Chunked = ChunkedArray<UInt8Type>;
/// A column of `u16`.
pub type UInt16Chunked = ChunkedArray<UInt16Type>;
/// A column of `u32`.
pub type UInt32Chunked = ChunkedArray<UInt32Type>;
/// A column of `u64`.
pub type UInt64Chunked = ChunkedArray<UInt64Type>;
/// A column of `f32`.
pub type Float32Chunked = ChunkedArray<Float32Type>;
/// A column of `f64`.
pub type Float64Chunked = ChunkedArray<Float64Type>;
/// A column of row indices.
pub type IdxCa = UInt32Chunked;

/// Resolve a Polars-style `(offset, length)` pair against an array length.
///
/// A negative offset counts from the end; the window is clamped to the array.
/// Returns `(start, len)`.
pub fn slice_offsets(offset: i64, length: usize, array_len: usize) -> (usize, usize) {
    let signed_len = array_len as i64;
    let start = if offset < 0 { offset.saturating_add(signed_len) } else { offset };
    let stop = start.saturating_add(length.min(i64::MAX as usize) as i64);
    let start = start.clamp(0, signed_len);
    let stop = stop.clamp(0, signed_len);
    (start as usize, (stop - start) as usize)
}

/// Normalise a validity bitmap: `None` when every bit is set.
pub(crate) fn normalize_validity(validity: Option<Bitmap>) -> Option<Bitmap> {
    validity.filter(|v| v.count_zeros() > 0)
}

/// AND two optional validity bitmaps.
pub(crate) fn combine_validity(a: Option<&Bitmap>, b: Option<&Bitmap>) -> Option<Bitmap> {
    match (a, b) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v.clone()),
        (Some(a), Some(b)) => Some(a.and(b)),
    }
}

/// A numeric column. See the [module documentation](self).
// `repr(C)` with the name first: every column type then keeps its name at the
// same offset, so `Series::name()` is one load instead of a dispatch on the type.
#[repr(C)]
pub struct ChunkedArray<T: PolarsNumericType> {
    pub(crate) name: PlSmallStr,
    pub(crate) values: Buffer<T::Native>,
    /// `None` means "no nulls". When present it has `values.len()` bits.
    pub(crate) validity: Option<Bitmap>,
}

impl<T: PolarsNumericType> Clone for ChunkedArray<T> {
    #[inline]
    fn clone(&self) -> Self {
        ChunkedArray { name: self.name.clone(), values: self.values.clone(), validity: self.validity.clone() }
    }
}

impl<T: PolarsNumericType> Default for ChunkedArray<T> {
    fn default() -> Self {
        ChunkedArray { name: PlSmallStr::EMPTY, values: Buffer::new(), validity: None }
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    // ----- construction ---------------------------------------------------

    /// Wrap existing storage without copying.
    ///
    /// # Panics
    /// If `validity` is present and its length differs from `values.len()`.
    pub fn from_buffer(name: PlSmallStr, values: Buffer<T::Native>, validity: Option<Bitmap>) -> Self {
        if let Some(v) = &validity {
            assert_eq!(v.len(), values.len(), "validity and values must have equal length");
        }
        ChunkedArray { name, values, validity: normalize_validity(validity) }
    }

    /// Freeze a column vector. O(1).
    #[inline]
    pub fn from_aligned(name: PlSmallStr, values: AlignedVec<T::Native>) -> Self {
        ChunkedArray { name, values: Buffer::from_aligned(values), validity: None }
    }

    /// Adopt a `Vec` without copying it.
    ///
    /// The values stay where the global allocator put them. Use
    /// [`from_slice`](Self::from_slice) to copy into cache-aligned,
    /// huge-page-backed column storage instead.
    #[inline]
    pub fn from_vec(name: PlSmallStr, values: Vec<T::Native>) -> Self {
        ChunkedArray { name, values: Buffer::from_vec(values), validity: None }
    }

    /// Polars alias of [`from_vec`](Self::from_vec).
    #[inline]
    pub fn new_vec(name: PlSmallStr, values: Vec<T::Native>) -> Self {
        Self::from_vec(name, values)
    }

    /// Adopt a `Vec` and a validity bitmap.
    pub fn from_vec_validity(name: PlSmallStr, values: Vec<T::Native>, validity: Option<Bitmap>) -> Self {
        Self::from_buffer(name, Buffer::from_vec(values), validity)
    }

    /// Copy a slice into column storage.
    pub fn from_slice(name: PlSmallStr, values: &[T::Native]) -> Self {
        ChunkedArray { name, values: Buffer::from_slice(values), validity: None }
    }

    /// Build from optional values; `None` becomes null.
    pub fn from_slice_options(name: PlSmallStr, values: &[Option<T::Native>]) -> Self {
        Self::from_iter_options(name, values.iter().copied())
    }

    /// Collect an iterator of values (no nulls).
    pub fn from_iter_values(name: PlSmallStr, iter: impl IntoIterator<Item = T::Native>) -> Self {
        ChunkedArray { name, values: iter.into_iter().collect(), validity: None }
    }

    /// Collect an iterator of optional values.
    pub fn from_iter_options(name: PlSmallStr, iter: impl IntoIterator<Item = Option<T::Native>>) -> Self {
        let iter = iter.into_iter();
        let cap = iter.size_hint().0;
        let mut values: AlignedVec<T::Native> = Vec::with_capacity_in(cap, ColumnAlloc);
        let mut validity = MutableBitmap::with_capacity(cap);
        for v in iter {
            values.push(v.unwrap_or_default());
            validity.push(v.is_some());
        }
        Self::from_buffer(name, values.into(), Some(validity.freeze()))
    }

    /// `len` copies of `value`.
    pub fn full(name: PlSmallStr, value: T::Native, len: usize) -> Self {
        let mut values: AlignedVec<T::Native> = Vec::with_capacity_in(len, ColumnAlloc);
        values.resize(len, value);
        Self::from_aligned(name, values)
    }

    /// `len` nulls.
    pub fn full_null(name: PlSmallStr, len: usize) -> Self {
        let mut out = Self::full(name, T::Native::default(), len);
        out.validity = (len > 0).then(|| Bitmap::new_zeroed(len));
        out
    }

    /// An empty column.
    pub fn new_empty(name: PlSmallStr) -> Self {
        ChunkedArray { name, values: Buffer::new(), validity: None }
    }

    // ----- metadata ---------------------------------------------------------

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

    /// Logical type.
    #[inline]
    pub fn dtype(&self) -> &'static DataType {
        &<T::Native as NativeType>::DTYPE
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

    /// The value buffer (null slots hold an unspecified value).
    #[inline]
    pub fn values(&self) -> &Buffer<T::Native> {
        &self.values
    }

    /// Always 1: columns are contiguous.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        1
    }

    /// No-op kept for Polars compatibility: columns are always contiguous.
    #[inline]
    pub fn rechunk(&self) -> Self {
        self.clone()
    }

    // ----- access -----------------------------------------------------------

    /// The values as one native slice.
    ///
    /// Fails if the column has nulls, because a slice cannot express them
    /// (same contract as Polars). Use [`values`](Self::values) to get the raw
    /// buffer regardless.
    #[inline]
    pub fn cont_slice(&self) -> PolarsResult<&[T::Native]> {
        if self.validity.is_some() {
            return Err(PolarsError::ComputeError("chunked array is not contiguous".into()));
        }
        Ok(self.values.as_slice())
    }

    /// The value at `index`, or `None` if it is null or out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T::Native> {
        let v = *self.values.get(index)?;
        match &self.validity {
            // SAFETY: `index < values.len() == validity.len()`.
            Some(valid) if !unsafe { valid.get_unchecked(index) } => None,
            _ => Some(v),
        }
    }

    /// The value at `index`, ignoring validity (a null row yields an
    /// unspecified value). The cheapest checked accessor: a bounds check and a
    /// load.
    ///
    /// # Panics
    /// If `index` is out of bounds.
    #[inline]
    pub fn value(&self, index: usize) -> T::Native {
        self.values[index]
    }

    /// The value at `index`, ignoring validity.
    ///
    /// # Safety
    /// `index < self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, index: usize) -> T::Native {
        // SAFETY: guaranteed by the caller.
        unsafe { *self.values.get_unchecked(index) }
    }

    /// `true` if row `index` is null. Panics if out of bounds.
    #[inline]
    pub fn is_null_at(&self, index: usize) -> bool {
        assert!(index < self.len(), "index {index} out of bounds for column of length {}", self.len());
        self.validity.as_ref().is_some_and(|v| !v.get(index))
    }

    /// First value.
    #[inline]
    pub fn first(&self) -> Option<T::Native> {
        self.get(0)
    }

    /// Last value.
    #[inline]
    pub fn last(&self) -> Option<T::Native> {
        self.len().checked_sub(1).and_then(|i| self.get(i))
    }

    /// Iterate over the rows as `Option<value>`.
    #[inline]
    pub fn iter(&self) -> ChunkedIter<'_, T::Native> {
        ChunkedIter { values: self.values.iter(), validity: self.validity.as_ref().map(Bitmap::iter) }
    }

    /// Iterate over the raw values, ignoring validity.
    #[inline]
    pub fn into_no_null_iter(&self) -> std::iter::Copied<std::slice::Iter<'_, T::Native>> {
        self.values.iter().copied()
    }

    /// Copy the values out (null slots hold an unspecified value).
    pub fn to_vec(&self) -> Vec<Option<T::Native>> {
        self.iter().collect()
    }

    /// Polars name for [`to_vec`](Self::to_vec).
    pub fn to_vec_null_aware(&self) -> Vec<Option<T::Native>> {
        self.to_vec()
    }

    // ----- slicing ----------------------------------------------------------

    /// A window of the column. O(1): shares the buffers.
    ///
    /// A negative `offset` counts from the end; the window is clamped.
    #[inline]
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let (start, len) = slice_offsets(offset, length, self.len());
        // SAFETY: `slice_offsets` clamps the window to the column.
        unsafe { self.slice_unchecked(start, len) }
    }

    /// # Safety
    /// `start + len <= self.len()`.
    #[inline]
    pub(crate) unsafe fn slice_unchecked(&self, start: usize, len: usize) -> Self {
        ChunkedArray {
            name: self.name.clone(),
            // SAFETY: guaranteed by the caller.
            values: unsafe { self.values.slice_unchecked(start, len) },
            validity: self.validity.as_ref().map(|v| v.slice(start, len)).and_then(|v| normalize_validity(Some(v))),
        }
    }

    /// The first `length` rows (10 if `None`).
    #[inline]
    pub fn head(&self, length: Option<usize>) -> Self {
        self.slice(0, length.unwrap_or(10))
    }

    /// The last `length` rows (10 if `None`).
    #[inline]
    pub fn tail(&self, length: Option<usize>) -> Self {
        let n = length.unwrap_or(10).min(self.len());
        self.slice(-(n as i64), n)
    }

    /// The first `num_elements` rows.
    #[inline]
    pub fn limit(&self, num_elements: usize) -> Self {
        self.slice(0, num_elements)
    }

    /// Rows in reverse order.
    pub fn reverse(&self) -> Self {
        let values: Buffer<T::Native> = Buffer::from_trusted_len_iter(self.values.iter().rev().copied());
        let validity = self.validity.as_ref().map(|v| v.iter().rev().collect::<Bitmap>());
        ChunkedArray { name: self.name.clone(), values, validity }
    }

    /// This column followed by `other`.
    pub fn append(&mut self, other: &Self) {
        if other.is_empty() {
            return;
        }
        let (n, m) = (self.len(), other.len());
        // First choice: extend the shared block in place (no copy even if
        // older snapshots still view the prefix). Otherwise thaw or copy; the
        // vector then grows geometrically, leaving room for the next append.
        if !self.values.append_in_place(&other.values) {
            let mut values = std::mem::take(&mut self.values).into_aligned();
            values.extend_from_slice(&other.values);
            self.values = values.into();
        }
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

    /// Bytes of heap memory viewed by this column.
    pub fn estimated_size(&self) -> usize {
        self.values.byte_len() + self.validity.as_ref().map_or(0, |v| v.len().div_ceil(8))
    }
}

impl<T: PolarsNumericType> PolarsDataType for ChunkedArray<T> {
    fn get_dtype() -> DataType {
        T::get_dtype()
    }
}

impl<T: PolarsNumericType> FromIterator<Option<T::Native>> for ChunkedArray<T> {
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        Self::from_iter_options(PlSmallStr::EMPTY, iter)
    }
}

impl<T: PolarsNumericType> PartialEq for ChunkedArray<T> {
    /// Same length, same nulls, and equal values at the non-null rows
    /// (floats compare in the total order, so NaN equals NaN).
    fn eq(&self, other: &Self) -> bool {
        use crate::kernels::Kernel;
        self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => a.tot_eq(b),
                (None, None) => true,
                _ => false,
            })
    }
}

impl<'a, T: PolarsNumericType> IntoIterator for &'a ChunkedArray<T> {
    type Item = Option<T::Native>;
    type IntoIter = ChunkedIter<'a, T::Native>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over the rows of a numeric column.
#[derive(Clone)]
pub struct ChunkedIter<'a, N> {
    values: std::slice::Iter<'a, N>,
    validity: Option<BitmapIter<'a>>,
}

impl<N: Copy> Iterator for ChunkedIter<'_, N> {
    type Item = Option<N>;

    #[inline]
    fn next(&mut self) -> Option<Option<N>> {
        let v = *self.values.next()?;
        match &mut self.validity {
            None => Some(Some(v)),
            Some(bits) => Some(bits.next().unwrap_or(false).then_some(v)),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl<N: Copy> ExactSizeIterator for ChunkedIter<'_, N> {}

impl<N: Copy> DoubleEndedIterator for ChunkedIter<'_, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Option<N>> {
        let v = *self.values.next_back()?;
        match &mut self.validity {
            None => Some(Some(v)),
            Some(bits) => Some(bits.next_back().unwrap_or(false).then_some(v)),
        }
    }
}

/// Shared `Debug` body for the array types.
pub(crate) fn fmt_array(
    f: &mut fmt::Formatter<'_>,
    kind: &str,
    name: &str,
    dtype: &DataType,
    len: usize,
    mut cell: impl FnMut(usize) -> String,
) -> fmt::Result {
    writeln!(f, "shape: ({len},)")?;
    writeln!(f, "{kind}: '{name}' [{dtype}]")?;
    writeln!(f, "[")?;
    const LIMIT: usize = 10;
    if len <= LIMIT {
        for i in 0..len {
            writeln!(f, "\t{}", cell(i))?;
        }
    } else {
        for i in 0..LIMIT / 2 {
            writeln!(f, "\t{}", cell(i))?;
        }
        writeln!(f, "\t…")?;
        for i in len - LIMIT / 2..len {
            writeln!(f, "\t{}", cell(i))?;
        }
    }
    write!(f, "]")
}

impl<T: PolarsNumericType> fmt::Debug for ChunkedArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_array(f, "ChunkedArray", &self.name, self.dtype(), self.len(), |i| match self.get(i) {
            Some(v) => v.into_any_value().to_string(),
            None => "null".to_string(),
        })
    }
}

/// Row indices as a typed slice.
pub(crate) fn idx_slice(indices: &IdxCa) -> PolarsResult<&[IdxSize]> {
    indices
        .cont_slice()
        .map_err(|_| PolarsError::ComputeError("index column must not contain nulls".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_offsets_follow_polars() {
        assert_eq!(slice_offsets(0, 3, 10), (0, 3));
        assert_eq!(slice_offsets(8, 5, 10), (8, 2));
        assert_eq!(slice_offsets(-3, 2, 10), (7, 2));
        assert_eq!(slice_offsets(-3, 10, 10), (7, 3));
        assert_eq!(slice_offsets(-20, 5, 10), (0, 0));
        assert_eq!(slice_offsets(-12, 5, 10), (0, 3));
        assert_eq!(slice_offsets(15, 5, 10), (10, 0));
        assert_eq!(slice_offsets(2, usize::MAX, 10), (2, 8));
    }

    #[test]
    fn construction_access_and_nulls() {
        let ca = Int64Chunked::from_slice("a".into(), &[1, 2, 3, 4]);
        assert_eq!(ca.name(), "a");
        assert_eq!((ca.len(), ca.null_count()), (4, 0));
        assert_eq!(ca.cont_slice().unwrap(), &[1, 2, 3, 4]);
        assert_eq!(ca.get(2), Some(3));
        assert_eq!(ca.get(9), None);
        assert_eq!(*ca.dtype(), DataType::Int64);

        let opt = Float64Chunked::from_slice_options("b".into(), &[Some(1.5), None, Some(3.0)]);
        assert_eq!(opt.null_count(), 1);
        assert!(opt.cont_slice().is_err());
        assert_eq!(opt.to_vec(), vec![Some(1.5), None, Some(3.0)]);
        assert_eq!(opt.get(1), None);
        assert!(opt.is_null_at(1));
        assert_eq!((opt.first(), opt.last()), (Some(1.5), Some(3.0)));
        let collected: Int32Chunked = [Some(1), None].into_iter().collect();
        assert_eq!(collected.to_vec(), vec![Some(1), None]);
        assert_eq!(UInt8Chunked::full("x".into(), 7, 3).cont_slice().unwrap(), &[7, 7, 7]);
        assert_eq!(Int16Chunked::full_null("n".into(), 3).null_count(), 3);
    }

    #[test]
    fn slices_share_storage() {
        let ca = Int32Chunked::from_slice("a".into(), &(0..100).collect::<Vec<_>>());
        let s = ca.slice(10, 5);
        assert_eq!(s.cont_slice().unwrap(), &[10, 11, 12, 13, 14]);
        assert_eq!(s.values().as_ptr(), unsafe { ca.values().as_ptr().add(10) });
        assert_eq!(ca.slice(-2, 5).cont_slice().unwrap(), &[98, 99]);
        assert_eq!(ca.head(Some(2)).len(), 2);
        assert_eq!(ca.tail(Some(3)).cont_slice().unwrap(), &[97, 98, 99]);
        assert_eq!(ca.tail(None).len(), 10);

        let opt = Int32Chunked::from_iter_options("o".into(), (0..100).map(|i| (i % 7 != 0).then_some(i)));
        let s = opt.slice(7, 8);
        assert_eq!(s.null_count(), 2);
        assert_eq!(s.get(0), None);
        assert_eq!(s.get(1), Some(8));
        // A window without nulls drops its validity entirely.
        assert!(opt.slice(1, 6).validity().is_none());
    }

    #[test]
    fn append_and_reverse() {
        let mut a = Int64Chunked::from_slice("a".into(), &[1, 2]);
        a.append(&Int64Chunked::from_slice_options("b".into(), &[None, Some(4)]));
        assert_eq!(a.to_vec(), vec![Some(1), Some(2), None, Some(4)]);
        assert_eq!(a.reverse().to_vec(), vec![Some(4), None, Some(2), Some(1)]);
        assert_eq!(a, a.clone());
        assert_ne!(a, a.reverse());
    }
}
