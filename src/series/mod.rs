//! Layer 4 — [`Series`]: a named column whose type is known at run time.
//!
//! A `Series` is a small enum over the typed columns of
//! [`crate::chunked_array`], stored inline. Dispatch is a `match` on a
//! discriminant — no heap box, no vtable — and cloning or slicing a `Series`
//! clones or slices the buffers of the column inside.
//!
//! The method names follow Polars; pandas spellings (`isna`, `fillna`,
//! `sort_values`, `nunique`, ...) are provided next to them in
//! the `pandas` submodule.

mod arithmetic;
mod comparison;
mod pandas;

pub use arithmetic::ArithOp;

use std::fmt;

use crate::chunked_array::{
    BooleanChunked, ChunkedArray, FillNullStrategy, Float32Chunked, Float64Chunked, IdxCa, Int8Chunked, Int16Chunked,
    Int32Chunked, Int64Chunked, Selection, SortOptions, StringChunked, UInt8Chunked, UInt16Chunked, UInt32Chunked,
    UInt64Chunked,
};
use crate::datatypes::{AnyValue, DataType, Field, IdxSize, NativeType, PlSmallStr, PolarsNumericType};
use crate::error::{PolarsError, PolarsResult};
use crate::kernels::aggregate::Num;
use crate::memory::Bitmap;
use crate::{polars_bail, polars_ensure};

/// A named column of any [`DataType`]. See the [module documentation](self).
#[derive(Clone)]
pub struct Series(pub(crate) SeriesInner);

/// Polars calls the element of a frame a `Column`; here it is the same type.
pub type Column = Series;

#[derive(Clone)]
pub(crate) enum SeriesInner {
    Boolean(BooleanChunked),
    UInt8(UInt8Chunked),
    UInt16(UInt16Chunked),
    UInt32(UInt32Chunked),
    UInt64(UInt64Chunked),
    Int8(Int8Chunked),
    Int16(Int16Chunked),
    Int32(Int32Chunked),
    Int64(Int64Chunked),
    Float32(Float32Chunked),
    Float64(Float64Chunked),
    String(StringChunked),
}

/// Run `$body` with `$ca` bound to the typed column, whatever its type.
macro_rules! for_all {
    ($series:expr, $ca:ident => $body:expr) => {
        match &$series.0 {
            $crate::series::SeriesInner::Boolean($ca) => $body,
            $crate::series::SeriesInner::UInt8($ca) => $body,
            $crate::series::SeriesInner::UInt16($ca) => $body,
            $crate::series::SeriesInner::UInt32($ca) => $body,
            $crate::series::SeriesInner::UInt64($ca) => $body,
            $crate::series::SeriesInner::Int8($ca) => $body,
            $crate::series::SeriesInner::Int16($ca) => $body,
            $crate::series::SeriesInner::Int32($ca) => $body,
            $crate::series::SeriesInner::Int64($ca) => $body,
            $crate::series::SeriesInner::Float32($ca) => $body,
            $crate::series::SeriesInner::Float64($ca) => $body,
            $crate::series::SeriesInner::String($ca) => $body,
        }
    };
}

/// Like `for_all!` with mutable access.
macro_rules! for_all_mut {
    ($series:expr, $ca:ident => $body:expr) => {
        match &mut $series.0 {
            $crate::series::SeriesInner::Boolean($ca) => $body,
            $crate::series::SeriesInner::UInt8($ca) => $body,
            $crate::series::SeriesInner::UInt16($ca) => $body,
            $crate::series::SeriesInner::UInt32($ca) => $body,
            $crate::series::SeriesInner::UInt64($ca) => $body,
            $crate::series::SeriesInner::Int8($ca) => $body,
            $crate::series::SeriesInner::Int16($ca) => $body,
            $crate::series::SeriesInner::Int32($ca) => $body,
            $crate::series::SeriesInner::Int64($ca) => $body,
            $crate::series::SeriesInner::Float32($ca) => $body,
            $crate::series::SeriesInner::Float64($ca) => $body,
            $crate::series::SeriesInner::String($ca) => $body,
        }
    };
}

/// Run `$body` for a numeric column, `$else` (with the pattern `$other`) otherwise.
macro_rules! for_numeric {
    ($series:expr, $ca:ident => $body:expr, $other:pat => $else:expr) => {
        match &$series.0 {
            $crate::series::SeriesInner::UInt8($ca) => $body,
            $crate::series::SeriesInner::UInt16($ca) => $body,
            $crate::series::SeriesInner::UInt32($ca) => $body,
            $crate::series::SeriesInner::UInt64($ca) => $body,
            $crate::series::SeriesInner::Int8($ca) => $body,
            $crate::series::SeriesInner::Int16($ca) => $body,
            $crate::series::SeriesInner::Int32($ca) => $body,
            $crate::series::SeriesInner::Int64($ca) => $body,
            $crate::series::SeriesInner::Float32($ca) => $body,
            $crate::series::SeriesInner::Float64($ca) => $body,
            $other => $else,
        }
    };
}

/// Call a generic function with the marker type matching a numeric `DataType`.
macro_rules! with_numeric_type {
    ($dtype:expr, $t:ident => $body:expr, _ => $else:expr) => {
        match $dtype {
            $crate::datatypes::DataType::UInt8 => { type $t = $crate::datatypes::UInt8Type; $body }
            $crate::datatypes::DataType::UInt16 => { type $t = $crate::datatypes::UInt16Type; $body }
            $crate::datatypes::DataType::UInt32 => { type $t = $crate::datatypes::UInt32Type; $body }
            $crate::datatypes::DataType::UInt64 => { type $t = $crate::datatypes::UInt64Type; $body }
            $crate::datatypes::DataType::Int8 => { type $t = $crate::datatypes::Int8Type; $body }
            $crate::datatypes::DataType::Int16 => { type $t = $crate::datatypes::Int16Type; $body }
            $crate::datatypes::DataType::Int32 => { type $t = $crate::datatypes::Int32Type; $body }
            $crate::datatypes::DataType::Int64 => { type $t = $crate::datatypes::Int64Type; $body }
            $crate::datatypes::DataType::Float32 => { type $t = $crate::datatypes::Float32Type; $body }
            $crate::datatypes::DataType::Float64 => { type $t = $crate::datatypes::Float64Type; $body }
            _ => $else,
        }
    };
}

pub(crate) use for_numeric;

/// Conversion into a [`Series`].
pub trait IntoSeries {
    /// Wrap as a dynamically typed column.
    fn into_series(self) -> Series;
}

/// Conversion into a [`Column`] (the same type as [`Series`] in this crate).
pub trait IntoColumn {
    /// Wrap as a frame column.
    fn into_column(self) -> Column;
}

impl<S: IntoSeries> IntoColumn for S {
    #[inline]
    fn into_column(self) -> Column {
        self.into_series()
    }
}

impl IntoSeries for Series {
    #[inline]
    fn into_series(self) -> Series {
        self
    }
}

impl<T: PolarsNumericType> IntoSeries for ChunkedArray<T> {
    #[inline]
    fn into_series(self) -> Series {
        T::wrap_series(self)
    }
}

impl IntoSeries for BooleanChunked {
    #[inline]
    fn into_series(self) -> Series {
        Series(SeriesInner::Boolean(self))
    }
}

impl IntoSeries for StringChunked {
    #[inline]
    fn into_series(self) -> Series {
        Series(SeriesInner::String(self))
    }
}

impl<T: PolarsNumericType> From<ChunkedArray<T>> for Series {
    fn from(ca: ChunkedArray<T>) -> Self {
        ca.into_series()
    }
}

impl From<BooleanChunked> for Series {
    fn from(ca: BooleanChunked) -> Self {
        ca.into_series()
    }
}

impl From<StringChunked> for Series {
    fn from(ca: StringChunked) -> Self {
        ca.into_series()
    }
}

/// Construction of a [`Series`] from a name and a collection, as in Polars.
///
/// `Phantom` only disambiguates the impls; it is inferred.
pub trait NamedFrom<T, Phantom: ?Sized> {
    /// Build a named column from `values`.
    fn new(name: PlSmallStr, values: T) -> Self;
}

// One impl per native type rather than a blanket `impl<N: NativeType>`: with
// concrete element types the compiler can discard the impls that do not apply
// and infer `Phantom`, including the `i32` / `f64` fallback for literals.
macro_rules! named_from_native {
    ($($native:ty),*) => {$(
        impl<T: AsRef<[$native]>> NamedFrom<T, [$native]> for Series {
            fn new(name: PlSmallStr, values: T) -> Self {
                ChunkedArray::<<$native as NativeType>::Polars>::from_slice(name, values.as_ref()).into_series()
            }
        }

        impl<T: AsRef<[Option<$native>]>> NamedFrom<T, [Option<$native>]> for Series {
            fn new(name: PlSmallStr, values: T) -> Self {
                ChunkedArray::<<$native as NativeType>::Polars>::from_slice_options(name, values.as_ref()).into_series()
            }
        }
    )*};
}
named_from_native!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl<T: AsRef<[bool]>> NamedFrom<T, [bool]> for Series {
    fn new(name: PlSmallStr, values: T) -> Self {
        BooleanChunked::from_slice(name, values.as_ref()).into_series()
    }
}

impl<T: AsRef<[Option<bool>]>> NamedFrom<T, [Option<bool>]> for Series {
    fn new(name: PlSmallStr, values: T) -> Self {
        BooleanChunked::from_slice_options(name, values.as_ref()).into_series()
    }
}

impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Series {
    fn new(name: PlSmallStr, values: T) -> Self {
        StringChunked::from_slice(name, values.as_ref()).into_series()
    }
}

impl<T: AsRef<[String]>> NamedFrom<T, [String]> for Series {
    fn new(name: PlSmallStr, values: T) -> Self {
        StringChunked::from_slice(name, values.as_ref()).into_series()
    }
}

impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Series {
    fn new(name: PlSmallStr, values: T) -> Self {
        StringChunked::from_slice_options(name, values.as_ref()).into_series()
    }
}

impl<T: AsRef<[Option<String>]>> NamedFrom<T, [Option<String>]> for Series {
    fn new(name: PlSmallStr, values: T) -> Self {
        StringChunked::from_slice_options(name, values.as_ref()).into_series()
    }
}

fn dtype_error(expected: &str, got: &DataType) -> PolarsError {
    PolarsError::SchemaMismatch(format!("invalid series dtype: expected `{expected}`, got `{got}`").into())
}

macro_rules! typed_accessors {
    ($($(#[$doc:meta])* $get:ident, $get_mut:ident, $variant:ident, $ca:ty, $label:literal;)*) => {$(
        $(#[$doc])*
        #[inline]
        pub fn $get(&self) -> PolarsResult<&$ca> {
            match &self.0 {
                SeriesInner::$variant(ca) => Ok(ca),
                _ => Err(dtype_error($label, self.dtype())),
            }
        }

        /// Mutable form of the accessor above (used for in-place updates).
        #[inline]
        pub fn $get_mut(&mut self) -> PolarsResult<&mut $ca> {
            let dtype = *self.dtype();
            match &mut self.0 {
                SeriesInner::$variant(ca) => Ok(ca),
                _ => Err(dtype_error($label, &dtype)),
            }
        }
    )*};
}

impl Series {
    // ----- construction ---------------------------------------------------

    /// Build a named column from a slice, array or `Vec` of values or
    /// `Option`s (see [`NamedFrom`]).
    ///
    /// ```
    /// use dataframes::prelude::*;
    /// let s = Series::new("a".into(), [1, 2, 3]);
    /// assert_eq!(s.i32().unwrap().sum(), Some(6));
    /// let t = Series::new("b".into(), &[Some("x"), None]);
    /// assert_eq!(t.null_count(), 1);
    /// ```
    #[allow(clippy::new_ret_no_self)]
    pub fn new<T, Phantom: ?Sized>(name: PlSmallStr, values: T) -> Series
    where
        Series: NamedFrom<T, Phantom>,
    {
        <Series as NamedFrom<T, Phantom>>::new(name, values)
    }

    /// Adopt a `Vec` of native values without copying it.
    pub fn from_vec<N: NativeType>(name: PlSmallStr, values: Vec<N>) -> Series {
        ChunkedArray::<N::Polars>::from_vec(name, values).into_series()
    }

    /// An empty column of the given type.
    pub fn new_empty(name: PlSmallStr, dtype: &DataType) -> Series {
        Self::full_null(name, 0, dtype)
    }

    /// `len` nulls of the given type (`Null` is materialised as `Boolean`).
    pub fn full_null(name: PlSmallStr, len: usize, dtype: &DataType) -> Series {
        with_numeric_type!(dtype, T => ChunkedArray::<T>::full_null(name, len).into_series(), _ => match dtype {
            DataType::String => StringChunked::full_null(name, len).into_series(),
            _ => BooleanChunked::full_null(name, len).into_series(),
        })
    }

    /// Build from dynamically typed values. The column type is the supertype
    /// of the non-null values (`Boolean` if all are null).
    pub fn from_any_values(name: PlSmallStr, values: &[AnyValue<'_>]) -> PolarsResult<Series> {
        let mut dtype: Option<DataType> = None;
        for v in values.iter().filter(|v| !v.is_null()) {
            let d = v.dtype();
            dtype = Some(match dtype {
                None => d,
                Some(prev) => DataType::supertype(&prev, &d).ok_or_else(|| {
                    PolarsError::SchemaMismatch(format!("cannot combine values of types {prev} and {d}").into())
                })?,
            });
        }
        let dtype = dtype.unwrap_or(DataType::Boolean);
        Ok(with_numeric_type!(dtype, T => {
            ChunkedArray::<T>::from_iter_options(name, values.iter().map(|v| v.extract())).into_series()
        }, _ => match dtype {
            DataType::String => StringChunked::from_iter_options(name, values.iter().map(|v| v.get_str())).into_series(),
            _ => BooleanChunked::from_iter_options(name, values.iter().map(|v| v.get_bool())).into_series(),
        }))
    }

    // ----- metadata ---------------------------------------------------------

    /// Column name.
    #[inline]
    pub fn name(&self) -> &PlSmallStr {
        for_all!(self, ca => ca.name())
    }

    /// Rename in place.
    pub fn rename(&mut self, name: PlSmallStr) -> &mut Series {
        for_all_mut!(self, ca => ca.rename(name));
        self
    }

    /// Rename, by value.
    pub fn with_name(mut self, name: PlSmallStr) -> Series {
        self.rename(name);
        self
    }

    /// Logical type.
    #[inline]
    pub fn dtype(&self) -> &DataType {
        for_all!(self, ca => ca.dtype())
    }

    /// Name and type.
    pub fn field(&self) -> Field {
        Field::new(self.name().clone(), *self.dtype())
    }

    /// Number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        for_all!(self, ca => ca.len())
    }

    /// `true` if there are no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of nulls.
    #[inline]
    pub fn null_count(&self) -> usize {
        for_all!(self, ca => ca.null_count())
    }

    /// `true` if at least one row is null.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// The validity bitmap, if the column has nulls.
    pub fn validity(&self) -> Option<&Bitmap> {
        for_all!(self, ca => ca.validity())
    }

    /// Always 1: columns are contiguous.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        1
    }

    /// No-op kept for Polars compatibility.
    #[inline]
    pub fn rechunk(&self) -> Series {
        self.clone()
    }

    /// Identity; exists so code written against Polars' `Column` compiles.
    #[inline]
    pub fn as_materialized_series(&self) -> &Series {
        self
    }

    /// Identity; exists so code written against Polars' `Column` compiles.
    #[inline]
    pub fn take_materialized_series(self) -> Series {
        self
    }

    /// Bytes of heap memory viewed by this column.
    pub fn estimated_size(&self) -> usize {
        for_all!(self, ca => ca.estimated_size())
    }

    // ----- typed access -----------------------------------------------------

    typed_accessors! {
        /// The column as `i8`s, if that is its type.
        i8, i8_mut, Int8, Int8Chunked, "Int8";
        /// The column as `i16`s, if that is its type.
        i16, i16_mut, Int16, Int16Chunked, "Int16";
        /// The column as `i32`s, if that is its type.
        i32, i32_mut, Int32, Int32Chunked, "Int32";
        /// The column as `i64`s, if that is its type.
        i64, i64_mut, Int64, Int64Chunked, "Int64";
        /// The column as `u8`s, if that is its type.
        u8, u8_mut, UInt8, UInt8Chunked, "UInt8";
        /// The column as `u16`s, if that is its type.
        u16, u16_mut, UInt16, UInt16Chunked, "UInt16";
        /// The column as `u32`s, if that is its type.
        u32, u32_mut, UInt32, UInt32Chunked, "UInt32";
        /// The column as `u64`s, if that is its type.
        u64, u64_mut, UInt64, UInt64Chunked, "UInt64";
        /// The column as `f32`s, if that is its type.
        f32, f32_mut, Float32, Float32Chunked, "Float32";
        /// The column as `f64`s, if that is its type.
        f64, f64_mut, Float64, Float64Chunked, "Float64";
        /// The column as booleans, if that is its type.
        bool, bool_mut, Boolean, BooleanChunked, "Boolean";
        /// The column as strings, if that is its type.
        str, str_mut, String, StringChunked, "String";
    }

    /// The column as row indices (`u32`), if that is its type.
    pub fn idx(&self) -> PolarsResult<&IdxCa> {
        self.u32()
    }

    /// The typed column, if `T` is its type.
    pub fn unpack<T: PolarsNumericType>(&self) -> PolarsResult<&ChunkedArray<T>> {
        T::unwrap_series(self).ok_or_else(|| dtype_error(&format!("{:?}", T::get_dtype()), self.dtype()))
    }

    /// The typed column, mutably, if `T` is its type.
    pub fn unpack_mut<T: PolarsNumericType>(&mut self) -> PolarsResult<&mut ChunkedArray<T>> {
        let dtype = *self.dtype();
        T::unwrap_series_mut(self).ok_or_else(|| dtype_error(&format!("{:?}", T::get_dtype()), &dtype))
    }

    /// The value at `index`.
    pub fn get(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(index < self.len(), OutOfBounds: "index {} is out of bounds for series of length {}", index, self.len());
        Ok(self.get_any(index))
    }

    /// The value at `index`; out-of-bounds indices read as null.
    pub(crate) fn get_any(&self, index: usize) -> AnyValue<'_> {
        for_numeric!(self, ca => ca.get(index).map_or(AnyValue::Null, NativeType::into_any_value), other => match other {
            SeriesInner::Boolean(ca) => ca.get(index).map_or(AnyValue::Null, AnyValue::Boolean),
            SeriesInner::String(ca) => ca.get(index).map_or(AnyValue::Null, AnyValue::String),
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// Iterate over the rows as dynamically typed values.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = AnyValue<'_>> + '_ {
        (0..self.len()).map(move |i| self.get_any(i))
    }

    // ----- slicing / selection --------------------------------------------

    /// A window of the column. O(1): shares the buffers.
    pub fn slice(&self, offset: i64, length: usize) -> Series {
        for_all!(self, ca => ca.slice(offset, length).into_series())
    }

    /// The first `length` rows (10 if `None`).
    pub fn head(&self, length: Option<usize>) -> Series {
        for_all!(self, ca => ca.head(length).into_series())
    }

    /// The last `length` rows (10 if `None`).
    pub fn tail(&self, length: Option<usize>) -> Series {
        for_all!(self, ca => ca.tail(length).into_series())
    }

    /// The first `num_elements` rows.
    pub fn limit(&self, num_elements: usize) -> Series {
        self.slice(0, num_elements)
    }

    /// Rows in reverse order.
    pub fn reverse(&self) -> Series {
        for_all!(self, ca => ca.reverse().into_series())
    }

    /// Keep the rows where `mask` is `true`.
    pub fn filter(&self, mask: &BooleanChunked) -> PolarsResult<Series> {
        for_all!(self, ca => ca.filter(mask).map(IntoSeries::into_series))
    }

    /// Apply a prepared selection (shared by every column of a frame filter).
    pub(crate) fn filter_selection(&self, sel: &Selection<'_>, tasks: Option<usize>, stream: Option<bool>) -> Series {
        for_numeric!(self, ca => ca.filter_selection(sel, tasks, stream).into_series(), other => match other {
            SeriesInner::Boolean(ca) => ca.filter_selection(sel).into_series(),
            SeriesInner::String(ca) => ca.filter_selection(sel, tasks).into_series(),
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// Gather rows at indices already known to be in bounds.
    ///
    /// # Safety
    /// Every index must be below `self.len()`.
    pub(crate) unsafe fn take_trusted(&self, indices: &[IdxSize]) -> Series {
        // SAFETY: forwarded.
        for_numeric!(self, ca => unsafe { ca.take_trusted(indices) }.into_series(), other => match other {
            SeriesInner::String(ca) => unsafe { ca.take_trusted(indices) }.into_series(),
            SeriesInner::Boolean(ca) => ca.take_slice(indices).expect("indices are in bounds").into_series(),
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// The rows at `indices`, in that order.
    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        self.take_slice(crate::chunked_array::idx_slice(indices)?)
    }

    /// The rows at `indices`, in that order.
    pub fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        for_all!(self, ca => ca.take_slice(indices).map(IntoSeries::into_series))
    }

    /// The rows at `indices`, in that order; `None` yields a null row.
    ///
    /// # Panics
    /// If an index is out of bounds.
    pub fn take_opt(&self, indices: &[Option<IdxSize>]) -> Series {
        let n = self.len();
        if let Some(bad) = indices.iter().flatten().find(|&&i| i as usize >= n) {
            panic!("index {bad} is out of bounds for series of length {n}");
        }
        fn numeric<T: PolarsNumericType>(ca: &ChunkedArray<T>, indices: &[Option<IdxSize>]) -> Series {
            ChunkedArray::<T>::from_iter_options(
                ca.name().clone(),
                indices.iter().map(|i| i.and_then(|i| ca.get(i as usize))),
            )
            .into_series()
        }
        for_numeric!(self, ca => numeric(ca, indices), other => match other {
            SeriesInner::Boolean(ca) => BooleanChunked::from_iter_options(
                ca.name().clone(),
                indices.iter().map(|i| i.and_then(|i| ca.get(i as usize))),
            )
            .into_series(),
            SeriesInner::String(ca) => {
                // Gather views (sharing the data buffers), then mark the nulls.
                let dense: Vec<IdxSize> = indices.iter().map(|i| i.unwrap_or(0)).collect();
                if n == 0 {
                    return StringChunked::full_null(ca.name().clone(), indices.len()).into_series();
                }
                let mut out = ca.take_slice(&dense).expect("indices were checked");
                let valid: Bitmap = indices.iter().map(|i| i.is_some_and(|i| ca.get(i as usize).is_some())).collect();
                out.validity = crate::chunked_array::normalize_validity(Some(valid));
                out.into_series()
            }
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// Append the rows of `other` (which must have the same type).
    pub fn append(&mut self, other: &Series) -> PolarsResult<&mut Series> {
        polars_ensure!(
            self.dtype() == other.dtype(),
            SchemaMismatch: "cannot append series, data types don't match: {} vs {}", self.dtype(), other.dtype()
        );
        match (&mut self.0, &other.0) {
            (SeriesInner::Boolean(a), SeriesInner::Boolean(b)) => a.append(b),
            (SeriesInner::String(a), SeriesInner::String(b)) => a.append(b),
            (SeriesInner::UInt8(a), SeriesInner::UInt8(b)) => a.append(b),
            (SeriesInner::UInt16(a), SeriesInner::UInt16(b)) => a.append(b),
            (SeriesInner::UInt32(a), SeriesInner::UInt32(b)) => a.append(b),
            (SeriesInner::UInt64(a), SeriesInner::UInt64(b)) => a.append(b),
            (SeriesInner::Int8(a), SeriesInner::Int8(b)) => a.append(b),
            (SeriesInner::Int16(a), SeriesInner::Int16(b)) => a.append(b),
            (SeriesInner::Int32(a), SeriesInner::Int32(b)) => a.append(b),
            (SeriesInner::Int64(a), SeriesInner::Int64(b)) => a.append(b),
            (SeriesInner::Float32(a), SeriesInner::Float32(b)) => a.append(b),
            (SeriesInner::Float64(a), SeriesInner::Float64(b)) => a.append(b),
            _ => unreachable!("data types were checked to be equal"),
        }
        Ok(self)
    }

    /// Polars alias of [`append`](Self::append).
    pub fn extend(&mut self, other: &Series) -> PolarsResult<&mut Series> {
        self.append(other)
    }

    // ----- nulls ------------------------------------------------------------

    /// `true` where the row is null.
    pub fn is_null(&self) -> BooleanChunked {
        for_all!(self, ca => ca.is_null())
    }

    /// `true` where the row is not null.
    pub fn is_not_null(&self) -> BooleanChunked {
        for_all!(self, ca => ca.is_not_null())
    }

    /// `true` where the value is NaN. Errors for non-float columns.
    pub fn is_nan(&self) -> PolarsResult<BooleanChunked> {
        match &self.0 {
            SeriesInner::Float32(ca) => Ok(ca.is_nan()),
            SeriesInner::Float64(ca) => Ok(ca.is_nan()),
            _ => polars_bail!(InvalidOperation: "`is_nan` operation not supported for dtype `{}`", self.dtype()),
        }
    }

    /// Drop the null rows.
    pub fn drop_nulls(&self) -> Series {
        if !self.has_nulls() {
            return self.clone();
        }
        self.filter(&self.is_not_null()).expect("mask has the series length")
    }

    /// Replace nulls according to `strategy`.
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Series> {
        for_numeric!(self, ca => Ok(ca.fill_null(strategy).into_series()), other => match (other, strategy) {
            (_, FillNullStrategy::Forward(_) | FillNullStrategy::Backward(_)) => {
                // Type-agnostic path: gather the row each null should copy.
                let n = self.len();
                let valid = self.is_not_null();
                let backward = matches!(strategy, FillNullStrategy::Backward(_));
                let limit = match strategy {
                    FillNullStrategy::Forward(l) | FillNullStrategy::Backward(l) => l,
                    _ => None,
                };
                let mut source: Vec<Option<IdxSize>> = vec![None; n];
                let (mut last, mut run) = (None, 0usize);
                let order: Box<dyn Iterator<Item = usize>> = if backward { Box::new((0..n).rev()) } else { Box::new(0..n) };
                for i in order {
                    if valid.get(i) == Some(true) {
                        (last, run) = (Some(i as IdxSize), 0);
                        source[i] = last;
                    } else {
                        run += 1;
                        source[i] = if limit.is_none_or(|l| run <= l) { last } else { None };
                    }
                }
                let values: Vec<AnyValue<'_>> =
                    source.iter().map(|s| s.map_or(AnyValue::Null, |i| self.get_any(i as usize))).collect();
                let mut out = Series::from_any_values(self.name().clone(), &values)?;
                if out.dtype() != self.dtype() {
                    out = out.cast(self.dtype())?;
                }
                Ok(out)
            }
            (SeriesInner::Boolean(ca), FillNullStrategy::Min | FillNullStrategy::Zero) => {
                Ok(ca.fill_null_with_values(false).into_series())
            }
            (SeriesInner::Boolean(ca), FillNullStrategy::Max | FillNullStrategy::One) => {
                Ok(ca.fill_null_with_values(true).into_series())
            }
            (SeriesInner::String(ca), FillNullStrategy::Min) => {
                Ok(ca.min_str().map_or_else(|| ca.clone(), |m| ca.fill_null_with_values(m)).into_series())
            }
            (SeriesInner::String(ca), FillNullStrategy::Max) => {
                Ok(ca.max_str().map_or_else(|| ca.clone(), |m| ca.fill_null_with_values(m)).into_series())
            }
            _ => polars_bail!(InvalidOperation: "fill-null strategy {:?} is not supported for dtype `{}`", strategy, self.dtype()),
        })
    }

    /// Replace nulls with a constant (cast to the column type).
    pub fn fill_null_with_value<'a>(&self, value: impl Into<AnyValue<'a>>) -> PolarsResult<Series> {
        let value = value.into();
        for_numeric!(self, ca => {
            let v = value.try_extract()?;
            Ok(ca.fill_null_with_values(v).into_series())
        }, other => match other {
            SeriesInner::Boolean(ca) => match value.get_bool() {
                Some(b) => Ok(ca.fill_null_with_values(b).into_series()),
                None => polars_bail!(SchemaMismatch: "cannot fill a boolean column with {}", value),
            },
            SeriesInner::String(ca) => match value.get_str() {
                Some(s) => Ok(ca.fill_null_with_values(s).into_series()),
                None => polars_bail!(SchemaMismatch: "cannot fill a string column with {}", value),
            },
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    // ----- aggregation ----------------------------------------------------

    /// Sum of the column, converted to `T`. Errors for string columns.
    ///
    /// Integer columns narrower than 64 bits are summed in `i64`, so the
    /// result does not overflow the column type (as in Polars).
    pub fn sum<T: NativeType>(&self) -> PolarsResult<T> {
        Ok(match &self.0 {
            SeriesInner::Boolean(ca) => T::from_i64(ca.num_trues() as i64),
            SeriesInner::Float32(ca) => T::from_f64(ca.sum_as_f64()),
            SeriesInner::Float64(ca) => T::from_f64(ca.sum().unwrap_or(0.0)),
            SeriesInner::UInt64(ca) => {
                let s = ca.sum().unwrap_or(0);
                if T::IS_FLOAT { T::from_f64(s as f64) } else { T::from_i64(s as i64) }
            }
            SeriesInner::String(_) => {
                polars_bail!(InvalidOperation: "`sum` operation not supported for dtype `{}`", self.dtype())
            }
            _ => {
                let s = self.sum_i64();
                if T::IS_FLOAT { T::from_f64(s as f64) } else { T::from_i64(s) }
            }
        })
    }

    /// Sum of an integer column in `i64` (0 for other types).
    fn sum_i64(&self) -> i64 {
        use crate::kernels::aggregate::{sum_i64, sum_i64_valid};
        for_numeric!(self, ca => match ca.validity() {
            None => sum_i64(&ca.values()[..]),
            Some(v) => sum_i64_valid(&ca.values()[..], &v.words()),
        }, _ => 0)
    }

    /// Smallest value, converted to `T`.
    pub fn min<T: NativeType>(&self) -> PolarsResult<Option<T>> {
        for_numeric!(self, ca => Ok(ca.min().map(|v| crate::chunked_array::NumericScalar::wide(Widen(v)).cast())), other => match other {
            SeriesInner::Boolean(ca) => Ok(ca.min().map(|b| T::from_i64(b as i64))),
            _ => polars_bail!(InvalidOperation: "`min` to a number is not supported for dtype `{}`", self.dtype()),
        })
    }

    /// Largest value, converted to `T`.
    pub fn max<T: NativeType>(&self) -> PolarsResult<Option<T>> {
        for_numeric!(self, ca => Ok(ca.max().map(|v| crate::chunked_array::NumericScalar::wide(Widen(v)).cast())), other => match other {
            SeriesInner::Boolean(ca) => Ok(ca.max().map(|b| T::from_i64(b as i64))),
            _ => polars_bail!(InvalidOperation: "`max` to a number is not supported for dtype `{}`", self.dtype()),
        })
    }

    /// Smallest value as a dynamically typed value (works for strings too).
    pub fn min_any(&self) -> AnyValue<'_> {
        for_numeric!(self, ca => ca.min().map_or(AnyValue::Null, NativeType::into_any_value), other => match other {
            SeriesInner::Boolean(ca) => ca.min().map_or(AnyValue::Null, AnyValue::Boolean),
            SeriesInner::String(ca) => ca.min_str().map_or(AnyValue::Null, AnyValue::String),
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// Largest value as a dynamically typed value (works for strings too).
    pub fn max_any(&self) -> AnyValue<'_> {
        for_numeric!(self, ca => ca.max().map_or(AnyValue::Null, NativeType::into_any_value), other => match other {
            SeriesInner::Boolean(ca) => ca.max().map_or(AnyValue::Null, AnyValue::Boolean),
            SeriesInner::String(ca) => ca.max_str().map_or(AnyValue::Null, AnyValue::String),
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// Arithmetic mean (`None` for strings and for columns without values).
    pub fn mean(&self) -> Option<f64> {
        for_numeric!(self, ca => ca.mean(), other => match other {
            SeriesInner::Boolean(ca) => ca.mean(),
            _ => None,
        })
    }

    /// Median (`None` for non-numeric columns).
    pub fn median(&self) -> Option<f64> {
        for_numeric!(self, ca => ca.median(), _ => None)
    }

    /// Standard deviation with `ddof` delta degrees of freedom.
    pub fn std(&self, ddof: u8) -> Option<f64> {
        for_numeric!(self, ca => ca.std(ddof), _ => None)
    }

    /// Variance with `ddof` delta degrees of freedom.
    pub fn var(&self, ddof: u8) -> Option<f64> {
        for_numeric!(self, ca => ca.var(ddof), _ => None)
    }

    /// The `quantile` (in `0..=1`) of a numeric column.
    pub fn quantile(&self, quantile: f64, method: crate::chunked_array::QuantileMethod) -> PolarsResult<Option<f64>> {
        for_numeric!(self, ca => ca.quantile(quantile, method), _ => {
            polars_bail!(InvalidOperation: "`quantile` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// Number of distinct values (null counts as one).
    pub fn n_unique(&self) -> PolarsResult<usize> {
        Ok(for_numeric!(self, ca => ca.n_unique(), other => match other {
            SeriesInner::Boolean(ca) => {
                let valid = ca.len() - ca.null_count();
                let trues = ca.num_trues();
                (trues > 0) as usize + (trues < valid) as usize + ca.has_nulls() as usize
            }
            SeriesInner::String(ca) => ca.n_unique(),
            _ => unreachable!("numeric variants are handled above"),
        }))
    }

    /// The distinct values.
    pub fn unique(&self) -> PolarsResult<Series> {
        Ok(for_numeric!(self, ca => ca.unique().into_series(), other => match other {
            SeriesInner::Boolean(ca) => {
                let mut seen = Vec::new();
                for v in ca.iter() {
                    if !seen.contains(&v) {
                        seen.push(v);
                    }
                }
                BooleanChunked::from_iter_options(ca.name().clone(), seen).into_series()
            }
            SeriesInner::String(ca) => ca.unique().into_series(),
            _ => unreachable!("numeric variants are handled above"),
        }))
    }

    /// Row index of the smallest value.
    pub fn arg_min(&self) -> Option<usize> {
        for_numeric!(self, ca => ca.arg_min(), _ => {
            let min = self.min_any();
            (!min.is_null()).then(|| self.iter().position(|v| v == min)).flatten()
        })
    }

    /// Row index of the largest value.
    pub fn arg_max(&self) -> Option<usize> {
        for_numeric!(self, ca => ca.arg_max(), _ => {
            let max = self.max_any();
            (!max.is_null()).then(|| self.iter().position(|v| v == max)).flatten()
        })
    }

    // ----- sort / cast ------------------------------------------------------

    /// Sort the column.
    pub fn sort(&self, options: SortOptions) -> PolarsResult<Series> {
        Ok(for_numeric!(self, ca => ca.sort_with(options).into_series(), other => match other {
            SeriesInner::String(ca) => ca.sort_with(options).into_series(),
            SeriesInner::Boolean(_) => self.take(&self.arg_sort(options))?,
            _ => unreachable!("numeric variants are handled above"),
        }))
    }

    /// The permutation that sorts the column.
    pub fn arg_sort(&self, options: SortOptions) -> IdxCa {
        for_numeric!(self, ca => ca.arg_sort(options), other => match other {
            SeriesInner::String(ca) => ca.arg_sort(options),
            SeriesInner::Boolean(ca) => {
                let as_int = UInt8Chunked::from_iter_options(ca.name().clone(), ca.iter().map(|b| b.map(u8::from)));
                as_int.arg_sort(options)
            }
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// Convert to another type.
    ///
    /// Numeric values that do not fit the target and strings that do not
    /// parse become null (Polars' non-strict cast).
    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        fn numeric_to<T: PolarsNumericType>(ca: &ChunkedArray<T>, dtype: &DataType) -> PolarsResult<Series> {
            Ok(with_numeric_type!(dtype, U => ca.cast_numeric::<U>().into_series(), _ => match dtype {
                DataType::Boolean => BooleanChunked::from_bitmap(
                    ca.name().clone(),
                    ca.values().iter().map(|v| *v != T::Native::ZERO).collect(),
                    ca.validity().cloned(),
                )
                .into_series(),
                DataType::String => StringChunked::from_iter_options(
                    ca.name().clone(),
                    ca.iter().map(|v| v.map(|v| match v.into_any_value() {
                        AnyValue::Float32(f) => crate::datatypes::fmt_float(f as f64),
                        AnyValue::Float64(f) => crate::datatypes::fmt_float(f),
                        other => other.to_string(),
                    })),
                )
                .into_series(),
                DataType::Null => polars_bail!(InvalidOperation: "cannot cast to the null type"),
                _ => unreachable!("numeric targets are handled above"),
            }))
        }
        for_numeric!(self, ca => numeric_to(ca, dtype), other => match other {
            SeriesInner::Boolean(ca) => Ok(with_numeric_type!(dtype, U => {
                ChunkedArray::<U>::from_iter_options(
                    ca.name().clone(),
                    ca.iter().map(|b| b.map(|b| <U as PolarsNumericType>::Native::from_i64(b as i64))),
                )
                .into_series()
            }, _ => match dtype {
                DataType::String => {
                    StringChunked::from_iter_options(ca.name().clone(), ca.iter().map(|b| b.map(|b| b.to_string()))).into_series()
                }
                _ => polars_bail!(InvalidOperation: "cannot cast `bool` to `{}`", dtype),
            })),
            SeriesInner::String(ca) => Ok(with_numeric_type!(dtype, U => {
                let parse = |s: &str| -> Option<<U as PolarsNumericType>::Native> {
                    let s = s.trim();
                    if <U as PolarsNumericType>::Native::IS_FLOAT {
                        s.parse::<f64>().ok().map(<U as PolarsNumericType>::Native::from_f64)
                    } else {
                        crate::chunked_array::Wide::Int(s.parse::<i128>().ok()?).checked_cast()
                    }
                };
                ChunkedArray::<U>::from_iter_options(ca.name().clone(), ca.iter().map(|s| s.and_then(parse))).into_series()
            }, _ => match dtype {
                DataType::Boolean => BooleanChunked::from_iter_options(
                    ca.name().clone(),
                    ca.iter().map(|s| s.and_then(|s| match s.trim() {
                        "true" | "True" | "TRUE" => Some(true),
                        "false" | "False" | "FALSE" => Some(false),
                        _ => None,
                    })),
                )
                .into_series(),
                _ => polars_bail!(InvalidOperation: "cannot cast `str` to `{}`", dtype),
            })),
            _ => unreachable!("numeric variants are handled above"),
        })
    }

    /// Like [`cast`](Self::cast), but values that would become null are an error.
    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        let out = self.cast(dtype)?;
        polars_ensure!(
            out.null_count() == self.null_count(),
            ComputeError: "strict conversion from `{}` to `{}` failed for {} value(s)",
            self.dtype(), dtype, out.null_count() - self.null_count()
        );
        Ok(out)
    }

    /// Cast to `Float64`.
    pub fn to_float(&self) -> PolarsResult<Series> {
        self.cast(&DataType::Float64)
    }

    /// Row-by-row equality: same type, same length, same values, and nulls in
    /// the same places. The name is not compared.
    pub fn equals_missing(&self, other: &Series) -> bool {
        match (&self.0, &other.0) {
            (SeriesInner::Boolean(a), SeriesInner::Boolean(b)) => a == b,
            (SeriesInner::String(a), SeriesInner::String(b)) => a == b,
            (SeriesInner::UInt8(a), SeriesInner::UInt8(b)) => a == b,
            (SeriesInner::UInt16(a), SeriesInner::UInt16(b)) => a == b,
            (SeriesInner::UInt32(a), SeriesInner::UInt32(b)) => a == b,
            (SeriesInner::UInt64(a), SeriesInner::UInt64(b)) => a == b,
            (SeriesInner::Int8(a), SeriesInner::Int8(b)) => a == b,
            (SeriesInner::Int16(a), SeriesInner::Int16(b)) => a == b,
            (SeriesInner::Int32(a), SeriesInner::Int32(b)) => a == b,
            (SeriesInner::Int64(a), SeriesInner::Int64(b)) => a == b,
            (SeriesInner::Float32(a), SeriesInner::Float32(b)) => a == b,
            (SeriesInner::Float64(a), SeriesInner::Float64(b)) => a == b,
            _ => false,
        }
    }

    /// Like [`equals_missing`](Self::equals_missing), but `false` if either side has nulls.
    pub fn equals(&self, other: &Series) -> bool {
        self.null_count() == 0 && other.null_count() == 0 && self.equals_missing(other)
    }
}

/// Adapter: widen a native value through [`NumericScalar`](crate::chunked_array::NumericScalar).
struct Widen<N>(N);
impl<N: NativeType> Clone for Widen<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<N: NativeType> Copy for Widen<N> {}
impl<N: NativeType> crate::chunked_array::NumericScalar for Widen<N> {
    fn wide(self) -> crate::chunked_array::Wide {
        use crate::chunked_array::Wide;
        if N::IS_FLOAT {
            Wide::Float(self.0.to_f64())
        } else if N::DTYPE == DataType::UInt64 {
            Wide::Int(self.0.to_i64() as u64 as i128)
        } else {
            Wide::Int(self.0.to_i64() as i128)
        }
    }
}

impl PartialEq for Series {
    /// Equal when the names match and [`equals_missing`](Series::equals_missing) holds.
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name() && self.equals_missing(other)
    }
}

impl fmt::Debug for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::chunked_array::fmt_array(f, "Series", self.name(), self.dtype(), self.len(), |i| {
            self.get_any(i).to_string()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_array::QuantileMethod;
    use crate::datatypes::{Float32Type, Int64Type};

    #[test]
    fn construction_infers_types_like_polars() {
        assert_eq!(*Series::new("a".into(), [1, 2, 3]).dtype(), DataType::Int32);
        assert_eq!(*Series::new("a".into(), &[1i64, 2]).dtype(), DataType::Int64);
        assert_eq!(*Series::new("a".into(), vec![1.5, 2.5]).dtype(), DataType::Float64);
        assert_eq!(*Series::new("a".into(), [Some(1u8), None]).dtype(), DataType::UInt8);
        assert_eq!(*Series::new("a".into(), [true, false]).dtype(), DataType::Boolean);
        assert_eq!(*Series::new("a".into(), ["x", "y"]).dtype(), DataType::String);
        assert_eq!(*Series::new("a".into(), vec!["x".to_string()]).dtype(), DataType::String);
        assert_eq!(*Series::new("a".into(), [Some("x"), None]).dtype(), DataType::String);
        let s = Series::from_vec("v".into(), vec![1.0f32, 2.0]);
        assert_eq!(s.f32().unwrap().cont_slice().unwrap(), &[1.0, 2.0]);
        assert_eq!(s.name(), "v");
        assert_eq!(s.field(), Field::new("v", DataType::Float32));
    }

    #[test]
    fn typed_accessors_and_any_values() {
        let s = Series::new("a".into(), [Some(1i64), None, Some(3)]);
        assert_eq!(s.i64().unwrap().get(2), Some(3));
        assert!(s.f64().is_err());
        assert_eq!(s.f64().unwrap_err().to_string(), "schema mismatch: invalid series dtype: expected `Float64`, got `i64`");
        assert_eq!(s.unpack::<Int64Type>().unwrap().len(), 3);
        assert!(s.unpack::<Float32Type>().is_err());
        assert_eq!(s.get(0).unwrap(), AnyValue::Int64(1));
        assert_eq!(s.get(1).unwrap(), AnyValue::Null);
        assert!(s.get(3).is_err());
        assert_eq!(s.iter().filter(|v| v.is_null()).count(), 1);
        let t = Series::new("s".into(), ["a", "b"]);
        assert_eq!(t.get(1).unwrap(), AnyValue::String("b"));
        assert_eq!(t.str().unwrap().get(0), Some("a"));
    }

    #[test]
    fn slicing_filtering_and_appending() {
        let s = Series::new("a".into(), [10i64, 20, 30, 40, 50]);
        assert_eq!(s.slice(1, 2).i64().unwrap().cont_slice().unwrap(), &[20, 30]);
        assert_eq!(s.head(Some(1)).len(), 1);
        assert_eq!(s.tail(Some(2)).i64().unwrap().cont_slice().unwrap(), &[40, 50]);
        assert_eq!(s.reverse().i64().unwrap().get(0), Some(50));
        let mask = BooleanChunked::from_slice("m".into(), &[true, false, true, false, true]);
        assert_eq!(s.filter(&mask).unwrap().i64().unwrap().cont_slice().unwrap(), &[10, 30, 50]);
        assert_eq!(s.take_slice(&[4, 0]).unwrap().i64().unwrap().cont_slice().unwrap(), &[50, 10]);
        let mut a = s.clone();
        a.append(&Series::new("b".into(), [60i64])).unwrap();
        assert_eq!(a.len(), 6);
        assert!(a.append(&Series::new("c".into(), [1.0])).is_err());
        assert_eq!(s.len(), 5, "append copied on write");
    }

    #[test]
    fn aggregations() {
        let s = Series::new("a".into(), [Some(3i16), None, Some(-1), Some(10)]);
        assert_eq!(s.sum::<i64>().unwrap(), 12);
        assert_eq!(s.sum::<f64>().unwrap(), 12.0);
        assert_eq!(s.min::<i32>().unwrap(), Some(-1));
        assert_eq!(s.max::<f64>().unwrap(), Some(10.0));
        assert_eq!(s.mean(), Some(4.0));
        assert_eq!(s.median(), Some(3.0));
        assert_eq!(s.n_unique().unwrap(), 4);
        assert_eq!((s.arg_min(), s.arg_max()), (Some(2), Some(3)));
        assert_eq!(s.quantile(1.0, QuantileMethod::Linear).unwrap(), Some(10.0));
        // Narrow integers do not overflow when summed.
        let big = Series::new("b".into(), vec![i8::MAX; 100]);
        assert_eq!(big.sum::<i64>().unwrap(), 12_700);
        let strs = Series::new("s".into(), ["b", "a", "b"]);
        assert!(strs.sum::<i64>().is_err());
        assert_eq!(strs.min_any(), AnyValue::String("a"));
        assert_eq!(strs.n_unique().unwrap(), 2);
        assert_eq!(strs.unique().unwrap().len(), 2);
        assert_eq!(strs.arg_max(), Some(0));
        let b = Series::new("b".into(), [true, true, false]);
        assert_eq!(b.sum::<u32>().unwrap(), 2);
        assert_eq!(b.n_unique().unwrap(), 2);
        assert_eq!(b.mean(), Some(2.0 / 3.0));
    }

    #[test]
    fn sorting_and_casting() {
        let s = Series::new("a".into(), [Some(3i64), None, Some(1)]);
        let sorted = s.sort(SortOptions::default()).unwrap();
        assert_eq!(sorted.i64().unwrap().to_vec(), vec![None, Some(1), Some(3)]);
        assert_eq!(s.arg_sort(SortOptions::default().with_order_descending(true)).cont_slice().unwrap(), &[1, 0, 2]);
        assert_eq!(s.cast(&DataType::Float64).unwrap().f64().unwrap().to_vec(), vec![Some(3.0), None, Some(1.0)]);
        assert_eq!(s.cast(&DataType::String).unwrap().str().unwrap().to_vec(), vec![Some("3"), None, Some("1")]);
        assert_eq!(s.cast(&DataType::Boolean).unwrap().bool().unwrap().to_vec(), vec![Some(true), None, Some(true)]);
        let strs = Series::new("s".into(), ["12", " 7 ", "x", "300"]);
        assert_eq!(strs.cast(&DataType::Int64).unwrap().i64().unwrap().to_vec(), vec![Some(12), Some(7), None, Some(300)]);
        assert_eq!(strs.cast(&DataType::UInt8).unwrap().u8().unwrap().to_vec(), vec![Some(12), Some(7), None, None]);
        assert_eq!(strs.cast(&DataType::Float64).unwrap().f64().unwrap().get(0), Some(12.0));
        assert!(strs.strict_cast(&DataType::Int64).is_err());
        let b = Series::new("b".into(), [true, false]);
        assert_eq!(b.cast(&DataType::Int32).unwrap().i32().unwrap().cont_slice().unwrap(), &[1, 0]);
        assert_eq!(b.sort(SortOptions::default()).unwrap().bool().unwrap().to_vec(), vec![Some(false), Some(true)]);
        let f = Series::new("f".into(), [1.5f64, 2.0]);
        assert_eq!(f.cast(&DataType::String).unwrap().str().unwrap().to_vec(), vec![Some("1.5"), Some("2.0")]);
    }

    #[test]
    fn null_handling_and_equality() {
        let s = Series::new("a".into(), [Some("x"), None, Some("z")]);
        assert_eq!(s.drop_nulls().len(), 2);
        assert_eq!(s.fill_null_with_value("y").unwrap().str().unwrap().to_vec(), vec![Some("x"), Some("y"), Some("z")]);
        assert_eq!(
            s.fill_null(FillNullStrategy::Forward(None)).unwrap().str().unwrap().to_vec(),
            vec![Some("x"), Some("x"), Some("z")]
        );
        assert!(s.fill_null(FillNullStrategy::Mean).is_err());
        let n = Series::new("n".into(), [Some(1i32), None]);
        assert_eq!(n.fill_null_with_value(9).unwrap().i32().unwrap().cont_slice().unwrap(), &[1, 9]);
        assert!(n.fill_null_with_value("no").is_err());
        assert!(n.equals_missing(&n.clone()));
        assert!(!n.equals(&n.clone()), "`equals` is false in the presence of nulls");
        assert_eq!(n, n.clone());
        assert_ne!(n, n.clone().with_name("other".into()));
        assert!(Series::new("f".into(), [1.0]).is_nan().is_ok());
        assert!(n.is_nan().is_err());
    }

    #[test]
    fn display_matches_polars_layout() {
        let s = Series::new("a".into(), [Some(1i64), None]);
        assert_eq!(s.to_string(), "shape: (2,)\nSeries: 'a' [i64]\n[\n\t1\n\tnull\n]");
        let t = Series::new("s".into(), ["x"]);
        assert_eq!(t.to_string(), "shape: (1,)\nSeries: 's' [str]\n[\n\t\"x\"\n]");
        let long = Series::new("l".into(), (0..20).collect::<Vec<i32>>());
        assert!(long.to_string().contains("\t…\n"));
    }

    #[test]
    fn any_value_round_trip() {
        let vals = [AnyValue::Int32(1), AnyValue::Null, AnyValue::Float64(2.5)];
        let s = Series::from_any_values("a".into(), &vals).unwrap();
        assert_eq!(*s.dtype(), DataType::Float64);
        assert_eq!(s.f64().unwrap().to_vec(), vec![Some(1.0), None, Some(2.5)]);
        assert!(Series::from_any_values("a".into(), &[AnyValue::Int32(1), AnyValue::String("x")]).is_err());
        assert_eq!(*Series::from_any_values("a".into(), &[AnyValue::Null]).unwrap().dtype(), DataType::Boolean);
    }
}
