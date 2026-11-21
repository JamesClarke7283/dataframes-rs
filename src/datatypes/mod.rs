//! Layer 2 — data types.
//!
//! The logical vocabulary of the crate: [`DataType`], the native element
//! trait [`NativeType`], the Polars-style marker types that parameterise
//! [`ChunkedArray`], the dynamically typed
//! cell [`AnyValue`], and [`Schema`].

mod any_value;
mod small_str;

use std::fmt;
use std::sync::Arc;

pub use any_value::AnyValue;
pub(crate) use any_value::fmt_float;
pub use small_str::PlSmallStr;

use crate::chunked_array::ChunkedArray;
use crate::kernels::aggregate::Num;
use crate::series::{Series, SeriesInner};

/// Row index type used by sort, take and group-by (matches Polars' default).
pub type IdxSize = u32;

/// The logical type of a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    /// `true` / `false`, one bit per value.
    Boolean,
    /// Unsigned 8-bit integer.
    UInt8,
    /// Unsigned 16-bit integer.
    UInt16,
    /// Unsigned 32-bit integer.
    UInt32,
    /// Unsigned 64-bit integer.
    UInt64,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// 32-bit float.
    Float32,
    /// 64-bit float.
    Float64,
    /// UTF-8 text.
    String,
    /// A column that holds only nulls and has no other type yet.
    Null,
}

impl DataType {
    /// Integers and floats.
    #[inline]
    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Polars name for [`is_numeric`](Self::is_numeric).
    #[inline]
    pub fn is_primitive_numeric(&self) -> bool {
        self.is_numeric()
    }

    /// Signed and unsigned integers.
    #[inline]
    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    /// `Int8` .. `Int64`.
    #[inline]
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64)
    }

    /// `UInt8` .. `UInt64`.
    #[inline]
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64)
    }

    /// `Float32` or `Float64`.
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    /// `Boolean`.
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(self, DataType::Boolean)
    }

    /// `String`.
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, DataType::String)
    }

    /// `Null`.
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, DataType::Null)
    }

    /// Bytes per value for fixed-width types (`None` for `String`; `Boolean` is a bit).
    pub fn byte_width(&self) -> Option<usize> {
        Some(match self {
            DataType::UInt8 | DataType::Int8 => 1,
            DataType::UInt16 | DataType::Int16 => 2,
            DataType::UInt32 | DataType::Int32 | DataType::Float32 => 4,
            DataType::UInt64 | DataType::Int64 | DataType::Float64 => 8,
            DataType::Boolean | DataType::String | DataType::Null => return None,
        })
    }

    /// The type both operands are cast to before arithmetic or comparison.
    ///
    /// Follows the usual lattice: equal types stay; any float makes the result
    /// a float (`Float64` unless both sides fit `Float32`); mixed integers widen
    /// to the smallest signed type that holds both, capped at `Int64` (or
    /// `Float64` when a `UInt64` meets a signed type); `Null` adopts the other side.
    pub fn supertype(a: &DataType, b: &DataType) -> Option<DataType> {
        use DataType::*;
        if a == b {
            return Some(*a);
        }
        match (a, b) {
            (Null, other) | (other, Null) => Some(*other),
            (String, _) | (_, String) => None,
            (Boolean, other) | (other, Boolean) if other.is_numeric() => Some(*other),
            _ if a.is_float() || b.is_float() => {
                let small = |t: &DataType| matches!(t, Float32 | Int8 | Int16 | UInt8 | UInt16);
                Some(if small(a) && small(b) { Float32 } else { Float64 })
            }
            _ if a.is_integer() && b.is_integer() => {
                let (wa, wb) = (a.byte_width()?, b.byte_width()?);
                if a.is_unsigned_integer() == b.is_unsigned_integer() {
                    return Some(if wa >= wb { *a } else { *b });
                }
                let (signed, unsigned, ws, wu) = if a.is_signed_integer() { (a, b, wa, wb) } else { (b, a, wb, wa) };
                let _ = unsigned;
                if ws > wu {
                    return Some(*signed);
                }
                Some(match wu {
                    1 => Int16,
                    2 => Int32,
                    4 => Int64,
                    _ => Float64,
                })
            }
            _ => None,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DataType::Boolean => "bool",
            DataType::UInt8 => "u8",
            DataType::UInt16 => "u16",
            DataType::UInt32 => "u32",
            DataType::UInt64 => "u64",
            DataType::Int8 => "i8",
            DataType::Int16 => "i16",
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            DataType::String => "str",
            DataType::Null => "null",
        })
    }
}

/// A native numeric element type and its mapping to the logical type system.
pub trait NativeType: Num + fmt::Display {
    /// The logical type of a column of `Self`.
    const DTYPE: DataType;
    /// The marker type with `Native = Self`.
    type Polars: PolarsNumericType<Native = Self>;

    /// Wrap as a dynamically typed value.
    fn into_any_value<'a>(self) -> AnyValue<'a>;
}

/// A logical type known at compile time.
pub trait PolarsDataType: Send + Sync + Sized + 'static {
    /// The run-time [`DataType`] this marker stands for.
    fn get_dtype() -> DataType;
}

/// A numeric logical type; `Native` is the element type stored in the column.
///
/// The trait is implemented for the ten marker types in this module and is
/// not meant to be implemented elsewhere.
pub trait PolarsNumericType: PolarsDataType {
    /// The element type.
    type Native: NativeType<Polars = Self>;

    /// Wrap a typed column as a dynamically typed [`Series`].
    #[doc(hidden)]
    fn wrap_series(ca: ChunkedArray<Self>) -> Series;
    /// The typed column inside `series`, if it has this type.
    #[doc(hidden)]
    fn unwrap_series(series: &Series) -> Option<&ChunkedArray<Self>>;
    /// The typed column inside `series`, mutably, if it has this type.
    #[doc(hidden)]
    fn unwrap_series_mut(series: &mut Series) -> Option<&mut ChunkedArray<Self>>;
}

macro_rules! numeric_types {
    ($($(#[$doc:meta])* $marker:ident, $native:ty, $variant:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $marker;

        impl PolarsDataType for $marker {
            #[inline]
            fn get_dtype() -> DataType {
                DataType::$variant
            }
        }

        impl PolarsNumericType for $marker {
            type Native = $native;

            #[inline]
            fn wrap_series(ca: ChunkedArray<Self>) -> Series {
                Series(SeriesInner::$variant(ca))
            }

            #[inline]
            fn unwrap_series(series: &Series) -> Option<&ChunkedArray<Self>> {
                match &series.0 {
                    SeriesInner::$variant(ca) => Some(ca),
                    _ => None,
                }
            }

            #[inline]
            fn unwrap_series_mut(series: &mut Series) -> Option<&mut ChunkedArray<Self>> {
                match &mut series.0 {
                    SeriesInner::$variant(ca) => Some(ca),
                    _ => None,
                }
            }
        }

        impl NativeType for $native {
            const DTYPE: DataType = DataType::$variant;
            type Polars = $marker;

            #[inline]
            fn into_any_value<'a>(self) -> AnyValue<'a> {
                AnyValue::$variant(self)
            }
        }
    )*};
}

numeric_types! {
    /// Marker for `i8` columns.
    Int8Type, i8, Int8;
    /// Marker for `i16` columns.
    Int16Type, i16, Int16;
    /// Marker for `i32` columns.
    Int32Type, i32, Int32;
    /// Marker for `i64` columns.
    Int64Type, i64, Int64;
    /// Marker for `u8` columns.
    UInt8Type, u8, UInt8;
    /// Marker for `u16` columns.
    UInt16Type, u16, UInt16;
    /// Marker for `u32` columns.
    UInt32Type, u32, UInt32;
    /// Marker for `u64` columns.
    UInt64Type, u64, UInt64;
    /// Marker for `f32` columns.
    Float32Type, f32, Float32;
    /// Marker for `f64` columns.
    Float64Type, f64, Float64;
}

/// Marker for boolean columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BooleanType;
impl PolarsDataType for BooleanType {
    fn get_dtype() -> DataType {
        DataType::Boolean
    }
}

/// Marker for string columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StringType;
impl PolarsDataType for StringType {
    fn get_dtype() -> DataType {
        DataType::String
    }
}

/// A column name paired with its type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    /// Column name.
    pub name: PlSmallStr,
    /// Column type.
    pub dtype: DataType,
}

impl Field {
    /// Create a field.
    pub fn new(name: impl Into<PlSmallStr>, dtype: DataType) -> Self {
        Field { name: name.into(), dtype }
    }

    /// The column name.
    #[inline]
    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    /// The column type.
    #[inline]
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
}

/// An ordered mapping from column name to type.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schema {
    fields: Vec<Field>,
}

/// A shared [`Schema`].
pub type SchemaRef = Arc<Schema>;

impl Schema {
    /// An empty schema.
    pub fn new() -> Self {
        Schema::default()
    }

    /// An empty schema with room for `n` fields.
    pub fn with_capacity(n: usize) -> Self {
        Schema { fields: Vec::with_capacity(n) }
    }

    /// Number of fields.
    #[inline]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// `true` if there are no fields.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Add or replace a field; returns the previous type if the name existed.
    pub fn with_column(&mut self, name: PlSmallStr, dtype: DataType) -> Option<DataType> {
        match self.index_of(&name) {
            Some(i) => Some(std::mem::replace(&mut self.fields[i].dtype, dtype)),
            None => {
                self.fields.push(Field { name, dtype });
                None
            }
        }
    }

    /// The type of column `name`.
    pub fn get(&self, name: &str) -> Option<&DataType> {
        self.fields.iter().find(|f| f.name == name).map(|f| &f.dtype)
    }

    /// The position of column `name`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    /// `true` if a column called `name` exists.
    pub fn contains(&self, name: &str) -> bool {
        self.index_of(name).is_some()
    }

    /// The field at position `index`.
    pub fn get_at_index(&self, index: usize) -> Option<(&PlSmallStr, &DataType)> {
        self.fields.get(index).map(|f| (&f.name, &f.dtype))
    }

    /// Iterate over `(name, type)` pairs in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&PlSmallStr, &DataType)> + '_ {
        self.fields.iter().map(|f| (&f.name, &f.dtype))
    }

    /// Iterate over the names in order.
    pub fn iter_names(&self) -> impl ExactSizeIterator<Item = &PlSmallStr> + '_ {
        self.fields.iter().map(|f| &f.name)
    }

    /// Iterate over the fields in order.
    pub fn iter_fields(&self) -> impl ExactSizeIterator<Item = Field> + '_ {
        self.fields.iter().cloned()
    }
}

impl FromIterator<Field> for Schema {
    fn from_iter<I: IntoIterator<Item = Field>>(iter: I) -> Self {
        Schema { fields: iter.into_iter().collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_match_polars() {
        let names: Vec<String> = [DataType::Boolean, DataType::UInt32, DataType::Int64, DataType::Float64, DataType::String]
            .iter()
            .map(|d| d.to_string())
            .collect();
        assert_eq!(names, ["bool", "u32", "i64", "f64", "str"]);
    }

    #[test]
    fn supertypes() {
        use DataType::*;
        let st = |a, b| DataType::supertype(&a, &b);
        assert_eq!(st(Int32, Int64), Some(Int64));
        assert_eq!(st(Int64, Float32), Some(Float64));
        assert_eq!(st(Int16, Float32), Some(Float32));
        assert_eq!(st(UInt8, Int8), Some(Int16));
        assert_eq!(st(UInt32, Int32), Some(Int64));
        assert_eq!(st(UInt64, Int64), Some(Float64));
        assert_eq!(st(UInt16, UInt64), Some(UInt64));
        assert_eq!(st(Int64, UInt8), Some(Int64));
        assert_eq!(st(String, Int64), None);
        assert_eq!(st(Null, String), Some(String));
        assert_eq!(st(Boolean, Int8), Some(Int8));
        assert_eq!(st(Float64, Float64), Some(Float64));
    }

    #[test]
    fn schema_keeps_order_and_replaces() {
        let mut s = Schema::new();
        assert_eq!(s.with_column("a".into(), DataType::Int64), None);
        assert_eq!(s.with_column("b".into(), DataType::String), None);
        assert_eq!(s.with_column("a".into(), DataType::Float64), Some(DataType::Int64));
        assert_eq!(s.len(), 2);
        assert_eq!(s.index_of("b"), Some(1));
        assert_eq!(s.get("a"), Some(&DataType::Float64));
        assert_eq!(s.iter_names().map(|n| n.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(<i64 as NativeType>::DTYPE, DataType::Int64);
        assert_eq!(Float32Type::get_dtype(), DataType::Float32);
    }
}
