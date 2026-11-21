//! [`AnyValue`]: one dynamically typed cell.

use std::cmp::Ordering;
use std::fmt;

use super::{DataType, NativeType, PlSmallStr};
use crate::error::{PolarsError, PolarsResult};

/// A single value of any [`DataType`], as returned by row-wise access
/// (`Series::get`, `DataFrame::get_row`, ...).
///
/// Row-wise access is the slow lane by design: use the typed accessors
/// (`Series::i64`, `ChunkedArray::cont_slice`) for anything hot.
#[derive(Clone, Debug)]
pub enum AnyValue<'a> {
    /// A missing value.
    Null,
    /// A boolean.
    Boolean(bool),
    /// An unsigned 8-bit integer.
    UInt8(u8),
    /// An unsigned 16-bit integer.
    UInt16(u16),
    /// An unsigned 32-bit integer.
    UInt32(u32),
    /// An unsigned 64-bit integer.
    UInt64(u64),
    /// A signed 8-bit integer.
    Int8(i8),
    /// A signed 16-bit integer.
    Int16(i16),
    /// A signed 32-bit integer.
    Int32(i32),
    /// A signed 64-bit integer.
    Int64(i64),
    /// A 32-bit float.
    Float32(f32),
    /// A 64-bit float.
    Float64(f64),
    /// A string borrowed from a column.
    String(&'a str),
    /// An owned string.
    StringOwned(PlSmallStr),
}

impl<'a> AnyValue<'a> {
    /// The type of the value (`Null` for a missing value).
    pub fn dtype(&self) -> DataType {
        match self {
            AnyValue::Null => DataType::Null,
            AnyValue::Boolean(_) => DataType::Boolean,
            AnyValue::UInt8(_) => DataType::UInt8,
            AnyValue::UInt16(_) => DataType::UInt16,
            AnyValue::UInt32(_) => DataType::UInt32,
            AnyValue::UInt64(_) => DataType::UInt64,
            AnyValue::Int8(_) => DataType::Int8,
            AnyValue::Int16(_) => DataType::Int16,
            AnyValue::Int32(_) => DataType::Int32,
            AnyValue::Int64(_) => DataType::Int64,
            AnyValue::Float32(_) => DataType::Float32,
            AnyValue::Float64(_) => DataType::Float64,
            AnyValue::String(_) | AnyValue::StringOwned(_) => DataType::String,
        }
    }

    /// `true` for [`AnyValue::Null`].
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, AnyValue::Null)
    }

    /// The value as `f64`, if it is numeric or boolean.
    pub fn to_f64(&self) -> Option<f64> {
        Some(match self {
            AnyValue::Boolean(b) => *b as u8 as f64,
            AnyValue::UInt8(v) => *v as f64,
            AnyValue::UInt16(v) => *v as f64,
            AnyValue::UInt32(v) => *v as f64,
            AnyValue::UInt64(v) => *v as f64,
            AnyValue::Int8(v) => *v as f64,
            AnyValue::Int16(v) => *v as f64,
            AnyValue::Int32(v) => *v as f64,
            AnyValue::Int64(v) => *v as f64,
            AnyValue::Float32(v) => *v as f64,
            AnyValue::Float64(v) => *v,
            _ => return None,
        })
    }

    /// The value as `i128`, if it is an integer or boolean.
    pub fn to_i128(&self) -> Option<i128> {
        Some(match self {
            AnyValue::Boolean(b) => *b as i128,
            AnyValue::UInt8(v) => *v as i128,
            AnyValue::UInt16(v) => *v as i128,
            AnyValue::UInt32(v) => *v as i128,
            AnyValue::UInt64(v) => *v as i128,
            AnyValue::Int8(v) => *v as i128,
            AnyValue::Int16(v) => *v as i128,
            AnyValue::Int32(v) => *v as i128,
            AnyValue::Int64(v) => *v as i128,
            _ => return None,
        })
    }

    /// Cast a numeric or boolean value to `T` (`None` for nulls and strings).
    ///
    /// Uses Rust `as` semantics, like Polars' `extract`.
    pub fn extract<T: NativeType>(&self) -> Option<T> {
        match self {
            AnyValue::Float32(_) | AnyValue::Float64(_) => self.to_f64().map(T::from_f64),
            AnyValue::UInt64(v) if T::IS_FLOAT => Some(T::from_f64(*v as f64)),
            AnyValue::UInt64(v) => Some(T::from_i64(*v as i64)),
            _ => {
                let v = self.to_i128()?;
                Some(if T::IS_FLOAT { T::from_f64(v as f64) } else { T::from_i64(v as i64) })
            }
        }
    }

    /// Like [`extract`](Self::extract), with an error instead of `None`.
    pub fn try_extract<T: NativeType>(&self) -> PolarsResult<T> {
        self.extract().ok_or_else(|| {
            PolarsError::ComputeError(format!("could not extract number from any-value of dtype: '{}'", self.dtype()).into())
        })
    }

    /// The string, if this is a string value.
    pub fn get_str(&self) -> Option<&str> {
        match self {
            AnyValue::String(s) => Some(s),
            AnyValue::StringOwned(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The boolean, if this is a boolean value.
    pub fn get_bool(&self) -> Option<bool> {
        match self {
            AnyValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Detach from the column the value was read from.
    pub fn into_static(self) -> AnyValue<'static> {
        match self {
            AnyValue::Null => AnyValue::Null,
            AnyValue::Boolean(v) => AnyValue::Boolean(v),
            AnyValue::UInt8(v) => AnyValue::UInt8(v),
            AnyValue::UInt16(v) => AnyValue::UInt16(v),
            AnyValue::UInt32(v) => AnyValue::UInt32(v),
            AnyValue::UInt64(v) => AnyValue::UInt64(v),
            AnyValue::Int8(v) => AnyValue::Int8(v),
            AnyValue::Int16(v) => AnyValue::Int16(v),
            AnyValue::Int32(v) => AnyValue::Int32(v),
            AnyValue::Int64(v) => AnyValue::Int64(v),
            AnyValue::Float32(v) => AnyValue::Float32(v),
            AnyValue::Float64(v) => AnyValue::Float64(v),
            AnyValue::String(s) => AnyValue::StringOwned(s.into()),
            AnyValue::StringOwned(s) => AnyValue::StringOwned(s),
        }
    }

    /// Total order across values: nulls first, then numbers (compared by
    /// value across numeric types, NaN last), then strings.
    pub fn total_cmp(&self, other: &AnyValue<'_>) -> Ordering {
        fn rank(v: &AnyValue<'_>) -> u8 {
            match v {
                AnyValue::Null => 0,
                AnyValue::String(_) | AnyValue::StringOwned(_) => 2,
                _ => 1,
            }
        }
        match (rank(self), rank(other)) {
            (0, 0) => Ordering::Equal,
            (2, 2) => self.get_str().cmp(&other.get_str()),
            (1, 1) => match (self.to_i128(), other.to_i128()) {
                (Some(a), Some(b)) => a.cmp(&b),
                _ => {
                    let (a, b) = (self.to_f64().unwrap_or(f64::NAN), other.to_f64().unwrap_or(f64::NAN));
                    match (a.is_nan(), b.is_nan()) {
                        (true, true) => Ordering::Equal,
                        (true, false) => Ordering::Greater,
                        (false, true) => Ordering::Less,
                        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
                    }
                }
            },
            (a, b) => a.cmp(&b),
        }
    }
}

impl PartialEq for AnyValue<'_> {
    /// Values are equal when they have the same kind and value; numbers of
    /// different widths compare by value (`Int32(1) == Int64(1)`), strings
    /// compare by content regardless of ownership, and `Null == Null`.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AnyValue::Null, AnyValue::Null) => true,
            (AnyValue::Null, _) | (_, AnyValue::Null) => false,
            (AnyValue::Boolean(a), AnyValue::Boolean(b)) => a == b,
            (AnyValue::Boolean(_), _) | (_, AnyValue::Boolean(_)) => false,
            _ => match (self.get_str(), other.get_str()) {
                (Some(a), Some(b)) => a == b,
                (None, None) => match (self.to_i128(), other.to_i128()) {
                    (Some(a), Some(b)) => a == b,
                    _ => self.to_f64() == other.to_f64(),
                },
                _ => false,
            },
        }
    }
}

/// Format a float the way Polars prints cells: integral values keep one decimal.
pub(crate) fn fmt_float(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "inf".to_string() } else { "-inf".to_string() }
    } else if v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{v:.1}")
    } else if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e16) {
        format!("{v:e}")
    } else {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0');
        if s.ends_with('.') { format!("{s}0") } else { s.to_string() }
    }
}

impl fmt::Display for AnyValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnyValue::Null => f.write_str("null"),
            AnyValue::Boolean(v) => write!(f, "{v}"),
            AnyValue::UInt8(v) => write!(f, "{v}"),
            AnyValue::UInt16(v) => write!(f, "{v}"),
            AnyValue::UInt32(v) => write!(f, "{v}"),
            AnyValue::UInt64(v) => write!(f, "{v}"),
            AnyValue::Int8(v) => write!(f, "{v}"),
            AnyValue::Int16(v) => write!(f, "{v}"),
            AnyValue::Int32(v) => write!(f, "{v}"),
            AnyValue::Int64(v) => write!(f, "{v}"),
            AnyValue::Float32(v) => f.write_str(&fmt_float(*v as f64)),
            AnyValue::Float64(v) => f.write_str(&fmt_float(*v)),
            AnyValue::String(s) => write!(f, "\"{s}\""),
            AnyValue::StringOwned(s) => write!(f, "\"{s}\""),
        }
    }
}

macro_rules! any_from {
    ($($t:ty => $variant:ident),*) => {$(
        impl From<$t> for AnyValue<'_> {
            #[inline]
            fn from(v: $t) -> Self {
                AnyValue::$variant(v)
            }
        }
    )*};
}
any_from!(bool => Boolean, u8 => UInt8, u16 => UInt16, u32 => UInt32, u64 => UInt64,
          i8 => Int8, i16 => Int16, i32 => Int32, i64 => Int64, f32 => Float32, f64 => Float64);

impl<'a> From<&'a str> for AnyValue<'a> {
    fn from(s: &'a str) -> Self {
        AnyValue::String(s)
    }
}

impl From<String> for AnyValue<'_> {
    fn from(s: String) -> Self {
        AnyValue::StringOwned(s.into())
    }
}

impl<'a, T: Into<AnyValue<'a>>> From<Option<T>> for AnyValue<'a> {
    fn from(v: Option<T>) -> Self {
        v.map_or(AnyValue::Null, Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_and_extraction() {
        assert_eq!(AnyValue::Int32(1), AnyValue::Int64(1));
        assert_eq!(AnyValue::Float64(2.0), AnyValue::Int8(2));
        assert_ne!(AnyValue::Int32(1), AnyValue::Null);
        assert_eq!(AnyValue::Null, AnyValue::Null);
        assert_eq!(AnyValue::String("a"), AnyValue::StringOwned("a".into()));
        assert_ne!(AnyValue::String("1"), AnyValue::Int32(1));
        assert_eq!(AnyValue::Int64(-3).extract::<f64>(), Some(-3.0));
        assert_eq!(AnyValue::Float64(2.9).extract::<i32>(), Some(2));
        assert_eq!(AnyValue::Boolean(true).extract::<u8>(), Some(1));
        assert_eq!(AnyValue::String("x").extract::<i64>(), None);
        assert!(AnyValue::Null.try_extract::<i64>().is_err());
        assert_eq!(AnyValue::from(Some(3i64)), AnyValue::Int64(3));
        assert_eq!(AnyValue::from(None::<i64>), AnyValue::Null);
    }

    #[test]
    fn display_matches_polars_conventions() {
        assert_eq!(AnyValue::Float64(1.0).to_string(), "1.0");
        assert_eq!(AnyValue::Float64(0.5).to_string(), "0.5");
        assert_eq!(AnyValue::Float64(1.0 / 3.0).to_string(), "0.333333");
        assert_eq!(AnyValue::Float64(f64::NAN).to_string(), "NaN");
        assert_eq!(AnyValue::Int64(-7).to_string(), "-7");
        assert_eq!(AnyValue::String("hi").to_string(), "\"hi\"");
        assert_eq!(AnyValue::Null.to_string(), "null");
    }

    #[test]
    fn total_order_puts_nulls_first_and_nan_last() {
        let mut v = vec![AnyValue::Float64(f64::NAN), AnyValue::Int64(3), AnyValue::Null, AnyValue::Float64(-1.5), AnyValue::String("a")];
        v.sort_by(|a, b| a.total_cmp(b));
        assert!(v[0].is_null());
        assert_eq!(v[1], AnyValue::Float64(-1.5));
        assert_eq!(v[2], AnyValue::Int64(3));
        assert!(matches!(v[3], AnyValue::Float64(x) if x.is_nan()));
        assert_eq!(v[4], AnyValue::String("a"));
    }
}
