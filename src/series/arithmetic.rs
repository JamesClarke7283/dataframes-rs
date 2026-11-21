//! Arithmetic on [`Series`].
//!
//! As in Polars, `series ∘ series` returns a `PolarsResult` (the types may be
//! incompatible) while `series ∘ scalar` returns a `Series` and casts the
//! scalar to the column type.

use std::ops::{Add, Div, Mul, Rem, Sub};

use super::{IntoSeries, Series, SeriesInner, for_numeric};
use crate::chunked_array::{NumericScalar, StringChunked, Wide};
use crate::datatypes::DataType;
use crate::error::PolarsResult;
use crate::polars_bail;

/// An arithmetic operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArithOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/` (truncating for integers; a zero divisor yields null)
    Div,
    /// `%`
    Rem,
}

impl ArithOp {
    /// The operator's symbol.
    pub fn symbol(self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Div => "/",
            ArithOp::Rem => "%",
        }
    }
}

macro_rules! apply_op {
    ($op:expr, $a:expr, $b:expr) => {
        match $op {
            ArithOp::Add => $a + $b,
            ArithOp::Sub => $a - $b,
            ArithOp::Mul => $a * $b,
            ArithOp::Div => $a / $b,
            ArithOp::Rem => $a % $b,
        }
    };
}

impl Series {
    /// `self ∘ rhs`, row by row, in the common supertype of both columns.
    pub fn arithmetic(&self, op: ArithOp, rhs: &Series) -> PolarsResult<Series> {
        if self.len() != rhs.len() && self.len() != 1 && rhs.len() != 1 {
            polars_bail!(ShapeMismatch: "cannot do arithmetic on series of lengths {} and {}", self.len(), rhs.len());
        }
        if let (SeriesInner::String(a), SeriesInner::String(b), ArithOp::Add) = (&self.0, &rhs.0, op) {
            let n = a.len().max(b.len());
            let pick = |ca: &'_ StringChunked, i: usize| ca.get(if ca.len() == 1 { 0 } else { i }).map(str::to_owned);
            let joined = (0..n).map(|i| match (pick(a, i), pick(b, i)) {
                (Some(x), Some(y)) => Some(x + &y),
                _ => None,
            });
            return Ok(StringChunked::from_iter_options(a.name().clone(), joined).into_series());
        }
        let (l, r) = (self.dtype(), rhs.dtype());
        if !l.is_numeric() || !r.is_numeric() {
            polars_bail!(InvalidOperation: "arithmetic `{}` is not supported between `{}` and `{}`", op.symbol(), l, r);
        }
        let common = DataType::supertype(l, r).expect("numeric types have a supertype");
        let (a, b) = (self.cast(&common)?, rhs.cast(&common)?);
        Ok(match (&a.0, &b.0) {
            (SeriesInner::UInt8(x), SeriesInner::UInt8(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::UInt16(x), SeriesInner::UInt16(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::UInt32(x), SeriesInner::UInt32(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::UInt64(x), SeriesInner::UInt64(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::Int8(x), SeriesInner::Int8(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::Int16(x), SeriesInner::Int16(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::Int32(x), SeriesInner::Int32(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::Int64(x), SeriesInner::Int64(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::Float32(x), SeriesInner::Float32(y)) => apply_op!(op, x, y).into_series(),
            (SeriesInner::Float64(x), SeriesInner::Float64(y)) => apply_op!(op, x, y).into_series(),
            _ => unreachable!("both sides were cast to the same numeric type"),
        })
    }

    /// `self ∘ rhs` for a scalar, which is cast to the column type.
    pub fn arithmetic_scalar(&self, op: ArithOp, rhs: impl NumericScalar) -> PolarsResult<Series> {
        let rhs = rhs.wide();
        for_numeric!(self, ca => Ok(match rhs {
            Wide::Int(i) => {
                // `i128` is not itself a scalar type; narrow through `i64`/`u64`.
                if i > i64::MAX as i128 { apply_op!(op, ca, i as u64).into_series() } else { apply_op!(op, ca, i as i64).into_series() }
            }
            Wide::Float(f) => apply_op!(op, ca, f).into_series(),
        }), _ => polars_bail!(InvalidOperation: "arithmetic `{}` is not supported for dtype `{}`", op.symbol(), self.dtype()))
    }

    /// True division: both sides are cast to `Float64` first (what `/` means
    /// in pandas and in Python Polars).
    pub fn true_div(&self, rhs: &Series) -> PolarsResult<Series> {
        self.cast(&DataType::Float64)?.arithmetic(ArithOp::Div, &rhs.cast(&DataType::Float64)?)
    }
}

macro_rules! series_ops {
    ($($trait:ident, $method:ident, $op:ident;)*) => {$(
        impl $trait<&Series> for &Series {
            type Output = PolarsResult<Series>;
            fn $method(self, rhs: &Series) -> PolarsResult<Series> {
                self.arithmetic(ArithOp::$op, rhs)
            }
        }
        impl $trait<Series> for Series {
            type Output = PolarsResult<Series>;
            fn $method(self, rhs: Series) -> PolarsResult<Series> {
                self.arithmetic(ArithOp::$op, &rhs)
            }
        }
        impl<R: NumericScalar> $trait<R> for &Series {
            type Output = Series;
            /// # Panics
            /// If the column is not numeric.
            fn $method(self, rhs: R) -> Series {
                self.arithmetic_scalar(ArithOp::$op, rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }
        impl<R: NumericScalar> $trait<R> for Series {
            type Output = Series;
            /// # Panics
            /// If the column is not numeric.
            fn $method(self, rhs: R) -> Series {
                (&self).$method(rhs)
            }
        }
    )*};
}
series_ops! { Add, add, Add; Sub, sub, Sub; Mul, mul, Mul; Div, div, Div; Rem, rem, Rem; }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_arithmetic_uses_supertypes() {
        let a = Series::new("a".into(), [1i32, 2, 3]);
        let b = Series::new("b".into(), [0.5f64, 1.5, 2.5]);
        let sum = (&a + &b).unwrap();
        assert_eq!(sum.name(), "a");
        assert_eq!(sum.f64().unwrap().cont_slice().unwrap(), &[1.5, 3.5, 5.5]);
        let prod = (&a * &a).unwrap();
        assert_eq!(prod.i32().unwrap().cont_slice().unwrap(), &[1, 4, 9]);
        let wide = (&a - &Series::new("c".into(), [10i64])).unwrap();
        assert_eq!(wide.i64().unwrap().cont_slice().unwrap(), &[-9, -8, -7]);
        let div = (&a / &Series::new("z".into(), [1i32, 0, 2])).unwrap();
        assert_eq!(div.i32().unwrap().to_vec(), vec![Some(1), None, Some(1)]);
        assert_eq!(a.true_div(&Series::new("d".into(), [2i32])).unwrap().f64().unwrap().cont_slice().unwrap(), &[0.5, 1.0, 1.5]);
        assert!((&a + &Series::new("s".into(), ["x", "y", "z"])).is_err());
        assert!((&a + &Series::new("s".into(), [1i32, 2])).is_err());
        let s = Series::new("s".into(), [Some("ab"), None]);
        let joined = (&s + &Series::new("t".into(), ["c"])).unwrap();
        assert_eq!(joined.str().unwrap().to_vec(), vec![Some("abc"), None]);
    }

    #[test]
    fn scalar_arithmetic_keeps_the_column_type() {
        let a = Series::new("a".into(), [1i32, 2, 3]);
        assert_eq!((&a + 1).i32().unwrap().cont_slice().unwrap(), &[2, 3, 4]);
        assert_eq!((&a * 2.9).i32().unwrap().cont_slice().unwrap(), &[2, 4, 6]);
        assert_eq!((&a % 2).i32().unwrap().cont_slice().unwrap(), &[1, 0, 1]);
        let f = Series::new("f".into(), [1.0f64, 2.0]);
        assert_eq!((f / 4).f64().unwrap().cont_slice().unwrap(), &[0.25, 0.5]);
        assert!(Series::new("s".into(), ["x"]).arithmetic_scalar(ArithOp::Add, 1).is_err());
    }
}
