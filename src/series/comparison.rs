//! Comparisons on [`Series`].

use super::{Series, SeriesInner, for_numeric};
use crate::chunked_array::{BooleanChunked, ChunkCompareEq, ChunkCompareIneq, NumericScalar, UInt8Chunked, Wide};
use crate::datatypes::DataType;
use crate::error::PolarsResult;
use crate::kernels::CmpOp;
use crate::polars_bail;

impl Series {
    /// Compare every row against a number.
    pub fn compare_scalar(&self, op: CmpOp, rhs: impl NumericScalar) -> PolarsResult<BooleanChunked> {
        self.compare_wide(op, rhs.wide())
    }

    pub(crate) fn compare_wide(&self, op: CmpOp, rhs: Wide) -> PolarsResult<BooleanChunked> {
        for_numeric!(self, ca => {
            let (words, count) = ca.cmp_scalar_words(op, rhs);
            Ok(BooleanChunked::from_bitmap(
                ca.name().clone(),
                crate::memory::Bitmap::from_words_with_count(words, ca.len(), count),
                ca.validity().cloned(),
            ))
        }, other => match other {
            SeriesInner::Boolean(ca) => {
                // Booleans compare as 0 / 1.
                let ints = UInt8Chunked::from_iter_options(ca.name().clone(), ca.iter().map(|b| b.map(u8::from)));
                let (words, count) = ints.cmp_scalar_words(op, rhs);
                Ok(BooleanChunked::from_bitmap(
                    ca.name().clone(),
                    crate::memory::Bitmap::from_words_with_count(words, ca.len(), count),
                    ca.validity().cloned(),
                ))
            }
            _ => polars_bail!(InvalidOperation: "cannot compare a `{}` column with a number", self.dtype()),
        })
    }

    /// Compare every row against a string.
    pub fn compare_str(&self, op: CmpOp, rhs: &str) -> PolarsResult<BooleanChunked> {
        match &self.0 {
            SeriesInner::String(ca) => Ok(ca.compare_scalar(op, rhs)),
            _ => polars_bail!(InvalidOperation: "cannot compare a `{}` column with a string", self.dtype()),
        }
    }

    /// Compare row-wise against another column. Numeric columns of different
    /// types are compared in their common supertype.
    pub fn compare(&self, op: CmpOp, rhs: &Series) -> PolarsResult<BooleanChunked> {
        if self.len() != rhs.len() && self.len() != 1 && rhs.len() != 1 {
            polars_bail!(ShapeMismatch: "cannot compare series of lengths {} and {}", self.len(), rhs.len());
        }
        match (&self.0, &rhs.0) {
            (SeriesInner::String(a), SeriesInner::String(b)) => {
                if a.len() == 1 && b.len() != 1 {
                    return Ok(b.compare(op.swap_operands(), a).with_name(a.name().clone()));
                }
                Ok(a.compare(op, b))
            }
            (SeriesInner::Boolean(a), SeriesInner::Boolean(b)) if matches!(op, CmpOp::Eq | CmpOp::Ne) && a.len() == b.len() => {
                Ok(if op == CmpOp::Eq { a.equal(b) } else { a.not_equal(b) })
            }
            _ => {
                let (l, r) = (self.dtype(), rhs.dtype());
                let numericish = |d: &DataType| d.is_numeric() || d.is_bool();
                if !numericish(l) || !numericish(r) {
                    polars_bail!(InvalidOperation: "cannot compare `{}` with `{}`", l, r);
                }
                let common = if l.is_bool() && r.is_bool() {
                    DataType::UInt8
                } else {
                    DataType::supertype(l, r).expect("numeric types have a supertype")
                };
                let (a, b) = (self.cast(&common)?, rhs.cast(&common)?);
                Ok(match (&a.0, &b.0) {
                    (SeriesInner::UInt8(x), SeriesInner::UInt8(y)) => x.compare(op, y),
                    (SeriesInner::UInt16(x), SeriesInner::UInt16(y)) => x.compare(op, y),
                    (SeriesInner::UInt32(x), SeriesInner::UInt32(y)) => x.compare(op, y),
                    (SeriesInner::UInt64(x), SeriesInner::UInt64(y)) => x.compare(op, y),
                    (SeriesInner::Int8(x), SeriesInner::Int8(y)) => x.compare(op, y),
                    (SeriesInner::Int16(x), SeriesInner::Int16(y)) => x.compare(op, y),
                    (SeriesInner::Int32(x), SeriesInner::Int32(y)) => x.compare(op, y),
                    (SeriesInner::Int64(x), SeriesInner::Int64(y)) => x.compare(op, y),
                    (SeriesInner::Float32(x), SeriesInner::Float32(y)) => x.compare(op, y),
                    (SeriesInner::Float64(x), SeriesInner::Float64(y)) => x.compare(op, y),
                    _ => unreachable!("both sides were cast to the same numeric type"),
                })
            }
        }
    }
}

impl<R: NumericScalar> ChunkCompareEq<R> for Series {
    type Item = PolarsResult<BooleanChunked>;
    fn equal(&self, rhs: R) -> Self::Item {
        self.compare_scalar(CmpOp::Eq, rhs)
    }
    fn not_equal(&self, rhs: R) -> Self::Item {
        self.compare_scalar(CmpOp::Ne, rhs)
    }
    fn equal_missing(&self, rhs: R) -> Self::Item {
        Ok(self.compare_scalar(CmpOp::Eq, rhs)?.fill_null_with_values(false))
    }
    fn not_equal_missing(&self, rhs: R) -> Self::Item {
        Ok(self.compare_scalar(CmpOp::Ne, rhs)?.fill_null_with_values(true))
    }
}

impl<R: NumericScalar> ChunkCompareIneq<R> for Series {
    type Item = PolarsResult<BooleanChunked>;
    fn gt(&self, rhs: R) -> Self::Item {
        self.compare_scalar(CmpOp::Gt, rhs)
    }
    fn gt_eq(&self, rhs: R) -> Self::Item {
        self.compare_scalar(CmpOp::Ge, rhs)
    }
    fn lt(&self, rhs: R) -> Self::Item {
        self.compare_scalar(CmpOp::Lt, rhs)
    }
    fn lt_eq(&self, rhs: R) -> Self::Item {
        self.compare_scalar(CmpOp::Le, rhs)
    }
}

impl ChunkCompareEq<&str> for Series {
    type Item = PolarsResult<BooleanChunked>;
    fn equal(&self, rhs: &str) -> Self::Item {
        self.compare_str(CmpOp::Eq, rhs)
    }
    fn not_equal(&self, rhs: &str) -> Self::Item {
        self.compare_str(CmpOp::Ne, rhs)
    }
    fn equal_missing(&self, rhs: &str) -> Self::Item {
        Ok(self.compare_str(CmpOp::Eq, rhs)?.fill_null_with_values(false))
    }
    fn not_equal_missing(&self, rhs: &str) -> Self::Item {
        Ok(self.compare_str(CmpOp::Ne, rhs)?.fill_null_with_values(true))
    }
}

impl ChunkCompareIneq<&str> for Series {
    type Item = PolarsResult<BooleanChunked>;
    fn gt(&self, rhs: &str) -> Self::Item {
        self.compare_str(CmpOp::Gt, rhs)
    }
    fn gt_eq(&self, rhs: &str) -> Self::Item {
        self.compare_str(CmpOp::Ge, rhs)
    }
    fn lt(&self, rhs: &str) -> Self::Item {
        self.compare_str(CmpOp::Lt, rhs)
    }
    fn lt_eq(&self, rhs: &str) -> Self::Item {
        self.compare_str(CmpOp::Le, rhs)
    }
}

impl ChunkCompareEq<&Series> for Series {
    type Item = PolarsResult<BooleanChunked>;
    fn equal(&self, rhs: &Series) -> Self::Item {
        self.compare(CmpOp::Eq, rhs)
    }
    fn not_equal(&self, rhs: &Series) -> Self::Item {
        self.compare(CmpOp::Ne, rhs)
    }
    fn equal_missing(&self, rhs: &Series) -> Self::Item {
        let eq = self.compare(CmpOp::Eq, rhs)?.fill_null_with_values(false);
        let both_null = &self.is_null() & &rhs.is_null();
        Ok(&eq | &both_null)
    }
    fn not_equal_missing(&self, rhs: &Series) -> Self::Item {
        Ok(!&self.equal_missing(rhs)?)
    }
}

impl ChunkCompareIneq<&Series> for Series {
    type Item = PolarsResult<BooleanChunked>;
    fn gt(&self, rhs: &Series) -> Self::Item {
        self.compare(CmpOp::Gt, rhs)
    }
    fn gt_eq(&self, rhs: &Series) -> Self::Item {
        self.compare(CmpOp::Ge, rhs)
    }
    fn lt(&self, rhs: &Series) -> Self::Item {
        self.compare(CmpOp::Lt, rhs)
    }
    fn lt_eq(&self, rhs: &Series) -> Self::Item {
        self.compare(CmpOp::Le, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_string_and_series_comparisons() {
        let s = Series::new("a".into(), [Some(1i64), None, Some(3)]);
        assert_eq!(s.gt(1).unwrap().to_vec(), vec![Some(false), None, Some(true)]);
        assert_eq!(s.equal(3.0).unwrap().to_vec(), vec![Some(false), None, Some(true)]);
        assert_eq!(s.not_equal_missing(1).unwrap().to_vec(), vec![Some(false), Some(true), Some(true)]);
        assert!(s.equal("x").is_err());

        let t = Series::new("s".into(), ["a", "b"]);
        assert_eq!(t.equal("b").unwrap().to_vec(), vec![Some(false), Some(true)]);
        assert_eq!(t.lt("b").unwrap().to_vec(), vec![Some(true), Some(false)]);
        assert!(t.gt(1).is_err());
        assert!(t.equal(&s).is_err());

        // Different numeric types meet in their supertype.
        let f = Series::new("f".into(), [1.0f64, 2.5, 3.0]);
        assert_eq!(s.equal(&f).unwrap().to_vec(), vec![Some(true), None, Some(true)]);
        assert_eq!(s.lt(&f).unwrap().to_vec(), vec![Some(false), None, Some(false)]);
        assert_eq!(s.equal_missing(&s).unwrap().to_vec(), vec![Some(true), Some(true), Some(true)]);
        assert!(s.equal(&Series::new("x".into(), [1i64, 2])).is_err());
        let b = Series::new("b".into(), [true, false]);
        assert_eq!(b.equal(&b).unwrap().to_vec(), vec![Some(true), Some(true)]);
        assert_eq!(b.gt(0).unwrap().to_vec(), vec![Some(true), Some(false)]);
        assert_eq!(t.equal(&Series::new("o".into(), ["a"])).unwrap().to_vec(), vec![Some(true), Some(false)]);
    }
}
