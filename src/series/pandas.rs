//! Element-wise helpers and the pandas vocabulary for [`Series`].
//!
//! Where pandas and Polars use different names for the same operation both
//! are available (`isna` / `is_null`, `fillna` / `fill_null_with_value`,
//! `nunique` / `n_unique`, `cumsum` / `cum_sum`, `isin` / `is_in`, ...).

use super::{IntoSeries, Series, SeriesInner, for_numeric};
use crate::chunked_array::{BooleanChunked, ChunkedArray, NumericScalar, SortOptions, Wide};
use crate::datatypes::{AnyValue, DataType, NativeType, PolarsNumericType};
use crate::error::PolarsResult;
use crate::kernels::aggregate::Num;
use crate::kernels::{CmpOp, Kernel};
use crate::polars_bail;

fn cumulative<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    reverse: bool,
    f: impl Fn(T::Native, T::Native) -> T::Native,
) -> ChunkedArray<T> {
    let mut acc: Option<T::Native> = None;
    let mut step = |v: Option<T::Native>| {
        v.map(|v| {
            let next = acc.map_or(v, |a| f(a, v));
            acc = Some(next);
            next
        })
    };
    if reverse {
        let mut out: Vec<Option<T::Native>> = ca.iter().rev().map(&mut step).collect();
        out.reverse();
        ChunkedArray::from_iter_options(ca.name().clone(), out)
    } else {
        ChunkedArray::from_iter_options(ca.name().clone(), ca.iter().map(step))
    }
}

impl Series {
    // ----- shape ------------------------------------------------------------

    /// `(len,)`, as in pandas.
    pub fn shape(&self) -> (usize,) {
        (self.len(),)
    }

    /// Number of rows, as in pandas.
    pub fn size(&self) -> usize {
        self.len()
    }

    /// Number of non-null rows.
    pub fn count(&self) -> usize {
        self.len() - self.null_count()
    }

    /// A handle to the same data (columns are immutable and shared, so no
    /// bytes are copied; mutation through either handle is copy-on-write).
    pub fn copy(&self) -> Series {
        self.clone()
    }

    /// The rows as dynamically typed values.
    pub fn to_list(&self) -> Vec<AnyValue<'_>> {
        self.iter().collect()
    }

    // ----- nulls ------------------------------------------------------------

    /// pandas name for [`is_null`](Series::is_null).
    pub fn isna(&self) -> BooleanChunked {
        self.is_null()
    }

    /// pandas name for [`is_null`](Series::is_null).
    pub fn isnull(&self) -> BooleanChunked {
        self.is_null()
    }

    /// pandas name for [`is_not_null`](Series::is_not_null).
    pub fn notna(&self) -> BooleanChunked {
        self.is_not_null()
    }

    /// pandas name for [`is_not_null`](Series::is_not_null).
    pub fn notnull(&self) -> BooleanChunked {
        self.is_not_null()
    }

    /// pandas name for [`fill_null_with_value`](Series::fill_null_with_value).
    pub fn fillna<'a>(&self, value: impl Into<AnyValue<'a>>) -> PolarsResult<Series> {
        self.fill_null_with_value(value)
    }

    /// pandas name for [`drop_nulls`](Series::drop_nulls).
    pub fn dropna(&self) -> Series {
        self.drop_nulls()
    }

    // ----- statistics -------------------------------------------------------

    /// pandas name for [`n_unique`](Series::n_unique), except that nulls are
    /// not counted (pandas' default `dropna=True`).
    pub fn nunique(&self) -> PolarsResult<usize> {
        Ok(self.n_unique()? - self.has_nulls() as usize)
    }

    /// pandas name for [`arg_min`](Series::arg_min).
    pub fn idxmin(&self) -> Option<usize> {
        self.arg_min()
    }

    /// pandas name for [`arg_max`](Series::arg_max).
    pub fn idxmax(&self) -> Option<usize> {
        self.arg_max()
    }

    /// Product of the non-null values as `f64`.
    pub fn product(&self) -> PolarsResult<f64> {
        for_numeric!(self, ca => Ok(ca.iter().flatten().map(|v| v.to_f64()).product()), _ => {
            polars_bail!(InvalidOperation: "`product` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// `true` if any value of a boolean column is `true`.
    pub fn any(&self) -> PolarsResult<bool> {
        Ok(self.bool()?.any())
    }

    /// `true` if every value of a boolean column is `true`.
    pub fn all(&self) -> PolarsResult<bool> {
        Ok(self.bool()?.all())
    }

    // ----- types / order / access -------------------------------------------

    /// pandas name for [`cast`](Series::cast).
    pub fn astype(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast(dtype)
    }

    /// pandas-style sort: `ascending` instead of sort options; nulls go last,
    /// as in pandas.
    pub fn sort_values(&self, ascending: bool) -> PolarsResult<Series> {
        self.sort(SortOptions::default().with_order_descending(!ascending).with_nulls_last(true))
    }

    /// The `n` largest values, largest first.
    pub fn nlargest(&self, n: usize) -> PolarsResult<Series> {
        Ok(self.drop_nulls().sort(SortOptions::default().with_order_descending(true))?.head(Some(n)))
    }

    /// The `n` smallest values, smallest first.
    pub fn nsmallest(&self, n: usize) -> PolarsResult<Series> {
        Ok(self.drop_nulls().sort(SortOptions::default())?.head(Some(n)))
    }

    /// The value at position `index`; negative positions count from the end.
    pub fn iloc(&self, index: i64) -> PolarsResult<AnyValue<'_>> {
        let n = self.len() as i64;
        let i = if index < 0 { index + n } else { index };
        if i < 0 || i >= n {
            polars_bail!(OutOfBounds: "index {} is out of bounds for series of length {}", index, n);
        }
        self.get(i as usize)
    }

    /// The value at position `index`.
    pub fn iat(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        self.get(index)
    }

    // ----- membership / ranges ----------------------------------------------

    /// `true` where the value occurs in `other`. Nulls stay null.
    pub fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        fn numeric<T: PolarsNumericType>(ca: &ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked> {
            if !other.dtype().is_numeric() && !other.dtype().is_bool() {
                polars_bail!(InvalidOperation: "cannot check membership of `{}` values in a `{}` column", ca.dtype(), other.dtype());
            }
            // Compare in f64 / i128 space so that mixed types behave mathematically.
            let mut keys: Vec<AnyValue<'_>> = other.iter().filter(|v| !v.is_null()).collect();
            keys.sort_by(|a, b| a.total_cmp(b));
            let found = |v: T::Native| {
                let probe = v.into_any_value();
                keys.binary_search_by(|k| k.total_cmp(&probe)).is_ok()
            };
            Ok(BooleanChunked::from_bitmap(
                ca.name().clone(),
                ca.values().iter().map(|&v| found(v)).collect(),
                ca.validity().cloned(),
            ))
        }
        for_numeric!(self, ca => numeric(ca, other), this => match (this, &other.0) {
            (SeriesInner::String(ca), SeriesInner::String(set)) => {
                let set: std::collections::HashSet<&str> = set.iter().flatten().collect();
                Ok(BooleanChunked::from_bitmap(
                    ca.name().clone(),
                    ca.into_no_null_iter().map(|s| set.contains(s)).collect(),
                    ca.validity().cloned(),
                ))
            }
            (SeriesInner::Boolean(ca), SeriesInner::Boolean(set)) => {
                let (has_true, has_false) = (set.iter().any(|v| v == Some(true)), set.iter().any(|v| v == Some(false)));
                Ok(BooleanChunked::from_iter_options(
                    ca.name().clone(),
                    ca.iter().map(|v| v.map(|v| if v { has_true } else { has_false })),
                ))
            }
            _ => polars_bail!(InvalidOperation: "cannot check membership of `{}` values in a `{}` column", self.dtype(), other.dtype()),
        })
    }

    /// pandas name for [`is_in`](Series::is_in).
    pub fn isin(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        self.is_in(other)
    }

    /// `lower <= value <= upper` (both ends inclusive).
    pub fn is_between(&self, lower: impl NumericScalar, upper: impl NumericScalar) -> PolarsResult<BooleanChunked> {
        Ok(&self.compare_scalar(CmpOp::Ge, lower)? & &self.compare_scalar(CmpOp::Le, upper)?)
    }

    /// pandas name for [`is_between`](Series::is_between).
    pub fn between(&self, lower: impl NumericScalar, upper: impl NumericScalar) -> PolarsResult<BooleanChunked> {
        self.is_between(lower, upper)
    }

    // ----- element-wise -------------------------------------------------------

    /// Absolute value.
    pub fn abs(&self) -> PolarsResult<Series> {
        for_numeric!(self, ca => Ok(ca.apply_values(|v| if v < Num::ZERO { Num::sub(Num::ZERO, v) } else { v }).into_series()), _ => {
            polars_bail!(InvalidOperation: "`abs` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// Clamp the values into `lower..=upper` (the bounds are cast to the column type).
    pub fn clip(&self, lower: impl NumericScalar, upper: impl NumericScalar) -> PolarsResult<Series> {
        let (lo, hi) = (lower.wide(), upper.wide());
        fn run<T: PolarsNumericType>(ca: &ChunkedArray<T>, lo: Wide, hi: Wide) -> Series {
            let (lo, hi): (T::Native, T::Native) = (lo.cast(), hi.cast());
            ca.apply_values(move |v| {
                if v.tot_lt(lo) {
                    lo
                } else if hi.tot_lt(v) {
                    hi
                } else {
                    v
                }
            })
            .into_series()
        }
        for_numeric!(self, ca => Ok(run(ca, lo, hi)), _ => {
            polars_bail!(InvalidOperation: "`clip` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// Round floats to `decimals` decimal places (integers are returned unchanged).
    pub fn round(&self, decimals: u32) -> PolarsResult<Series> {
        let scale = 10f64.powi(decimals as i32);
        match &self.0 {
            SeriesInner::Float64(ca) => Ok(ca.apply_values(move |v| (v * scale).round() / scale).into_series()),
            SeriesInner::Float32(ca) => {
                Ok(ca.apply_values(move |v| ((v as f64 * scale).round() / scale) as f32).into_series())
            }
            _ if self.dtype().is_integer() => Ok(self.clone()),
            _ => polars_bail!(InvalidOperation: "`round` operation not supported for dtype `{}`", self.dtype()),
        }
    }

    /// Cumulative sum; nulls stay null and are skipped by the running total.
    pub fn cum_sum(&self, reverse: bool) -> PolarsResult<Series> {
        for_numeric!(self, ca => Ok(cumulative(ca, reverse, Num::add).into_series()), _ => {
            polars_bail!(InvalidOperation: "`cum_sum` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// Cumulative product.
    pub fn cum_prod(&self, reverse: bool) -> PolarsResult<Series> {
        for_numeric!(self, ca => Ok(cumulative(ca, reverse, Num::mul).into_series()), _ => {
            polars_bail!(InvalidOperation: "`cum_prod` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// Cumulative minimum.
    pub fn cum_min(&self, reverse: bool) -> PolarsResult<Series> {
        for_numeric!(self, ca => Ok(cumulative(ca, reverse, |a, b| if b.tot_lt(a) { b } else { a }).into_series()), _ => {
            polars_bail!(InvalidOperation: "`cum_min` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// Cumulative maximum.
    pub fn cum_max(&self, reverse: bool) -> PolarsResult<Series> {
        for_numeric!(self, ca => Ok(cumulative(ca, reverse, |a, b| if a.tot_lt(b) { b } else { a }).into_series()), _ => {
            polars_bail!(InvalidOperation: "`cum_max` operation not supported for dtype `{}`", self.dtype())
        })
    }

    /// pandas name for [`cum_sum`](Series::cum_sum).
    pub fn cumsum(&self) -> PolarsResult<Series> {
        self.cum_sum(false)
    }

    /// pandas name for [`cum_prod`](Series::cum_prod).
    pub fn cumprod(&self) -> PolarsResult<Series> {
        self.cum_prod(false)
    }

    /// pandas name for [`cum_min`](Series::cum_min).
    pub fn cummin(&self) -> PolarsResult<Series> {
        self.cum_min(false)
    }

    /// pandas name for [`cum_max`](Series::cum_max).
    pub fn cummax(&self) -> PolarsResult<Series> {
        self.cum_max(false)
    }

    /// Shift the rows by `periods` positions (down if positive), filling the
    /// vacated rows with nulls.
    pub fn shift(&self, periods: i64) -> Series {
        let n = self.len();
        let k = (periods.unsigned_abs() as usize).min(n);
        let nulls = Series::full_null(self.name().clone(), k, self.dtype());
        let mut out;
        if periods >= 0 {
            out = nulls;
            out.append(&self.slice(0, n - k)).expect("same dtype");
        } else {
            out = self.slice(k as i64, n - k);
            out.append(&nulls).expect("same dtype");
        }
        out
    }

    /// `value[i] - value[i - n]`; the first `n` rows are null.
    pub fn diff(&self, n: i64) -> PolarsResult<Series> {
        if !self.dtype().is_numeric() {
            polars_bail!(InvalidOperation: "`diff` operation not supported for dtype `{}`", self.dtype());
        }
        self.arithmetic(super::arithmetic::ArithOp::Sub, &self.shift(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pandas_aliases() {
        let s = Series::new("a".into(), [Some(3i64), None, Some(1), Some(3)]);
        assert_eq!(s.shape(), (4,));
        assert_eq!((s.size(), s.count()), (4, 3));
        assert_eq!(s.isna().to_vec(), s.is_null().to_vec());
        assert_eq!(s.notna().num_trues(), 3);
        assert_eq!(s.fillna(0).unwrap().i64().unwrap().cont_slice().unwrap(), &[3, 0, 1, 3]);
        assert_eq!(s.dropna().len(), 3);
        assert_eq!(s.nunique().unwrap(), 2);
        assert_eq!(s.n_unique().unwrap(), 3);
        assert_eq!((s.idxmin(), s.idxmax()), (Some(2), Some(0)));
        assert_eq!(*s.astype(&DataType::Float32).unwrap().dtype(), DataType::Float32);
        assert_eq!(s.sort_values(true).unwrap().i64().unwrap().to_vec(), vec![Some(1), Some(3), Some(3), None]);
        assert_eq!(s.sort_values(false).unwrap().i64().unwrap().to_vec(), vec![Some(3), Some(3), Some(1), None]);
        assert_eq!(s.iloc(-1).unwrap(), AnyValue::Int64(3));
        assert_eq!(s.iloc(1).unwrap(), AnyValue::Null);
        assert!(s.iloc(4).is_err() && s.iloc(-5).is_err());
        assert_eq!(s.iat(2).unwrap(), AnyValue::Int64(1));
        assert_eq!(s.to_list().len(), 4);
        assert_eq!(s.nlargest(2).unwrap().i64().unwrap().cont_slice().unwrap(), &[3, 3]);
        assert_eq!(s.nsmallest(1).unwrap().i64().unwrap().cont_slice().unwrap(), &[1]);
        assert_eq!(s.product().unwrap(), 9.0);
        assert!(s.copy().equals_missing(&s));
    }

    #[test]
    fn membership_and_ranges() {
        let s = Series::new("a".into(), [Some(1i32), Some(5), None, Some(9)]);
        let set = Series::new("set".into(), [5.0f64, 9.0, 100.0]);
        assert_eq!(s.is_in(&set).unwrap().to_vec(), vec![Some(false), Some(true), None, Some(true)]);
        assert_eq!(s.isin(&set).unwrap().num_trues(), 2);
        assert_eq!(s.between(1, 5).unwrap().to_vec(), vec![Some(true), Some(true), None, Some(false)]);
        let t = Series::new("t".into(), ["x", "y", "z"]);
        assert_eq!(t.is_in(&Series::new("s".into(), ["z", "x"])).unwrap().to_vec(), vec![Some(true), Some(false), Some(true)]);
        assert!(t.is_in(&set).is_err());
        assert!(s.is_in(&t).is_err());
        let b = Series::new("b".into(), [true, false]);
        assert_eq!(b.is_in(&Series::new("s".into(), [true])).unwrap().to_vec(), vec![Some(true), Some(false)]);
        assert!(b.any().unwrap() && !b.all().unwrap());
    }

    #[test]
    fn element_wise_helpers() {
        let s = Series::new("a".into(), [Some(-2i64), None, Some(3), Some(-1)]);
        assert_eq!(s.abs().unwrap().i64().unwrap().to_vec(), vec![Some(2), None, Some(3), Some(1)]);
        assert_eq!(s.clip(-1, 2).unwrap().i64().unwrap().to_vec(), vec![Some(-1), None, Some(2), Some(-1)]);
        assert_eq!(s.cumsum().unwrap().i64().unwrap().to_vec(), vec![Some(-2), None, Some(1), Some(0)]);
        assert_eq!(s.cum_sum(true).unwrap().i64().unwrap().to_vec(), vec![Some(0), None, Some(2), Some(-1)]);
        assert_eq!(s.cummax().unwrap().i64().unwrap().to_vec(), vec![Some(-2), None, Some(3), Some(3)]);
        assert_eq!(s.cummin().unwrap().i64().unwrap().to_vec(), vec![Some(-2), None, Some(-2), Some(-2)]);
        assert_eq!(s.cumprod().unwrap().i64().unwrap().to_vec(), vec![Some(-2), None, Some(-6), Some(6)]);
        assert_eq!(s.shift(1).i64().unwrap().to_vec(), vec![None, Some(-2), None, Some(3)]);
        assert_eq!(s.shift(-2).i64().unwrap().to_vec(), vec![Some(3), Some(-1), None, None]);
        assert_eq!(s.shift(10).null_count(), 4);
        assert_eq!(s.diff(1).unwrap().i64().unwrap().to_vec(), vec![None, None, None, Some(-4)]);
        let f = Series::new("f".into(), [1.2345f64, -2.5]);
        assert_eq!(f.round(2).unwrap().f64().unwrap().cont_slice().unwrap(), &[1.23, -2.5]);
        assert_eq!(f.round(0).unwrap().f64().unwrap().cont_slice().unwrap(), &[1.0, -3.0]);
        assert!(Series::new("s".into(), ["x"]).abs().is_err());
        assert!(Series::new("s".into(), ["x"]).cumsum().is_err());
    }
}
