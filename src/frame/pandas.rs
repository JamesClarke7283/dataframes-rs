//! The pandas vocabulary for [`DataFrame`], plus whole-frame reductions.
//!
//! pandas indexes with `df.iloc[...]`, `df.loc[...]`, `df.at[...]`; Rust has
//! no indexer properties, so those become methods taking the same arguments.
//! Everything here is a thin layer over the Polars-named core.

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use super::{DataFrame, GroupBy, JoinArgs, JoinType, Row, SortMultipleOptions, UniqueKeepStrategy};
use crate::chunked_array::{BooleanChunked, QuantileMethod, StringChunked, UInt32Chunked};
use crate::datatypes::{AnyValue, DataType, IdxSize, PlSmallStr};
use crate::error::{PolarsError, PolarsResult};
use crate::series::{Column, IntoSeries, Series};
use crate::{polars_bail, polars_ensure};

/// What `DataFrame::iloc` accepts: a position, a range of positions or a
/// list of positions.
///
/// Ranges are clamped to the frame (as pandas does) and share the frame's
/// buffers; lists gather the rows.
pub trait RowSelector {
    /// Select rows of `df`.
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame>;
}

fn one_row(df: &DataFrame, index: i64) -> PolarsResult<DataFrame> {
    let n = df.height() as i64;
    let i = if index < 0 { index + n } else { index };
    polars_ensure!(i >= 0 && i < n, OutOfBounds: "single positional indexer {} is out-of-bounds for DataFrame of height {}", index, n);
    Ok(df.slice(i, 1))
}

macro_rules! scalar_selector {
    ($($t:ty),*) => {$(
        impl RowSelector for $t {
            fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
                one_row(df, self as i64)
            }
        }
    )*};
}
scalar_selector!(usize, i32, i64);

impl RowSelector for Range<usize> {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        Ok(df.slice(self.start as i64, self.end.saturating_sub(self.start)))
    }
}

impl RowSelector for RangeInclusive<usize> {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        Ok(df.slice(*self.start() as i64, (self.end() + 1).saturating_sub(*self.start())))
    }
}

impl RowSelector for RangeTo<usize> {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        Ok(df.slice(0, self.end))
    }
}

impl RowSelector for RangeToInclusive<usize> {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        Ok(df.slice(0, self.end + 1))
    }
}

impl RowSelector for RangeFrom<usize> {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        Ok(df.slice(self.start as i64, usize::MAX))
    }
}

impl RowSelector for RangeFull {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        Ok(df.clone())
    }
}

impl RowSelector for &[usize] {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let idx: Vec<IdxSize> = self
            .iter()
            .map(|&i| IdxSize::try_from(i).map_err(|_| PolarsError::OutOfBounds(format!("index {i} is out of bounds").into())))
            .collect::<PolarsResult<_>>()?;
        df.take_slice(&idx)
    }
}

impl RowSelector for Vec<usize> {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        self.as_slice().select_rows(df)
    }
}

impl<const N: usize> RowSelector for [usize; N] {
    fn select_rows(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        self.as_slice().select_rows(df)
    }
}

impl DataFrame {
    // ----- indexing -----------------------------------------------------------

    /// Positional row selection, as `df.iloc[...]`.
    ///
    /// ```
    /// use dataframes::prelude::*;
    /// let df = df!("a" => [10, 20, 30, 40])?;
    /// assert_eq!(df.iloc(1..3)?.column("a")?.i32()?.cont_slice()?, &[20, 30]);
    /// assert_eq!(df.iloc(-1)?.column("a")?.i32()?.get(0), Some(40));
    /// assert_eq!(df.iloc([3, 0])?.height(), 2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn iloc<R: RowSelector>(&self, rows: R) -> PolarsResult<DataFrame> {
        rows.select_rows(self)
    }

    /// Boolean-mask row selection, as `df.loc[mask]`.
    pub fn loc(&self, mask: &BooleanChunked) -> PolarsResult<DataFrame> {
        self.filter(mask)
    }

    /// Boolean-mask row selection with a column subset, as `df.loc[mask, cols]`.
    pub fn loc_columns<I, S>(&self, mask: &BooleanChunked, columns: I) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Select first: only the requested columns are compacted.
        self.select(columns)?.filter(mask)
    }

    /// One cell by row position and column name, as `df.at[row, col]`.
    pub fn at(&self, row: usize, column: &str) -> PolarsResult<AnyValue<'_>> {
        self.column(column)?.get(row)
    }

    /// One cell by row and column position, as `df.iat[row, col]`.
    pub fn iat(&self, row: usize, column: usize) -> PolarsResult<AnyValue<'_>> {
        match self.select_at_idx(column) {
            Some(c) => c.get(row),
            None => polars_bail!(OutOfBounds: "column index {} is out of bounds for DataFrame of width {}", column, self.width()),
        }
    }

    // ----- metadata -----------------------------------------------------------

    /// Column names, as `df.keys()` / `df.columns` in pandas.
    pub fn keys(&self) -> Vec<&str> {
        self.get_column_names_str()
    }

    /// Number of cells (`height * width`).
    pub fn size(&self) -> usize {
        self.height() * self.width()
    }

    /// A handle to the same data. Nothing is copied: columns are immutable and
    /// shared, and mutation through either handle is copy-on-write.
    pub fn copy(&self) -> DataFrame {
        self.clone()
    }

    /// Bytes of heap memory viewed by the frame.
    pub fn memory_usage(&self) -> usize {
        self.estimated_size()
    }

    /// A summary of the columns, like `df.info()`.
    pub fn info(&self) -> String {
        let mut out = format!("DataFrame: {} rows x {} columns\n", self.height(), self.width());
        for (i, c) in self.columns().iter().enumerate() {
            out.push_str(&format!(
                "{i:>3}  {:<24} {:>10} non-null  {}\n",
                c.name(),
                c.len() - c.null_count(),
                c.dtype()
            ));
        }
        out.push_str(&format!("memory usage: {} bytes", self.estimated_size()));
        out
    }

    // ----- ordering / grouping / joining ---------------------------------------

    /// Sort by `by`, as `df.sort_values(by, ascending=...)`. Nulls go last.
    pub fn sort_values(&self, by: impl IntoIterator<Item = impl AsRef<str>>, ascending: bool) -> PolarsResult<DataFrame> {
        self.sort(by, SortMultipleOptions::default().with_order_descending(!ascending).with_nulls_last(true))
    }

    /// The `n` rows with the largest values in `column`.
    pub fn nlargest(&self, n: usize, column: &str) -> PolarsResult<DataFrame> {
        Ok(self.sort([column], SortMultipleOptions::default().with_order_descending(true).with_nulls_last(true))?.head(n))
    }

    /// The `n` rows with the smallest values in `column`.
    pub fn nsmallest(&self, n: usize, column: &str) -> PolarsResult<DataFrame> {
        Ok(self.sort([column], SortMultipleOptions::default().with_nulls_last(true))?.head(n))
    }

    /// Group by `by` the pandas way: aggregated columns keep their names and
    /// the result is ordered by the keys.
    ///
    /// ```
    /// use dataframes::prelude::*;
    /// let df = df!("k" => ["b", "a", "b"], "v" => [1, 2, 3])?;
    /// let out = df.groupby(["k"])?.sum()?;
    /// assert_eq!(out.column("k")?.str()?.to_vec(), vec![Some("a"), Some("b")]);
    /// assert_eq!(out.column("v")?.i32()?.cont_slice()?, &[2, 4]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn groupby<I, S>(&self, by: I) -> PolarsResult<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        GroupBy::new(self, by.into_iter().map(Into::into).collect(), false, true)
    }

    /// Join on the columns `on` (same names in both frames), as `df.merge(other, on=..., how=...)`.
    pub fn merge(&self, other: &DataFrame, on: &[&str], how: JoinType) -> PolarsResult<DataFrame> {
        self.join(other, on, on, JoinArgs::new(how), None)
    }

    // ----- nulls --------------------------------------------------------------

    /// A boolean frame marking the nulls.
    pub fn isna(&self) -> DataFrame {
        DataFrame::from_parts(self.height(), self.columns().iter().map(|c| c.is_null().into_series()).collect())
    }

    /// pandas alias of [`isna`](Self::isna).
    pub fn isnull(&self) -> DataFrame {
        self.isna()
    }

    /// A boolean frame marking the non-null cells.
    pub fn notna(&self) -> DataFrame {
        DataFrame::from_parts(self.height(), self.columns().iter().map(|c| c.is_not_null().into_series()).collect())
    }

    /// pandas alias of [`notna`](Self::notna).
    pub fn notnull(&self) -> DataFrame {
        self.notna()
    }

    /// Replace nulls by `value` in every column whose type can hold it
    /// (a number fills numeric columns, a string fills string columns, ...).
    pub fn fillna<'a>(&self, value: impl Into<AnyValue<'a>>) -> PolarsResult<DataFrame> {
        let value = value.into();
        let fits = |dtype: &DataType| match &value {
            AnyValue::Null => false,
            AnyValue::Boolean(_) => dtype.is_bool(),
            AnyValue::String(_) | AnyValue::StringOwned(_) => dtype.is_string(),
            _ => dtype.is_numeric(),
        };
        let columns = self
            .columns()
            .iter()
            .map(|c| if c.has_nulls() && fits(c.dtype()) { c.fill_null_with_value(value.clone()) } else { Ok(c.clone()) })
            .collect::<PolarsResult<_>>()?;
        Ok(DataFrame::from_parts(self.height(), columns))
    }

    /// Drop the rows containing any null.
    pub fn dropna(&self) -> PolarsResult<DataFrame> {
        self.drop_nulls::<&str>(None)
    }

    // ----- duplicates ----------------------------------------------------------

    /// Distinct rows, keeping the first occurrence.
    pub fn drop_duplicates(&self) -> PolarsResult<DataFrame> {
        self.unique_stable(None, UniqueKeepStrategy::First, None)
    }

    /// `true` for every row that repeats an earlier row.
    pub fn duplicated(&self) -> PolarsResult<BooleanChunked> {
        let first = self.unique_stable_mask()?;
        Ok(!&first)
    }

    fn unique_stable_mask(&self) -> PolarsResult<BooleanChunked> {
        // A row is a first occurrence iff it survives `drop_duplicates`.
        let tagged = self.with_row_index("__row__".into(), None)?;
        let names: Vec<String> = self.get_column_names_str().iter().map(|s| s.to_string()).collect();
        let kept = tagged.unique_stable(Some(&names), UniqueKeepStrategy::First, None)?;
        let mut keep = vec![false; self.height()];
        for &i in kept.column("__row__")?.u32()?.cont_slice()? {
            keep[i as usize] = true;
        }
        Ok(BooleanChunked::from_slice(PlSmallStr::EMPTY, &keep))
    }

    // ----- columns -------------------------------------------------------------

    /// A new frame with `column` added (or replaced), as `df.assign(name=values)`.
    pub fn assign(&self, column: impl IntoSeries) -> PolarsResult<DataFrame> {
        let mut out = self.clone();
        out.with_column(column)?;
        Ok(out)
    }

    /// A new frame with `column` cast to `dtype`.
    pub fn astype(&self, column: &str, dtype: &DataType) -> PolarsResult<DataFrame> {
        let mut out = self.clone();
        out.try_apply(column, |c| c.cast(dtype))?;
        Ok(out)
    }

    /// Rename several columns: `[("old", "new"), ...]`.
    pub fn rename_many<'a>(&mut self, renames: impl IntoIterator<Item = (&'a str, PlSmallStr)>) -> PolarsResult<&mut Self> {
        for (old, new) in renames {
            self.rename(old, new)?;
        }
        Ok(self)
    }

    // ----- rows ----------------------------------------------------------------

    /// Iterate over the rows as dynamically typed values (the slow lane).
    pub fn iter_rows(&self) -> impl ExactSizeIterator<Item = Row<'_>> + '_ {
        (0..self.height()).map(move |i| Row(self.columns().iter().map(|c| c.get_any(i)).collect()))
    }

    /// Iterate over `(row number, row)`, as `df.iterrows()`.
    pub fn iterrows(&self) -> impl ExactSizeIterator<Item = (usize, Row<'_>)> + '_ {
        self.iter_rows().enumerate()
    }

    /// Iterate over the rows, as `df.itertuples(index=False)`.
    pub fn itertuples(&self) -> impl ExactSizeIterator<Item = Row<'_>> + '_ {
        self.iter_rows()
    }

    /// `n` rows drawn without replacement, reproducibly from `seed`.
    pub fn sample_n(&self, n: usize, seed: u64) -> PolarsResult<DataFrame> {
        polars_ensure!(n <= self.height(), ShapeMismatch: "cannot take a larger sample than the total population");
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        // Partial Fisher-Yates shuffle.
        let mut rows: Vec<IdxSize> = (0..self.height() as IdxSize).collect();
        for i in 0..n {
            let j = i + (next() % (rows.len() - i) as u64) as usize;
            rows.swap(i, j);
        }
        self.take_slice(&rows[..n])
    }

    // ----- reductions ------------------------------------------------------------

    fn reduce(&self, f: impl Fn(&Column) -> (AnyValue<'_>, DataType)) -> PolarsResult<DataFrame> {
        let columns = self
            .columns()
            .iter()
            .map(|c| {
                let (value, dtype) = f(c);
                if value.is_null() {
                    return Ok(Series::full_null(c.name().clone(), 1, &dtype));
                }
                Series::from_any_values(c.name().clone(), &[value])?.cast(&dtype)
            })
            .collect::<PolarsResult<_>>()?;
        Ok(DataFrame::from_parts(1, columns))
    }

    /// Column sums as a one-row frame (null for non-numeric columns).
    pub fn sum(&self) -> PolarsResult<DataFrame> {
        self.reduce(|c| {
            let d = *c.dtype();
            if d.is_float() {
                (c.sum::<f64>().map_or(AnyValue::Null, AnyValue::Float64), d)
            } else if d == DataType::UInt64 {
                (c.sum::<u64>().map_or(AnyValue::Null, AnyValue::UInt64), d)
            } else if d.is_integer() || d.is_bool() {
                (c.sum::<i64>().map_or(AnyValue::Null, AnyValue::Int64), DataType::Int64)
            } else {
                (AnyValue::Null, d)
            }
        })
    }

    /// Column means as a one-row frame.
    pub fn mean(&self) -> PolarsResult<DataFrame> {
        self.reduce(|c| (c.mean().map_or(AnyValue::Null, AnyValue::Float64), float_or(c.dtype())))
    }

    /// Column medians as a one-row frame.
    pub fn median(&self) -> PolarsResult<DataFrame> {
        self.reduce(|c| (c.median().map_or(AnyValue::Null, AnyValue::Float64), float_or(c.dtype())))
    }

    /// Column standard deviations as a one-row frame.
    pub fn std(&self, ddof: u8) -> PolarsResult<DataFrame> {
        self.reduce(|c| (c.std(ddof).map_or(AnyValue::Null, AnyValue::Float64), float_or(c.dtype())))
    }

    /// Column variances as a one-row frame.
    pub fn var(&self, ddof: u8) -> PolarsResult<DataFrame> {
        self.reduce(|c| (c.var(ddof).map_or(AnyValue::Null, AnyValue::Float64), float_or(c.dtype())))
    }

    /// Column minima as a one-row frame.
    pub fn min(&self) -> PolarsResult<DataFrame> {
        self.reduce(|c| (c.min_any(), *c.dtype()))
    }

    /// Column maxima as a one-row frame.
    pub fn max(&self) -> PolarsResult<DataFrame> {
        self.reduce(|c| (c.max_any(), *c.dtype()))
    }

    /// Non-null counts as a one-row frame.
    pub fn count(&self) -> DataFrame {
        let columns = self
            .columns()
            .iter()
            .map(|c| UInt32Chunked::from_slice(c.name().clone(), &[(c.len() - c.null_count()) as IdxSize]).into_series())
            .collect();
        DataFrame::from_parts(1, columns)
    }

    /// Distinct-value counts as a one-row frame (nulls are not counted, as in pandas).
    pub fn nunique(&self) -> PolarsResult<DataFrame> {
        let columns = self
            .columns()
            .iter()
            .map(|c| Ok(UInt32Chunked::from_slice(c.name().clone(), &[c.nunique()? as IdxSize]).into_series()))
            .collect::<PolarsResult<_>>()?;
        Ok(DataFrame::from_parts(1, columns))
    }

    /// Summary statistics of the numeric columns: count, null count, mean,
    /// standard deviation, minimum, quartiles and maximum, one per row.
    pub fn describe(&self) -> PolarsResult<DataFrame> {
        const STATS: [&str; 9] = ["count", "null_count", "mean", "std", "min", "25%", "50%", "75%", "max"];
        let mut columns = vec![StringChunked::from_slice("statistic".into(), &STATS).into_series()];
        for c in self.columns().iter().filter(|c| c.dtype().is_numeric()) {
            let q = |p: f64| c.quantile(p, QuantileMethod::Linear).ok().flatten();
            let values = [
                Some((c.len() - c.null_count()) as f64),
                Some(c.null_count() as f64),
                c.mean(),
                c.std(1),
                c.min::<f64>()?,
                q(0.25),
                q(0.5),
                q(0.75),
                c.max::<f64>()?,
            ];
            columns.push(Series::new(c.name().clone(), values));
        }
        DataFrame::new(STATS.len(), columns)
    }
}

fn float_or(dtype: &DataType) -> DataType {
    if dtype.is_numeric() || dtype.is_bool() { DataType::Float64 } else { *dtype }
}

impl Series {
    /// How often each distinct value occurs: a frame with the values and a
    /// `count` column, most frequent first when `sort` is set.
    pub fn value_counts(&self, sort: bool) -> PolarsResult<DataFrame> {
        let frame = self.clone().into_frame();
        let name = self.name().clone();
        let counts = frame.group_by([name])?.len()?;
        let mut counts = counts;
        counts.rename("len", "count".into())?;
        if sort {
            return counts.sort(["count"], SortMultipleOptions::default().with_order_descending(true).with_maintain_order(true));
        }
        Ok(counts)
    }

    /// Summary statistics of a numeric column (see [`DataFrame::describe`]).
    pub fn describe(&self) -> PolarsResult<DataFrame> {
        self.to_frame().describe()
    }
}

/// Stack frames vertically (`pd.concat` / `pl.concat`). All frames must have
/// the same columns in the same order.
pub fn concat(frames: &[DataFrame]) -> PolarsResult<DataFrame> {
    let mut out = DataFrame::empty();
    for f in frames {
        out.vstack_mut(f)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_array::ChunkCompareIneq;
    use crate::df;

    fn sample() -> DataFrame {
        df!(
            "k" => ["b", "a", "b", "c"],
            "x" => [Some(4i64), Some(1), None, Some(7)],
            "y" => [0.5f64, 1.5, 2.5, 3.5],
        )
        .unwrap()
    }

    #[test]
    fn positional_and_boolean_indexing() {
        let df = sample();
        assert_eq!(df.iloc(1).unwrap()["k"].str().unwrap().get(0), Some("a"));
        assert_eq!(df.iloc(-1).unwrap()["x"].i64().unwrap().get(0), Some(7));
        assert_eq!(df.iloc(1..3).unwrap().height(), 2);
        assert_eq!(df.iloc(2..).unwrap().height(), 2);
        assert_eq!(df.iloc(..=1).unwrap().height(), 2);
        assert_eq!(df.iloc(..10).unwrap().height(), 4);
        assert_eq!(df.iloc(..).unwrap(), df);
        assert_eq!(df.iloc([3usize, 0]).unwrap()["y"].f64().unwrap().cont_slice().unwrap(), &[3.5, 0.5]);
        assert_eq!(df.iloc(vec![0usize]).unwrap().height(), 1);
        assert!(df.iloc(4).is_err() && df.iloc(-5).is_err() && df.iloc([9usize]).is_err());
        // Range selection is a view.
        assert_eq!(
            df.iloc(1..3).unwrap()["y"].f64().unwrap().values().as_ptr(),
            unsafe { df["y"].f64().unwrap().values().as_ptr().add(1) }
        );
        let mask = df["y"].gt(1.0).unwrap();
        assert_eq!(df.loc(&mask).unwrap().height(), 3);
        assert_eq!(df.loc_columns(&mask, ["k"]).unwrap().shape(), (3, 1));
        assert_eq!(df.at(3, "x").unwrap(), AnyValue::Int64(7));
        assert_eq!(df.iat(0, 2).unwrap(), AnyValue::Float64(0.5));
        assert!(df.iat(0, 9).is_err() && df.at(9, "x").is_err());
        assert_eq!(df.keys(), ["k", "x", "y"]);
        assert_eq!(df.size(), 12);
        assert!(df.info().contains("x") && df.info().contains("i64"));
        assert!(df.memory_usage() > 0);
        assert_eq!(df.copy(), df);
    }

    #[test]
    fn nulls_duplicates_and_columns() {
        let df = sample();
        assert_eq!(df.isna()["x"].bool().unwrap().num_trues(), 1);
        assert_eq!(df.notna()["x"].bool().unwrap().num_trues(), 3);
        assert_eq!(df.fillna(0).unwrap()["x"].i64().unwrap().cont_slice().unwrap(), &[4, 1, 0, 7]);
        assert_eq!(df.fillna("z").unwrap()["x"].null_count(), 1, "a string cannot fill a numeric column");
        assert_eq!(df.dropna().unwrap().height(), 3);
        let dup = df!("a" => [1, 1, 2, 1], "b" => ["x", "x", "y", "z"]).unwrap();
        assert_eq!(dup.drop_duplicates().unwrap().height(), 3);
        assert_eq!(dup.duplicated().unwrap().to_vec(), vec![Some(false), Some(true), Some(false), Some(false)]);
        let assigned = df.assign(Series::new("z".into(), [1u8, 2, 3, 4])).unwrap();
        assert_eq!((assigned.width(), df.width()), (4, 3));
        assert_eq!(*df.astype("x", &DataType::Float64).unwrap()["x"].dtype(), DataType::Float64);
        let mut renamed = df.clone();
        renamed.rename_many([("k", "key".into()), ("x", "value".into())]).unwrap();
        assert_eq!(renamed.keys(), ["key", "value", "y"]);
    }

    #[test]
    fn ordering_grouping_merging() {
        let df = sample();
        assert_eq!(df.sort_values(["x"], true).unwrap()["x"].i64().unwrap().to_vec(), vec![Some(1), Some(4), Some(7), None]);
        assert_eq!(df.sort_values(["x"], false).unwrap()["x"].i64().unwrap().to_vec(), vec![Some(7), Some(4), Some(1), None]);
        assert_eq!(df.nlargest(2, "y").unwrap()["y"].f64().unwrap().cont_slice().unwrap(), &[3.5, 2.5]);
        assert_eq!(df.nsmallest(1, "x").unwrap()["k"].str().unwrap().get(0), Some("a"));
        let g = df.groupby(["k"]).unwrap().sum().unwrap();
        assert_eq!(g.keys(), ["k", "x", "y"]);
        assert_eq!(g["k"].str().unwrap().to_vec(), vec![Some("a"), Some("b"), Some("c")]);
        assert_eq!(g["y"].f64().unwrap().cont_slice().unwrap(), &[1.5, 3.0, 3.5]);
        assert_eq!(df.groupby(["k"]).unwrap().size().unwrap()["size"].u32().unwrap().cont_slice().unwrap(), &[1, 2, 1]);
        let other = df!("k" => ["a", "b"], "w" => [100, 200]).unwrap();
        let m = df.merge(&other, &["k"], JoinType::Inner).unwrap();
        assert_eq!(m["w"].i32().unwrap().cont_slice().unwrap(), &[200, 100, 200]);
        assert_eq!(concat(&[df.clone(), df.clone()]).unwrap().height(), 8);
        assert!(concat(&[df.clone(), other]).is_err());
    }

    #[test]
    fn reductions_and_describe() {
        let df = sample();
        let s = df.sum().unwrap();
        assert_eq!(s.shape(), (1, 3));
        assert_eq!(s["x"].i64().unwrap().get(0), Some(12));
        assert_eq!(s["y"].f64().unwrap().get(0), Some(8.0));
        assert_eq!((s["k"].null_count(), *s["k"].dtype()), (1, DataType::String));
        assert_eq!(df.mean().unwrap()["x"].f64().unwrap().get(0), Some(4.0));
        assert_eq!(df.min().unwrap()["k"].str().unwrap().get(0), Some("a"));
        assert_eq!(df.max().unwrap()["y"].f64().unwrap().get(0), Some(3.5));
        assert_eq!(df.median().unwrap()["y"].f64().unwrap().get(0), Some(2.0));
        assert_eq!(df.var(1).unwrap()["x"].f64().unwrap().get(0), Some(9.0));
        assert_eq!(df.std(1).unwrap()["x"].f64().unwrap().get(0), Some(3.0));
        assert_eq!(df.count()["x"].u32().unwrap().get(0), Some(3));
        assert_eq!(df.nunique().unwrap()["k"].u32().unwrap().get(0), Some(3));
        let d = df.describe().unwrap();
        assert_eq!(d.keys(), ["statistic", "x", "y"]);
        assert_eq!(d.height(), 9);
        assert_eq!(d["x"].f64().unwrap().to_vec()[..3], [Some(3.0), Some(1.0), Some(4.0)]);
        assert_eq!(d["y"].f64().unwrap().get(6), Some(2.0));
        assert_eq!(df["k"].value_counts(true).unwrap()["count"].u32().unwrap().cont_slice().unwrap(), &[2, 1, 1]);
        assert_eq!(df["k"].value_counts(true).unwrap()["k"].str().unwrap().get(0), Some("b"));
        assert_eq!(df["y"].describe().unwrap().height(), 9);
    }

    #[test]
    fn row_iteration_and_sampling() {
        let df = sample();
        let rows: Vec<Row> = df.iter_rows().collect();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[2].0, vec![AnyValue::String("b"), AnyValue::Null, AnyValue::Float64(2.5)]);
        assert_eq!(df.iterrows().last().unwrap().0, 3);
        assert_eq!(df.itertuples().count(), 4);
        let s = df.sample_n(3, 42).unwrap();
        assert_eq!(s.height(), 3);
        assert_eq!(s, df.sample_n(3, 42).unwrap(), "sampling is reproducible");
        assert_eq!(s["y"].n_unique().unwrap(), 3, "without replacement");
        assert!(df.sample_n(5, 0).is_err());
    }
}
