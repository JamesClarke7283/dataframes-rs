//! Sorting frames by one or several key columns.

use std::cmp::Ordering;

use rayon::prelude::*;

use super::DataFrame;
use crate::chunked_array::{ChunkedArray, SortOptions};
use crate::datatypes::{IdxSize, PolarsNumericType};
use crate::error::PolarsResult;
use crate::kernels::{Kernel, par};
use crate::polars_ensure;
use crate::series::{Column, SeriesInner, for_numeric};

/// Options for sorting by several columns (mirrors Polars).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortMultipleOptions {
    /// Per key: largest first. A single entry applies to every key.
    pub descending: Vec<bool>,
    /// Per key: nulls after the values. A single entry applies to every key.
    pub nulls_last: Vec<bool>,
    /// Sort on all cores when the frame is large.
    pub multithreaded: bool,
    /// Keep rows with equal keys in their original order.
    pub maintain_order: bool,
    /// Keep only the first `limit` rows of the result.
    pub limit: Option<IdxSize>,
}

impl Default for SortMultipleOptions {
    fn default() -> Self {
        SortMultipleOptions {
            descending: vec![false],
            nulls_last: vec![false],
            multithreaded: true,
            maintain_order: false,
            limit: None,
        }
    }
}

impl SortMultipleOptions {
    /// Default options: ascending, nulls first.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sort every key in the given direction.
    pub fn with_order_descending(mut self, descending: bool) -> Self {
        self.descending = vec![descending];
        self
    }

    /// Set the direction per key.
    pub fn with_order_descending_multi(mut self, descending: impl IntoIterator<Item = bool>) -> Self {
        self.descending = descending.into_iter().collect();
        self
    }

    /// Put nulls last for every key.
    pub fn with_nulls_last(mut self, nulls_last: bool) -> Self {
        self.nulls_last = vec![nulls_last];
        self
    }

    /// Set the position of nulls per key.
    pub fn with_nulls_last_multi(mut self, nulls_last: impl IntoIterator<Item = bool>) -> Self {
        self.nulls_last = nulls_last.into_iter().collect();
        self
    }

    /// Allow or forbid parallel sorting.
    pub fn with_multithreaded(mut self, multithreaded: bool) -> Self {
        self.multithreaded = multithreaded;
        self
    }

    /// Require a stable sort.
    pub fn with_maintain_order(mut self, maintain_order: bool) -> Self {
        self.maintain_order = maintain_order;
        self
    }

    /// Keep only the first `limit` rows.
    pub fn with_limit(mut self, limit: Option<IdxSize>) -> Self {
        self.limit = limit;
        self
    }

    fn per_key(flags: &[bool], keys: usize, what: &str) -> PolarsResult<Vec<bool>> {
        match flags.len() {
            1 => Ok(vec![flags[0]; keys]),
            n => {
                polars_ensure!(n == keys, ComputeError: "the length of `{}` ({}) does not match the number of sort keys ({})", what, n, keys);
                Ok(flags.to_vec())
            }
        }
    }
}

type RowCmp<'a> = Box<dyn Fn(IdxSize, IdxSize) -> Ordering + Sync + 'a>;

/// Order two rows of one key column: nulls go where `nulls_last` says
/// regardless of the direction, values follow `descending`.
fn row_comparator<'a>(column: &'a Column, descending: bool, nulls_last: bool) -> RowCmp<'a> {
    fn with_nulls<V>(a: Option<V>, b: Option<V>, nulls_last: bool, descending: bool, cmp: impl Fn(V, V) -> Ordering) -> Ordering {
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => if nulls_last { Ordering::Greater } else { Ordering::Less },
            (Some(_), None) => if nulls_last { Ordering::Less } else { Ordering::Greater },
            (Some(a), Some(b)) => {
                let ord = cmp(a, b);
                if descending { ord.reverse() } else { ord }
            }
        }
    }
    fn numeric<'a, T: PolarsNumericType>(ca: &'a ChunkedArray<T>, descending: bool, nulls_last: bool) -> RowCmp<'a> {
        if ca.validity().is_none() {
            let values = ca.values().as_slice();
            return Box::new(move |a, b| {
                let ord = values[a as usize].tot_cmp(values[b as usize]);
                if descending { ord.reverse() } else { ord }
            });
        }
        Box::new(move |a, b| with_nulls(ca.get(a as usize), ca.get(b as usize), nulls_last, descending, |x, y| x.tot_cmp(y)))
    }
    for_numeric!(column, ca => numeric(ca, descending, nulls_last), other => match other {
        SeriesInner::String(ca) => {
            Box::new(move |a, b| with_nulls(ca.get(a as usize), ca.get(b as usize), nulls_last, descending, |x, y| x.cmp(y)))
        }
        SeriesInner::Boolean(ca) => {
            Box::new(move |a, b| with_nulls(ca.get(a as usize), ca.get(b as usize), nulls_last, descending, |x, y| x.cmp(&y)))
        }
        _ => unreachable!("numeric variants are handled above"),
    })
}

impl DataFrame {
    /// The permutation that sorts the frame by the columns `by`.
    pub fn arg_sort_multiple(
        &self,
        by: &[impl AsRef<str>],
        options: &SortMultipleOptions,
    ) -> PolarsResult<Vec<IdxSize>> {
        let keys = self.key_columns(by)?;
        let descending = SortMultipleOptions::per_key(&options.descending, keys.len(), "descending")?;
        let nulls_last = SortMultipleOptions::per_key(&options.nulls_last, keys.len(), "nulls_last")?;
        if let [key] = keys[..] {
            let single = SortOptions {
                descending: descending[0],
                nulls_last: nulls_last[0],
                multithreaded: options.multithreaded,
                maintain_order: options.maintain_order,
            };
            return Ok(key.arg_sort(single).values().to_vec());
        }
        let comparators: Vec<RowCmp<'_>> =
            keys.iter().zip(descending.iter().zip(&nulls_last)).map(|(k, (&d, &n))| row_comparator(k, d, n)).collect();
        let cmp = |a: &IdxSize, b: &IdxSize| {
            for c in &comparators {
                let ord = c(*a, *b);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            // Ties resolve by row index: the result is stable either way.
            a.cmp(b)
        };
        let mut idx: Vec<IdxSize> = (0..self.height as IdxSize).collect();
        if options.multithreaded && idx.len() >= 1 << 15 && par::threads() > 1 {
            idx.par_sort_unstable_by(cmp);
        } else {
            idx.sort_unstable_by(cmp);
        }
        Ok(idx)
    }

    /// Sort the rows by the columns `by`.
    ///
    /// ```
    /// use dataframes::prelude::*;
    /// let df = df!("k" => ["b", "a", "b"], "v" => [1, 2, 0])?;
    /// let sorted = df.sort(["k", "v"], SortMultipleOptions::default())?;
    /// assert_eq!(sorted.column("v")?.i32()?.cont_slice()?, &[2, 0, 1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn sort(
        &self,
        by: impl IntoIterator<Item = impl AsRef<str>>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<Self> {
        let by: Vec<_> = by.into_iter().collect();
        let mut idx = self.arg_sort_multiple(&by, &sort_options)?;
        if let Some(limit) = sort_options.limit {
            idx.truncate(limit as usize);
        }
        self.take_slice(&idx)
    }

    /// Sort the rows in place.
    pub fn sort_in_place(
        &mut self,
        by: impl IntoIterator<Item = impl AsRef<str>>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<&mut Self> {
        *self = self.sort(by, sort_options)?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    #[test]
    fn single_and_multi_key_sorts() {
        let df = df!(
            "k" => [Some("b"), Some("a"), None, Some("b"), Some("a")],
            "v" => [1i64, 2, 3, 0, 5],
        )
        .unwrap();
        let by_v = df.sort(["v"], SortMultipleOptions::default()).unwrap();
        assert_eq!(by_v["v"].i64().unwrap().cont_slice().unwrap(), &[0, 1, 2, 3, 5]);
        assert_eq!(by_v["k"].str().unwrap().to_vec(), vec![Some("b"), Some("b"), Some("a"), None, Some("a")]);

        let both = df.sort(["k", "v"], SortMultipleOptions::default()).unwrap();
        assert_eq!(both["k"].str().unwrap().to_vec(), vec![None, Some("a"), Some("a"), Some("b"), Some("b")]);
        assert_eq!(both["v"].i64().unwrap().cont_slice().unwrap(), &[3, 2, 5, 0, 1]);

        let mixed = df
            .sort(["k", "v"], SortMultipleOptions::default().with_order_descending_multi([false, true]).with_nulls_last(true))
            .unwrap();
        assert_eq!(mixed["v"].i64().unwrap().cont_slice().unwrap(), &[5, 2, 1, 0, 3]);

        let desc = df.sort(["v"], SortMultipleOptions::default().with_order_descending(true).with_limit(Some(2))).unwrap();
        assert_eq!(desc["v"].i64().unwrap().cont_slice().unwrap(), &[5, 3]);
        assert!(df.sort(["missing"], SortMultipleOptions::default()).is_err());
        assert!(df.sort(["k", "v"], SortMultipleOptions::default().with_order_descending_multi([true, false, true])).is_err());
        let mut m = df.clone();
        m.sort_in_place(["v"], SortMultipleOptions::default()).unwrap();
        assert_eq!(m, by_v);
    }

    #[test]
    fn large_multi_key_sort_is_correct_and_stable() {
        let n = 60_000i64;
        let a: Vec<i64> = (0..n).map(|i| (i * 7919) % 13).collect();
        let b: Vec<f64> = (0..n).map(|i| ((i * 104_729) % 1000) as f64).collect();
        let row: Vec<i64> = (0..n).collect();
        let df = df!("a" => &a, "b" => &b, "row" => &row).unwrap();
        let sorted = df.sort(["a", "b"], SortMultipleOptions::default()).unwrap();
        let (sa, sb, sr) = (
            sorted["a"].i64().unwrap().cont_slice().unwrap(),
            sorted["b"].f64().unwrap().cont_slice().unwrap(),
            sorted["row"].i64().unwrap().cont_slice().unwrap(),
        );
        for i in 1..n as usize {
            let prev = (sa[i - 1], sb[i - 1], sr[i - 1]);
            let cur = (sa[i], sb[i], sr[i]);
            assert!(prev.0 < cur.0 || (prev.0 == cur.0 && (prev.1 < cur.1 || (prev.1 == cur.1 && prev.2 < cur.2))));
        }
    }
}
