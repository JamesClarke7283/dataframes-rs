//! Group-by: assign every row a dense group id, then aggregate by id.
//!
//! Grouping and aggregating are decoupled by a `Vec<IdxSize>` of group ids
//! (numbered in order of first appearance). Ids for a single key come from a
//! direct-address table when the key range is small and from an open
//! addressing table otherwise; several keys are folded pairwise, the pair of
//! ids so far being just another integer key. Aggregations are then tight
//! typed loops `acc[id[i]] ⊕= value[i]`.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use super::DataFrame;
use crate::chunked_array::{ChunkedArray, Float64Chunked, Int64Chunked, StringChunked, UInt32Chunked};
use crate::datatypes::{DataType, IdxSize, NativeType, PlSmallStr, PolarsNumericType};
use crate::error::PolarsResult;
use crate::kernels::Kernel;
use crate::kernels::aggregate::Num;
use crate::memory::Bitmap;
use crate::polars_bail;
use crate::series::{Column, IntoSeries, Series, SeriesInner, for_numeric};

/// The FxHash function (as used inside rustc): fast and good enough for
/// hash tables that are not exposed to adversarial keys.
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.write_u64(u64::from_le_bytes(c.try_into().expect("chunk of 8")));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.write_u64(u64::from_le_bytes(buf));
        }
        self.write_u64(bytes.len() as u64);
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(FX_SEED);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) type FxBuild = BuildHasherDefault<FxHasher>;

/// Row → group assignment.
#[derive(Clone, Debug)]
pub struct Groups {
    /// Group id of every row; ids are dense and numbered by first appearance.
    pub ids: Vec<IdxSize>,
    /// Row index of the first row of every group.
    pub first: Vec<IdxSize>,
}

impl Groups {
    /// Number of groups.
    #[inline]
    pub fn len(&self) -> usize {
        self.first.len()
    }

    /// `true` if there are no groups (no rows).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.first.is_empty()
    }

    /// Number of rows in every group.
    pub fn sizes(&self) -> Vec<IdxSize> {
        let mut sizes = vec![0 as IdxSize; self.len()];
        for &g in &self.ids {
            sizes[g as usize] += 1;
        }
        sizes
    }

    /// The rows of every group, in row order.
    pub fn rows(&self) -> Vec<Vec<IdxSize>> {
        let sizes = self.sizes();
        let mut rows: Vec<Vec<IdxSize>> = sizes.iter().map(|&s| Vec::with_capacity(s as usize)).collect();
        for (i, &g) in self.ids.iter().enumerate() {
            rows[g as usize].push(i as IdxSize);
        }
        rows
    }
}

/// Group rows by a `u64` key per row (`None` = null, its own group).
fn group_u64(keys: impl ExactSizeIterator<Item = Option<u64>> + Clone, hint_range: Option<(u64, u64)>) -> Groups {
    let n = keys.len();
    let mut ids = Vec::with_capacity(n);
    let mut first: Vec<IdxSize> = Vec::new();
    let mut null_id: Option<IdxSize> = None;
    let mut assign_null = |row: usize, first: &mut Vec<IdxSize>| {
        *null_id.get_or_insert_with(|| {
            first.push(row as IdxSize);
            (first.len() - 1) as IdxSize
        })
    };

    // Small key range: index a table directly, no hashing at all.
    if let Some((min, max)) = hint_range {
        let range = max - min;
        if range < (4 * n as u64).max(1 << 16) {
            let mut table = vec![IdxSize::MAX; range as usize + 1];
            for (row, key) in keys.enumerate() {
                let id = match key {
                    None => assign_null(row, &mut first),
                    Some(k) => {
                        let slot = &mut table[(k - min) as usize];
                        if *slot == IdxSize::MAX {
                            *slot = first.len() as IdxSize;
                            first.push(row as IdxSize);
                        }
                        *slot
                    }
                };
                ids.push(id);
            }
            return Groups { ids, first };
        }
    }

    // Open addressing with linear probing; grows at 50 % load.
    const EMPTY: IdxSize = IdxSize::MAX;
    let mut cap = 1024usize;
    let mut slot_keys = vec![0u64; cap];
    let mut slot_ids = vec![EMPTY; cap];
    let mut used = 0usize;
    let hash = |k: u64, cap: usize| ((k.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as usize & (cap - 1);
    for (row, key) in keys.enumerate() {
        let id = match key {
            None => assign_null(row, &mut first),
            Some(k) => {
                if used * 2 >= cap {
                    let new_cap = cap * 2;
                    let mut nk = vec![0u64; new_cap];
                    let mut ni = vec![EMPTY; new_cap];
                    for s in 0..cap {
                        if slot_ids[s] != EMPTY {
                            let mut h = hash(slot_keys[s], new_cap);
                            while ni[h] != EMPTY {
                                h = (h + 1) & (new_cap - 1);
                            }
                            (nk[h], ni[h]) = (slot_keys[s], slot_ids[s]);
                        }
                    }
                    (slot_keys, slot_ids, cap) = (nk, ni, new_cap);
                }
                let mut h = hash(k, cap);
                loop {
                    if slot_ids[h] == EMPTY {
                        slot_keys[h] = k;
                        slot_ids[h] = first.len() as IdxSize;
                        first.push(row as IdxSize);
                        used += 1;
                        break slot_ids[h];
                    }
                    if slot_keys[h] == k {
                        break slot_ids[h];
                    }
                    h = (h + 1) & (cap - 1);
                }
            }
        };
        ids.push(id);
    }
    Groups { ids, first }
}

/// Group ids for one key column.
fn group_single(column: &Column) -> Groups {
    fn numeric<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Groups {
        // Map every value to a `u64` such that equal values (in the total
        // order) map to equal keys: floats are canonicalised first.
        let key = |v: T::Native| -> u64 {
            if T::Native::IS_FLOAT {
                let f = v.to_f64();
                (if f.is_nan() { f64::NAN } else if f == 0.0 { 0.0 } else { f }).to_bits()
            } else {
                v.to_i64() as u64
            }
        };
        let range = if T::Native::IS_FLOAT || <T::Native as NativeType>::DTYPE == DataType::UInt64 {
            None
        } else {
            // Bias signed keys so that the range is computed without overflow.
            ca.min_max().and_then(|(lo, hi)| {
                let (lo, hi) = (lo.to_i64(), hi.to_i64());
                hi.checked_sub(lo).map(|_| (lo as u64, hi as u64))
            })
        };
        let n = ca.len();
        match range {
            // Direct addressing needs keys relative to `min` in wrapping arithmetic.
            Some((min, max)) => {
                let span = max.wrapping_sub(min);
                let keys = (0..n).map(|i| ca.get(i).map(|v| key(v).wrapping_sub(min)));
                group_u64(keys, Some((0, span)))
            }
            None => group_u64((0..n).map(|i| ca.get(i).map(key)), None),
        }
    }
    for_numeric!(column, ca => numeric(ca), other => match other {
        SeriesInner::Boolean(ca) => group_u64((0..ca.len()).map(|i| ca.get(i).map(u64::from)), Some((0, 1))),
        SeriesInner::String(ca) => group_strings(ca),
        _ => unreachable!("numeric variants are handled above"),
    })
}

fn group_strings(ca: &StringChunked) -> Groups {
    let mut map: HashMap<&str, IdxSize, FxBuild> = HashMap::with_capacity_and_hasher(1024, FxBuild::default());
    let mut ids = Vec::with_capacity(ca.len());
    let mut first: Vec<IdxSize> = Vec::new();
    let mut null_id: Option<IdxSize> = None;
    for (row, s) in ca.iter().enumerate() {
        let id = match s {
            None => *null_id.get_or_insert_with(|| {
                first.push(row as IdxSize);
                (first.len() - 1) as IdxSize
            }),
            Some(s) => *map.entry(s).or_insert_with(|| {
                first.push(row as IdxSize);
                (first.len() - 1) as IdxSize
            }),
        };
        ids.push(id);
    }
    Groups { ids, first }
}

/// Group ids for one or more key columns.
pub(crate) fn compute_groups(keys: &[&Column]) -> Groups {
    let mut groups = group_single(keys[0]);
    for key in &keys[1..] {
        let next = group_single(key);
        // The pair (group so far, group in this column) is the new key.
        let pairs = groups.ids.iter().zip(&next.ids).map(|(&a, &b)| Some((a as u64) << 32 | b as u64));
        groups = group_u64(pairs, None);
    }
    groups
}

/// An aggregation applied to the rows of each group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggKind {
    /// Sum of the non-null values.
    Sum,
    /// Mean of the non-null values.
    Mean,
    /// Smallest non-null value.
    Min,
    /// Largest non-null value.
    Max,
    /// Value of the first row.
    First,
    /// Value of the last row.
    Last,
    /// Number of non-null values.
    Count,
    /// Number of rows.
    Len,
    /// Number of distinct values.
    NUnique,
    /// Median of the non-null values.
    Median,
    /// Variance with the given delta degrees of freedom.
    Var(u8),
    /// Standard deviation with the given delta degrees of freedom.
    Std(u8),
}

impl AggKind {
    /// The suffix Polars' eager group-by appends to the column name.
    pub fn suffix(self) -> &'static str {
        match self {
            AggKind::Sum => "sum",
            AggKind::Mean => "mean",
            AggKind::Min => "min",
            AggKind::Max => "max",
            AggKind::First => "first",
            AggKind::Last => "last",
            AggKind::Count => "count",
            AggKind::Len => "len",
            AggKind::NUnique => "n_unique",
            AggKind::Median => "median",
            AggKind::Var(_) => "var",
            AggKind::Std(_) => "std",
        }
    }

    /// `true` if the aggregation is defined for a column of type `dtype`.
    pub fn supports(self, dtype: &DataType) -> bool {
        match self {
            AggKind::Sum | AggKind::Mean => dtype.is_numeric() || dtype.is_bool(),
            AggKind::Median | AggKind::Var(_) | AggKind::Std(_) => dtype.is_numeric(),
            _ => true,
        }
    }
}

/// Validity bitmap marking the groups that saw at least one value.
fn seen_validity(counts: &[IdxSize]) -> Option<Bitmap> {
    counts.contains(&0).then(|| counts.iter().map(|&c| c > 0).collect())
}

/// Number of non-null values per group, and `f` folded over them.
fn fold_valid<T: PolarsNumericType, A: Clone>(
    ca: &ChunkedArray<T>,
    groups: &Groups,
    init: A,
    mut f: impl FnMut(&mut A, T::Native),
) -> (Vec<A>, Vec<IdxSize>) {
    let mut acc = vec![init; groups.len()];
    let mut counts = vec![0 as IdxSize; groups.len()];
    let values = ca.values().as_slice();
    match ca.validity() {
        None => {
            for (&v, &g) in values.iter().zip(&groups.ids) {
                f(&mut acc[g as usize], v);
                counts[g as usize] += 1;
            }
        }
        Some(valid) => {
            for i in valid.true_indices() {
                let g = groups.ids[i] as usize;
                f(&mut acc[g], values[i]);
                counts[g] += 1;
            }
        }
    }
    (acc, counts)
}

fn numeric_agg<T: PolarsNumericType>(ca: &ChunkedArray<T>, groups: &Groups, kind: AggKind, name: PlSmallStr) -> Series {
    let dtype = <T::Native as NativeType>::DTYPE;
    match kind {
        AggKind::Sum => {
            if matches!(dtype, DataType::Int8 | DataType::Int16 | DataType::UInt8 | DataType::UInt16) {
                // Narrow integers are summed in 64 bits, as in Polars.
                let (acc, _) = fold_valid(ca, groups, 0i64, |a, v| *a = a.wrapping_add(v.to_i64()));
                Int64Chunked::from_vec(name, acc).into_series()
            } else {
                let (acc, _) = fold_valid(ca, groups, T::Native::ZERO, |a, v| *a = a.add(v));
                ChunkedArray::<T>::from_vec(name, acc).into_series()
            }
        }
        AggKind::Mean => {
            let (acc, counts) = fold_valid(ca, groups, 0.0f64, |a, v| *a += v.to_f64());
            let means: Vec<f64> = acc.iter().zip(&counts).map(|(&s, &c)| s / c as f64).collect();
            Float64Chunked::from_vec_validity(name, means, seen_validity(&counts)).into_series()
        }
        AggKind::Min | AggKind::Max => {
            let want_min = kind == AggKind::Min;
            let (acc, counts) = fold_valid(ca, groups, None::<T::Native>, |a, v| {
                // NaN never replaces a number, and any number replaces NaN.
                let better = match *a {
                    None => true,
                    Some(cur) if cur.is_nan() => !v.is_nan(),
                    Some(cur) => !v.is_nan() && if want_min { v.tot_lt(cur) } else { cur.tot_lt(v) },
                };
                if better {
                    *a = Some(v);
                }
            });
            let _ = counts;
            ChunkedArray::<T>::from_iter_options(name, acc).into_series()
        }
        AggKind::Var(ddof) | AggKind::Std(ddof) => {
            let (sums, counts) = fold_valid(ca, groups, 0.0f64, |a, v| *a += v.to_f64());
            let means: Vec<f64> = sums.iter().zip(&counts).map(|(&s, &c)| s / c.max(1) as f64).collect();
            let mut ss = vec![0.0f64; groups.len()];
            let values = ca.values().as_slice();
            let mut add = |i: usize| {
                let g = groups.ids[i] as usize;
                let d = values[i].to_f64() - means[g];
                ss[g] += d * d;
            };
            match ca.validity() {
                None => (0..values.len()).for_each(&mut add),
                Some(valid) => valid.true_indices().for_each(&mut add),
            }
            let out = ss.iter().zip(&counts).map(|(&s, &c)| {
                (c as usize > ddof as usize).then(|| {
                    let var = s / (c as usize - ddof as usize) as f64;
                    if matches!(kind, AggKind::Std(_)) { var.sqrt() } else { var }
                })
            });
            Float64Chunked::from_iter_options(name, out).into_series()
        }
        AggKind::Median => {
            let mut buckets: Vec<Vec<T::Native>> = vec![Vec::new(); groups.len()];
            let values = ca.values().as_slice();
            let mut push = |i: usize| buckets[groups.ids[i] as usize].push(values[i]);
            match ca.validity() {
                None => (0..values.len()).for_each(&mut push),
                Some(valid) => valid.true_indices().for_each(&mut push),
            }
            let medians = buckets.into_iter().map(|mut b| {
                if b.is_empty() {
                    return None;
                }
                let mid = b.len() / 2;
                let even = b.len() % 2 == 0;
                let (lower, pivot, _) = b.select_nth_unstable_by(mid, |x, y| x.tot_cmp(*y));
                let hi = pivot.to_f64();
                if !even {
                    return Some(hi);
                }
                // Even count: the other middle value is the largest of the lower half.
                let lo = lower.iter().copied().max_by(|x, y| x.tot_cmp(*y)).map_or(hi, |v| v.to_f64());
                Some((lo + hi) / 2.0)
            });
            Float64Chunked::from_iter_options(name, medians).into_series()
        }
        _ => unreachable!("type-agnostic aggregations are handled by the caller"),
    }
}

/// Aggregate `column` per group. The result has one row per group, in group order.
pub(crate) fn aggregate(column: &Column, groups: &Groups, kind: AggKind, name: PlSmallStr) -> PolarsResult<Series> {
    if !kind.supports(column.dtype()) {
        polars_bail!(InvalidOperation: "`{}` aggregation is not supported for dtype `{}`", kind.suffix(), column.dtype());
    }
    let n_groups = groups.len();
    Ok(match kind {
        AggKind::Len => UInt32Chunked::from_vec(name, groups.sizes()).into_series(),
        AggKind::Count => {
            let mut counts = vec![0 as IdxSize; n_groups];
            match column.validity() {
                None => return Ok(UInt32Chunked::from_vec(name, groups.sizes()).into_series()),
                Some(valid) => valid.true_indices().for_each(|i| counts[groups.ids[i] as usize] += 1),
            }
            UInt32Chunked::from_vec(name, counts).into_series()
        }
        AggKind::First => column.take_slice(&groups.first)?.with_name(name),
        AggKind::Last => {
            let mut last = groups.first.clone();
            for (i, &g) in groups.ids.iter().enumerate() {
                last[g as usize] = i as IdxSize;
            }
            column.take_slice(&last)?.with_name(name)
        }
        AggKind::NUnique => {
            // Distinct (group, value) pairs, counted per group.
            let values = group_single(column);
            let pairs = groups.ids.iter().zip(&values.ids).map(|(&g, &v)| Some((g as u64) << 32 | v as u64));
            let distinct = group_u64(pairs, None);
            let mut counts = vec![0 as IdxSize; n_groups];
            for &row in &distinct.first {
                counts[groups.ids[row as usize] as usize] += 1;
            }
            UInt32Chunked::from_vec(name, counts).into_series()
        }
        _ => for_numeric!(column, ca => numeric_agg(ca, groups, kind, name), other => match (other, kind) {
            (SeriesInner::Boolean(ca), AggKind::Sum | AggKind::Mean) => {
                let ints = UInt32Chunked::from_iter_options(name.clone(), ca.iter().map(|b| b.map(IdxSize::from)));
                numeric_agg(&ints, groups, kind, name)
            }
            (_, AggKind::Min | AggKind::Max) => {
                // Type-agnostic: find the row holding the extreme of each group.
                let want_min = kind == AggKind::Min;
                let mut best: Vec<Option<IdxSize>> = vec![None; n_groups];
                for (i, &g) in groups.ids.iter().enumerate() {
                    let v = column.get_any(i);
                    if v.is_null() {
                        continue;
                    }
                    let slot = &mut best[g as usize];
                    let better = match *slot {
                        None => true,
                        Some(cur) => {
                            let ord = v.total_cmp(&column.get_any(cur as usize));
                            if want_min { ord.is_lt() } else { ord.is_gt() }
                        }
                    };
                    if better {
                        *slot = Some(i as IdxSize);
                    }
                }
                column.take_opt(&best).with_name(name)
            }
            _ => unreachable!("`supports` rejected the remaining combinations"),
        }),
    })
}

/// The result of [`DataFrame::group_by`]: a frame plus a row → group mapping.
///
/// ```
/// use dataframes::prelude::*;
/// let df = df!("k" => ["a", "b", "a"], "v" => [1, 2, 3])?;
/// let out = df.group_by(["k"])?.select(["v"]).sum()?;
/// assert_eq!(out.column("v_sum")?.i32()?.cont_slice()?, &[4, 2]);
/// # Ok::<(), PolarsError>(())
/// ```
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    by: Vec<PlSmallStr>,
    groups: Groups,
    selected: Option<Vec<PlSmallStr>>,
    /// Append `_sum`, `_mean`, ... to the aggregated column names (Polars' eager API).
    suffix: bool,
    /// Order the result by the key columns (pandas) instead of first appearance.
    sort_keys: bool,
}

impl<'a> GroupBy<'a> {
    pub(crate) fn new(df: &'a DataFrame, by: Vec<PlSmallStr>, suffix: bool, sort_keys: bool) -> PolarsResult<Self> {
        let keys = df.key_columns(&by)?;
        let groups = compute_groups(&keys);
        Ok(GroupBy { df, by, groups, selected: None, suffix, sort_keys })
    }

    /// Restrict the aggregation to these columns.
    pub fn select<I, S>(mut self, selection: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        self.selected = Some(selection.into_iter().map(Into::into).collect());
        self
    }

    /// The row → group mapping.
    pub fn get_groups(&self) -> &Groups {
        &self.groups
    }

    /// Number of groups.
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// The key columns, one row per group.
    pub fn keys(&self) -> Vec<Series> {
        self.by
            .iter()
            .map(|name| self.df.column(name).and_then(|c| c.take_slice(&self.groups.first)).expect("keys were validated"))
            .collect()
    }

    fn finish(&self, mut columns: Vec<Series>) -> PolarsResult<DataFrame> {
        let mut out = self.keys();
        out.append(&mut columns);
        let df = DataFrame::new(self.groups.len(), out)?;
        if self.sort_keys {
            return df.sort(&self.by, super::SortMultipleOptions::default());
        }
        Ok(df)
    }

    /// Apply one aggregation to every selected (or every eligible) column.
    pub fn agg_kind(&self, kind: AggKind) -> PolarsResult<DataFrame> {
        let mut columns = Vec::new();
        match &self.selected {
            Some(names) => {
                for name in names {
                    let col = self.df.column(name)?;
                    columns.push(aggregate(col, &self.groups, kind, self.out_name(name, kind))?);
                }
            }
            None => {
                for col in self.df.columns().iter().filter(|c| !self.by.contains(c.name()) && kind.supports(c.dtype())) {
                    columns.push(aggregate(col, &self.groups, kind, self.out_name(col.name(), kind))?);
                }
            }
        }
        self.finish(columns)
    }

    fn out_name(&self, name: &PlSmallStr, kind: AggKind) -> PlSmallStr {
        if self.suffix { format!("{name}_{}", kind.suffix()).into() } else { name.clone() }
    }

    /// Several aggregations at once: `[("col", &[AggKind::Sum, AggKind::Mean])]`.
    /// Output columns are named `col_sum`, `col_mean`, ...
    pub fn agg(&self, spec: &[(&str, &[AggKind])]) -> PolarsResult<DataFrame> {
        let mut columns = Vec::new();
        for (name, kinds) in spec {
            let col = self.df.column(name)?;
            for &kind in *kinds {
                columns.push(aggregate(col, &self.groups, kind, format!("{name}_{}", kind.suffix()).into())?);
            }
        }
        self.finish(columns)
    }

    /// Per-group sum.
    pub fn sum(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Sum)
    }

    /// Per-group mean.
    pub fn mean(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Mean)
    }

    /// Per-group minimum.
    pub fn min(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Min)
    }

    /// Per-group maximum.
    pub fn max(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Max)
    }

    /// First row of every group.
    pub fn first(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::First)
    }

    /// Last row of every group.
    pub fn last(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Last)
    }

    /// Per-group number of non-null values.
    pub fn count(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Count)
    }

    /// Per-group number of distinct values.
    pub fn n_unique(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::NUnique)
    }

    /// Per-group median.
    pub fn median(&self) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Median)
    }

    /// Per-group variance.
    pub fn var(&self, ddof: u8) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Var(ddof))
    }

    /// Per-group standard deviation.
    pub fn std(&self, ddof: u8) -> PolarsResult<DataFrame> {
        self.agg_kind(AggKind::Std(ddof))
    }

    /// Number of rows per group, in a column called `len`.
    pub fn len(&self) -> PolarsResult<DataFrame> {
        self.finish(vec![UInt32Chunked::from_vec("len".into(), self.groups.sizes()).into_series()])
    }

    /// pandas name for [`len`](Self::len); the column is called `size`.
    pub fn size(&self) -> PolarsResult<DataFrame> {
        self.finish(vec![UInt32Chunked::from_vec("size".into(), self.groups.sizes()).into_series()])
    }

    /// Call `f` on the sub-frame of every group and stack the results.
    pub fn apply<F>(&self, f: F) -> PolarsResult<DataFrame>
    where
        F: Fn(DataFrame) -> PolarsResult<DataFrame>,
    {
        let mut out = DataFrame::empty();
        for rows in self.groups.rows() {
            out.vstack_mut(&f(self.df.take_slice(&rows)?)?)?;
        }
        Ok(out)
    }
}

impl DataFrame {
    /// Group the rows by the columns `by`.
    ///
    /// Groups come out in order of first appearance.
    pub fn group_by<I, S>(&self, by: I) -> PolarsResult<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        GroupBy::new(self, by.into_iter().map(Into::into).collect(), true, false)
    }

    /// Same as [`group_by`](Self::group_by), which is always stable.
    pub fn group_by_stable<I, S>(&self, by: I) -> PolarsResult<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        self.group_by(by)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    fn sample() -> DataFrame {
        df!(
            "k" => ["a", "b", "a", "c", "b", "a"],
            "g" => [1i32, 1, 2, 1, 1, 1],
            "v" => [Some(1.0f64), Some(2.0), Some(3.0), None, Some(6.0), Some(5.0)],
            "n" => [10i64, 20, 30, 40, 50, 60],
        )
        .unwrap()
    }

    #[test]
    fn group_ids_are_dense_and_ordered_by_first_appearance() {
        let df = sample();
        let g = compute_groups(&[df.column("k").unwrap()]);
        assert_eq!(g.ids, [0, 1, 0, 2, 1, 0]);
        assert_eq!(g.first, [0, 1, 3]);
        assert_eq!(g.sizes(), [3, 2, 1]);
        let two = compute_groups(&[df.column("k").unwrap(), df.column("g").unwrap()]);
        assert_eq!(two.ids, [0, 1, 2, 3, 1, 0]);
        let nulls = Series::new("x".into(), [Some(5i64), None, Some(5), None]);
        assert_eq!(compute_groups(&[&nulls]).ids, [0, 1, 0, 1]);
        let floats = Series::new("f".into(), [0.0f64, -0.0, f64::NAN, f64::NAN, 1.5]);
        assert_eq!(compute_groups(&[&floats]).ids, [0, 0, 1, 1, 2]);
        // Wide key range forces the hash path.
        let wide = Series::new("w".into(), [i64::MIN, i64::MAX, 0, i64::MAX, i64::MIN]);
        assert_eq!(compute_groups(&[&wide]).ids, [0, 1, 2, 1, 0]);
        let many: Vec<u64> = (0..50_000u64).map(|i| (i * 2_654_435_761) % 7001 * 1_000_003).collect();
        let g = compute_groups(&[&Series::new("m".into(), &many)]);
        assert_eq!(g.len(), 7001);
        assert!(g.ids.iter().zip(&many).all(|(&id, &v)| many[g.first[id as usize] as usize] == v));
    }

    #[test]
    fn eager_aggregations_use_polars_names() {
        let df = sample();
        let out = df.group_by(["k"]).unwrap().select(["v", "n"]).sum().unwrap();
        assert_eq!(out.get_column_names_str(), ["k", "v_sum", "n_sum"]);
        assert_eq!(out["k"].str().unwrap().to_vec(), vec![Some("a"), Some("b"), Some("c")]);
        assert_eq!(out["v_sum"].f64().unwrap().cont_slice().unwrap(), &[9.0, 8.0, 0.0]);
        assert_eq!(out["n_sum"].i64().unwrap().cont_slice().unwrap(), &[100, 70, 40]);

        let gb = df.group_by(["k"]).unwrap();
        assert_eq!(gb.n_groups(), 3);
        assert_eq!(gb.mean().unwrap()["v_mean"].f64().unwrap().to_vec(), vec![Some(3.0), Some(4.0), None]);
        assert_eq!(gb.min().unwrap()["v_min"].f64().unwrap().to_vec(), vec![Some(1.0), Some(2.0), None]);
        assert_eq!(gb.max().unwrap()["n_max"].i64().unwrap().cont_slice().unwrap(), &[60, 50, 40]);
        assert_eq!(gb.first().unwrap()["n_first"].i64().unwrap().cont_slice().unwrap(), &[10, 20, 40]);
        assert_eq!(gb.last().unwrap()["n_last"].i64().unwrap().cont_slice().unwrap(), &[60, 50, 40]);
        assert_eq!(gb.count().unwrap()["v_count"].u32().unwrap().cont_slice().unwrap(), &[3, 2, 0]);
        assert_eq!(gb.len().unwrap()["len"].u32().unwrap().cont_slice().unwrap(), &[3, 2, 1]);
        assert_eq!(gb.n_unique().unwrap()["g_n_unique"].u32().unwrap().cont_slice().unwrap(), &[2, 1, 1]);
        assert_eq!(gb.median().unwrap()["v_median"].f64().unwrap().to_vec(), vec![Some(3.0), Some(4.0), None]);
        assert_eq!(gb.var(1).unwrap()["v_var"].f64().unwrap().to_vec(), vec![Some(4.0), Some(8.0), None]);
        assert_eq!(gb.std(0).unwrap()["n_std"].f64().unwrap().get(1), Some(15.0));
        // Unsupported columns are skipped unless asked for explicitly.
        assert!(!gb.sum().unwrap().get_column_names_str().contains(&"k_sum"));
        assert!(df.group_by(["g"]).unwrap().select(["k"]).sum().is_err());
        assert_eq!(df.group_by(["g"]).unwrap().select(["k"]).max().unwrap()["k_max"].str().unwrap().to_vec(), vec![Some("c"), Some("a")]);
        assert!(df.group_by(["nope"]).is_err());
    }

    #[test]
    fn multiple_keys_agg_spec_and_apply() {
        let df = sample();
        let out = df.group_by(["k", "g"]).unwrap().agg(&[("n", &[AggKind::Sum, AggKind::Len]), ("v", &[AggKind::Max])]).unwrap();
        assert_eq!(out.get_column_names_str(), ["k", "g", "n_sum", "n_len", "v_max"]);
        assert_eq!(out.height(), 4);
        assert_eq!(out["n_sum"].i64().unwrap().cont_slice().unwrap(), &[70, 70, 30, 40]);
        assert_eq!(out["n_len"].u32().unwrap().cont_slice().unwrap(), &[2, 2, 1, 1]);
        let heads = df.group_by(["k"]).unwrap().apply(|g| Ok(g.head(1))).unwrap();
        assert_eq!(heads["n"].i64().unwrap().cont_slice().unwrap(), &[10, 20, 40]);
        let narrow = df!("k" => [1u8, 1, 2], "x" => [200u8, 100, 7]).unwrap();
        let s = narrow.group_by(["k"]).unwrap().sum().unwrap();
        assert_eq!(s["x_sum"].i64().unwrap().cont_slice().unwrap(), &[300, 7]);
    }

    #[test]
    fn median_handles_even_and_odd_groups() {
        let df = df!("k" => [1, 1, 1, 1, 2, 2, 2, 3], "v" => [4.0, 1.0, 3.0, 2.0, 9.0, 7.0, 8.0, 5.0]).unwrap();
        let out = df.group_by(["k"]).unwrap().median().unwrap();
        assert_eq!(out["v_median"].f64().unwrap().cont_slice().unwrap(), &[2.5, 8.0, 5.0]);
    }
}
