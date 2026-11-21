//! [`LazyFrame`]: a query plan that runs on `collect`.
//!
//! The plan is a small tree. Running it applies three optimisations that
//! matter for the memory layout this crate is built around:
//!
//! * **projection pushdown** — every node is told which columns its parent
//!   needs, so a filter only compacts the columns that survive to the output;
//! * **filter fusion** — consecutive filters become one predicate, which is
//!   compiled and fused with the compaction (one pass over the data);
//! * **filtered aggregation** — `filter(..).select([aggregates])` reduces the
//!   selected rows block by block without materialising them.

use super::dsl::Expr;
use super::eval::{broadcast, eval};
use super::exec::{aggregate_filtered, filter_frame};
use crate::datatypes::{IdxSize, PlSmallStr};
use crate::error::PolarsResult;
use crate::frame::{
    AggKind, DataFrame, FilterPredicate, JoinArgs, JoinType, SortMultipleOptions, UniqueKeepStrategy, aggregate,
    compute_groups,
};
use crate::series::{Column, IntoSeries, Series};
use crate::{polars_bail, polars_ensure};

#[derive(Clone)]
enum Plan {
    Scan(DataFrame),
    Filter { input: Box<Plan>, predicate: Expr },
    Select { input: Box<Plan>, exprs: Vec<Expr> },
    WithColumns { input: Box<Plan>, exprs: Vec<Expr> },
    GroupBy { input: Box<Plan>, keys: Vec<Expr>, aggs: Vec<Expr> },
    Sort { input: Box<Plan>, by: Vec<PlSmallStr>, options: SortMultipleOptions },
    Slice { input: Box<Plan>, offset: i64, len: usize },
    Unique { input: Box<Plan>, subset: Option<Vec<String>>, keep: UniqueKeepStrategy },
    DropNulls { input: Box<Plan>, subset: Option<Vec<PlSmallStr>> },
    Join { left: Box<Plan>, right: Box<Plan>, left_on: Vec<PlSmallStr>, right_on: Vec<PlSmallStr>, args: JoinArgs },
    Rename { input: Box<Plan>, existing: Vec<PlSmallStr>, new: Vec<PlSmallStr> },
    Drop { input: Box<Plan>, columns: Vec<PlSmallStr> },
    Reverse { input: Box<Plan> },
    Reduce { input: Box<Plan>, kind: AggKind },
}

/// A lazily evaluated query over a [`DataFrame`].
///
/// ```
/// use dataframes::prelude::*;
/// let df = df!("k" => ["a", "b", "a"], "v" => [1.0, 2.0, 3.0])?;
/// let out = df
///     .lazy()
///     .filter(col("v").gt(1.0))
///     .group_by([col("k")])
///     .agg([col("v").sum().alias("total")])
///     .collect()?;
/// assert_eq!(out.column("total")?.f64()?.cont_slice()?, &[2.0, 3.0]);
/// # Ok::<(), PolarsError>(())
/// ```
#[derive(Clone)]
pub struct LazyFrame {
    plan: Plan,
}

/// Conversion into a [`LazyFrame`].
pub trait IntoLazy {
    /// Start a lazy query.
    fn lazy(self) -> LazyFrame;
}

impl IntoLazy for DataFrame {
    fn lazy(self) -> LazyFrame {
        LazyFrame { plan: Plan::Scan(self) }
    }
}

/// The result of [`LazyFrame::group_by`], waiting for its aggregations.
#[derive(Clone)]
pub struct LazyGroupBy {
    input: Plan,
    keys: Vec<Expr>,
}

impl LazyGroupBy {
    /// Aggregate each group. Every expression must be an aggregation
    /// (`col("x").sum()`, `len()`, ...), optionally aliased.
    pub fn agg<E: AsRef<[Expr]>>(self, aggs: E) -> LazyFrame {
        LazyFrame { plan: Plan::GroupBy { input: Box::new(self.input), keys: self.keys, aggs: aggs.as_ref().to_vec() } }
    }
}

fn column_names(exprs: &[Expr], what: &str) -> PolarsResult<Vec<PlSmallStr>> {
    exprs
        .iter()
        .map(|e| match e {
            Expr::Column(name) => Ok(name.clone()),
            other => polars_bail!(InvalidOperation: "{} must be plain columns, got {}", what, other),
        })
        .collect()
}

impl LazyFrame {
    fn wrap(self, f: impl FnOnce(Box<Plan>) -> Plan) -> Self {
        LazyFrame { plan: f(Box::new(self.plan)) }
    }

    /// Keep the rows for which `predicate` is true.
    pub fn filter(self, predicate: Expr) -> Self {
        self.wrap(|input| Plan::Filter { input, predicate })
    }

    /// Compute the given expressions as the new set of columns.
    pub fn select<E: AsRef<[Expr]>>(self, exprs: E) -> Self {
        self.wrap(|input| Plan::Select { input, exprs: exprs.as_ref().to_vec() })
    }

    /// Add (or replace) columns.
    pub fn with_columns<E: AsRef<[Expr]>>(self, exprs: E) -> Self {
        self.wrap(|input| Plan::WithColumns { input, exprs: exprs.as_ref().to_vec() })
    }

    /// Add (or replace) one column.
    pub fn with_column(self, expr: Expr) -> Self {
        self.with_columns([expr])
    }

    /// Group by the given key columns.
    pub fn group_by<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, by: E) -> LazyGroupBy {
        LazyGroupBy { input: self.plan, keys: by.as_ref().iter().cloned().map(Into::into).collect() }
    }

    /// Same as [`group_by`](Self::group_by): groups always keep their order of first appearance.
    pub fn group_by_stable<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, by: E) -> LazyGroupBy {
        self.group_by(by)
    }

    /// Sort by the given columns.
    pub fn sort(self, by: impl IntoIterator<Item = impl Into<PlSmallStr>>, sort_options: SortMultipleOptions) -> Self {
        let by = by.into_iter().map(Into::into).collect();
        self.wrap(|input| Plan::Sort { input, by, options: sort_options })
    }

    /// A window of rows.
    pub fn slice(self, offset: i64, len: IdxSize) -> Self {
        self.wrap(|input| Plan::Slice { input, offset, len: len as usize })
    }

    /// The first `n` rows.
    pub fn limit(self, n: IdxSize) -> Self {
        self.slice(0, n)
    }

    /// The first `n` rows.
    pub fn head(self, n: IdxSize) -> Self {
        self.slice(0, n)
    }

    /// The last `n` rows.
    pub fn tail(self, n: IdxSize) -> Self {
        self.slice(-(n as i64), n)
    }

    /// The first row.
    pub fn first(self) -> Self {
        self.slice(0, 1)
    }

    /// The last row.
    pub fn last(self) -> Self {
        self.slice(-1, 1)
    }

    /// Rows in reverse order.
    pub fn reverse(self) -> Self {
        self.wrap(|input| Plan::Reverse { input })
    }

    /// Distinct rows (by `subset`, or all columns).
    pub fn unique(self, subset: Option<Vec<String>>, keep: UniqueKeepStrategy) -> Self {
        self.wrap(|input| Plan::Unique { input, subset, keep })
    }

    /// Same as [`unique`](Self::unique), which always keeps the row order.
    pub fn unique_stable(self, subset: Option<Vec<String>>, keep: UniqueKeepStrategy) -> Self {
        self.unique(subset, keep)
    }

    /// Drop the rows with a null in any of the `subset` columns (all columns if `None`).
    pub fn drop_nulls(self, subset: Option<Vec<Expr>>) -> Self {
        let subset = subset.map(|exprs| {
            exprs.iter().map(|e| e.output_name()).collect::<Vec<_>>()
        });
        self.wrap(|input| Plan::DropNulls { input, subset })
    }

    /// Join with `other` on equality of the key columns.
    pub fn join<E: AsRef<[Expr]>>(self, other: LazyFrame, left_on: E, right_on: E, args: JoinArgs) -> Self {
        let names = |e: &[Expr]| e.iter().map(Expr::output_name).collect::<Vec<_>>();
        let (left_on, right_on) = (names(left_on.as_ref()), names(right_on.as_ref()));
        LazyFrame { plan: Plan::Join { left: Box::new(self.plan), right: Box::new(other.plan), left_on, right_on, args } }
    }

    /// Inner join on one key per side.
    pub fn inner_join(self, other: LazyFrame, left_on: Expr, right_on: Expr) -> Self {
        self.join(other, [left_on], [right_on], JoinArgs::new(JoinType::Inner))
    }

    /// Left join on one key per side.
    pub fn left_join(self, other: LazyFrame, left_on: Expr, right_on: Expr) -> Self {
        self.join(other, [left_on], [right_on], JoinArgs::new(JoinType::Left))
    }

    /// Full join on one key per side.
    pub fn full_join(self, other: LazyFrame, left_on: Expr, right_on: Expr) -> Self {
        self.join(other, [left_on], [right_on], JoinArgs::new(JoinType::Full))
    }

    /// Rename columns. With `strict`, a missing column is an error.
    pub fn rename<I, J, T, S>(self, existing: I, new: J, strict: bool) -> Self
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator<Item = S>,
        T: AsRef<str>,
        S: AsRef<str>,
    {
        let existing: Vec<PlSmallStr> = existing.into_iter().map(|s| s.as_ref().into()).collect();
        let new: Vec<PlSmallStr> = new.into_iter().map(|s| s.as_ref().into()).collect();
        let _ = strict;
        self.wrap(|input| Plan::Rename { input, existing, new })
    }

    /// Remove columns.
    pub fn drop<I, S>(self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let columns = columns.into_iter().map(Into::into).collect();
        self.wrap(|input| Plan::Drop { input, columns })
    }

    fn reduce(self, kind: AggKind) -> Self {
        self.wrap(|input| Plan::Reduce { input, kind })
    }

    /// Column sums (one row).
    pub fn sum(self) -> Self {
        self.reduce(AggKind::Sum)
    }

    /// Column means (one row).
    pub fn mean(self) -> Self {
        self.reduce(AggKind::Mean)
    }

    /// Column minima (one row).
    pub fn min(self) -> Self {
        self.reduce(AggKind::Min)
    }

    /// Column maxima (one row).
    pub fn max(self) -> Self {
        self.reduce(AggKind::Max)
    }

    /// Column medians (one row).
    pub fn median(self) -> Self {
        self.reduce(AggKind::Median)
    }

    /// Column standard deviations (one row).
    pub fn std(self, ddof: u8) -> Self {
        self.reduce(AggKind::Std(ddof))
    }

    /// Column variances (one row).
    pub fn var(self, ddof: u8) -> Self {
        self.reduce(AggKind::Var(ddof))
    }

    /// Non-null counts (one row).
    pub fn count(self) -> Self {
        self.reduce(AggKind::Count)
    }

    /// No-op kept for Polars compatibility (results are not cached implicitly).
    pub fn cache(self) -> Self {
        self
    }

    /// Run the query.
    pub fn collect(self) -> PolarsResult<DataFrame> {
        run(&self.plan, None)
    }

    /// Describe the plan. With `optimized`, adjacent filters are shown merged
    /// and annotated the way they will execute.
    pub fn explain(&self, optimized: bool) -> PolarsResult<String> {
        let mut out = String::new();
        describe(&self.plan, 0, optimized, &mut out);
        Ok(out.trim_end().to_string())
    }
}

/// Merge `Filter(Filter(..))` chains into one predicate and return the node below them.
fn peel_filters(plan: &Plan) -> (Option<Expr>, &Plan) {
    let mut predicate: Option<Expr> = None;
    let mut node = plan;
    while let Plan::Filter { input, predicate: p } = node {
        // The inner filter runs first, so it goes on the left of the AND.
        predicate = Some(match predicate {
            None => p.clone(),
            Some(outer) => p.clone().and(outer),
        });
        node = input;
    }
    (predicate, node)
}

fn describe(plan: &Plan, depth: usize, optimized: bool, out: &mut String) {
    let pad = "  ".repeat(depth);
    let list = |exprs: &[Expr]| exprs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ");
    match plan {
        Plan::Scan(df) => out.push_str(&format!("{pad}DF {:?}; shape: {:?}\n", df.get_column_names_str(), df.shape())),
        Plan::Filter { input, predicate } => {
            if optimized {
                let (merged, below) = peel_filters(plan);
                out.push_str(&format!("{pad}FILTER {} [fused with compaction]\n", merged.expect("at least one filter")));
                describe(below, depth + 1, optimized, out);
            } else {
                out.push_str(&format!("{pad}FILTER {predicate}\n"));
                describe(input, depth + 1, optimized, out);
            }
        }
        Plan::Select { input, exprs } => {
            let fused = optimized && matches!(**input, Plan::Filter { .. }) && exprs.iter().all(Expr::is_aggregation);
            out.push_str(&format!("{pad}SELECT [{}]{}\n", list(exprs), if fused { " [aggregated while filtering]" } else { "" }));
            describe(input, depth + 1, optimized, out);
        }
        Plan::WithColumns { input, exprs } => {
            out.push_str(&format!("{pad}WITH_COLUMNS [{}]\n", list(exprs)));
            describe(input, depth + 1, optimized, out);
        }
        Plan::GroupBy { input, keys, aggs } => {
            out.push_str(&format!("{pad}AGGREGATE [{}] BY [{}]\n", list(aggs), list(keys)));
            describe(input, depth + 1, optimized, out);
        }
        Plan::Sort { input, by, .. } => {
            out.push_str(&format!("{pad}SORT BY {by:?}\n"));
            describe(input, depth + 1, optimized, out);
        }
        Plan::Slice { input, offset, len } => {
            out.push_str(&format!("{pad}SLICE offset: {offset}, len: {len}\n"));
            describe(input, depth + 1, optimized, out);
        }
        Plan::Unique { input, subset, .. } => {
            out.push_str(&format!("{pad}UNIQUE BY {subset:?}\n"));
            describe(input, depth + 1, optimized, out);
        }
        Plan::DropNulls { input, subset } => {
            out.push_str(&format!("{pad}DROP_NULLS {subset:?}\n"));
            describe(input, depth + 1, optimized, out);
        }
        Plan::Join { left, right, left_on, right_on, args } => {
            out.push_str(&format!("{pad}{:?} JOIN ON {left_on:?} = {right_on:?}\n", args.how));
            describe(left, depth + 1, optimized, out);
            describe(right, depth + 1, optimized, out);
        }
        Plan::Rename { input, existing, new } => {
            out.push_str(&format!("{pad}RENAME {existing:?} -> {new:?}\n"));
            describe(input, depth + 1, optimized, out);
        }
        Plan::Drop { input, columns } => {
            out.push_str(&format!("{pad}DROP {columns:?}\n"));
            describe(input, depth + 1, optimized, out);
        }
        Plan::Reverse { input } => {
            out.push_str(&format!("{pad}REVERSE\n"));
            describe(input, depth + 1, optimized, out);
        }
        Plan::Reduce { input, kind } => {
            out.push_str(&format!("{pad}REDUCE {}\n", kind.suffix()));
            describe(input, depth + 1, optimized, out);
        }
    }
}

fn union(a: &[PlSmallStr], b: &[PlSmallStr]) -> Vec<PlSmallStr> {
    let mut out = a.to_vec();
    for name in b {
        if !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

fn expr_columns(exprs: &[Expr]) -> Vec<PlSmallStr> {
    let mut out = Vec::new();
    for e in exprs {
        e.collect_columns(&mut out);
    }
    out
}

/// Replace top-level wildcards by one expression per column of `df`.
fn expand_wildcards(df: &DataFrame, exprs: &[Expr]) -> PolarsResult<Vec<Expr>> {
    fn substitute(e: &Expr, name: &PlSmallStr) -> Expr {
        match e {
            Expr::Wildcard => Expr::Column(name.clone()),
            Expr::Column(_) | Expr::Literal(_) | Expr::Len => e.clone(),
            Expr::Binary { left, op, right } => {
                Expr::Binary { left: Box::new(substitute(left, name)), op: *op, right: Box::new(substitute(right, name)) }
            }
            Expr::Ternary { predicate, truthy, falsy } => Expr::Ternary {
                predicate: Box::new(substitute(predicate, name)),
                truthy: Box::new(substitute(truthy, name)),
                falsy: Box::new(substitute(falsy, name)),
            },
            Expr::Not(x) => Expr::Not(Box::new(substitute(x, name))),
            Expr::Neg(x) => Expr::Neg(Box::new(substitute(x, name))),
            Expr::IsNull(x) => Expr::IsNull(Box::new(substitute(x, name))),
            Expr::IsNotNull(x) => Expr::IsNotNull(Box::new(substitute(x, name))),
            Expr::Alias(x, alias) => Expr::Alias(Box::new(substitute(x, name)), alias.clone()),
            Expr::Agg { input, kind } => Expr::Agg { input: Box::new(substitute(input, name)), kind: *kind },
            Expr::Cast { expr, dtype } => Expr::Cast { expr: Box::new(substitute(expr, name)), dtype: *dtype },
        }
    }
    let mut out = Vec::with_capacity(exprs.len());
    for e in exprs {
        if e.has_wildcard() {
            out.extend(df.columns().iter().map(|c| substitute(e, c.name())));
        } else {
            out.push(e.clone());
        }
    }
    Ok(out)
}

/// Evaluate projection expressions against `df` into a new frame.
pub(crate) fn project(df: &DataFrame, exprs: &[Expr]) -> PolarsResult<DataFrame> {
    let exprs = expand_wildcards(df, exprs)?;
    let mut columns: Vec<Series> = Vec::with_capacity(exprs.len());
    for e in &exprs {
        let name = e.output_name();
        columns.push(eval(df, e)?.with_name(name));
    }
    // Scalars (aggregates, literals) stretch to the height of the longest column.
    let all_scalar = exprs.iter().all(|e| e.is_aggregation() || matches!(e, Expr::Literal(_)));
    let height = if all_scalar { 1 } else { df.height() };
    let columns = columns.into_iter().map(|c| broadcast(c, height)).collect::<PolarsResult<Vec<_>>>()?;
    DataFrame::new(height, columns)
}

/// Add or replace columns computed from `exprs`.
pub(crate) fn with_columns(df: &DataFrame, exprs: &[Expr]) -> PolarsResult<DataFrame> {
    let exprs = expand_wildcards(df, exprs)?;
    let mut out = df.clone();
    for e in &exprs {
        let name = e.output_name();
        // Every expression sees the input frame, not earlier outputs.
        let column = broadcast(eval(df, e)?.with_name(name), df.height())?;
        out.with_column(column)?;
    }
    Ok(out)
}

fn group_by(df: &DataFrame, keys: &[Expr], aggs: &[Expr]) -> PolarsResult<DataFrame> {
    let key_names = column_names(keys, "group-by keys")?;
    let key_cols: Vec<&Column> = key_names.iter().map(|n| df.column(n)).collect::<PolarsResult<_>>()?;
    polars_ensure!(!key_cols.is_empty(), ComputeError: "at least one key is required in a group_by operation");
    let groups = compute_groups(&key_cols);
    let mut columns: Vec<Series> =
        key_cols.iter().map(|c| c.take_slice(&groups.first)).collect::<PolarsResult<_>>()?;
    for e in aggs {
        let name = e.output_name();
        let inner = match e {
            Expr::Alias(inner, _) => &**inner,
            other => other,
        };
        let column = match inner {
            Expr::Len => crate::chunked_array::UInt32Chunked::from_vec(name, groups.sizes()).into_series(),
            Expr::Agg { input, kind } => {
                let values = eval(df, input)?;
                polars_ensure!(
                    values.len() == df.height(),
                    ComputeError: "cannot aggregate {}: its input does not have one value per row", e
                );
                aggregate(&values, &groups, *kind, name)?
            }
            other => polars_bail!(InvalidOperation: "{} is not an aggregation; use e.g. `.sum()`, `.first()` or `len()`", other),
        };
        columns.push(column);
    }
    DataFrame::new(groups.len(), columns)
}

/// Execute `plan`, producing at least the columns in `needed` (all if `None`).
fn run(plan: &Plan, needed: Option<&[PlSmallStr]>) -> PolarsResult<DataFrame> {
    match plan {
        Plan::Scan(df) => match needed {
            // Zero-copy: only the buffer handles of the needed columns are cloned.
            Some(names) => df.select(names.iter().map(|n| n.as_str())),
            None => Ok(df.clone()),
        },
        Plan::Filter { .. } => {
            let (predicate, below) = peel_filters(plan);
            let predicate = predicate.expect("a filter node has a predicate");
            let below_needed = needed.map(|n| union(n, &expr_columns(std::slice::from_ref(&predicate))));
            let df = run(below, below_needed.as_deref())?;
            filter_frame(&df, &predicate, needed)
        }
        Plan::Select { input, exprs } => {
            if exprs.iter().any(Expr::has_wildcard) {
                return project(&run(input, None)?, exprs);
            }
            let cols = expr_columns(exprs);
            if let (Plan::Filter { .. }, true) = (&**input, !exprs.is_empty() && exprs.iter().all(Expr::is_aggregation)) {
                // Aggregates over a filter: reduce while filtering.
                let (predicate, below) = peel_filters(input);
                let predicate = predicate.expect("a filter node has a predicate");
                let df = run(below, Some(&union(&cols, &expr_columns(std::slice::from_ref(&predicate)))))?;
                if let Some(out) = aggregate_filtered(&df, &predicate, exprs)? {
                    return Ok(out);
                }
                return project(&filter_frame(&df, &predicate, Some(&cols))?, exprs);
            }
            project(&run(input, Some(&cols))?, exprs)
        }
        Plan::WithColumns { input, exprs } => {
            if exprs.iter().any(Expr::has_wildcard) {
                return with_columns(&run(input, None)?, exprs);
            }
            let below_needed = needed.map(|n| union(n, &expr_columns(exprs)));
            // Columns that only exist as outputs of this node are not asked of the input.
            let outputs: Vec<PlSmallStr> = exprs.iter().map(Expr::output_name).collect();
            let inputs = expr_columns(exprs);
            let below_needed =
                below_needed.map(|n| n.into_iter().filter(|c| !outputs.contains(c) || inputs.contains(c)).collect::<Vec<_>>());
            with_columns(&run(input, below_needed.as_deref())?, exprs)
        }
        Plan::GroupBy { input, keys, aggs } => {
            let wildcard = keys.iter().chain(aggs).any(Expr::has_wildcard);
            polars_ensure!(!wildcard, InvalidOperation: "wildcards are not supported in group_by");
            let cols = union(&expr_columns(keys), &expr_columns(aggs));
            group_by(&run(input, Some(&cols))?, keys, aggs)
        }
        Plan::Sort { input, by, options } => {
            let below_needed = needed.map(|n| union(n, by));
            run(input, below_needed.as_deref())?.sort(by, options.clone())
        }
        Plan::Slice { input, offset, len } => {
            if let (Plan::Sort { input: sort_input, by, options }, true) = (&**input, *offset == 0) {
                // Top-k: only the first `len` rows of the permutation are gathered.
                let below_needed = needed.map(|n| union(n, by));
                let limit = IdxSize::try_from(*len).unwrap_or(IdxSize::MAX);
                return run(sort_input, below_needed.as_deref())?.sort(by, options.clone().with_limit(Some(limit)));
            }
            Ok(run(input, needed)?.slice(*offset, *len))
        }
        Plan::Unique { input, subset, keep } => run(input, None)?.unique_stable(subset.as_deref(), *keep, None),
        Plan::DropNulls { input, subset } => {
            let below_needed = match (needed, subset) {
                (Some(n), Some(s)) => Some(union(n, s)),
                _ => None,
            };
            run(input, below_needed.as_deref())?.drop_nulls(subset.as_deref())
        }
        Plan::Join { left, right, left_on, right_on, args } => {
            let (l, r) = (run(left, None)?, run(right, None)?);
            l.join(&r, left_on, right_on, args.clone(), None)
        }
        Plan::Rename { input, existing, new } => {
            polars_ensure!(existing.len() == new.len(), ComputeError: "`rename` needs as many new names as existing ones");
            let mut df = run(input, None)?;
            for (old, new) in existing.iter().zip(new) {
                df.rename(old, new.clone())?;
            }
            Ok(df)
        }
        Plan::Drop { input, columns } => Ok(run(input, None)?.drop_many(columns.iter().cloned())),
        Plan::Reverse { input } => Ok(run(input, needed)?.reverse()),
        Plan::Reduce { input, kind } => {
            let df = run(input, needed)?;
            match kind {
                AggKind::Sum => df.sum(),
                AggKind::Mean => df.mean(),
                AggKind::Min => df.min(),
                AggKind::Max => df.max(),
                AggKind::Median => df.median(),
                AggKind::Std(ddof) => df.std(*ddof),
                AggKind::Var(ddof) => df.var(*ddof),
                AggKind::Count => Ok(df.count()),
                other => polars_bail!(InvalidOperation: "`{}` is not a whole-frame reduction", other.suffix()),
            }
        }
    }
}

impl FilterPredicate for Expr {
    fn filter_frame(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        filter_frame(df, &self, None)
    }
}

impl FilterPredicate for &Expr {
    fn filter_frame(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        filter_frame(df, self, None)
    }
}

impl DataFrame {
    /// Compute expressions as a new frame (Python Polars' `df.select(exprs)`).
    pub fn select_exprs<E: AsRef<[Expr]>>(&self, exprs: E) -> PolarsResult<DataFrame> {
        project(self, exprs.as_ref())
    }

    /// Add or replace columns computed from expressions (Python Polars'
    /// `df.with_columns(...)`, pandas' `df.assign(...)`).
    pub fn with_columns<E: AsRef<[Expr]>>(&self, exprs: E) -> PolarsResult<DataFrame> {
        with_columns(self, exprs.as_ref())
    }

    /// Evaluate one expression against the frame.
    pub fn eval_expr(&self, expr: &Expr) -> PolarsResult<Series> {
        Ok(eval(self, expr)?.with_name(expr.output_name()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::dsl::{all, col, len, lit, when};
    use super::*;
    use crate::chunked_array::ChunkCompareIneq;
    use crate::df;

    fn sample() -> DataFrame {
        df!(
            "k" => ["a", "b", "a", "c", "b", "a"],
            "v" => [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            "n" => [Some(10i64), None, Some(30), Some(40), Some(50), Some(60)],
        )
        .unwrap()
    }

    #[test]
    fn filter_select_and_projection_pushdown() {
        let df = sample();
        let out = df.clone().lazy().filter(col("v").gt(2.0)).select([col("k"), col("v") * 10]).collect().unwrap();
        assert_eq!(out.get_column_names_str(), ["k", "v"]);
        assert_eq!(out["v"].f64().unwrap().cont_slice().unwrap(), &[30.0, 40.0, 50.0, 60.0]);
        // Two filters are merged; the result equals filtering twice.
        let twice = df.clone().lazy().filter(col("v").gt(1.0)).filter(col("k").neq("b")).collect().unwrap();
        assert_eq!(twice["v"].f64().unwrap().cont_slice().unwrap(), &[3.0, 4.0, 6.0]);
        assert_eq!(twice.width(), 3);
        // The predicate column need not be in the output.
        let narrow = df.clone().lazy().filter(col("n").gt(20)).select([col("k")]).collect().unwrap();
        assert_eq!(narrow["k"].str().unwrap().to_vec(), vec![Some("a"), Some("c"), Some("b"), Some("a")]);
        assert_eq!(df.clone().lazy().select([all()]).collect().unwrap(), df);
        let doubled = df.clone().lazy().select([col("v"), (col("v") * 2).alias("w")]).collect().unwrap();
        assert_eq!(doubled.get_column_names_str(), ["v", "w"]);
        assert!(df.clone().lazy().select([col("nope")]).collect().is_err());
        assert!(df.clone().lazy().select([col("v"), col("v")]).collect().is_err());
    }

    #[test]
    fn aggregates_broadcast_and_fuse() {
        let df = sample();
        let out = df.clone().lazy().filter(col("v").gt(2.0)).select([col("v").sum(), len(), col("n").max().alias("m")]).collect().unwrap();
        assert_eq!(out.shape(), (1, 3));
        assert_eq!(out["v"].f64().unwrap().get(0), Some(18.0));
        assert_eq!(out["len"].u32().unwrap().get(0), Some(4));
        assert_eq!(out["m"].i64().unwrap().get(0), Some(60));
        let plain = df.clone().lazy().select([col("v").mean(), col("k").n_unique().alias("ks")]).collect().unwrap();
        assert_eq!(plain["v"].f64().unwrap().get(0), Some(3.5));
        assert_eq!(plain["ks"].u32().unwrap().get(0), Some(3));
        let mixed = df.clone().lazy().select([col("v"), col("v").max().alias("top"), lit(1).alias("one")]).collect().unwrap();
        assert_eq!(mixed.height(), 6);
        assert_eq!(mixed["top"].f64().unwrap().cont_slice().unwrap(), &[6.0; 6]);
        assert_eq!(mixed["one"].i64().unwrap().cont_slice().unwrap(), &[1; 6]);
        assert_eq!(df.clone().lazy().sum().collect().unwrap()["v"].f64().unwrap().get(0), Some(21.0));
        assert_eq!(df.clone().lazy().count().collect().unwrap()["n"].u32().unwrap().get(0), Some(5));
        assert_eq!(df.clone().lazy().max().collect().unwrap()["k"].str().unwrap().get(0), Some("c"));
    }

    #[test]
    fn with_columns_group_by_sort_slice() {
        let df = sample();
        let out = df
            .clone()
            .lazy()
            .with_columns([(col("v") * col("v")).alias("sq"), when(col("n").is_null()).then(lit(0)).otherwise(col("n")).alias("n")])
            .collect()
            .unwrap();
        assert_eq!(out.get_column_names_str(), ["k", "v", "n", "sq"]);
        assert_eq!(out["n"].i64().unwrap().cont_slice().unwrap(), &[10, 0, 30, 40, 50, 60]);
        assert_eq!(out["sq"].f64().unwrap().get(5), Some(36.0));

        let g = df
            .clone()
            .lazy()
            .group_by([col("k")])
            .agg([col("v").sum(), col("n").count().alias("cnt"), len(), (col("v") * 2).max().alias("twice")])
            .collect()
            .unwrap();
        assert_eq!(g.get_column_names_str(), ["k", "v", "cnt", "len", "twice"]);
        assert_eq!(g["k"].str().unwrap().to_vec(), vec![Some("a"), Some("b"), Some("c")]);
        assert_eq!(g["v"].f64().unwrap().cont_slice().unwrap(), &[10.0, 7.0, 4.0]);
        assert_eq!(g["cnt"].u32().unwrap().cont_slice().unwrap(), &[3, 1, 1]);
        assert_eq!(g["twice"].f64().unwrap().cont_slice().unwrap(), &[12.0, 10.0, 8.0]);
        assert!(df.clone().lazy().group_by([col("k")]).agg([col("v")]).collect().is_err());
        assert!(df.clone().lazy().group_by([col("k") + 1]).agg([len()]).collect().is_err());

        let top = df.clone().lazy().sort(["v"], SortMultipleOptions::default().with_order_descending(true)).limit(2).collect().unwrap();
        assert_eq!(top["v"].f64().unwrap().cont_slice().unwrap(), &[6.0, 5.0]);
        assert_eq!(df.clone().lazy().tail(2).collect().unwrap()["v"].f64().unwrap().cont_slice().unwrap(), &[5.0, 6.0]);
        assert_eq!(df.clone().lazy().first().collect().unwrap().height(), 1);
        assert_eq!(df.clone().lazy().last().collect().unwrap()["k"].str().unwrap().get(0), Some("a"));
        assert_eq!(df.clone().lazy().reverse().collect().unwrap()["v"].f64().unwrap().get(0), Some(6.0));
        assert_eq!(df.clone().lazy().slice(1, 2).select([col("v")]).collect().unwrap()["v"].f64().unwrap().cont_slice().unwrap(), &[2.0, 3.0]);
    }

    #[test]
    fn joins_unique_rename_drop() {
        let df = sample();
        let prices = df!("k" => ["a", "b"], "p" => [100, 200]).unwrap();
        let j = df.clone().lazy().inner_join(prices.clone().lazy(), col("k"), col("k")).collect().unwrap();
        assert_eq!(j.height(), 5);
        assert_eq!(j["p"].i32().unwrap().get(1), Some(200));
        let l = df.clone().lazy().left_join(prices.lazy(), col("k"), col("k")).collect().unwrap();
        assert_eq!(l["p"].null_count(), 1);
        let u = df.clone().lazy().unique(Some(vec!["k".to_string()]), UniqueKeepStrategy::First).collect().unwrap();
        assert_eq!(u.height(), 3);
        let r = df.clone().lazy().rename(["v"], ["value"], true).drop(["n"]).collect().unwrap();
        assert_eq!(r.get_column_names_str(), ["k", "value"]);
        assert_eq!(df.clone().lazy().drop_nulls(None).collect().unwrap().height(), 5);
        assert_eq!(df.clone().lazy().drop_nulls(Some(vec![col("v")])).collect().unwrap().height(), 6);
    }

    #[test]
    fn eager_expression_helpers_and_explain() {
        let df = sample();
        assert_eq!(df.filter(col("v").gt(4.0)).unwrap().height(), 2);
        assert_eq!(df.filter(col("v").gt(4.0)).unwrap().height(), 2);
        assert_eq!(df.filter(&df["v"].gt(4.0).unwrap()).unwrap().height(), 2);
        assert_eq!(df.select_exprs([col("v").sum()]).unwrap()["v"].f64().unwrap().get(0), Some(21.0));
        assert_eq!(df.with_columns([(col("v") + 1).alias("w")]).unwrap().width(), 4);
        assert_eq!(df.eval_expr(&(col("v") * 2)).unwrap().f64().unwrap().get(0), Some(2.0));
        let lf = df.lazy().filter(col("v").gt(1.0)).filter(col("k").eq("a")).select([col("v").sum()]);
        let plan = lf.explain(true).unwrap();
        assert!(plan.contains("aggregated while filtering"), "{plan}");
        assert_eq!(plan.matches("FILTER").count(), 1, "{plan}");
        assert_eq!(lf.explain(false).unwrap().matches("FILTER").count(), 2);
        assert_eq!(lf.collect().unwrap()["v"].f64().unwrap().get(0), Some(9.0));
    }
}
