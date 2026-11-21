//! Equality joins.
//!
//! Both tables' key columns are stacked and run through the group-by
//! machinery, which gives every row of either table an id such that two rows
//! match exactly when their ids are equal. The right table's rows are bucketed
//! by id with a counting sort; probing the left table is then two array
//! look-ups per row. Rows whose key contains a null never match, as in SQL and
//! Polars.

use super::DataFrame;
use super::group_by::compute_groups;
use crate::datatypes::{DataType, IdxSize, PlSmallStr};
use crate::error::PolarsResult;
use crate::series::{Column, Series};
use crate::{polars_bail, polars_ensure};

/// The kind of join.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum JoinType {
    /// Rows with a match in both tables.
    #[default]
    Inner,
    /// Every left row; unmatched ones get nulls on the right.
    Left,
    /// Every right row; unmatched ones get nulls on the left.
    Right,
    /// Every row of both tables.
    Full,
    /// Left rows that have a match (left columns only).
    Semi,
    /// Left rows that have no match (left columns only).
    Anti,
    /// Every combination of a left and a right row; takes no keys.
    Cross,
}

/// Join configuration (mirrors Polars).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JoinArgs {
    /// The kind of join.
    pub how: JoinType,
    /// Suffix for right-hand columns whose name clashes (default `_right`).
    pub suffix: Option<PlSmallStr>,
}

impl JoinArgs {
    /// Arguments for a join of kind `how`.
    pub fn new(how: JoinType) -> Self {
        JoinArgs { how, suffix: None }
    }

    /// Set the suffix for clashing right-hand column names.
    pub fn with_suffix(mut self, suffix: Option<PlSmallStr>) -> Self {
        self.suffix = suffix;
        self
    }

    fn suffix(&self) -> &str {
        self.suffix.as_deref().unwrap_or("_right")
    }
}

/// Placeholder for Polars' per-join-type options (none are needed here).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoinTypeOptions;

/// Matching row pairs. `None` on a side means "no partner: emit nulls".
struct Matches {
    left: Vec<Option<IdxSize>>,
    right: Vec<Option<IdxSize>>,
}

fn match_rows(left_keys: &[Column], right_keys: &[Column], how: JoinType) -> PolarsResult<Matches> {
    let n_left = left_keys[0].len();
    let n_right = right_keys[0].len();
    let mut stacked: Vec<Series> = Vec::with_capacity(left_keys.len());
    let mut usable = vec![true; n_left + n_right];
    for (l, r) in left_keys.iter().zip(right_keys) {
        let common = if l.dtype() == r.dtype() {
            *l.dtype()
        } else {
            match DataType::supertype(l.dtype(), r.dtype()) {
                Some(t) if l.dtype().is_numeric() && r.dtype().is_numeric() => t,
                _ => polars_bail!(SchemaMismatch: "datatypes of join keys don't match - `{}`: {} on left does not match `{}`: {} on right", l.name(), l.dtype(), r.name(), r.dtype()),
            }
        };
        let mut both = l.cast(&common)?;
        both.append(&r.cast(&common)?)?;
        if let Some(valid) = both.validity() {
            for (u, v) in usable.iter_mut().zip(valid.iter()) {
                *u &= v;
            }
        }
        stacked.push(both);
    }
    let refs: Vec<&Column> = stacked.iter().collect();
    let groups = compute_groups(&refs);
    let (left_ids, right_ids) = groups.ids.split_at(n_left);

    // Counting sort of the usable right rows by group id.
    let mut starts = vec![0usize; groups.len() + 1];
    for (r, &g) in right_ids.iter().enumerate() {
        if usable[n_left + r] {
            starts[g as usize + 1] += 1;
        }
    }
    for g in 0..groups.len() {
        starts[g + 1] += starts[g];
    }
    let mut cursor = starts.clone();
    let mut bucketed = vec![0 as IdxSize; starts[groups.len()]];
    for (r, &g) in right_ids.iter().enumerate() {
        if usable[n_left + r] {
            bucketed[cursor[g as usize]] = r as IdxSize;
            cursor[g as usize] += 1;
        }
    }

    let mut out = Matches { left: Vec::new(), right: Vec::new() };
    let mut right_hit = vec![false; n_right];
    for (l, &g) in left_ids.iter().enumerate() {
        let partners: &[IdxSize] =
            if usable[l] { &bucketed[starts[g as usize]..starts[g as usize + 1]] } else { &[] };
        match how {
            JoinType::Semi => {
                if !partners.is_empty() {
                    out.left.push(Some(l as IdxSize));
                }
            }
            JoinType::Anti => {
                if partners.is_empty() {
                    out.left.push(Some(l as IdxSize));
                }
            }
            _ => {
                for &r in partners {
                    out.left.push(Some(l as IdxSize));
                    out.right.push(Some(r));
                    right_hit[r as usize] = true;
                }
                if partners.is_empty() && matches!(how, JoinType::Left | JoinType::Full) {
                    out.left.push(Some(l as IdxSize));
                    out.right.push(None);
                }
            }
        }
    }
    if how == JoinType::Full {
        for (r, hit) in right_hit.iter().enumerate() {
            if !hit {
                out.left.push(None);
                out.right.push(Some(r as IdxSize));
            }
        }
    }
    Ok(out)
}

impl DataFrame {
    /// Join with `other` on equality of `left_on` and `right_on`.
    ///
    /// The key columns appear once in the result (taken from the left table,
    /// or from the right where the left has no row). Other right-hand columns
    /// whose name already exists on the left get the suffix `_right`.
    ///
    /// ```
    /// use dataframes::prelude::*;
    /// let orders = df!("id" => [1, 2, 3], "item" => ["pen", "ink", "pad"])?;
    /// let prices = df!("item" => ["pen", "pad"], "price" => [1.5, 3.0])?;
    /// let out = orders.join(&prices, ["item"], ["item"], JoinArgs::new(JoinType::Left), None)?;
    /// assert_eq!(out.column("price")?.f64()?.to_vec(), vec![Some(1.5), None, Some(3.0)]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn join(
        &self,
        other: &DataFrame,
        left_on: impl IntoIterator<Item = impl AsRef<str>>,
        right_on: impl IntoIterator<Item = impl AsRef<str>>,
        args: JoinArgs,
        _options: Option<JoinTypeOptions>,
    ) -> PolarsResult<DataFrame> {
        let left_on: Vec<String> = left_on.into_iter().map(|s| s.as_ref().to_string()).collect();
        let right_on: Vec<String> = right_on.into_iter().map(|s| s.as_ref().to_string()).collect();
        let suffix = args.suffix().to_string();
        let disambiguate = |taken: &dyn Fn(&str) -> bool, name: &PlSmallStr| -> PlSmallStr {
            if taken(name) { format!("{name}{suffix}").into() } else { name.clone() }
        };

        if args.how == JoinType::Cross {
            polars_ensure!(left_on.is_empty() && right_on.is_empty(), ComputeError: "a cross join takes no join keys");
            let (n, m) = (self.height(), other.height());
            let left_idx: Vec<IdxSize> = (0..n as IdxSize).flat_map(|i| std::iter::repeat_n(i, m)).collect();
            let right_idx: Vec<IdxSize> = (0..n).flat_map(|_| 0..m as IdxSize).collect();
            let mut columns = self.take_slice(&left_idx)?.into_columns();
            for c in other.take_slice(&right_idx)?.into_columns() {
                let name = disambiguate(&|n| self.get_column_index(n).is_some(), c.name());
                columns.push(c.with_name(name));
            }
            return DataFrame::new(n * m, columns);
        }

        polars_ensure!(!left_on.is_empty(), ComputeError: "at least one join key is required");
        polars_ensure!(
            left_on.len() == right_on.len(),
            ComputeError: "the number of columns given as join key (left: {}, right: {}) should be equal", left_on.len(), right_on.len()
        );
        if args.how == JoinType::Right {
            // A right join is a left join with the tables swapped (so the
            // right table's columns come first in the result).
            return other.join(self, &right_on, &left_on, JoinArgs { how: JoinType::Left, ..args.clone() }, None);
        }

        let left_keys: Vec<Column> = left_on.iter().map(|n| self.column(n).cloned()).collect::<PolarsResult<_>>()?;
        let right_keys: Vec<Column> = right_on.iter().map(|n| other.column(n).cloned()).collect::<PolarsResult<_>>()?;
        let m = match_rows(&left_keys, &right_keys, args.how)?;
        let height = m.left.len();

        let mut columns: Vec<Series> = Vec::with_capacity(self.width() + other.width());
        for c in self.columns() {
            let mut out = c.take_opt(&m.left);
            if args.how == JoinType::Full {
                // Coalesce the key: where the left has no row, use the right key.
                if let Some(k) = left_on.iter().position(|n| n == c.name().as_str()) {
                    let from_right = right_keys[k].cast(out.dtype())?.take_opt(&m.right);
                    let values: Vec<_> = (0..height)
                        .map(|i| if m.left[i].is_some() { out.get_any(i) } else { from_right.get_any(i) })
                        .collect();
                    out = Series::from_any_values(c.name().clone(), &values)?.cast(c.dtype())?;
                }
            }
            columns.push(out);
        }
        if !matches!(args.how, JoinType::Semi | JoinType::Anti) {
            for c in other.columns().iter().filter(|c| !right_on.iter().any(|n| n == c.name().as_str())) {
                let name = disambiguate(&|n| self.get_column_index(n).is_some(), c.name());
                columns.push(c.take_opt(&m.right).with_name(name));
            }
        }
        DataFrame::new(height, columns)
    }

    /// Inner join on equality of `left_on` and `right_on`.
    pub fn inner_join(
        &self,
        other: &DataFrame,
        left_on: impl IntoIterator<Item = impl AsRef<str>>,
        right_on: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> PolarsResult<DataFrame> {
        self.join(other, left_on, right_on, JoinArgs::new(JoinType::Inner), None)
    }

    /// Left join on equality of `left_on` and `right_on`.
    pub fn left_join(
        &self,
        other: &DataFrame,
        left_on: impl IntoIterator<Item = impl AsRef<str>>,
        right_on: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> PolarsResult<DataFrame> {
        self.join(other, left_on, right_on, JoinArgs::new(JoinType::Left), None)
    }

    /// Full (outer) join on equality of `left_on` and `right_on`.
    pub fn full_join(
        &self,
        other: &DataFrame,
        left_on: impl IntoIterator<Item = impl AsRef<str>>,
        right_on: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> PolarsResult<DataFrame> {
        self.join(other, left_on, right_on, JoinArgs::new(JoinType::Full), None)
    }

    /// Cartesian product with `other`.
    pub fn cross_join(&self, other: &DataFrame, suffix: Option<PlSmallStr>) -> PolarsResult<DataFrame> {
        let none: [&str; 0] = [];
        self.join(other, none, none, JoinArgs::new(JoinType::Cross).with_suffix(suffix), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    fn tables() -> (DataFrame, DataFrame) {
        let left = df!("k" => [Some(1i64), Some(2), Some(3), None, Some(2)], "l" => ["a", "b", "c", "d", "e"]).unwrap();
        let right = df!("k" => [Some(2i32), Some(2), Some(4), None], "r" => [10.0, 20.0, 40.0, 0.0], "l" => ["w", "x", "y", "z"]).unwrap();
        (left, right)
    }

    #[test]
    fn inner_and_left() {
        let (left, right) = tables();
        let inner = left.inner_join(&right, ["k"], ["k"]).unwrap();
        assert_eq!(inner.get_column_names_str(), ["k", "l", "r", "l_right"]);
        assert_eq!(inner["k"].i64().unwrap().cont_slice().unwrap(), &[2, 2, 2, 2]);
        assert_eq!(inner["l"].str().unwrap().to_vec(), vec![Some("b"), Some("b"), Some("e"), Some("e")]);
        assert_eq!(inner["r"].f64().unwrap().cont_slice().unwrap(), &[10.0, 20.0, 10.0, 20.0]);

        let l = left.left_join(&right, ["k"], ["k"]).unwrap();
        assert_eq!(l.height(), 7);
        assert_eq!(l["l"].str().unwrap().to_vec(), vec![Some("a"), Some("b"), Some("b"), Some("c"), Some("d"), Some("e"), Some("e")]);
        assert_eq!(l["r"].f64().unwrap().to_vec(), vec![None, Some(10.0), Some(20.0), None, None, Some(10.0), Some(20.0)]);
        assert_eq!(l["l_right"].null_count(), 3);
        // Null keys never match.
        assert!(l["k"].i64().unwrap().get(4).is_none() && l["r"].f64().unwrap().get(4).is_none());
    }

    #[test]
    fn full_right_semi_anti_cross() {
        let (left, right) = tables();
        let full = left.full_join(&right, ["k"], ["k"]).unwrap();
        assert_eq!(full.height(), 9);
        assert_eq!(full["k"].i64().unwrap().to_vec()[7..], [Some(4), None]);
        assert_eq!(full["l"].str().unwrap().to_vec()[7..], [None, None]);
        assert_eq!(full["r"].f64().unwrap().to_vec()[7..], [Some(40.0), Some(0.0)]);

        let r = left.join(&right, ["k"], ["k"], JoinArgs::new(JoinType::Right), None).unwrap();
        assert_eq!(r.height(), 6);
        assert_eq!(r.get_column_names_str(), ["k", "r", "l", "l_right"]);

        let semi = left.join(&right, ["k"], ["k"], JoinArgs::new(JoinType::Semi), None).unwrap();
        assert_eq!(semi["l"].str().unwrap().to_vec(), vec![Some("b"), Some("e")]);
        assert_eq!(semi.width(), 2);
        let anti = left.join(&right, ["k"], ["k"], JoinArgs::new(JoinType::Anti), None).unwrap();
        assert_eq!(anti["l"].str().unwrap().to_vec(), vec![Some("a"), Some("c"), Some("d")]);

        let cross = left.head(2).cross_join(&right.head(3), Some("_r".into())).unwrap();
        assert_eq!(cross.shape(), (6, 5));
        assert_eq!(cross.get_column_names_str(), ["k", "l", "k_r", "r", "l_r"]);
        assert_eq!(cross["k_r"].i32().unwrap().cont_slice().unwrap(), &[2, 2, 4, 2, 2, 4]);
    }

    #[test]
    fn multi_key_and_errors() {
        let a = df!("x" => [1, 1, 2], "y" => ["p", "q", "p"], "v" => [1, 2, 3]).unwrap();
        let b = df!("x" => [1, 2, 2], "y" => ["q", "p", "z"], "w" => [10, 20, 30]).unwrap();
        let j = a.inner_join(&b, ["x", "y"], ["x", "y"]).unwrap();
        assert_eq!(j["v"].i32().unwrap().cont_slice().unwrap(), &[2, 3]);
        assert_eq!(j["w"].i32().unwrap().cont_slice().unwrap(), &[10, 20]);
        assert!(a.inner_join(&b, ["x"], ["y"]).is_err());
        assert!(a.inner_join(&b, ["x", "y"], ["x"]).is_err());
        assert!(a.inner_join(&b, ["nope"], ["x"]).is_err());
        let custom = a.join(&b.rename_cols(), ["x"], ["x"], JoinArgs::new(JoinType::Inner).with_suffix(Some("_b".into())), None).unwrap();
        assert!(custom.get_column_names_str().contains(&"v_b"));
    }

    impl DataFrame {
        fn rename_cols(&self) -> DataFrame {
            let mut out = self.clone();
            out.rename("w", "v".into()).unwrap();
            out
        }
    }
}
