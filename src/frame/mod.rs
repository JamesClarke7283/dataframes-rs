//! Layer 5 — [`DataFrame`]: equal-length named columns.
//!
//! A frame is a `Vec` of [`Column`]s and a height. Every structural operation
//! that does not have to touch data doesn't: `select`, `slice`, `head`,
//! `tail`, `clone`, `hstack` and `drop` only move the three-word buffer
//! handles of the columns involved.
//!
//! The method names and signatures follow Polars 0.55. pandas spellings live
//! in the `pandas` submodule.

mod display;
mod group_by;
mod join;
mod pandas;
mod sort;

use std::ops::Index;
use std::sync::{Arc, OnceLock};


pub use group_by::{AggKind, GroupBy, Groups};
pub(crate) use group_by::{aggregate, compute_groups};
pub use join::{JoinArgs, JoinType, JoinTypeOptions};
pub use pandas::{RowSelector, concat};
pub use sort::SortMultipleOptions;

use crate::chunked_array::{BooleanChunked, FillNullStrategy, IdxCa, Selection, UInt32Chunked, slice_offsets};
use crate::datatypes::{AnyValue, DataType, Field, IdxSize, PlSmallStr, Schema, SchemaRef};
use crate::error::{PolarsError, PolarsResult};
use crate::kernels::{cache, par};
use crate::series::{Column, IntoSeries, Series};
use crate::{polars_bail, polars_ensure};

/// Frame filters sparser than this go through a shared index list. The
/// break-even is higher than for a single column because the cost of decoding
/// the mask is paid once instead of once per column.
const FRAME_SPARSE_SELECTIVITY: f64 = 0.25;

/// Default number of rows returned by `head` / `tail`.
pub const HEAD_DEFAULT_LENGTH: usize = 10;

/// Which of several duplicate rows `unique` keeps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UniqueKeepStrategy {
    /// Keep the first occurrence.
    First,
    /// Keep the last occurrence.
    Last,
    /// Keep no row that has a duplicate.
    None,
    /// Keep any one occurrence (here: the first).
    #[default]
    Any,
}

/// A table: named columns of equal length.
///
/// ```
/// use dataframes::prelude::*;
///
/// let df = df!("city" => ["Oslo", "Lima", "Pune"], "temp" => [4.5, 19.0, 31.5])?;
/// let warm = df.filter(&df.column("temp")?.gt(10.0)?)?;
/// assert_eq!(warm.shape(), (2, 2));
/// assert_eq!(warm.column("city")?.str()?.get(0), Some("Lima"));
/// # Ok::<(), PolarsError>(())
/// ```
pub struct DataFrame {
    /// Number of rows in this frame's window.
    pub(crate) height: usize,
    /// First row of the window inside `base`.
    offset: usize,
    /// `true` when the window is a strict part of `base`.
    windowed: bool,
    /// The columns at their full extent. Shared by every slice and clone of
    /// the frame, so both are a single reference-count bump however wide the
    /// frame is.
    base: Arc<Vec<Column>>,
    /// `base` narrowed to the window, built on the first column access and
    /// never for a frame that is not a slice.
    view: OnceLock<Vec<Column>>,
}

impl Clone for DataFrame {
    /// O(1): shares the column set.
    #[inline]
    fn clone(&self) -> Self {
        DataFrame {
            height: self.height,
            offset: self.offset,
            windowed: self.windowed,
            base: self.base.clone(),
            view: OnceLock::new(),
        }
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        DataFrame::empty()
    }
}

/// Something [`DataFrame::filter`] can filter by.
pub trait FilterPredicate {
    /// Apply the predicate to `df`.
    fn filter_frame(self, df: &DataFrame) -> PolarsResult<DataFrame>;
}

impl FilterPredicate for &BooleanChunked {
    fn filter_frame(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        df.filter_mask(self)
    }
}

impl FilterPredicate for &Series {
    fn filter_frame(self, df: &DataFrame) -> PolarsResult<DataFrame> {
        df.filter_mask(self.bool()?)
    }
}

/// One row of a frame as dynamically typed values.
#[derive(Clone, Debug, PartialEq)]
pub struct Row<'a>(pub Vec<AnyValue<'a>>);

fn duplicate_err(name: &str) -> PolarsError {
    PolarsError::Duplicate(format!("column with name '{name}' has more than one occurrence").into())
}

fn not_found(name: &str) -> PolarsError {
    PolarsError::ColumnNotFound(format!("{name:?} not found").into())
}

fn check_unique(columns: &[Column]) -> PolarsResult<()> {
    if columns.len() <= 16 {
        for (i, c) in columns.iter().enumerate() {
            if columns[..i].iter().any(|p| p.name() == c.name()) {
                return Err(duplicate_err(c.name()));
            }
        }
    } else {
        let mut seen = std::collections::HashSet::with_capacity(columns.len());
        for c in columns {
            if !seen.insert(c.name().as_str()) {
                return Err(duplicate_err(c.name()));
            }
        }
    }
    Ok(())
}

impl DataFrame {
    // ----- construction ---------------------------------------------------

    /// Create a frame of `height` rows from columns of that length with
    /// distinct names.
    pub fn new(height: usize, columns: Vec<Column>) -> PolarsResult<Self> {
        for c in &columns {
            polars_ensure!(
                c.len() == height,
                ShapeMismatch: "could not create a new DataFrame: height of column '{}' ({}) does not match height of DataFrame ({})",
                c.name(), c.len(), height
            );
        }
        check_unique(&columns)?;
        Ok(DataFrame::from_parts(height, columns))
    }

    /// Like [`new`](Self::new), taking the height from the first column.
    pub fn new_infer_height(columns: Vec<Column>) -> PolarsResult<Self> {
        Self::new(columns.first().map_or(0, Series::len), columns)
    }

    /// Like [`new`](Self::new), repeating columns of length 1 to `height`.
    pub fn new_with_broadcast(height: usize, mut columns: Vec<Column>) -> PolarsResult<Self> {
        for c in &mut columns {
            if c.len() == 1 && height != 1 {
                let idx = vec![0 as IdxSize; height];
                *c = c.take_slice(&idx)?;
            }
        }
        Self::new(height, columns)
    }

    /// Skip the checks of [`new`](Self::new).
    ///
    /// # Safety
    /// Not memory-unsafe, but the caller must guarantee equal lengths and
    /// unique names or later operations misbehave (the signature mirrors Polars).
    pub unsafe fn new_unchecked(height: usize, columns: Vec<Column>) -> Self {
        debug_assert!(columns.iter().all(|c| c.len() == height));
        DataFrame::from_parts(height, columns)
    }

    /// Internal constructor for columns already known to be consistent.
    #[inline]
    pub(crate) fn from_parts(height: usize, columns: Vec<Column>) -> Self {
        debug_assert!(columns.iter().all(|c| c.len() == height));
        DataFrame { height, offset: 0, windowed: false, base: Arc::new(columns), view: OnceLock::new() }
    }

    /// A frame with no columns and no rows.
    pub fn empty() -> Self {
        DataFrame::from_parts(0, Vec::new())
    }

    /// A frame with no columns and `height` rows.
    pub fn empty_with_height(height: usize) -> Self {
        DataFrame::from_parts(height, Vec::new())
    }

    /// A frame with the columns of `schema` and no rows.
    pub fn empty_with_schema(schema: &Schema) -> Self {
        DataFrame::from_parts(0, schema.iter().map(|(name, dtype)| Series::new_empty(name.clone(), dtype)).collect())
    }

    /// The window columns: narrows `base` (O(width) handle operations, no
    /// data) the first time a slice is asked for its columns.
    #[cold]
    fn narrowed(&self) -> Vec<Column> {
        self.base.iter().map(|c| c.slice(self.offset as i64, self.height)).collect()
    }

    /// Exclusive access to the column vector. Resolves a pending window and
    /// un-shares the vector (cloning column handles, never data).
    pub(crate) fn columns_mut(&mut self) -> &mut Vec<Column> {
        if self.windowed {
            let narrowed = match self.view.take() {
                Some(view) => view,
                None => self.narrowed(),
            };
            self.base = Arc::new(narrowed);
            self.offset = 0;
            self.windowed = false;
        }
        Arc::make_mut(&mut self.base)
    }

    // ----- shape and schema -------------------------------------------------

    /// Number of rows.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of rows (alias of [`height`](Self::height)).
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.height
    }

    /// Number of columns.
    #[inline]
    pub fn width(&self) -> usize {
        self.base.len()
    }

    /// `(height, width)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.base.len())
    }

    /// `true` if the frame has no rows or no columns.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.height == 0 || self.base.is_empty()
    }

    /// The columns.
    #[inline]
    pub fn columns(&self) -> &[Column] {
        if !self.windowed {
            return &self.base;
        }
        self.view.get_or_init(|| self.narrowed())
    }

    /// The columns (older Polars name of [`columns`](Self::columns)).
    #[inline]
    pub fn get_columns(&self) -> &[Column] {
        self.columns()
    }

    /// Consume the frame, returning its columns.
    pub fn into_columns(mut self) -> Vec<Column> {
        std::mem::take(self.columns_mut())
    }

    /// Iterate over the columns.
    pub fn iter(&self) -> std::slice::Iter<'_, Column> {
        self.columns().iter()
    }

    /// Column names, in order.
    pub fn get_column_names(&self) -> Vec<&PlSmallStr> {
        self.base.iter().map(Series::name).collect()
    }

    /// Column names, in order, as `&str`.
    pub fn get_column_names_str(&self) -> Vec<&str> {
        self.base.iter().map(|c| c.name().as_str()).collect()
    }

    /// Column names, in order, owned.
    pub fn get_column_names_owned(&self) -> Vec<PlSmallStr> {
        self.base.iter().map(|c| c.name().clone()).collect()
    }

    /// Rename every column.
    pub fn set_column_names<T: AsRef<str>>(&mut self, new_names: &[T]) -> PolarsResult<()> {
        polars_ensure!(
            new_names.len() == self.width(),
            ShapeMismatch: "{} column names provided for a DataFrame of width {}", new_names.len(), self.width()
        );
        let unique: std::collections::HashSet<&str> = new_names.iter().map(AsRef::as_ref).collect();
        polars_ensure!(unique.len() == new_names.len(), Duplicate: "duplicate column names found");
        for (c, name) in self.columns_mut().iter_mut().zip(new_names) {
            c.rename(name.as_ref().into());
        }
        Ok(())
    }

    /// Column types, in order.
    pub fn dtypes(&self) -> Vec<DataType> {
        self.base.iter().map(|c| *c.dtype()).collect()
    }

    /// Column names and types, in order.
    pub fn fields(&self) -> Vec<Field> {
        self.base.iter().map(Series::field).collect()
    }

    /// The schema of the frame.
    pub fn schema(&self) -> SchemaRef {
        Arc::new(self.base.iter().map(Series::field).collect())
    }

    /// Bytes of heap memory viewed by the frame.
    pub fn estimated_size(&self) -> usize {
        self.columns().iter().map(Series::estimated_size).sum()
    }

    /// Always 1: columns are contiguous.
    pub fn first_col_n_chunks(&self) -> usize {
        1
    }

    /// No-op kept for Polars compatibility.
    pub fn rechunk_mut(&mut self) -> &mut Self {
        self
    }

    // ----- column access ----------------------------------------------------

    /// Position of the column called `name`.
    #[inline]
    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.base.iter().position(|c| c.name() == name)
    }

    /// Position of the column called `name`, or an error.
    pub fn try_get_column_index(&self, name: &str) -> PolarsResult<usize> {
        self.get_column_index(name).ok_or_else(|| not_found(name))
    }

    /// The column called `name`.
    #[inline]
    pub fn column(&self, name: &str) -> PolarsResult<&Column> {
        self.columns().iter().find(|c| c.name() == name).ok_or_else(|| not_found(name))
    }

    /// The column at position `idx`.
    #[inline]
    pub fn select_at_idx(&self, idx: usize) -> Option<&Column> {
        self.columns().get(idx)
    }

    /// The columns called `names`, in that order.
    pub fn select_to_vec(&self, selection: impl IntoIterator<Item = impl AsRef<str>>) -> PolarsResult<Vec<Column>> {
        selection.into_iter().map(|n| self.column(n.as_ref()).cloned()).collect()
    }

    /// A frame of the columns called `names`, in that order. Shares the data.
    pub fn select<I, S>(&self, names: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let columns = self.select_to_vec(names)?;
        check_unique(&columns)?;
        Ok(DataFrame::from_parts(self.height, columns))
    }

    /// Row `idx` as dynamically typed values, or `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<Vec<AnyValue<'_>>> {
        (idx < self.height).then(|| self.columns().iter().map(|c| c.get_any(idx)).collect())
    }

    /// Row `idx` as a [`Row`].
    pub fn get_row(&self, idx: usize) -> PolarsResult<Row<'_>> {
        self.get(idx).map(Row).ok_or_else(|| {
            PolarsError::OutOfBounds(format!("index {idx} is out of bounds for DataFrame of height {}", self.height).into())
        })
    }

    // ----- structural edits -------------------------------------------------

    /// Add a column, or replace the one with the same name.
    pub fn with_column(&mut self, column: impl IntoSeries) -> PolarsResult<&mut Self> {
        let mut column = column.into_series();
        if column.len() == 1 && self.height != 1 && self.width() > 0 {
            let idx = vec![0 as IdxSize; self.height];
            column = column.take_slice(&idx)?;
        }
        if self.width() == 0 {
            self.height = column.len();
        }
        polars_ensure!(
            column.len() == self.height,
            ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}", column.len(), self.height
        );
        match self.get_column_index(column.name()) {
            Some(i) => self.columns_mut()[i] = column,
            None => self.columns_mut().push(column),
        }
        Ok(self)
    }

    /// Insert a column at position `index`.
    pub fn insert_column(&mut self, index: usize, column: impl IntoSeries) -> PolarsResult<&mut Self> {
        let column = column.into_series();
        polars_ensure!(index <= self.width(), OutOfBounds: "column index {} is out of bounds", index);
        polars_ensure!(self.get_column_index(column.name()).is_none(), Duplicate: "column with name '{}' already exists", column.name());
        polars_ensure!(
            column.len() == self.height || self.width() == 0,
            ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}", column.len(), self.height
        );
        if self.width() == 0 {
            self.height = column.len();
        }
        self.columns_mut().insert(index, column);
        Ok(self)
    }

    /// Replace the column called `column` by `new_col` (which is renamed to match).
    pub fn replace(&mut self, column: &str, new_col: impl IntoSeries) -> PolarsResult<&mut Self> {
        let idx = self.try_get_column_index(column)?;
        self.replace_column(idx, new_col.into_series().with_name(column.into()))
    }

    /// Replace the column at position `index`.
    pub fn replace_column(&mut self, index: usize, new_column: impl IntoSeries) -> PolarsResult<&mut Self> {
        let new_column = new_column.into_series();
        polars_ensure!(index < self.width(), OutOfBounds: "column index {} is out of bounds", index);
        polars_ensure!(
            new_column.len() == self.height,
            ShapeMismatch: "unable to replace a column, series length {} doesn't match the DataFrame height {}",
            new_column.len(), self.height
        );
        self.columns_mut()[index] = new_column;
        check_unique(self.columns())?;
        Ok(self)
    }

    /// Replace the column called `name` by `f(column)`.
    pub fn apply<F, C>(&mut self, name: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> C,
        C: IntoSeries,
    {
        let idx = self.try_get_column_index(name)?;
        self.apply_at_idx(idx, f)
    }

    /// Replace the column at `idx` by `f(column)`.
    pub fn apply_at_idx<F, C>(&mut self, idx: usize, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> C,
        C: IntoSeries,
    {
        polars_ensure!(idx < self.width(), OutOfBounds: "column index {} is out of bounds", idx);
        let name = self.columns()[idx].name().clone();
        let new = f(&self.columns()[idx]).into_series().with_name(name);
        self.replace_column(idx, new)
    }

    /// Replace the column called `column` by `f(column)?`.
    pub fn try_apply<F, C>(&mut self, column: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Series) -> PolarsResult<C>,
        C: IntoSeries,
    {
        let idx = self.try_get_column_index(column)?;
        let name = self.columns()[idx].name().clone();
        let new = f(&self.columns()[idx])?.into_series().with_name(name);
        self.replace_column(idx, new)
    }

    /// Mutable access to one column for in-place updates.
    ///
    /// The frame's height is fixed, so only operations that keep the length
    /// (such as `apply_mut` on the typed column) are meaningful here.
    pub fn column_mut(&mut self, name: &str) -> PolarsResult<&mut Column> {
        let idx = self.try_get_column_index(name)?;
        Ok(&mut self.columns_mut()[idx])
    }

    /// Rename the column called `column` to `name`.
    pub fn rename(&mut self, column: &str, name: PlSmallStr) -> PolarsResult<&mut Self> {
        if column == name.as_str() {
            return Ok(self);
        }
        polars_ensure!(self.get_column_index(&name).is_none(), Duplicate: "column rename attempted with already existing name \"{}\"", name);
        let idx = self.try_get_column_index(column)?;
        self.columns_mut()[idx].rename(name);
        Ok(self)
    }

    /// The frame without the column called `name`.
    pub fn drop(&self, name: &str) -> PolarsResult<Self> {
        let idx = self.try_get_column_index(name)?;
        let mut columns = self.columns().to_vec();
        columns.remove(idx);
        Ok(DataFrame::from_parts(self.height, columns))
    }

    /// Remove the column called `name` and return it.
    pub fn drop_in_place(&mut self, name: &str) -> PolarsResult<Column> {
        let idx = self.try_get_column_index(name)?;
        Ok(self.columns_mut().remove(idx))
    }

    /// The frame without the columns called `names` (unknown names are ignored).
    pub fn drop_many<I, S>(&self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let names: Vec<PlSmallStr> = names.into_iter().map(Into::into).collect();
        DataFrame::from_parts(self.height, self.columns().iter().filter(|c| !names.contains(c.name())).cloned().collect())
    }

    /// This frame with `columns` appended on the right.
    pub fn hstack(&self, columns: &[Column]) -> PolarsResult<Self> {
        let mut out = self.columns().to_vec();
        out.extend_from_slice(columns);
        let height = if self.width() == 0 { out.first().map_or(0, Series::len) } else { self.height };
        DataFrame::new(height, out)
    }

    /// This frame with the rows of `other` appended.
    pub fn vstack(&self, other: &DataFrame) -> PolarsResult<Self> {
        let mut out = self.clone();
        out.vstack_mut(other)?;
        Ok(out)
    }

    /// Append the rows of `other` in place.
    pub fn vstack_mut(&mut self, other: &DataFrame) -> PolarsResult<&mut Self> {
        if self.width() == 0 {
            *self = other.clone();
            return Ok(self);
        }
        polars_ensure!(
            self.width() == other.width(),
            ShapeMismatch: "unable to append to a DataFrame of width {} with a DataFrame of width {}", self.width(), other.width()
        );
        for (a, b) in self.columns().iter().zip(other.columns()) {
            polars_ensure!(a.name() == b.name(), ShapeMismatch: "unable to vstack, column names don't match: {:?} and {:?}", a.name(), b.name());
            polars_ensure!(a.dtype() == b.dtype(), SchemaMismatch: "unable to vstack, dtypes for column {:?} don't match: `{}` and `{}`", a.name(), a.dtype(), b.dtype());
        }
        for (a, b) in self.columns_mut().iter_mut().zip(other.columns()) {
            a.append(b)?;
        }
        self.height += other.height;
        Ok(self)
    }

    /// Append the rows of `other` in place.
    pub fn extend(&mut self, other: &DataFrame) -> PolarsResult<()> {
        self.vstack_mut(other).map(|_| ())
    }

    /// Add a column of row numbers starting at `offset` as the first column.
    pub fn with_row_index(&self, name: PlSmallStr, offset: Option<IdxSize>) -> PolarsResult<Self> {
        polars_ensure!(self.get_column_index(&name).is_none(), Duplicate: "column with name '{}' already exists", name);
        let start = offset.unwrap_or(0);
        let idx = UInt32Chunked::from_iter_values(name, (0..self.height as IdxSize).map(|i| i + start));
        let mut columns = Vec::with_capacity(self.width() + 1);
        columns.push(idx.into_series());
        columns.extend(self.columns().iter().cloned());
        Ok(DataFrame::from_parts(self.height, columns))
    }

    // ----- rows: slicing ------------------------------------------------------

    /// A window of rows. O(1), whatever the width: the result shares the
    /// column set and records the window; the columns are narrowed (handle
    /// operations only, no data) the first time they are asked for.
    ///
    /// A negative `offset` counts from the end; the window is clamped.
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let (start, len) = slice_offsets(offset, length, self.height);
        let partial = start != 0 || len != self.height;
        DataFrame {
            height: len,
            offset: self.offset + start,
            windowed: self.windowed || partial,
            base: self.base.clone(),
            view: OnceLock::new(),
        }
    }

    /// Split into the rows before and from `offset`.
    pub fn split_at(&self, offset: i64) -> (Self, Self) {
        let (start, _) = slice_offsets(offset, usize::MAX, self.height);
        (self.slice(0, start), self.slice(start as i64, self.height - start))
    }

    /// The same columns with no rows.
    pub fn clear(&self) -> Self {
        self.slice(0, 0)
    }

    /// The first rows (10 by default). Accepts `Some(n)`, `None` or `n`.
    pub fn head(&self, length: impl Into<Option<usize>>) -> Self {
        self.slice(0, length.into().unwrap_or(HEAD_DEFAULT_LENGTH))
    }

    /// The last rows (10 by default). Accepts `Some(n)`, `None` or `n`.
    pub fn tail(&self, length: impl Into<Option<usize>>) -> Self {
        let n = length.into().unwrap_or(HEAD_DEFAULT_LENGTH).min(self.height);
        self.slice(-(n as i64), n)
    }

    /// The first `n` rows.
    pub fn limit(&self, n: usize) -> Self {
        self.slice(0, n)
    }

    /// Rows in reverse order.
    pub fn reverse(&self) -> Self {
        DataFrame::from_parts(self.height, self.map_columns(Series::reverse))
    }

    /// Shift every column by `periods` rows, filling with nulls.
    pub fn shift(&self, periods: i64) -> Self {
        DataFrame::from_parts(self.height, self.map_columns(|c| c.shift(periods)))
    }

    // ----- rows: filter / take ------------------------------------------------

    /// Threads to spread over the columns of this frame: 1 for a small or
    /// single-column frame (whose kernels then split the rows instead).
    fn column_threads(&self) -> usize {
        if self.width() < 2 {
            return 1;
        }
        par::tasks_for(self.height, self.row_bytes()).min(self.width())
    }

    /// Apply `f` to every column, on several cores when the frame is large.
    pub(crate) fn map_columns(&self, f: impl Fn(&Column) -> Column + Send + Sync) -> Vec<Column> {
        let columns = self.columns();
        par::map(columns.len(), self.column_threads(), |i| f(&columns[i]))
    }

    pub(crate) fn try_map_columns(
        &self,
        f: impl Fn(&Column) -> PolarsResult<Column> + Send + Sync,
    ) -> PolarsResult<Vec<Column>> {
        let columns = self.columns();
        par::map(columns.len(), self.column_threads(), |i| f(&columns[i])).into_iter().collect()
    }

    /// Keep the rows selected by `predicate`: a boolean mask
    /// (`&BooleanChunked`, as in Polars' Rust API) or an expression
    /// (`col("a").gt(5)`, as in Python Polars). Null counts as `false`.
    ///
    /// A mask is reduced to a [`Selection`] once; every column is then
    /// compacted by the vector kernels, columns and row ranges in parallel,
    /// each range writing straight into its final position. An expression is
    /// compiled and fused with the compaction block by block, so the predicate
    /// columns are read from memory only once.
    pub fn filter<P: FilterPredicate>(&self, predicate: P) -> PolarsResult<Self> {
        predicate.filter_frame(self)
    }

    /// [`filter`](Self::filter) for a boolean mask.
    pub fn filter_mask(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        if mask.len() == 1 && self.height != 1 {
            return Ok(if mask.get(0) == Some(true) { self.clone() } else { self.clear() });
        }
        polars_ensure!(
            mask.len() == self.height,
            ShapeMismatch: "filter's length: {} differs from that of the DataFrame: {}", mask.len(), self.height
        );
        Ok(self.filter_selection(&Selection::from_mask(mask)))
    }

    /// Same as [`filter`](Self::filter) on a single core.
    pub fn filter_seq(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        polars_ensure!(
            mask.len() == self.height,
            ShapeMismatch: "filter's length: {} differs from that of the DataFrame: {}", mask.len(), self.height
        );
        let sel = Selection::from_mask(mask);
        Ok(DataFrame::from_parts(
            sel.count,
            self.columns().iter().map(|c| c.filter_selection(&sel, Some(1), None)).collect(),
        ))
    }

    /// Bytes per row over the fixed-width parts of all columns.
    pub(crate) fn row_bytes(&self) -> usize {
        self.columns().iter().map(|c| c.dtype().byte_width().unwrap_or(16)).sum::<usize>().max(1)
    }

    pub(crate) fn filter_selection(&self, sel: &Selection<'_>) -> Self {
        if sel.is_all() {
            return self.clone();
        }
        if sel.is_none() {
            return self.clear();
        }
        // Block-wise over all columns at once: every core compacts blocks of
        // the whole row set into the shared outputs, through the mask or
        // through a per-block index list, whichever is cheaper for the block.
        if let Some(out) = crate::lazy::exec::filter_by_words(self, &sel.words, sel.count) {
            return out;
        }
        // Nullable or bit-packed columns: column by column.
        if (sel.count as f64) < FRAME_SPARSE_SELECTIVITY * sel.len as f64 && sel.len <= IdxSize::MAX as usize {
            // Sparse selection: decode the mask into row indices once, then
            // every column only gathers the rows it keeps (instead of every
            // column decoding the mask and reading all of its rows).
            let idx = crate::kernels::mask_indices(&sel.words, sel.len, 0);
            // SAFETY: the indices are positions of set bits below the height.
            let columns = self.map_columns(|c| unsafe { c.take_trusted(&idx) });
            return DataFrame::from_parts(sel.count, columns);
        }
        let row_bytes = self.row_bytes();
        // One decision for the whole frame: if the output as a whole exceeds
        // the cache, no column's share of it will stay cached either.
        let stream = sel.count.saturating_mul(row_bytes) >= cache::stream_threshold();
        let budget = par::tasks_for(self.height, row_bytes);
        let columns = self.columns();
        let filtered = if budget > 1 && columns.len() >= budget {
            // Enough columns to keep every thread busy with one each.
            par::map(columns.len(), budget, |i| columns[i].filter_selection(sel, Some(1), Some(stream)))
        } else {
            // A narrow frame: one column after the other, each split by rows.
            columns.iter().map(|c| c.filter_selection(sel, Some(budget), Some(stream))).collect()
        };
        DataFrame::from_parts(sel.count, filtered)
    }

    /// The rows at `indices`, in that order.
    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        self.take_slice(crate::chunked_array::idx_slice(indices)?)
    }

    /// The rows at `indices`, in that order.
    pub fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Self> {
        if let Some(&max) = indices.iter().max() {
            polars_ensure!((max as usize) < self.height, OutOfBounds: "index {} is out of bounds for DataFrame of height {}", max, self.height);
        }
        Ok(DataFrame::from_parts(indices.len(), self.try_map_columns(|c| c.take_slice(indices))?))
    }

    // ----- nulls --------------------------------------------------------------

    /// Number of nulls per column, as a one-row frame.
    pub fn null_count(&self) -> Self {
        let columns = self
            .columns()
            .iter()
            .map(|c| UInt32Chunked::from_slice(c.name().clone(), &[c.null_count() as IdxSize]).into_series())
            .collect();
        DataFrame::from_parts(1, columns)
    }

    /// Drop the rows that have a null in any of the `subset` columns (all
    /// columns if `None`).
    pub fn drop_nulls<S: AsRef<str>>(&self, subset: Option<&[S]>) -> PolarsResult<Self> {
        let selected: Vec<&Column> = match subset {
            Some(names) => names.iter().map(|n| self.column(n.as_ref())).collect::<PolarsResult<_>>()?,
            None => self.columns().iter().collect(),
        };
        let mut keep: Option<BooleanChunked> = None;
        for c in selected.into_iter().filter(|c| c.has_nulls()) {
            let valid = c.is_not_null();
            keep = Some(match keep {
                None => valid,
                Some(k) => &k & &valid,
            });
        }
        match keep {
            None => Ok(self.clone()),
            Some(mask) => self.filter(&mask),
        }
    }

    /// Replace nulls in every column according to `strategy`.
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        Ok(DataFrame::from_parts(self.height, self.try_map_columns(|c| c.fill_null(strategy))?))
    }

    // ----- duplicates ---------------------------------------------------------

    fn row_keys(&self, subset: Option<&[String]>) -> PolarsResult<Vec<Vec<AnyValue<'_>>>> {
        let cols: Vec<&Column> = match subset {
            Some(names) => names.iter().map(|n| self.column(n)).collect::<PolarsResult<_>>()?,
            None => self.columns().iter().collect(),
        };
        Ok((0..self.height).map(|i| cols.iter().map(|c| c.get_any(i)).collect()).collect())
    }

    fn duplicate_mask(&self, subset: Option<&[String]>, keep: UniqueKeepStrategy) -> PolarsResult<BooleanChunked> {
        let keys = self.row_keys(subset)?;
        let mut order: Vec<IdxSize> = (0..self.height as IdxSize).collect();
        let cmp = |a: &IdxSize, b: &IdxSize| {
            let (ka, kb) = (&keys[*a as usize], &keys[*b as usize]);
            ka.iter().zip(kb).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
        };
        order.sort_unstable_by(cmp);
        let same = |a: IdxSize, b: IdxSize| keys[a as usize].iter().zip(&keys[b as usize]).all(|(x, y)| x.total_cmp(y).is_eq());
        let mut keep_row = vec![false; self.height];
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && same(order[start], order[end]) {
                end += 1;
            }
            match keep {
                UniqueKeepStrategy::First | UniqueKeepStrategy::Any => keep_row[order[start] as usize] = true,
                UniqueKeepStrategy::Last => keep_row[order[end - 1] as usize] = true,
                UniqueKeepStrategy::None if end - start == 1 => keep_row[order[start] as usize] = true,
                UniqueKeepStrategy::None => {}
            }
            start = end;
        }
        Ok(BooleanChunked::from_slice(PlSmallStr::EMPTY, &keep_row))
    }

    /// Distinct rows (by `subset`, or all columns), in their original order.
    pub fn unique_stable(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        let out = self.filter(&self.duplicate_mask(subset, keep)?)?;
        Ok(match slice {
            Some((offset, len)) => out.slice(offset, len),
            None => out,
        })
    }

    /// Distinct rows (by `subset`, or all columns).
    pub fn unique(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        self.unique_stable(subset, keep, slice)
    }

    /// `true` for rows that occur exactly once.
    pub fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        self.duplicate_mask(None, UniqueKeepStrategy::None)
    }

    /// `true` for rows that occur more than once.
    pub fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        Ok(!&self.is_unique()?)
    }

    // ----- comparison ---------------------------------------------------------

    /// Same shape, names, types and values (nulls equal nulls).
    pub fn equals_missing(&self, other: &DataFrame) -> bool {
        self.shape() == other.shape() && self.columns().iter().zip(other.columns()).all(|(a, b)| a == b)
    }

    /// Like [`equals_missing`](Self::equals_missing), but `false` if any null is present.
    pub fn equals(&self, other: &DataFrame) -> bool {
        self.equals_missing(other) && self.columns().iter().all(|c| !c.has_nulls())
    }

    /// Thread the frame through `f` (handy in method chains).
    pub fn pipe<F, B>(self, f: F) -> PolarsResult<B>
    where
        F: Fn(DataFrame) -> PolarsResult<B>,
    {
        f(self)
    }

    /// Reject frames whose key columns are not usable (shared by sort/group/join).
    pub(crate) fn key_columns(&self, by: &[impl AsRef<str>]) -> PolarsResult<Vec<&Column>> {
        if by.is_empty() {
            polars_bail!(ComputeError: "at least one key column is required");
        }
        by.iter().map(|n| self.column(n.as_ref())).collect()
    }
}

impl PartialEq for DataFrame {
    fn eq(&self, other: &Self) -> bool {
        self.equals_missing(other)
    }
}

impl Index<&str> for DataFrame {
    type Output = Column;
    /// # Panics
    /// If there is no column called `name`.
    fn index(&self, name: &str) -> &Column {
        self.column(name).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl Index<usize> for DataFrame {
    type Output = Column;
    /// # Panics
    /// If `idx` is out of bounds.
    fn index(&self, idx: usize) -> &Column {
        &self.columns()[idx]
    }
}

impl<'a> IntoIterator for &'a DataFrame {
    type Item = &'a Column;
    type IntoIter = std::slice::Iter<'a, Column>;
    fn into_iter(self) -> Self::IntoIter {
        self.columns().iter()
    }
}

impl FromIterator<Column> for DataFrame {
    /// # Panics
    /// If the columns differ in length or share a name.
    fn from_iter<I: IntoIterator<Item = Column>>(iter: I) -> Self {
        DataFrame::new_infer_height(iter.into_iter().collect()).expect("columns must be consistent")
    }
}

impl Series {
    /// A one-column frame holding this series.
    pub fn into_frame(self) -> DataFrame {
        DataFrame::from_parts(self.len(), vec![self])
    }

    /// pandas name for [`into_frame`](Series::into_frame) (shares the data).
    pub fn to_frame(&self) -> DataFrame {
        self.clone().into_frame()
    }
}

/// Build a [`DataFrame`] from `name => values` pairs.
///
/// ```
/// use dataframes::prelude::*;
/// let df = df!("a" => [1, 2, 3], "b" => ["x", "y", "z"])?;
/// assert_eq!(df.shape(), (3, 2));
/// # Ok::<(), PolarsError>(())
/// ```
#[macro_export]
macro_rules! df {
    ($($name:expr => $values:expr),+ $(,)?) => {
        $crate::frame::DataFrame::new_infer_height(vec![
            $($crate::series::Series::new($name.into(), $values)),+
        ])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked_array::ChunkCompareIneq;

    fn sample() -> DataFrame {
        df!(
            "id" => [1i64, 2, 3, 4, 5],
            "score" => [Some(0.5f64), None, Some(2.5), Some(3.5), None],
            "tag" => ["a", "b", "a", "c", "b"],
        )
        .unwrap()
    }

    #[test]
    fn construction_checks_shape_and_names() {
        let df = sample();
        assert_eq!(df.shape(), (5, 3));
        assert_eq!((df.height(), df.width(), df.len()), (5, 3, 5));
        assert_eq!(df.get_column_names_str(), ["id", "score", "tag"]);
        assert_eq!(df.dtypes(), [DataType::Int64, DataType::Float64, DataType::String]);
        assert!(df!("a" => [1, 2], "b" => [1]).is_err());
        assert!(df!("a" => [1], "a" => [2]).is_err());
        assert!(DataFrame::new(3, vec![Series::new("a".into(), [1, 2])]).is_err());
        assert_eq!(DataFrame::empty().shape(), (0, 0));
        assert_eq!(DataFrame::empty_with_schema(&df.schema()).shape(), (0, 3));
        let b = DataFrame::new_with_broadcast(3, vec![Series::new("a".into(), [1, 2, 3]), Series::new("k".into(), [9])]).unwrap();
        assert_eq!(b.column("k").unwrap().i32().unwrap().cont_slice().unwrap(), &[9, 9, 9]);
        assert_eq!(df.schema().get("score"), Some(&DataType::Float64));
    }

    #[test]
    fn column_access_and_selection_share_buffers() {
        let df = sample();
        assert_eq!(df.column("id").unwrap().len(), 5);
        assert!(matches!(df.column("nope"), Err(PolarsError::ColumnNotFound(_))));
        assert_eq!(df["tag"].str().unwrap().get(3), Some("c"));
        assert_eq!(df[0].name(), "id");
        assert_eq!(df.get_column_index("score"), Some(1));
        let sel = df.select(["tag", "id"]).unwrap();
        assert_eq!(sel.get_column_names_str(), ["tag", "id"]);
        assert_eq!(
            sel["id"].i64().unwrap().values().as_ptr(),
            df["id"].i64().unwrap().values().as_ptr(),
            "select must not copy"
        );
        assert!(df.select(["id", "id"]).is_err());
        assert_eq!(df.get(1).unwrap(), vec![AnyValue::Int64(2), AnyValue::Null, AnyValue::String("b")]);
        assert_eq!(df.get(5), None);
        assert!(df.get_row(9).is_err());
    }

    #[test]
    fn slicing_is_zero_copy() {
        let df = sample();
        let s = df.slice(1, 3);
        assert_eq!(s.shape(), (3, 3));
        assert_eq!(s["id"].i64().unwrap().cont_slice().unwrap(), &[2, 3, 4]);
        assert_eq!(s["id"].i64().unwrap().values().as_ptr(), unsafe { df["id"].i64().unwrap().values().as_ptr().add(1) });
        assert_eq!(df.head(2).height(), 2);
        assert_eq!(df.head(Some(2)).height(), 2);
        assert_eq!(df.head(None).height(), 5);
        assert_eq!(df.tail(2)["tag"].str().unwrap().to_vec(), vec![Some("c"), Some("b")]);
        assert_eq!(df.slice(-2, 10).height(), 2);
        let (a, b) = df.split_at(2);
        assert_eq!((a.height(), b.height()), (2, 3));
        assert_eq!(df.clear().shape(), (0, 3));
        assert_eq!(df.reverse()["id"].i64().unwrap().get(0), Some(5));
    }

    #[test]
    fn slices_are_windows_over_a_shared_column_set() {
        let df = sample();
        let s = df.slice(1, 3);
        // A slice of a slice composes without narrowing anything in between.
        let t = s.slice(1, 5);
        assert_eq!(t.shape(), (2, 3));
        assert_eq!(t["id"].i64().unwrap().cont_slice().unwrap(), &[3, 4]);
        assert_eq!(s.get_column_names_str(), ["id", "score", "tag"]);
        assert_eq!(s.dtypes(), df.dtypes());
        assert_eq!(s.get(0).unwrap()[0], AnyValue::Int64(2));
        assert_eq!(df.clone(), df);
        assert_eq!(df.slice(0, 5), df);
        assert_eq!(df.slice(4, 1).into_columns()[0].len(), 1);

        // Mutating a slice resolves its window and leaves the parent alone.
        let mut m = df.slice(2, 2);
        m.with_column(Series::new("extra".into(), [1, 2])).unwrap();
        m.rename("id", "key".into()).unwrap();
        assert_eq!(m.shape(), (2, 4));
        assert_eq!(m["key"].i64().unwrap().cont_slice().unwrap(), &[3, 4]);
        assert_eq!((df.shape(), df.get_column_names_str()), ((5, 3), vec!["id", "score", "tag"]));
        let mut c = df.clone();
        c.drop_in_place("tag").unwrap();
        assert_eq!((c.width(), df.width()), (2, 3));

        // Everything downstream sees the window, not the base.
        let stacked = df.head(2).vstack(&df.tail(1)).unwrap();
        assert_eq!(stacked["id"].i64().unwrap().cont_slice().unwrap(), &[1, 2, 5]);
        let w = df.slice(1, 4);
        assert_eq!(w.filter(&w["id"].gt(3).unwrap()).unwrap()["id"].i64().unwrap().cont_slice().unwrap(), &[4, 5]);
        assert_eq!(w.take_slice(&[3, 0]).unwrap()["id"].i64().unwrap().cont_slice().unwrap(), &[5, 2]);
        assert_eq!(w.sort(["id"], SortMultipleOptions::default().with_order_descending(true)).unwrap()["id"].i64().unwrap().get(0), Some(5));
        assert_eq!(w.null_count()["score"].u32().unwrap().get(0), Some(2));
        assert_eq!(w.estimated_size(), w.columns().iter().map(Series::estimated_size).sum::<usize>());
        assert_eq!(df.slice(1, 0).shape(), (0, 3));
        assert_eq!(DataFrame::empty_with_height(7).slice(2, 3).shape(), (3, 0));
    }

    #[test]
    fn filter_take_and_null_handling() {
        let df = sample();
        let mask = df["id"].gt(2).unwrap();
        let f = df.filter(&mask).unwrap();
        assert_eq!(f.shape(), (3, 3));
        assert_eq!(f["score"].f64().unwrap().to_vec(), vec![Some(2.5), Some(3.5), None]);
        assert_eq!(f["tag"].str().unwrap().to_vec(), vec![Some("a"), Some("c"), Some("b")]);
        assert_eq!(df.filter_seq(&mask).unwrap(), f);
        assert!(df.filter(&BooleanChunked::from_slice("m".into(), &[true, false])).is_err());
        assert_eq!(df.filter(&BooleanChunked::from_slice("m".into(), &[false])).unwrap().height(), 0);
        let t = df.take_slice(&[4, 0]).unwrap();
        assert_eq!(t["id"].i64().unwrap().cont_slice().unwrap(), &[5, 1]);
        assert!(df.take_slice(&[5]).is_err());
        assert_eq!(df.null_count()["score"].u32().unwrap().get(0), Some(2));
        assert_eq!(df.drop_nulls::<&str>(None).unwrap().height(), 3);
        assert_eq!(df.drop_nulls(Some(&["id"])).unwrap().height(), 5);
        let filled = df.fill_null(FillNullStrategy::Forward(None)).unwrap();
        assert_eq!(filled["score"].f64().unwrap().to_vec(), vec![Some(0.5), Some(0.5), Some(2.5), Some(3.5), Some(3.5)]);
    }

    #[test]
    fn large_frames_filter_in_parallel_and_agree_with_serial() {
        let n = 400_000usize;
        let a: Vec<i64> = (0..n as i64).map(|i| (i * 7919) % 1000).collect();
        let b: Vec<f64> = a.iter().map(|&v| v as f64 * 0.5).collect();
        let c: Vec<i32> = a.iter().map(|&v| v as i32 - 500).collect();
        let s: Vec<String> = a.iter().map(|v| format!("row-{v}-with-a-long-suffix")).collect();
        let df = df!("a" => &a, "b" => &b, "c" => &c, "s" => &s).unwrap();
        let mask = df["a"].lt(337).unwrap();
        let out = df.filter(&mask).unwrap();
        let want: Vec<usize> = (0..n).filter(|&i| a[i] < 337).collect();
        assert_eq!(out.height(), want.len());
        assert!(out["a"].i64().unwrap().cont_slice().unwrap().iter().zip(&want).all(|(&v, &i)| v == a[i]));
        assert!(out["b"].f64().unwrap().cont_slice().unwrap().iter().zip(&want).all(|(&v, &i)| v == b[i]));
        assert!(out["c"].i32().unwrap().cont_slice().unwrap().iter().zip(&want).all(|(&v, &i)| v == c[i]));
        assert!(out["s"].str().unwrap().into_no_null_iter().zip(&want).all(|(v, &i)| v == s[i]));
        assert_eq!(df.filter_seq(&mask).unwrap(), out);
    }

    #[test]
    fn filters_agree_with_a_row_by_row_model_at_every_density() {
        use crate::lazy::{IntoLazy, col, lit};
        // Heights below, around and above the block and parallel thresholds;
        // densities on both sides of every strategy switch (skip, index list,
        // permutation tables, copy).
        for &n in &[0usize, 1, 63, 1000, 70_001, 300_000] {
            let key: Vec<i64> = (0..n as i64).map(|i| (i * 7919 + 13) % 1000).collect();
            let b: Vec<f64> = key.iter().map(|&v| v as f64 * 0.5).collect();
            let c: Vec<i32> = key.iter().map(|&v| v as i32 - 500).collect();
            let d: Vec<u8> = key.iter().map(|&v| v as u8).collect();
            let s: Vec<String> = key.iter().map(|v| if v % 3 == 0 { format!("k{v}") } else { format!("key-{v}-long-enough-to-leave-the-view") }).collect();
            let plain = df!("key" => &key, "b" => &b, "c" => &c, "d" => &d, "s" => &s).unwrap();
            // The same frame with a nullable column takes the column-wise path.
            let nullable: Vec<Option<i64>> = key.iter().map(|&v| (v % 7 != 0).then_some(v)).collect();
            let mut with_nulls = plain.clone();
            with_nulls.with_column(Series::new("n".into(), nullable.as_slice())).unwrap();
            for threshold in [0i64, 1, 10, 100, 390, 410, 500, 900, 999, 1000] {
                let want: Vec<usize> = (0..n).filter(|&i| key[i] < threshold).collect();
                let mask = plain["key"].lt(threshold).unwrap();
                let lazy = plain.clone().lazy().filter(col("key").lt(lit(threshold))).collect().unwrap();
                for (label, out) in [("eager", plain.filter(&mask).unwrap()), ("lazy", lazy), ("nullable", with_nulls.filter(&mask).unwrap())] {
                    let ctx = format!("{label} n {n} threshold {threshold}");
                    assert_eq!(out.height(), want.len(), "{ctx}");
                    assert!(out["key"].i64().unwrap().cont_slice().unwrap().iter().zip(&want).all(|(&v, &i)| v == key[i]), "{ctx}");
                    assert!(out["b"].f64().unwrap().cont_slice().unwrap().iter().zip(&want).all(|(&v, &i)| v == b[i]), "{ctx}");
                    assert!(out["c"].i32().unwrap().cont_slice().unwrap().iter().zip(&want).all(|(&v, &i)| v == c[i]), "{ctx}");
                    assert!(out["d"].u8().unwrap().cont_slice().unwrap().iter().zip(&want).all(|(&v, &i)| v == d[i]), "{ctx}");
                    assert!(out["s"].str().unwrap().into_no_null_iter().zip(&want).all(|(v, &i)| v == s[i]), "{ctx}");
                    if label == "nullable" {
                        let got: Vec<Option<i64>> = out["n"].i64().unwrap().iter().collect();
                        assert!(got.iter().zip(&want).all(|(v, &i)| *v == nullable[i]), "{ctx}");
                    }
                }
            }
        }
    }

    #[test]
    fn structural_edits() {
        let mut df = sample();
        df.with_column(Series::new("flag".into(), [true, false, true, false, true])).unwrap();
        assert_eq!(df.width(), 4);
        df.with_column(Series::new("id".into(), [10i64, 20, 30, 40, 50])).unwrap();
        assert_eq!((df.width(), df["id"].i64().unwrap().get(0)), (4, Some(10)));
        df.with_column(Series::new("one".into(), [7])).unwrap();
        assert_eq!(df["one"].i32().unwrap().cont_slice().unwrap(), &[7; 5]);
        assert!(df.with_column(Series::new("bad".into(), [1, 2])).is_err());
        df.rename("flag", "f".into()).unwrap();
        assert!(df.rename("f", "id".into()).is_err());
        assert_eq!(df.drop("one").unwrap().width(), 4);
        assert_eq!(df.drop_many(["one", "f", "missing"]).width(), 3);
        assert_eq!(df.drop_in_place("one").unwrap().name(), "one");
        df.apply("id", |c| c + 1).unwrap();
        assert_eq!(df["id"].i64().unwrap().get(0), Some(11));
        df.replace("f", Series::new("x".into(), [false; 5])).unwrap();
        assert_eq!(df["f"].bool().unwrap().num_trues(), 0);
        df.insert_column(0, Series::new("first".into(), [0u8; 5])).unwrap();
        assert_eq!(df[0].name(), "first");
        let indexed = df.with_row_index("row".into(), Some(100)).unwrap();
        assert_eq!(indexed["row"].u32().unwrap().get(4), Some(104));
        let mut names = df.get_column_names_owned();
        names.reverse();
        assert!(df.set_column_names(&["a"]).is_err());
        df.set_column_names(&names).unwrap();
    }

    #[test]
    fn stacking() {
        let df = sample();
        let two = df.vstack(&df).unwrap();
        assert_eq!(two.shape(), (10, 3));
        assert_eq!(two["tag"].str().unwrap().get(7), Some("a"));
        assert_eq!(two["score"].null_count(), 4);
        assert!(df.vstack(&df.select(["id"]).unwrap()).is_err());
        let wide = df.hstack(&[Series::new("extra".into(), [1u8; 5])]).unwrap();
        assert_eq!(wide.width(), 4);
        assert!(df.hstack(&[Series::new("id".into(), [1u8; 5])]).is_err());
        let mut empty = DataFrame::empty();
        empty.vstack_mut(&df).unwrap();
        assert_eq!(empty, df);
    }

    #[test]
    fn duplicates() {
        let df = df!("k" => [1, 2, 1, 3, 2], "v" => ["a", "b", "a", "c", "z"]).unwrap();
        assert_eq!(df.unique_stable(None, UniqueKeepStrategy::First, None).unwrap().height(), 4);
        let by_k = df.unique_stable(Some(&["k".to_string()]), UniqueKeepStrategy::First, None).unwrap();
        assert_eq!(by_k["v"].str().unwrap().to_vec(), vec![Some("a"), Some("b"), Some("c")]);
        let last = df.unique(Some(&["k".to_string()]), UniqueKeepStrategy::Last, None).unwrap();
        assert_eq!(last["v"].str().unwrap().to_vec(), vec![Some("a"), Some("c"), Some("z")]);
        let none = df.unique(Some(&["k".to_string()]), UniqueKeepStrategy::None, None).unwrap();
        assert_eq!(none["k"].i32().unwrap().cont_slice().unwrap(), &[3]);
        assert_eq!(df.is_duplicated().unwrap().to_vec(), vec![Some(true), Some(false), Some(true), Some(false), Some(false)]);
        assert_eq!(df.is_unique().unwrap().num_trues(), 3);
    }

    #[test]
    fn equality() {
        let df = sample();
        assert!(df.equals_missing(&df.clone()));
        assert!(!df.equals(&df.clone()), "nulls make `equals` false");
        assert_ne!(df, df.slice(0, 4));
        assert_eq!(df.select(["id"]).unwrap(), Series::new("id".into(), [1i64, 2, 3, 4, 5]).into_frame());
    }
}
