//! Operations on numeric columns.

use std::ops::{Add, Div, Mul, Rem, Sub};

use rayon::prelude::*;

use super::{BooleanChunked, ChunkedArray, IdxCa, Selection, combine_validity, idx_slice, normalize_validity};
use crate::datatypes::{DataType, IdxSize, NativeType, PlSmallStr, PolarsNumericType};
use crate::error::PolarsResult;
use crate::kernels::aggregate::{self, Num};
use crate::kernels::{self, CmpOp, Kernel, arith, par, pipeline};
use crate::memory::{AlignedVec, Bitmap, Buffer, ColumnAlloc, MutableBitmap};
use crate::polars_ensure;

// ---------------------------------------------------------------------------
// scalars
// ---------------------------------------------------------------------------

/// A scalar widened to a type that holds every native value exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wide {
    /// Any integer.
    Int(i128),
    /// Any float.
    Float(f64),
}

impl Wide {
    /// Convert with Rust `as` semantics.
    #[inline]
    pub fn cast<N: NativeType>(self) -> N {
        match self {
            Wide::Int(i) if N::IS_FLOAT => N::from_f64(i as f64),
            Wide::Int(i) => N::from_i64(i as i64),
            Wide::Float(f) => N::from_f64(f),
        }
    }
}

impl Wide {
    /// Convert exactly: `None` if an integer target cannot hold the value
    /// (out of range, NaN or infinite). Floats are truncated toward zero.
    pub fn checked_cast<N: NativeType>(self) -> Option<N> {
        if N::IS_FLOAT {
            return Some(self.cast());
        }
        let wide = match self {
            Wide::Int(i) => i,
            Wide::Float(f) if f.is_finite() => f.trunc() as i128,
            Wide::Float(_) => return None,
        };
        let (lo, hi) = int_bounds(&N::DTYPE);
        (lo..=hi).contains(&wide).then(|| N::from_i64(wide as i64))
    }
}

/// A Rust number that may appear as the scalar operand of a comparison or an
/// arithmetic operation on a column of any numeric type.
pub trait NumericScalar: Copy {
    /// Widen without loss.
    fn wide(self) -> Wide;
}

macro_rules! numeric_scalar {
    (int: $($i:ty),*; float: $($f:ty),*) => {
        $(impl NumericScalar for $i { #[inline] fn wide(self) -> Wide { Wide::Int(self as i128) } })*
        $(impl NumericScalar for $f { #[inline] fn wide(self) -> Wide { Wide::Float(self as f64) } })*
    };
}
numeric_scalar!(int: i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; float: f32, f64);

fn int_bounds(dtype: &DataType) -> (i128, i128) {
    match dtype {
        DataType::Int8 => (i8::MIN as i128, i8::MAX as i128),
        DataType::Int16 => (i16::MIN as i128, i16::MAX as i128),
        DataType::Int32 => (i32::MIN as i128, i32::MAX as i128),
        DataType::Int64 => (i64::MIN as i128, i64::MAX as i128),
        DataType::UInt8 => (0, u8::MAX as i128),
        DataType::UInt16 => (0, u16::MAX as i128),
        DataType::UInt32 => (0, u32::MAX as i128),
        _ => (0, u64::MAX as i128),
    }
}

/// How `column op scalar` is evaluated once the scalar is fitted to the column type.
pub(crate) enum Resolved<N> {
    /// Run the comparison kernel with this operator and operand.
    Kernel(CmpOp, N),
    /// Every row compares the same way (the scalar is outside the column's range).
    All(bool),
}

/// Fit a wide scalar to a column of `N`, preserving the mathematical meaning
/// of the comparison (`u8_column > 300` is false everywhere, `int_column > 2.5`
/// means `> 2`).
pub(crate) fn resolve_cmp<N: NativeType>(op: CmpOp, rhs: Wide) -> Resolved<N> {
    if N::IS_FLOAT {
        return Resolved::Kernel(op, rhs.cast());
    }
    match rhs {
        Wide::Int(i) => {
            let (lo, hi) = int_bounds(&N::DTYPE);
            if i < lo {
                Resolved::All(matches!(op, CmpOp::Gt | CmpOp::Ge | CmpOp::Ne))
            } else if i > hi {
                Resolved::All(matches!(op, CmpOp::Lt | CmpOp::Le | CmpOp::Ne))
            } else {
                Resolved::Kernel(op, N::from_i64(i as i64))
            }
        }
        Wide::Float(f) if f.is_nan() => Resolved::All(matches!(op, CmpOp::Lt | CmpOp::Le | CmpOp::Ne)),
        Wide::Float(f) => {
            let floor = f.floor();
            let integral = floor == f;
            let int = |x: f64| Wide::Int(x as i128);
            match op {
                CmpOp::Eq if !integral => Resolved::All(false),
                CmpOp::Ne if !integral => Resolved::All(true),
                CmpOp::Eq | CmpOp::Ne => resolve_cmp(op, int(f)),
                CmpOp::Lt if integral => resolve_cmp(CmpOp::Lt, int(f)),
                CmpOp::Lt | CmpOp::Le => resolve_cmp(CmpOp::Le, int(floor)),
                CmpOp::Ge if integral => resolve_cmp(CmpOp::Ge, int(f)),
                CmpOp::Ge | CmpOp::Gt => resolve_cmp(CmpOp::Gt, int(floor)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// comparisons
// ---------------------------------------------------------------------------

/// `==` / `!=` producing a boolean column.
pub trait ChunkCompareEq<Rhs> {
    /// The mask type.
    type Item;
    /// Row-wise `==`; null where an operand is null.
    fn equal(&self, rhs: Rhs) -> Self::Item;
    /// Row-wise `!=`; null where an operand is null.
    fn not_equal(&self, rhs: Rhs) -> Self::Item;
    /// Row-wise `==` where null equals null and nothing else; never null.
    fn equal_missing(&self, rhs: Rhs) -> Self::Item;
    /// Row-wise `!=` where null equals null and nothing else; never null.
    fn not_equal_missing(&self, rhs: Rhs) -> Self::Item;
}

/// `<`, `<=`, `>`, `>=` producing a boolean column.
pub trait ChunkCompareIneq<Rhs> {
    /// The mask type.
    type Item;
    /// Row-wise `>`.
    fn gt(&self, rhs: Rhs) -> Self::Item;
    /// Row-wise `>=`.
    fn gt_eq(&self, rhs: Rhs) -> Self::Item;
    /// Row-wise `<`.
    fn lt(&self, rhs: Rhs) -> Self::Item;
    /// Row-wise `<=`.
    fn lt_eq(&self, rhs: Rhs) -> Self::Item;
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// `self op rhs` for a scalar, as mask words and the number of matches.
    /// Null rows are *not* cleared.
    pub(crate) fn cmp_scalar_words(&self, op: CmpOp, rhs: Wide) -> (AlignedVec<u64>, usize) {
        match resolve_cmp::<T::Native>(op, rhs) {
            Resolved::Kernel(op, v) => kernels::cmp_scalar_mask(&self.values, op, v),
            Resolved::All(value) => {
                let n = self.len();
                let mut words: AlignedVec<u64> = Vec::with_capacity_in(n.div_ceil(64), ColumnAlloc);
                words.resize(n.div_ceil(64), if value { u64::MAX } else { 0 });
                if let Some(last) = words.last_mut() {
                    *last &= crate::memory::tail_mask(n);
                }
                (words, if value { n } else { 0 })
            }
        }
    }

    /// Compare every row against a scalar.
    pub fn compare_scalar<R: NumericScalar>(&self, op: CmpOp, rhs: R) -> BooleanChunked {
        let (words, count) = self.cmp_scalar_words(op, rhs.wide());
        BooleanChunked {
            name: self.name.clone(),
            values: Bitmap::from_words_with_count(words, self.len(), count),
            validity: self.validity.clone(),
        }
    }

    /// Compare row-wise against another column of the same type.
    ///
    /// # Panics
    /// If the lengths differ (a length-1 operand is broadcast).
    pub fn compare(&self, op: CmpOp, rhs: &ChunkedArray<T>) -> BooleanChunked {
        if rhs.len() == 1 && self.len() != 1 {
            return match rhs.get(0) {
                Some(v) => {
                    let (words, count) = kernels::cmp_scalar_mask(&self.values, op, v);
                    BooleanChunked {
                        name: self.name.clone(),
                        values: Bitmap::from_words_with_count(words, self.len(), count),
                        validity: self.validity.clone(),
                    }
                }
                None => BooleanChunked::full_null(self.name.clone(), self.len()),
            };
        }
        if self.len() == 1 && rhs.len() != 1 {
            return rhs.compare(op.swap_operands(), self).with_name(self.name.clone());
        }
        let (words, count) = kernels::cmp_values_mask(&self.values, &rhs.values, op);
        BooleanChunked {
            name: self.name.clone(),
            values: Bitmap::from_words_with_count(words, self.len(), count),
            validity: combine_validity(self.validity.as_ref(), rhs.validity.as_ref()),
        }
    }
}

/// Turn a nullable comparison into the "missing" flavour: null == null.
fn missing(eq: BooleanChunked, a: Option<&Bitmap>, b: Option<&Bitmap>, negate: bool) -> BooleanChunked {
    let n = eq.len();
    let ones = || Bitmap::new_ones(n);
    let (va, vb) = (a.cloned().unwrap_or_else(ones), b.cloned().unwrap_or_else(ones));
    // Equal when both valid and values equal, or both null.
    let both_valid_eq = eq.values.and(&va).and(&vb);
    let both_null = va.not().and(&vb.not());
    let equal = both_valid_eq.or(&both_null);
    BooleanChunked { name: eq.name, values: if negate { equal.not() } else { equal }, validity: None }
}

impl<T: PolarsNumericType, R: NumericScalar> ChunkCompareEq<R> for ChunkedArray<T> {
    type Item = BooleanChunked;
    fn equal(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Eq, rhs)
    }
    fn not_equal(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Ne, rhs)
    }
    fn equal_missing(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Eq, rhs).fill_null_with_values(false)
    }
    fn not_equal_missing(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Ne, rhs).fill_null_with_values(true)
    }
}

impl<T: PolarsNumericType, R: NumericScalar> ChunkCompareIneq<R> for ChunkedArray<T> {
    type Item = BooleanChunked;
    fn gt(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Gt, rhs)
    }
    fn gt_eq(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Ge, rhs)
    }
    fn lt(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Lt, rhs)
    }
    fn lt_eq(&self, rhs: R) -> BooleanChunked {
        self.compare_scalar(CmpOp::Le, rhs)
    }
}

impl<T: PolarsNumericType> ChunkCompareEq<&ChunkedArray<T>> for ChunkedArray<T> {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        self.compare(CmpOp::Eq, rhs)
    }
    fn not_equal(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        self.compare(CmpOp::Ne, rhs)
    }
    fn equal_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        missing(self.compare(CmpOp::Eq, rhs), self.validity.as_ref(), rhs.validity.as_ref(), false)
    }
    fn not_equal_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        missing(self.compare(CmpOp::Eq, rhs), self.validity.as_ref(), rhs.validity.as_ref(), true)
    }
}

impl<T: PolarsNumericType> ChunkCompareIneq<&ChunkedArray<T>> for ChunkedArray<T> {
    type Item = BooleanChunked;
    fn gt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        self.compare(CmpOp::Gt, rhs)
    }
    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        self.compare(CmpOp::Ge, rhs)
    }
    fn lt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        self.compare(CmpOp::Lt, rhs)
    }
    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        self.compare(CmpOp::Le, rhs)
    }
}

// ---------------------------------------------------------------------------
// filter
// ---------------------------------------------------------------------------

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Keep the rows where `mask` is `true` (null mask rows count as `false`).
    ///
    /// A mask of length 1 is broadcast. If every row is kept the result shares
    /// this column's buffers.
    pub fn filter(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        if mask.len() == 1 && self.len() != 1 {
            return Ok(if mask.get(0) == Some(true) { self.clone() } else { Self::new_empty(self.name.clone()) });
        }
        polars_ensure!(
            mask.len() == self.len(),
            ShapeMismatch: "filter's length: {} differs from that of the series: {}", mask.len(), self.len()
        );
        Ok(self.filter_selection(&Selection::from_mask(mask), None, None))
    }

    /// Apply a prepared selection. `tasks`/`stream` override the defaults
    /// (which are derived from the column size).
    pub(crate) fn filter_selection(&self, sel: &Selection<'_>, tasks: Option<usize>, stream: Option<bool>) -> Self {
        debug_assert_eq!(sel.len, self.len());
        if sel.is_all() {
            return self.clone();
        }
        if sel.is_none() {
            return Self::new_empty(self.name.clone());
        }
        let tasks = tasks.unwrap_or_else(|| par::tasks_for(self.len(), size_of::<T::Native>()));
        let values = kernels::filter_by_mask_with(&self.values, &sel.words, tasks, stream);
        let validity = self.validity.as_ref().map(|v| {
            Bitmap::from_words(kernels::compact_bits(&v.words(), &sel.words, sel.len, sel.count), sel.count)
        });
        ChunkedArray { name: self.name.clone(), values: values.into(), validity: normalize_validity(validity) }
    }

    /// Keep the rows satisfying `value op rhs`, in one pass over the column.
    ///
    /// Equivalent to `self.filter(&self.compare_scalar(op, rhs))` without ever
    /// materialising the mask: the predicate and the compaction are fused
    /// block by block (see [`crate::kernels::pipeline`]).
    pub fn filter_cmp<R: NumericScalar>(&self, op: CmpOp, rhs: R) -> Self {
        let rhs = rhs.wide();
        if self.validity.is_some() {
            let (words, _) = self.cmp_scalar_words(op, rhs);
            let mask = BooleanChunked {
                name: PlSmallStr::EMPTY,
                values: Bitmap::from_words(words, self.len()),
                validity: self.validity.clone(),
            };
            return self.filter_selection(&Selection::from_mask(&mask), None, None);
        }
        match resolve_cmp::<T::Native>(op, rhs) {
            Resolved::All(true) => self.clone(),
            Resolved::All(false) => Self::new_empty(self.name.clone()),
            Resolved::Kernel(op, v) => {
                let tasks = par::tasks_for(self.len(), size_of::<T::Native>());
                let values = if tasks <= 1 {
                    kernels::filter_by_cmp_scalar(&self.values, op, v)
                } else {
                    pipeline::filter_fused(&self.values, op, v, tasks, None)
                };
                ChunkedArray { name: self.name.clone(), values: values.into(), validity: None }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// aggregation
// ---------------------------------------------------------------------------

/// How to interpolate when a quantile falls between two rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantileMethod {
    /// The row nearest to the quantile position.
    Nearest,
    /// The row below.
    Lower,
    /// The row above.
    Higher,
    /// The mean of the rows below and above.
    Midpoint,
    /// Linear interpolation between the rows below and above.
    #[default]
    Linear,
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Number of non-null rows.
    #[inline]
    pub fn count(&self) -> usize {
        self.len() - self.null_count()
    }

    /// Sum of the non-null values (integers wrap on overflow). `Some(0)` when
    /// there is nothing to add, like Polars.
    pub fn sum(&self) -> Option<T::Native> {
        Some(match &self.validity {
            None => aggregate::sum(&self.values[..]),
            Some(v) => aggregate::sum_valid(&self.values[..], &v.words()),
        })
    }

    /// Sum of the non-null values as `f64`.
    pub fn sum_as_f64(&self) -> f64 {
        match &self.validity {
            None => aggregate::sum_f64(&self.values[..]),
            Some(v) => aggregate::sum_f64_valid(&self.values[..], &v.words()),
        }
    }

    /// Smallest and largest non-null value. NaN is ignored unless every value is NaN.
    pub fn min_max(&self) -> Option<(T::Native, T::Native)> {
        match &self.validity {
            None => aggregate::min_max(&self.values[..]),
            Some(v) => aggregate::min_max_valid(&self.values[..], &v.words()),
        }
    }

    /// Smallest non-null value.
    pub fn min(&self) -> Option<T::Native> {
        match &self.validity {
            None => aggregate::min(&self.values[..]),
            Some(_) => self.min_max().map(|(lo, _)| lo),
        }
    }

    /// Largest non-null value.
    pub fn max(&self) -> Option<T::Native> {
        match &self.validity {
            None => aggregate::max(&self.values[..]),
            Some(_) => self.min_max().map(|(_, hi)| hi),
        }
    }

    /// Arithmetic mean of the non-null values.
    pub fn mean(&self) -> Option<f64> {
        let n = self.count();
        (n > 0).then(|| self.sum_as_f64() / n as f64)
    }

    /// Variance with `ddof` delta degrees of freedom (1 for the sample variance).
    pub fn var(&self, ddof: u8) -> Option<f64> {
        let n = self.count();
        if n <= ddof as usize {
            return None;
        }
        let mean = self.sum_as_f64() / n as f64;
        let ss = match &self.validity {
            None => aggregate::sum_sq_dev(&self.values[..], mean),
            Some(v) => aggregate::sum_sq_dev_valid(&self.values[..], &v.words(), mean),
        };
        Some(ss / (n - ddof as usize) as f64)
    }

    /// Standard deviation with `ddof` delta degrees of freedom.
    pub fn std(&self, ddof: u8) -> Option<f64> {
        self.var(ddof).map(f64::sqrt)
    }

    /// The non-null values, copied out.
    pub(crate) fn valid_values(&self) -> Vec<T::Native> {
        match &self.validity {
            None => self.values.to_vec(),
            Some(v) => v.true_indices().map(|i| self.values[i]).collect(),
        }
    }

    /// The `quantile` (in `0..=1`) of the non-null values.
    pub fn quantile(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Option<f64>> {
        polars_ensure!((0.0..=1.0).contains(&quantile), ComputeError: "quantile should be between 0.0 and 1.0");
        let mut v = self.valid_values();
        if v.is_empty() {
            return Ok(None);
        }
        let pos = quantile * (v.len() - 1) as f64;
        let (lo_idx, hi_idx) = (pos.floor() as usize, pos.ceil() as usize);
        let idx = match method {
            QuantileMethod::Nearest => pos.round() as usize,
            QuantileMethod::Higher => hi_idx,
            _ => lo_idx,
        };
        let (_, pivot, upper) = v.select_nth_unstable_by(idx, |a, b| a.tot_cmp(*b));
        let lo = pivot.to_f64();
        if matches!(method, QuantileMethod::Nearest | QuantileMethod::Lower | QuantileMethod::Higher) || hi_idx == lo_idx {
            return Ok(Some(lo));
        }
        // The next order statistic is the minimum of the upper partition.
        let hi = upper.iter().copied().min_by(|a, b| a.tot_cmp(*b)).map_or(lo, Num::to_f64);
        Ok(Some(match method {
            QuantileMethod::Midpoint => (lo + hi) / 2.0,
            _ => lo + (hi - lo) * (pos - lo_idx as f64),
        }))
    }

    /// Median of the non-null values.
    pub fn median(&self) -> Option<f64> {
        self.quantile(0.5, QuantileMethod::Linear).ok().flatten()
    }

    /// Row index of the smallest non-null value (first occurrence).
    pub fn arg_min(&self) -> Option<usize> {
        let (lo, _) = self.min_max()?;
        self.iter().position(|v| v.is_some_and(|v| v.tot_eq(lo)))
    }

    /// Row index of the largest non-null value (first occurrence).
    pub fn arg_max(&self) -> Option<usize> {
        let (_, hi) = self.min_max()?;
        self.iter().position(|v| v.is_some_and(|v| v.tot_eq(hi)))
    }

    /// Number of distinct values; null counts as one value if present.
    pub fn n_unique(&self) -> usize {
        let mut v = self.valid_values();
        v.sort_unstable_by(|a, b| a.tot_cmp(*b));
        v.dedup_by(|a, b| a.tot_eq(*b));
        v.len() + self.has_nulls() as usize
    }

    /// The distinct values in ascending order (null first if present).
    pub fn unique(&self) -> Self {
        let mut v = self.valid_values();
        v.sort_unstable_by(|a, b| a.tot_cmp(*b));
        v.dedup_by(|a, b| a.tot_eq(*b));
        if self.has_nulls() {
            Self::from_iter_options(self.name.clone(), std::iter::once(None).chain(v.into_iter().map(Some)))
        } else {
            Self::from_slice(self.name.clone(), &v)
        }
    }
}

// ---------------------------------------------------------------------------
// element-wise maps and arithmetic
// ---------------------------------------------------------------------------

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// `f` applied to every value; nulls stay null.
    pub fn apply_values(&self, f: impl Fn(T::Native) -> T::Native + Sync) -> Self {
        ChunkedArray {
            name: self.name.clone(),
            values: arith::unary(&self.values, f).into(),
            validity: self.validity.clone(),
        }
    }

    /// `f` applied to every row including nulls.
    pub fn apply(&self, f: impl Fn(Option<T::Native>) -> Option<T::Native>) -> Self {
        Self::from_iter_options(self.name.clone(), self.iter().map(f))
    }

    /// `f` applied to every value, producing a column of another type.
    pub fn apply_cast<U: PolarsNumericType>(&self, f: impl Fn(T::Native) -> U::Native + Sync) -> ChunkedArray<U> {
        ChunkedArray {
            name: self.name.clone(),
            values: arith::unary(&self.values, f).into(),
            validity: self.validity.clone(),
        }
    }

    /// Overwrite every value with `f(value)`.
    ///
    /// When this handle is the only owner of the buffer the update happens in
    /// place, in parallel over disjoint row ranges; otherwise the column is
    /// rebuilt (copy-on-write) and other holders keep seeing the old values.
    pub fn apply_mut(&mut self, f: impl Fn(T::Native) -> T::Native + Sync) {
        match self.values.get_mut() {
            Some(values) => arith::unary_in_place(values, f),
            None => self.values = arith::unary(&self.values, f).into(),
        }
    }

    /// By-value form of [`apply_mut`](Self::apply_mut).
    pub fn apply_in_place(mut self, f: impl Fn(T::Native) -> T::Native + Sync) -> Self {
        self.apply_mut(f);
        self
    }

    /// Mutable access to the values, copying them first if they are shared.
    ///
    /// The returned slice may be split (`split_at_mut`, `chunks_mut`,
    /// `par_chunks_mut`) to mutate disjoint row ranges from several threads.
    pub fn values_mut(&mut self) -> &mut [T::Native] {
        self.values.make_mut()
    }

    /// Set row `index` to `value` (`None` makes it null).
    ///
    /// # Panics
    /// If `index` is out of bounds.
    pub fn set_at(&mut self, index: usize, value: Option<T::Native>) {
        let n = self.len();
        assert!(index < n, "index {index} out of bounds for column of length {n}");
        if let Some(v) = value {
            self.values.make_mut()[index] = v;
        }
        if value.is_none() || self.validity.is_some() {
            let mut bits = match self.validity.take() {
                Some(v) => v.into_mut(),
                None => MutableBitmap::from_len_set(n),
            };
            bits.set(index, value.is_some());
            self.validity = normalize_validity(Some(bits.freeze()));
        }
    }

    /// Replace the rows where `mask` is `true` by `value`.
    pub fn set(&self, mask: &BooleanChunked, value: Option<T::Native>) -> PolarsResult<Self> {
        polars_ensure!(
            mask.len() == self.len(),
            ShapeMismatch: "mask length {} differs from column length {}", mask.len(), self.len()
        );
        let sel = mask.true_mask();
        let mut out = self.clone();
        match value {
            Some(v) => {
                let values = out.values.make_mut();
                for i in sel.true_indices() {
                    values[i] = v;
                }
                out.validity = normalize_validity(out.validity.map(|valid| valid.or(&sel)));
            }
            None => {
                let valid = self.validity.clone().unwrap_or_else(|| Bitmap::new_ones(self.len()));
                out.validity = normalize_validity(Some(valid.and_not(&sel)));
            }
        }
        Ok(out)
    }

    fn arithmetic(&self, rhs: &Self, f: impl Fn(T::Native, T::Native) -> T::Native + Sync) -> Self {
        if rhs.len() == 1 && self.len() != 1 {
            return match rhs.get(0) {
                Some(r) => self.apply_values(move |a| f(a, r)),
                None => Self::full_null(self.name.clone(), self.len()),
            };
        }
        if self.len() == 1 && rhs.len() != 1 {
            return match self.get(0) {
                Some(l) => rhs.apply_values(move |b| f(l, b)).with_name(self.name.clone()),
                None => Self::full_null(self.name.clone(), rhs.len()),
            };
        }
        ChunkedArray {
            name: self.name.clone(),
            values: arith::binary(&self.values, &rhs.values, f).into(),
            validity: combine_validity(self.validity.as_ref(), rhs.validity.as_ref()),
        }
    }

    /// Division or remainder: for integers a zero divisor yields null.
    fn checked_arithmetic(&self, rhs: &Self, f: impl Fn(T::Native, T::Native) -> Option<T::Native> + Sync) -> Self {
        if T::Native::IS_FLOAT {
            return self.arithmetic(rhs, |a, b| f(a, b).unwrap_or_default());
        }
        let (a, b) = if rhs.len() == 1 && self.len() != 1 {
            (self.clone(), Self::from_iter_options("".into(), std::iter::repeat_n(rhs.get(0), self.len())))
        } else if self.len() == 1 && rhs.len() != 1 {
            (Self::from_iter_options(self.name.clone(), std::iter::repeat_n(self.get(0), rhs.len())), rhs.clone())
        } else {
            (self.clone(), rhs.clone())
        };
        assert_eq!(a.len(), b.len(), "operands must have equal length");
        Self::from_iter_options(
            self.name.clone(),
            a.iter().zip(b.iter()).map(|(x, y)| match (x, y) {
                (Some(x), Some(y)) => f(x, y),
                _ => None,
            }),
        )
    }
}

macro_rules! arithmetic_ops {
    ($($trait:ident, $method:ident, $num:ident;)*) => {$(
        impl<T: PolarsNumericType> $trait<&ChunkedArray<T>> for &ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: &ChunkedArray<T>) -> ChunkedArray<T> {
                self.arithmetic(rhs, <T::Native as Num>::$num)
            }
        }
        impl<T: PolarsNumericType> $trait<ChunkedArray<T>> for ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: ChunkedArray<T>) -> ChunkedArray<T> {
                (&self).$method(&rhs)
            }
        }
        impl<T: PolarsNumericType, R: NumericScalar> $trait<R> for &ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: R) -> ChunkedArray<T> {
                let r: T::Native = rhs.wide().cast();
                self.apply_values(move |a| <T::Native as Num>::$num(a, r))
            }
        }
        impl<T: PolarsNumericType, R: NumericScalar> $trait<R> for ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: R) -> ChunkedArray<T> {
                (&self).$method(rhs)
            }
        }
    )*};
}
arithmetic_ops! { Add, add, add; Sub, sub, sub; Mul, mul, mul; }

macro_rules! checked_ops {
    ($($trait:ident, $method:ident, $num:ident;)*) => {$(
        impl<T: PolarsNumericType> $trait<&ChunkedArray<T>> for &ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: &ChunkedArray<T>) -> ChunkedArray<T> {
                self.checked_arithmetic(rhs, <T::Native as Num>::$num)
            }
        }
        impl<T: PolarsNumericType> $trait<ChunkedArray<T>> for ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: ChunkedArray<T>) -> ChunkedArray<T> {
                (&self).$method(&rhs)
            }
        }
        impl<T: PolarsNumericType, R: NumericScalar> $trait<R> for &ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: R) -> ChunkedArray<T> {
                let r: T::Native = rhs.wide().cast();
                if T::Native::IS_FLOAT {
                    return self.apply_values(move |a| <T::Native as Num>::$num(a, r).unwrap_or_default());
                }
                match <T::Native as Num>::$num(T::Native::ONE, r) {
                    // A valid integer divisor never fails except for MIN / -1.
                    Some(_) => self.apply(|a| a.and_then(|a| <T::Native as Num>::$num(a, r))),
                    None => ChunkedArray::full_null(self.name.clone(), self.len()),
                }
            }
        }
        impl<T: PolarsNumericType, R: NumericScalar> $trait<R> for ChunkedArray<T> {
            type Output = ChunkedArray<T>;
            fn $method(self, rhs: R) -> ChunkedArray<T> {
                (&self).$method(rhs)
            }
        }
    )*};
}
checked_ops! { Div, div, checked_div; Rem, rem, checked_rem; }

// ---------------------------------------------------------------------------
// sort / take
// ---------------------------------------------------------------------------

/// Options for sorting a single column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SortOptions {
    /// Largest first.
    pub descending: bool,
    /// Put nulls after the values instead of before.
    pub nulls_last: bool,
    /// Sort on all cores when the column is large.
    pub multithreaded: bool,
    /// Keep equal rows in their original order.
    pub maintain_order: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        SortOptions { descending: false, nulls_last: false, multithreaded: true, maintain_order: false }
    }
}

impl SortOptions {
    /// Default options: ascending, nulls first.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the direction.
    pub fn with_order_descending(mut self, descending: bool) -> Self {
        self.descending = descending;
        self
    }

    /// Set where nulls go.
    pub fn with_nulls_last(mut self, nulls_last: bool) -> Self {
        self.nulls_last = nulls_last;
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
}

/// Rows per thread below which a parallel sort is not worth starting.
const PAR_SORT_MIN: usize = 1 << 15;

pub(crate) fn sort_slice<N: NativeType>(v: &mut [N], options: SortOptions) {
    let cmp = move |a: &N, b: &N| if options.descending { b.tot_cmp(*a) } else { a.tot_cmp(*b) };
    let parallel = options.multithreaded && v.len() >= PAR_SORT_MIN && par::threads() > 1;
    match (parallel, options.maintain_order) {
        (true, true) => v.par_sort_by(cmp),
        (true, false) => v.par_sort_unstable_by(cmp),
        (false, true) => v.sort_by(cmp),
        (false, false) => v.sort_unstable_by(cmp),
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Sort the column (nulls first).
    pub fn sort(&self, descending: bool) -> Self {
        self.sort_with(SortOptions::default().with_order_descending(descending))
    }

    /// Sort the column.
    pub fn sort_with(&self, options: SortOptions) -> Self {
        let nulls = self.null_count();
        let mut v = self.valid_values();
        sort_slice(&mut v, options);
        if nulls == 0 {
            return Self::from_slice(self.name.clone(), &v);
        }
        let null_iter = std::iter::repeat_n(None, nulls);
        let values = v.into_iter().map(Some);
        if options.nulls_last {
            Self::from_iter_options(self.name.clone(), values.chain(null_iter))
        } else {
            Self::from_iter_options(self.name.clone(), null_iter.chain(values))
        }
    }

    /// The permutation that sorts the column.
    pub fn arg_sort(&self, options: SortOptions) -> IdxCa {
        let mut keyed: Vec<(T::Native, IdxSize)> = match &self.validity {
            None => self.values.iter().copied().zip(0..).collect(),
            Some(v) => v.true_indices().map(|i| (self.values[i], i as IdxSize)).collect(),
        };
        let cmp = move |a: &(T::Native, IdxSize), b: &(T::Native, IdxSize)| {
            let ord = if options.descending { b.0.tot_cmp(a.0) } else { a.0.tot_cmp(b.0) };
            // Ties resolve by row index, which makes the unstable sorts stable.
            ord.then(a.1.cmp(&b.1))
        };
        if options.multithreaded && keyed.len() >= PAR_SORT_MIN && par::threads() > 1 {
            keyed.par_sort_unstable_by(cmp);
        } else {
            keyed.sort_unstable_by(cmp);
        }
        let sorted = keyed.into_iter().map(|(_, i)| i);
        let null_rows: Vec<IdxSize> = match &self.validity {
            None => Vec::new(),
            Some(v) => (0..self.len() as IdxSize).filter(|&i| !v.get(i as usize)).collect(),
        };
        let idx: Buffer<IdxSize> = if options.nulls_last {
            sorted.chain(null_rows).collect()
        } else {
            null_rows.into_iter().chain(sorted).collect()
        };
        IdxCa { name: self.name.clone(), values: idx, validity: None }
    }

    /// The rows at `indices`, in that order.
    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        self.take_slice(idx_slice(indices)?)
    }

    /// The rows at `indices`, in that order.
    pub fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Self> {
        if let Some(&max) = indices.iter().max() {
            polars_ensure!((max as usize) < self.len(), OutOfBounds: "index {} is out of bounds for column of length {}", max, self.len());
        }
        Ok(self.take_unchecked(indices))
    }

    /// Gather rows at indices already known to be in bounds (the positions of
    /// the set bits of a mask over this column).
    ///
    /// # Safety
    /// Every index must be below `self.len()`.
    pub(crate) unsafe fn take_trusted(&self, indices: &[IdxSize]) -> Self {
        ChunkedArray {
            name: self.name.clone(),
            // SAFETY: guaranteed by the caller.
            values: unsafe { arith::take_unchecked(&self.values, indices) }.into(),
            validity: normalize_validity(
                // SAFETY: as above.
                self.validity.as_ref().map(|v| indices.iter().map(|&i| unsafe { v.get_unchecked(i as usize) }).collect::<Bitmap>()),
            ),
        }
    }

    /// Gather without reporting out-of-bounds indices as an error (they panic).
    pub(crate) fn take_unchecked(&self, indices: &[IdxSize]) -> Self {
        ChunkedArray {
            name: self.name.clone(),
            values: arith::take(&self.values, indices).into(),
            validity: normalize_validity(
                self.validity.as_ref().map(|v| indices.iter().map(|&i| v.get(i as usize)).collect::<Bitmap>()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// nulls and casts
// ---------------------------------------------------------------------------

/// How to replace nulls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillNullStrategy {
    /// The previous non-null value.
    Forward(Option<usize>),
    /// The next non-null value.
    Backward(Option<usize>),
    /// The column mean.
    Mean,
    /// The column minimum.
    Min,
    /// The column maximum.
    Max,
    /// Zero.
    Zero,
    /// One.
    One,
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// `true` where the row is null.
    pub fn is_null(&self) -> BooleanChunked {
        let values = match &self.validity {
            None => Bitmap::new_zeroed(self.len()),
            Some(v) => v.not(),
        };
        BooleanChunked { name: self.name.clone(), values, validity: None }
    }

    /// `true` where the row is not null.
    pub fn is_not_null(&self) -> BooleanChunked {
        let values = match &self.validity {
            None => Bitmap::new_ones(self.len()),
            Some(v) => v.clone(),
        };
        BooleanChunked { name: self.name.clone(), values, validity: None }
    }

    /// `true` where the value is NaN (always `false` for integers).
    pub fn is_nan(&self) -> BooleanChunked {
        BooleanChunked {
            name: self.name.clone(),
            values: self.values.iter().map(|v| v.is_nan()).collect(),
            validity: self.validity.clone(),
        }
    }

    /// Drop the null rows.
    pub fn drop_nulls(&self) -> Self {
        match &self.validity {
            None => self.clone(),
            Some(v) => {
                let sel = Selection { words: v.words(), len: v.len(), count: v.count_ones() };
                let mut out = self.filter_selection(&sel, None, None);
                out.validity = None;
                out
            }
        }
    }

    /// Replace nulls with `value`.
    pub fn fill_null_with_values(&self, value: T::Native) -> Self {
        match &self.validity {
            None => self.clone(),
            Some(v) => {
                let mut values = self.values.clone();
                let slots = values.make_mut();
                for (i, valid) in v.iter().enumerate() {
                    if !valid {
                        slots[i] = value;
                    }
                }
                ChunkedArray { name: self.name.clone(), values, validity: None }
            }
        }
    }

    /// Replace nulls according to `strategy`.
    pub fn fill_null(&self, strategy: FillNullStrategy) -> Self {
        if !self.has_nulls() {
            return self.clone();
        }
        let directional = |limit: Option<usize>, iter: &mut dyn Iterator<Item = Option<T::Native>>| {
            let (mut last, mut run) = (None, 0usize);
            iter.map(|v| match v {
                Some(v) => {
                    (last, run) = (Some(v), 0);
                    Some(v)
                }
                None => {
                    run += 1;
                    if limit.is_none_or(|l| run <= l) { last } else { None }
                }
            })
            .collect::<Vec<_>>()
        };
        match strategy {
            FillNullStrategy::Forward(limit) => {
                Self::from_iter_options(self.name.clone(), directional(limit, &mut self.iter()))
            }
            FillNullStrategy::Backward(limit) => {
                let mut filled = directional(limit, &mut self.iter().rev());
                filled.reverse();
                Self::from_iter_options(self.name.clone(), filled)
            }
            FillNullStrategy::Mean => match self.mean() {
                Some(m) => self.fill_null_with_values(T::Native::from_f64(m)),
                None => self.clone(),
            },
            FillNullStrategy::Min => self.min().map_or_else(|| self.clone(), |m| self.fill_null_with_values(m)),
            FillNullStrategy::Max => self.max().map_or_else(|| self.clone(), |m| self.fill_null_with_values(m)),
            FillNullStrategy::Zero => self.fill_null_with_values(T::Native::ZERO),
            FillNullStrategy::One => self.fill_null_with_values(T::Native::ONE),
        }
    }

    /// Convert to another numeric type.
    ///
    /// Values that do not fit the target (out of range, or NaN/fractional
    /// overflow for float → integer) become null, as in Polars' non-strict cast.
    pub fn cast_numeric<U: PolarsNumericType>(&self) -> ChunkedArray<U> {
        let (src_float, dst_float) = (T::Native::IS_FLOAT, U::Native::IS_FLOAT);
        if dst_float {
            // Every source value has a float image (possibly rounded).
            return ChunkedArray {
                name: self.name.clone(),
                values: arith::unary(&self.values, move |v| {
                    if src_float { U::Native::from_f64(v.to_f64()) } else { int_to_float::<T::Native, U::Native>(v) }
                })
                .into(),
                validity: self.validity.clone(),
            };
        }
        let (lo, hi) = int_bounds(&U::get_dtype());
        let convert = move |v: T::Native| -> Option<U::Native> {
            let wide: i128 = if src_float {
                let f = v.to_f64();
                if !f.is_finite() {
                    return None;
                }
                f.trunc() as i128
            } else {
                int_to_i128(v)
            };
            (lo..=hi).contains(&wide).then(|| U::Native::from_i64(wide as i64))
        };
        // Fast path: nothing overflows, so the validity is unchanged.
        let fits = match self.min_max() {
            None => true,
            Some((min, max)) => convert(min).is_some() && convert(max).is_some() && !max.is_nan(),
        };
        if fits && !(src_float && aggregate::count_nan(&self.values[..]) > 0) {
            return ChunkedArray {
                name: self.name.clone(),
                values: arith::unary(&self.values, move |v| convert(v).unwrap_or_default()).into(),
                validity: self.validity.clone(),
            };
        }
        ChunkedArray::from_iter_options(self.name.clone(), self.iter().map(|v| v.and_then(convert)))
    }
}

/// Exact integer widening (`u64` does not go through `i64`).
#[inline]
fn int_to_i128<N: NativeType>(v: N) -> i128 {
    if N::DTYPE == DataType::UInt64 { v.to_i64() as u64 as i128 } else { v.to_i64() as i128 }
}

#[inline]
fn int_to_float<N: NativeType, F: NativeType>(v: N) -> F {
    if N::DTYPE == DataType::UInt64 { F::from_f64(v.to_i64() as u64 as f64) } else { F::from_f64(v.to_f64()) }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    fn i64s(v: &[i64]) -> Int64Chunked {
        Int64Chunked::from_slice("a".into(), v)
    }

    #[test]
    fn scalar_comparisons_respect_type_ranges() {
        let u = UInt8Chunked::from_slice("u".into(), &[0, 10, 255]);
        let t = Some(true);
        let f = Some(false);
        assert_eq!(u.gt(300).to_vec(), vec![f, f, f]);
        assert_eq!(u.lt(300).to_vec(), vec![t, t, t]);
        assert_eq!(u.gt(-1).to_vec(), vec![t, t, t]);
        assert_eq!(u.not_equal(1000).to_vec(), vec![t, t, t]);
        assert_eq!(u.equal(255u64).to_vec(), vec![f, f, t]);
        let i = i64s(&[1, 2, 3, 4]);
        assert_eq!(i.gt(2.5).to_vec(), vec![f, f, t, t]);
        assert_eq!(i.gt_eq(2.5).to_vec(), vec![f, f, t, t]);
        assert_eq!(i.lt(2.5).to_vec(), vec![t, t, f, f]);
        assert_eq!(i.lt_eq(2.0).to_vec(), vec![t, t, f, f]);
        assert_eq!(i.equal(2.5).to_vec(), vec![f, f, f, f]);
        assert_eq!(i.equal(3.0).to_vec(), vec![f, f, t, f]);
        assert_eq!(i.lt(f64::NAN).to_vec(), vec![t, t, t, t]);
        let big = UInt64Chunked::from_slice("b".into(), &[u64::MAX, 1]);
        assert_eq!(big.equal(u64::MAX).to_vec(), vec![t, f]);
        assert_eq!(big.gt(i64::MAX).to_vec(), vec![t, f]);
    }

    #[test]
    fn comparisons_propagate_nulls() {
        let a = Int64Chunked::from_slice_options("a".into(), &[Some(1), None, Some(3)]);
        let b = Int64Chunked::from_slice_options("b".into(), &[Some(1), None, None]);
        assert_eq!(a.gt(1).to_vec(), vec![Some(false), None, Some(true)]);
        assert_eq!(a.equal(&b).to_vec(), vec![Some(true), None, None]);
        assert_eq!(a.equal_missing(&b).to_vec(), vec![Some(true), Some(true), Some(false)]);
        assert_eq!(a.not_equal_missing(&b).to_vec(), vec![Some(false), Some(false), Some(true)]);
        assert_eq!(a.lt(&i64s(&[2])).to_vec(), vec![Some(true), None, Some(false)]);
    }

    #[test]
    fn filter_variants_agree() {
        let n = 10_000;
        let ca = i64s(&(0..n).map(|i| (i * 7919) % 1000).collect::<Vec<_>>());
        let want: Vec<i64> = ca.cont_slice().unwrap().iter().copied().filter(|&v| v < 300).collect();
        let mask = ca.lt(300);
        assert_eq!(ca.filter(&mask).unwrap().cont_slice().unwrap(), &want[..]);
        assert_eq!(ca.filter_cmp(CmpOp::Lt, 300).cont_slice().unwrap(), &want[..]);
        assert_eq!(ca.filter_cmp(CmpOp::Lt, 300.0).cont_slice().unwrap(), &want[..]);
        // All selected: the buffer is shared, not copied.
        let all = ca.filter(&ca.lt(5000)).unwrap();
        assert_eq!(all.values().as_ptr(), ca.values().as_ptr());
        assert!(ca.filter(&ca.lt(-1)).unwrap().is_empty());
        assert!(ca.filter(&BooleanChunked::from_slice("m".into(), &[true, false])).is_err());
        assert_eq!(ca.filter(&BooleanChunked::from_slice("m".into(), &[true])).unwrap().len(), n as usize);

        let nullable = Int64Chunked::from_iter_options("n".into(), (0..100).map(|i| (i % 3 != 0).then_some(i)));
        let kept = nullable.filter(&nullable.gt_eq(50)).unwrap();
        assert_eq!(kept.null_count(), 0);
        assert_eq!(kept.len(), (50..100).filter(|i| i % 3 != 0).count());
        assert_eq!(nullable.filter_cmp(CmpOp::Ge, 50), kept);
        let keep_nulls = nullable.filter(&BooleanChunked::full("m".into(), true, 100).slice(0, 100)).unwrap();
        assert_eq!(keep_nulls.null_count(), 34);
        let some = nullable.filter(&BooleanChunked::from_iter_values("m".into(), (0..100).map(|i| i < 10))).unwrap();
        assert_eq!(some.to_vec(), (0..10).map(|i| (i % 3 != 0).then_some(i)).collect::<Vec<_>>());
    }

    #[test]
    fn aggregates_skip_nulls() {
        let ca = Float64Chunked::from_slice_options("f".into(), &[Some(1.0), None, Some(4.0), Some(7.0)]);
        assert_eq!(ca.sum(), Some(12.0));
        assert_eq!(ca.mean(), Some(4.0));
        assert_eq!((ca.min(), ca.max()), (Some(1.0), Some(7.0)));
        assert_eq!(ca.count(), 3);
        assert_eq!(ca.var(1), Some(9.0));
        assert_eq!(ca.std(1), Some(3.0));
        assert_eq!(ca.median(), Some(4.0));
        assert_eq!(ca.quantile(0.25, QuantileMethod::Linear).unwrap(), Some(2.5));
        assert_eq!(ca.quantile(0.25, QuantileMethod::Lower).unwrap(), Some(1.0));
        assert_eq!(ca.quantile(0.25, QuantileMethod::Higher).unwrap(), Some(4.0));
        assert!(ca.quantile(1.5, QuantileMethod::Linear).is_err());
        assert_eq!((ca.arg_min(), ca.arg_max()), (Some(0), Some(3)));
        let empty = Int32Chunked::new_empty("e".into());
        assert_eq!((empty.sum(), empty.min(), empty.mean(), empty.median()), (Some(0), None, None, None));
        let ints = i64s(&[5, 3, 5, 1, 3]);
        assert_eq!(ints.n_unique(), 3);
        assert_eq!(ints.unique().cont_slice().unwrap(), &[1, 3, 5]);
        assert_eq!(ints.median(), Some(3.0));
    }

    #[test]
    fn arithmetic_and_nulls() {
        let a = Int64Chunked::from_slice_options("a".into(), &[Some(10), None, Some(-7)]);
        let b = i64s(&[3, 4, 0]);
        assert_eq!((&a + &b).to_vec(), vec![Some(13), None, Some(-7)]);
        assert_eq!((&a - 1).to_vec(), vec![Some(9), None, Some(-8)]);
        assert_eq!((&a * 2).to_vec(), vec![Some(20), None, Some(-14)]);
        assert_eq!((&a / &b).to_vec(), vec![Some(3), None, None]);
        assert_eq!((&a % 4).to_vec(), vec![Some(2), None, Some(-3)]);
        assert_eq!((&a / 0).null_count(), 3);
        let f = Float64Chunked::from_slice("f".into(), &[1.0, 2.0]);
        assert_eq!((&f / 0.0).cont_slice().unwrap(), &[f64::INFINITY, f64::INFINITY]);
        assert_eq!((&f * &Float64Chunked::from_slice("s".into(), &[10.0])).cont_slice().unwrap(), &[10.0, 20.0]);
        assert_eq!((f + 0.5).cont_slice().unwrap(), &[1.5, 2.5]);
    }

    #[test]
    fn apply_mut_is_in_place_only_when_unique() {
        let mut ca = Float64Chunked::from_slice("f".into(), &[1.0, 2.0, 3.0]);
        let p = ca.values().as_ptr();
        ca.apply_mut(|v| v * 2.0);
        assert_eq!(ca.values().as_ptr(), p, "unique: updated in place");
        let snapshot = ca.clone();
        ca.apply_mut(|v| v + 1.0);
        assert_ne!(ca.values().as_ptr(), p, "shared: copy-on-write");
        assert_eq!(snapshot.cont_slice().unwrap(), &[2.0, 4.0, 6.0]);
        assert_eq!(ca.cont_slice().unwrap(), &[3.0, 5.0, 7.0]);
        ca.set_at(1, None);
        ca.set_at(0, Some(-1.0));
        assert_eq!(ca.to_vec(), vec![Some(-1.0), None, Some(7.0)]);
        ca.set_at(1, Some(0.0));
        assert!(ca.validity().is_none());
        let masked = ca.set(&BooleanChunked::from_slice("m".into(), &[true, false, true]), Some(9.0)).unwrap();
        assert_eq!(masked.cont_slice().unwrap(), &[9.0, 0.0, 9.0]);
        let nulled = ca.set(&BooleanChunked::from_slice("m".into(), &[false, true, false]), None).unwrap();
        assert_eq!(nulled.to_vec(), vec![Some(-1.0), None, Some(7.0)]);
    }

    #[test]
    fn sort_take_and_arg_sort() {
        let ca = Int64Chunked::from_slice_options("a".into(), &[Some(3), None, Some(1), Some(2), None]);
        assert_eq!(ca.sort(false).to_vec(), vec![None, None, Some(1), Some(2), Some(3)]);
        assert_eq!(ca.sort(true).to_vec(), vec![None, None, Some(3), Some(2), Some(1)]);
        let last = SortOptions::default().with_nulls_last(true);
        assert_eq!(ca.sort_with(last).to_vec(), vec![Some(1), Some(2), Some(3), None, None]);
        let idx = ca.arg_sort(SortOptions::default());
        assert_eq!(idx.cont_slice().unwrap(), &[1, 4, 2, 3, 0]);
        assert_eq!(ca.take(&idx).unwrap(), ca.sort(false));
        assert!(ca.take_slice(&[5]).is_err());
        let f = Float64Chunked::from_slice("f".into(), &[2.0, f64::NAN, -1.0]);
        let sorted = f.sort(false);
        assert_eq!(&sorted.cont_slice().unwrap()[..2], &[-1.0, 2.0]);
        assert!(sorted.get(2).unwrap().is_nan());
        let big: Vec<i64> = (0..100_000).map(|i| (i * 7919) % 10_007).collect();
        let sorted = i64s(&big).sort(false);
        assert!(sorted.cont_slice().unwrap().windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn null_handling() {
        let ca = Int32Chunked::from_slice_options("a".into(), &[None, Some(2), None, None, Some(5)]);
        assert_eq!(ca.is_null().to_vec(), vec![Some(true), Some(false), Some(true), Some(true), Some(false)]);
        assert_eq!(ca.drop_nulls().cont_slice().unwrap(), &[2, 5]);
        assert_eq!(ca.fill_null_with_values(0).cont_slice().unwrap(), &[0, 2, 0, 0, 5]);
        assert_eq!(ca.fill_null(FillNullStrategy::Forward(None)).to_vec(), vec![None, Some(2), Some(2), Some(2), Some(5)]);
        assert_eq!(ca.fill_null(FillNullStrategy::Forward(Some(1))).to_vec(), vec![None, Some(2), Some(2), None, Some(5)]);
        assert_eq!(ca.fill_null(FillNullStrategy::Backward(None)).to_vec(), vec![Some(2), Some(2), Some(5), Some(5), Some(5)]);
        assert_eq!(ca.fill_null(FillNullStrategy::Max).cont_slice().unwrap(), &[5, 2, 5, 5, 5]);
        assert_eq!(ca.fill_null(FillNullStrategy::Zero).cont_slice().unwrap(), &[0, 2, 0, 0, 5]);
    }

    #[test]
    fn casts() {
        let ca = i64s(&[1, 300, -1]);
        assert_eq!(ca.cast_numeric::<UInt8Type>().to_vec(), vec![Some(1), None, None]);
        assert_eq!(ca.cast_numeric::<Float64Type>().cont_slice().unwrap(), &[1.0, 300.0, -1.0]);
        assert_eq!(ca.cast_numeric::<Int32Type>().cont_slice().unwrap(), &[1, 300, -1]);
        let f = Float64Chunked::from_slice("f".into(), &[1.9, -2.9, f64::NAN, 1e30]);
        assert_eq!(f.cast_numeric::<Int64Type>().to_vec(), vec![Some(1), Some(-2), None, None]);
        let u = UInt64Chunked::from_slice("u".into(), &[u64::MAX]);
        assert_eq!(u.cast_numeric::<Int64Type>().to_vec(), vec![None]);
        assert_eq!(u.cast_numeric::<Float64Type>().cont_slice().unwrap(), &[u64::MAX as f64]);
    }
}
