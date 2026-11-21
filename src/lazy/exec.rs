//! Block-wise execution of filters and filtered aggregations.
//!
//! A predicate is compiled against a frame into a small tree whose leaves are
//! kernels over row ranges ("`a[range] < 500` → mask words"). The executor
//! then walks the frame in cache-sized blocks:
//!
//! * [`filter_frame`] evaluates the predicate for a block and immediately
//!   compacts that block of every output column, using the ordered-commit
//!   pipeline so that all cores take part and the output stays contiguous;
//! * [`aggregate_filtered`] evaluates the predicate for a block, compacts the
//!   aggregated column into a scratch buffer that never leaves L1/L2 and
//!   reduces it there — the filtered rows are never materialised at all.
//!
//! Anything the compiler cannot express as a leaf is evaluated once with the
//! whole-column evaluator and becomes a "precomputed mask" leaf, so every
//! predicate is supported and the common ones are fast.

use std::borrow::Cow;
use std::ops::Range;


use super::dsl::{Expr, LiteralValue, Operator};
use super::eval::eval_mask;
use crate::chunked_array::{ChunkedArray, Resolved, Selection, StringChunked, Wide, resolve_cmp};
use crate::datatypes::{AnyValue, DataType, PlSmallStr, PolarsNumericType};
use crate::error::PolarsResult;
use crate::frame::filter::{Sink, emit_block, sink_for};
use crate::frame::{AggKind, DataFrame};
use crate::kernels::aggregate::Num;
use crate::kernels::pipeline::{self, FenceOnDrop};
use crate::kernels::{self, CmpOp, Kernel, cache, par};
use crate::memory::{APRON, AlignedVec, Bitmap, tail_mask, words_for};
use crate::polars_bail;
use crate::series::{Column, Series, SeriesInner, for_numeric};

type Leaf<'a> = Box<dyn Fn(Range<usize>, &mut [u64]) -> usize + Send + Sync + 'a>;

/// A predicate compiled for evaluation over row ranges.
enum Node<'a> {
    /// The same answer for every row.
    Const(bool),
    /// A kernel over a row range.
    Leaf(Leaf<'a>),
    /// Mask words for the whole frame, computed up front.
    Bits(Cow<'a, [u64]>),
    /// Both children hold.
    And(Box<Node<'a>>, Box<Node<'a>>),
    /// Either child holds.
    Or(Box<Node<'a>>, Box<Node<'a>>),
}

impl Node<'_> {
    /// Number of scratch mask buffers needed below this node.
    fn depth(&self) -> usize {
        match self {
            Node::And(a, b) | Node::Or(a, b) => a.depth().max(b.depth() + 1),
            _ => 0,
        }
    }

    /// Write the mask of `rows` (which starts on a multiple of 64) to `out`
    /// and return the number of set bits. Bits past the range are cleared.
    fn eval(&self, rows: Range<usize>, out: &mut [u64], scratch: &mut [Vec<u64>]) -> usize {
        let n = rows.len();
        let nw = words_for(n);
        match self {
            Node::Const(false) => {
                out[..nw].fill(0);
                0
            }
            Node::Const(true) => {
                out[..nw].fill(u64::MAX);
                if nw > 0 {
                    out[nw - 1] &= tail_mask(n);
                }
                n
            }
            Node::Leaf(f) => f(rows, out),
            Node::Bits(bits) => {
                let first = rows.start / 64;
                out[..nw].copy_from_slice(&bits[first..first + nw]);
                if nw > 0 {
                    out[nw - 1] &= tail_mask(n);
                }
                out[..nw].iter().map(|w| w.count_ones() as usize).sum()
            }
            Node::And(a, b) => {
                if a.eval(rows.clone(), out, scratch) == 0 {
                    return 0;
                }
                let (tmp, rest) = scratch.split_first_mut().expect("scratch sized by depth()");
                b.eval(rows, tmp, rest);
                let mut count = 0;
                for (o, t) in out[..nw].iter_mut().zip(&tmp[..nw]) {
                    *o &= *t;
                    count += o.count_ones() as usize;
                }
                count
            }
            Node::Or(a, b) => {
                if a.eval(rows.clone(), out, scratch) == n {
                    return n;
                }
                let (tmp, rest) = scratch.split_first_mut().expect("scratch sized by depth()");
                b.eval(rows, tmp, rest);
                let mut count = 0;
                for (o, t) in out[..nw].iter_mut().zip(&tmp[..nw]) {
                    *o |= *t;
                    count += o.count_ones() as usize;
                }
                count
            }
        }
    }
}

/// AND the mask in `out` (covering `rows`) with the validity words of the column.
#[inline]
fn and_validity(valid: &[u64], rows: &Range<usize>, out: &mut [u64]) -> usize {
    let nw = words_for(rows.len());
    let first = rows.start / 64;
    let mut count = 0;
    for (o, v) in out[..nw].iter_mut().zip(&valid[first..first + nw]) {
        *o &= *v;
        count += o.count_ones() as usize;
    }
    count
}

fn validity_words(validity: Option<&Bitmap>) -> Option<Cow<'_, [u64]>> {
    validity.map(Bitmap::words)
}

fn numeric_scalar_leaf<'a, T: PolarsNumericType>(ca: &'a ChunkedArray<T>, op: CmpOp, rhs: Wide) -> Node<'a> {
    let valid = validity_words(ca.validity());
    match resolve_cmp::<T::Native>(op, rhs) {
        Resolved::All(false) => Node::Const(false),
        Resolved::All(true) => match valid {
            None => Node::Const(true),
            Some(bits) => Node::Bits(bits),
        },
        Resolved::Kernel(op, v) => {
            let values = ca.values().as_slice();
            match valid {
                None => Node::Leaf(Box::new(move |rows, out| T::Native::cmp_scalar(&values[rows], op, v, out))),
                Some(valid) => Node::Leaf(Box::new(move |rows, out| {
                    T::Native::cmp_scalar(&values[rows.clone()], op, v, out);
                    and_validity(&valid, &rows, out)
                })),
            }
        }
    }
}

fn numeric_column_leaf<'a, T: PolarsNumericType>(a: &'a ChunkedArray<T>, op: CmpOp, b: &'a ChunkedArray<T>) -> Node<'a> {
    let (va, vb) = (a.values().as_slice(), b.values().as_slice());
    let valid: Option<Cow<'a, [u64]>> = match (a.validity(), b.validity()) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x.words()),
        (Some(x), Some(y)) => Some(Cow::Owned(x.and(y).words().into_owned())),
    };
    Node::Leaf(Box::new(move |rows, out| {
        let count = T::Native::cmp_values(&va[rows.clone()], &vb[rows.clone()], op, out);
        match &valid {
            None => count,
            Some(valid) => and_validity(valid, &rows, out),
        }
    }))
}

fn string_leaf<'a>(ca: &'a StringChunked, op: CmpOp, rhs: &'a str) -> Node<'a> {
    let valid = validity_words(ca.validity());
    Node::Leaf(Box::new(move |rows, out| {
        let count = ca.compare_scalar_words(op, rhs, rows.clone(), out);
        match &valid {
            None => count,
            Some(valid) => and_validity(valid, &rows, out),
        }
    }))
}

/// The rows where a boolean column is `true` (or `false`, when negated).
fn boolean_node(column: &Column, negate: bool) -> Option<Node<'_>> {
    let ca = column.bool().ok()?;
    let bits = if negate { (!ca).true_mask() } else { ca.true_mask() };
    Some(Node::Bits(Cow::Owned(bits.words().into_owned())))
}

fn compile<'a>(df: &'a DataFrame, expr: &'a Expr, negate: bool) -> PolarsResult<Node<'a>> {
    let fallback = |expr: &Expr| -> PolarsResult<Node<'a>> {
        let mask = eval_mask(df, expr)?;
        let mask = if negate { !&mask } else { mask };
        Ok(Node::Bits(Cow::Owned(mask.true_mask().words().into_owned())))
    };
    match expr {
        Expr::Not(inner) => compile(df, inner, !negate),
        Expr::Alias(inner, _) => compile(df, inner, negate),
        Expr::Literal(LiteralValue::Boolean(b)) => Ok(Node::Const(*b != negate)),
        Expr::Literal(LiteralValue::Null) => Ok(Node::Const(false)),
        Expr::Column(name) => {
            let column = df.column(name)?;
            match boolean_node(column, negate) {
                Some(node) => Ok(node),
                None => polars_bail!(InvalidOperation: "filter predicate must be of type `Boolean`, got `{}`", column.dtype()),
            }
        }
        Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            let Expr::Column(name) = &**inner else { return fallback(expr) };
            let want_null = matches!(expr, Expr::IsNull(_)) != negate;
            Ok(match (df.column(name)?.validity(), want_null) {
                (None, true) => Node::Const(false),
                (None, false) => Node::Const(true),
                (Some(valid), false) => Node::Bits(valid.words()),
                (Some(valid), true) => Node::Bits(Cow::Owned(valid.not().words().into_owned())),
            })
        }
        Expr::Binary { left, op: Operator::And, right } | Expr::Binary { left, op: Operator::Or, right } => {
            let is_and = matches!(expr, Expr::Binary { op: Operator::And, .. });
            let (a, b) = (compile(df, left, negate)?, compile(df, right, negate)?);
            // De Morgan: a negated AND is an OR of the negated children.
            Ok(match (is_and != negate, a, b) {
                (true, Node::Const(false), _) | (true, _, Node::Const(false)) => Node::Const(false),
                (true, Node::Const(true), other) | (true, other, Node::Const(true)) => other,
                (false, Node::Const(true), _) | (false, _, Node::Const(true)) => Node::Const(true),
                (false, Node::Const(false), other) | (false, other, Node::Const(false)) => other,
                (true, a, b) => Node::And(Box::new(a), Box::new(b)),
                (false, a, b) => Node::Or(Box::new(a), Box::new(b)),
            })
        }
        Expr::Binary { left, op: Operator::Cmp(op), right } => {
            let op = if negate { op.negate() } else { *op };
            match (&**left, &**right) {
                (Expr::Column(name), Expr::Literal(value)) | (Expr::Literal(value), Expr::Column(name)) => {
                    let op = if matches!(&**left, Expr::Literal(_)) { op.swap_operands() } else { op };
                    let column = df.column(name)?;
                    let wide = match value {
                        LiteralValue::Null => return Ok(Node::Const(false)),
                        LiteralValue::Int(i) => Some(Wide::Int(*i)),
                        LiteralValue::Float(f) => Some(Wide::Float(*f)),
                        LiteralValue::Boolean(_) | LiteralValue::String(_) => None,
                    };
                    match (&column.0, wide, value) {
                        (SeriesInner::String(ca), _, LiteralValue::String(s)) => Ok(string_leaf(ca, op, s.as_str())),
                        (_, Some(wide), _) if column.dtype().is_numeric() => Ok(for_numeric!(
                            column,
                            ca => numeric_scalar_leaf(ca, op, wide),
                            _ => unreachable!("guarded by `is_numeric`")
                        )),
                        _ => fallback(expr),
                    }
                }
                (Expr::Column(a), Expr::Column(b)) => {
                    let (ca, cb) = (df.column(a)?, df.column(b)?);
                    macro_rules! same_type {
                        ($($variant:ident),*) => {
                            match (&ca.0, &cb.0) {
                                $((SeriesInner::$variant(x), SeriesInner::$variant(y)) => Ok(numeric_column_leaf(x, op, y)),)*
                                _ => fallback(expr),
                            }
                        };
                    }
                    same_type!(UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float32, Float64)
                }
                _ => fallback(expr),
            }
        }
        _ => fallback(expr),
    }
}

/// A predicate bound to a frame, ready to be evaluated block by block.
pub(crate) struct Predicate<'a> {
    root: Node<'a>,
    depth: usize,
}

/// Per-worker state for evaluating a [`Predicate`].
pub(crate) struct Scratch {
    /// The mask of the current block.
    pub(crate) words: Vec<u64>,
    /// Row indices of the current block, when it is emitted through a list.
    idx: Vec<u32>,
    stack: Vec<Vec<u64>>,
    /// Fences this worker's non-temporal stores when it finishes.
    _fence: FenceOnDrop,
}

impl<'a> Predicate<'a> {
    pub(crate) fn compile(df: &'a DataFrame, expr: &'a Expr) -> PolarsResult<Self> {
        let root = compile(df, expr, false)?;
        let depth = root.depth();
        Ok(Predicate { root, depth })
    }

    /// `Some(answer)` if the predicate is the same for every row.
    pub(crate) fn constant(&self) -> Option<bool> {
        match self.root {
            Node::Const(b) => Some(b),
            _ => None,
        }
    }

    pub(crate) fn scratch(&self, block_rows: usize, stream: bool) -> Scratch {
        let nw = words_for(block_rows);
        Scratch {
            words: vec![0; nw],
            idx: Vec::new(),
            stack: vec![vec![0; nw]; self.depth],
            _fence: FenceOnDrop { words: Vec::new(), stream },
        }
    }

    /// Evaluate `rows` into `scratch.words`; returns the number of matches.
    #[inline]
    pub(crate) fn eval(&self, rows: Range<usize>, scratch: &mut Scratch) -> usize {
        self.root.eval(rows, &mut scratch.words, &mut scratch.stack)
    }

    /// Fraction of rows expected to match, from 64 evenly spaced 64-row stripes.
    pub(crate) fn sample_selectivity(&self, height: usize) -> f64 {
        const STRIPES: usize = 64;
        if height <= 2 * STRIPES * 64 {
            let mut scratch = self.scratch(height.max(64), false);
            return self.eval(0..height, &mut scratch) as f64 / height.max(1) as f64;
        }
        let mut scratch = self.scratch(64, false);
        let step = (height / STRIPES) / 64 * 64;
        let hits: usize = (0..STRIPES).map(|s| self.eval(s * step..s * step + 64, &mut scratch)).sum();
        hits as f64 / (STRIPES * 64) as f64
    }

    /// The mask of the whole frame as aligned words, computed on all cores.
    pub(crate) fn full_mask(&self, height: usize, tasks: usize) -> AlignedVec<u64> {
        let block = pipeline::block_rows(8);
        let mut words: AlignedVec<u64> = crate::memory::zeroed_vec(words_for(height));
        par::for_each_chunk_mut(&mut words, block / 64, tasks, |i, out| {
            let mut scratch = self.scratch(block, false);
            let start = i * block;
            self.eval(start..(start + block).min(height), &mut scratch);
            let n = out.len();
            out.copy_from_slice(&scratch.words[..n]);
        });
        words
    }
}

// ---------------------------------------------------------------------------
// filter
// ---------------------------------------------------------------------------

/// The rows of `df` satisfying `predicate`, restricted to `projection` (all
/// columns if `None`).
pub(crate) fn filter_frame(df: &DataFrame, predicate: &Expr, projection: Option<&[PlSmallStr]>) -> PolarsResult<DataFrame> {
    let projected;
    let out_df = match projection {
        Some(names) => {
            projected = df.select(names.iter().map(|n| n.as_str()))?;
            &projected
        }
        None => df,
    };
    let height = df.height();
    let compiled = Predicate::compile(df, predicate)?;
    match compiled.constant() {
        Some(true) => return Ok(out_df.clone()),
        Some(false) => return Ok(out_df.clear()),
        None => {}
    }
    if out_df.width() == 0 || height == 0 {
        let tasks = par::tasks_for(height, 8);
        let mask = compiled.full_mask(height, tasks);
        let count = crate::memory::bitmap::count_ones_range(&mask, 0, height);
        return Ok(DataFrame::from_parts(count, out_df.columns().iter().map(|c| c.slice(0, 0)).collect()));
    }

    let row_bytes = out_df.row_bytes();
    let tasks = par::tasks_for(height, row_bytes);
    let two_phase = |tasks: usize| -> DataFrame {
        let mask = compiled.full_mask(height, tasks);
        out_df.filter_selection(&Selection::from_words(Cow::Borrowed(&mask[..]), height))
    };

    // Estimate the output size, then try the single-pass pipeline.
    let block = pipeline::frame_block_rows(height, tasks);
    let p = compiled.sample_selectivity(height);
    let margin = 4.0 * 0.5 / 4096f64.sqrt();
    let capacity = ((((p + margin) * height as f64) as usize).saturating_add(block)).min(height);
    let stream = capacity.saturating_mul(row_bytes) >= cache::stream_threshold();
    let sinks: Option<Vec<Box<dyn Sink + '_>>> = out_df.columns().iter().map(|c| sink_for(c, capacity)).collect();
    let Some(sinks) = sinks else {
        // Nullable or boolean payload: materialise the mask, then compact.
        return Ok(two_phase(tasks));
    };

    let done = pipeline::scan_then_emit(
        height,
        block,
        tasks,
        capacity,
        || compiled.scratch(block, stream),
        |rows, scratch| compiled.eval(rows, scratch),
        |rows, scratch, offset, count| {
            let Scratch { words, idx, .. } = scratch;
            emit_block(&sinks, rows, words, offset, count, stream, idx);
        },
    );
    if !done.complete {
        // The sample underestimated the selectivity; redo with exact sizes.
        drop(sinks);
        return Ok(two_phase(tasks));
    }
    let columns = sinks.into_iter().map(|s| s.finish(done.total)).collect();
    Ok(DataFrame::from_parts(done.total, columns))
}

// ---------------------------------------------------------------------------
// filtered aggregation
// ---------------------------------------------------------------------------

/// Running state of one aggregate over the blocks seen so far.
#[derive(Clone, Copy)]
struct Partial {
    sum_f: f64,
    sum_i: i64,
    min: f64,
    max: f64,
    min_i: i64,
    max_i: i64,
    count: usize,
}

impl Partial {
    const EMPTY: Partial =
        Partial { sum_f: 0.0, sum_i: 0, min: f64::INFINITY, max: f64::NEG_INFINITY, min_i: i64::MAX, max_i: i64::MIN, count: 0 };

    fn merge(self, o: Partial) -> Partial {
        Partial {
            sum_f: self.sum_f + o.sum_f,
            sum_i: self.sum_i.wrapping_add(o.sum_i),
            min: if o.min < self.min { o.min } else { self.min },
            max: if o.max > self.max { o.max } else { self.max },
            min_i: self.min_i.min(o.min_i),
            max_i: self.max_i.max(o.max_i),
            count: self.count + o.count,
        }
    }
}

type BlockReducer<'a> = Box<dyn Fn(Range<usize>, &[u64], usize, &mut Vec<u8>) -> Partial + Send + Sync + 'a>;

fn reducer_for<'a, T: PolarsNumericType>(ca: &'a ChunkedArray<T>, kind: AggKind) -> BlockReducer<'a> {
    let values = ca.values().as_slice();
    let need_sum = matches!(kind, AggKind::Sum | AggKind::Mean);
    let need_min_max = matches!(kind, AggKind::Min | AggKind::Max);
    let reduce = move |selected: &[T::Native]| -> Partial {
        let mut p = Partial { count: selected.len(), ..Partial::EMPTY };
        if need_sum {
            if T::Native::IS_FLOAT {
                p.sum_f = kernels::aggregate::sum_f64(selected);
            } else {
                p.sum_i = kernels::aggregate::sum_i64(selected);
                p.sum_f = p.sum_i as f64;
            }
        }
        if need_min_max && let Some((lo, hi)) = kernels::aggregate::min_max(selected) {
            (p.min, p.max) = (lo.to_f64(), hi.to_f64());
            (p.min_i, p.max_i) = (lo.to_i64(), hi.to_i64());
        }
        p
    };
    Box::new(move |rows, mask, count, scratch| {
        let block = &values[rows];
        if count == block.len() {
            // Everything selected: reduce the source directly.
            return reduce(block);
        }
        if !need_min_max {
            // A sum does not need the selected rows next to each other: add
            // every row, with the unselected ones counting as zero.
            let mut p = Partial { count, ..Partial::EMPTY };
            if T::Native::IS_FLOAT {
                p.sum_f = kernels::aggregate::sum_f64_selected(block, mask);
            } else {
                p.sum_i = kernels::aggregate::sum_i64_selected(block, mask);
                p.sum_f = p.sum_i as f64;
            }
            return p;
        }
        // Compact into a buffer that stays in cache, then reduce that.
        let need = (count + APRON / size_of::<T::Native>() + 1) * size_of::<T::Native>() + 64;
        if scratch.len() < need {
            scratch.resize(need, 0);
        }
        // SAFETY: `scratch` holds at least `count + apron` elements past the
        // aligned start; `align_to_mut` yields a correctly aligned view of it.
        let (_, aligned, _) = unsafe { scratch.align_to_mut::<T::Native>() };
        let cap = aligned.len();
        // SAFETY: `aligned` is valid for `cap >= count` writes and does not
        // overlap `block`.
        let written = unsafe { kernels::compact(block.as_ptr(), mask, block.len(), aligned.as_mut_ptr(), cap, false) };
        reduce(&aligned[..written])
    })
}

/// One aggregate the fused path can compute.
struct FusedAgg<'a> {
    name: PlSmallStr,
    kind: AggKind,
    dtype: DataType,
    reducer: Option<BlockReducer<'a>>,
}

fn plan_agg<'a>(df: &'a DataFrame, expr: &Expr) -> Option<FusedAgg<'a>> {
    let (inner, alias) = match expr {
        Expr::Alias(inner, name) => (&**inner, Some(name.clone())),
        other => (other, None),
    };
    match inner {
        Expr::Len => Some(FusedAgg { name: alias.unwrap_or_else(|| "len".into()), kind: AggKind::Len, dtype: DataType::UInt32, reducer: None }),
        Expr::Agg { input, kind } => {
            let Expr::Column(col_name) = &**input else { return None };
            let column = df.column(col_name).ok()?;
            let name = alias.unwrap_or_else(|| col_name.clone());
            let dtype = *column.dtype();
            if matches!(kind, AggKind::Len) {
                return Some(FusedAgg { name, kind: *kind, dtype, reducer: None });
            }
            if column.validity().is_some() || !dtype.is_numeric() {
                return None;
            }
            match kind {
                AggKind::Count => Some(FusedAgg { name, kind: *kind, dtype, reducer: None }),
                // `u64` does not fit the integer side of `Partial`.
                AggKind::Sum | AggKind::Mean | AggKind::Min | AggKind::Max if dtype != DataType::UInt64 => {
                    let reducer = for_numeric!(column, ca => reducer_for(ca, *kind), _ => unreachable!("numeric checked above"));
                    Some(FusedAgg { name, kind: *kind, dtype, reducer: Some(reducer) })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// `select(aggs)` over `filter(predicate)` without materialising the filtered
/// rows. Returns `Ok(None)` when some aggregate is outside the fused subset
/// (the caller then filters and aggregates the ordinary way).
pub(crate) fn aggregate_filtered(df: &DataFrame, predicate: &Expr, aggs: &[Expr]) -> PolarsResult<Option<DataFrame>> {
    let Some(plans) = aggs.iter().map(|e| plan_agg(df, e)).collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    let height = df.height();
    let compiled = Predicate::compile(df, predicate)?;
    let block = pipeline::block_rows(8);
    let n_blocks = height.div_ceil(block);
    let tasks = par::tasks_for(height, 8 * plans.len().max(1));

    // One vector of partials per block, in block order, so that the fold below
    // is the same whatever the number of workers.
    let run_block = |b: usize, scratch: &mut Scratch, buf: &mut Vec<u8>| -> Vec<Partial> {
        let rows = b * block..((b + 1) * block).min(height);
        let count = compiled.eval(rows.clone(), scratch);
        plans
            .iter()
            .map(|plan| match (&plan.reducer, count) {
                (_, 0) => Partial::EMPTY,
                (None, _) => Partial { count, ..Partial::EMPTY },
                (Some(reduce), _) => reduce(rows.clone(), &scratch.words, count, buf),
            })
            .collect()
    };
    // The threads claim runs of consecutive blocks; each run brings its own
    // scratch space and returns its blocks' partials in order.
    let run_len = n_blocks.div_ceil((tasks * 4).max(1)).max(1);
    let per_run: Vec<Vec<Vec<Partial>>> = par::map(n_blocks.div_ceil(run_len), tasks, |r| {
        let (mut scratch, mut buf) = (compiled.scratch(block, false), Vec::new());
        (r * run_len..((r + 1) * run_len).min(n_blocks)).map(|b| run_block(b, &mut scratch, &mut buf)).collect()
    });
    let per_block: Vec<Vec<Partial>> = per_run.into_iter().flatten().collect();
    let totals: Vec<Partial> = (0..plans.len())
        .map(|i| per_block.iter().fold(Partial::EMPTY, |acc, parts| acc.merge(parts[i])))
        .collect();

    let mut columns = Vec::with_capacity(plans.len());
    for (plan, t) in plans.iter().zip(&totals) {
        let name = plan.name.clone();
        let is_float = plan.dtype.is_float();
        let number = |f: f64, i: i64| if is_float { AnyValue::Float64(f) } else { AnyValue::Int64(i) };
        let (value, dtype) = match plan.kind {
            AggKind::Len | AggKind::Count => (AnyValue::UInt32(t.count as u32), DataType::UInt32),
            AggKind::Sum => (number(t.sum_f, t.sum_i), if is_float || plan.dtype == DataType::UInt64 { plan.dtype } else { DataType::Int64 }),
            AggKind::Mean if t.count == 0 => (AnyValue::Null, DataType::Float64),
            AggKind::Mean => (AnyValue::Float64(t.sum_f / t.count as f64), DataType::Float64),
            // Nothing selected: null. Only NaNs selected: the lanes still hold
            // their sentinels (`min > max`), and the answer is NaN.
            AggKind::Min | AggKind::Max if t.count == 0 => (AnyValue::Null, plan.dtype),
            AggKind::Min | AggKind::Max if is_float && t.min > t.max => (AnyValue::Float64(f64::NAN), plan.dtype),
            AggKind::Min => (number(t.min, t.min_i), plan.dtype),
            AggKind::Max => (number(t.max, t.max_i), plan.dtype),
            _ => unreachable!("`plan_agg` only admits the kinds above"),
        };
        columns.push(if value.is_null() {
            Series::full_null(name, 1, &dtype)
        } else {
            Series::from_any_values(name, &[value])?.cast(&dtype)?
        });
    }
    Ok(Some(DataFrame::new(1, columns)?))
}

#[cfg(test)]
mod tests {
    use super::super::dsl::{col, len, lit};
    use super::*;
    use crate::df;
    use crate::testutil::Rng;

    fn big(n: usize) -> DataFrame {
        let mut rng = Rng::new(5);
        let a: Vec<i64> = (0..n).map(|_| rng.below(1000) as i64).collect();
        let b: Vec<f64> = (0..n).map(|_| rng.below(10_000) as f64 / 10_000.0).collect();
        let c: Vec<i32> = (0..n).map(|_| rng.below(100) as i32).collect();
        let s: Vec<String> = a.iter().map(|v| format!("key-{}", v % 7)).collect();
        let o: Vec<Option<i64>> = a.iter().map(|&v| (v % 5 != 0).then_some(v)).collect();
        df!("a" => &a, "b" => &b, "c" => &c, "s" => &s, "o" => &o).unwrap()
    }

    fn reference(df: &DataFrame, e: &Expr) -> DataFrame {
        df.filter(&eval_mask(df, e).unwrap()).unwrap()
    }

    #[test]
    fn fused_filter_agrees_with_the_mask_path() {
        for n in [0usize, 1, 100, 5_000, 300_000] {
            let df = big(n);
            let predicates = [
                col("a").lt(500),
                col("a").lt(500).and(col("b").lt(0.5)),
                col("a").lt(100).or(col("c").gt_eq(90)),
                col("a").lt(500).and(col("b").lt(0.5)).not(),
                col("s").eq("key-3"),
                col("s").neq("key-3").and(col("a").gt(col("o"))),
                col("o").is_null(),
                col("o").is_not_null().and(col("o").lt(300)),
                lit(5).lt(col("c")),
                col("a").gt(2000),
                col("a").gt_eq(0),
                (col("a") + col("c")).gt(600),
                col("a").lt(2.5),
                lit(true),
                lit(false).or(col("c").eq(7)),
            ];
            for e in &predicates {
                let want = reference(&df, e);
                // All columns: the nullable one forces the two-phase path.
                assert_eq!(filter_frame(&df, e, None).unwrap(), want, "n {n} predicate {e}");
                // Null-free projection: the single-pass pipeline.
                let cols: Vec<PlSmallStr> = ["s", "b", "a"].iter().map(|&c| c.into()).collect();
                let got = filter_frame(&df, e, Some(&cols)).unwrap();
                assert_eq!(got, want.select(["s", "b", "a"]).unwrap(), "n {n} predicate {e} (projected)");
            }
        }
        let df = big(100);
        assert!(filter_frame(&df, &col("a"), None).is_err());
        assert!(filter_frame(&df, &col("nope").lt(1), None).is_err());
    }

    #[test]
    fn fused_aggregation_agrees_with_filter_then_aggregate() {
        for n in [0usize, 100, 70_000, 300_000] {
            let df = big(n);
            let predicate = col("a").lt(500).and(col("c").gt(10));
            let aggs = [
                col("b").sum(),
                col("b").mean().alias("mean_b"),
                col("a").sum().alias("sum_a"),
                col("c").min().alias("min_c"),
                col("b").max().alias("max_b"),
                col("a").count().alias("n"),
                len(),
            ];
            let got = aggregate_filtered(&df, &predicate, &aggs).unwrap().expect("fused subset");
            let filtered = reference(&df, &predicate);
            let want_sum: f64 = filtered["b"].f64().unwrap().cont_slice().unwrap().iter().sum();
            let got_sum = got["b"].f64().unwrap().get(0).unwrap();
            assert!((got_sum - want_sum).abs() <= 1e-9 * want_sum.abs().max(1.0), "n {n}: {got_sum} vs {want_sum}");
            assert_eq!(got["sum_a"].i64().unwrap().get(0), Some(filtered["a"].i64().unwrap().sum().unwrap()));
            assert_eq!(got["min_c"].i32().unwrap().get(0), filtered["c"].i32().unwrap().min());
            assert_eq!(got["max_b"].f64().unwrap().get(0), filtered["b"].f64().unwrap().max());
            assert_eq!(got["n"].u32().unwrap().get(0), Some(filtered.height() as u32));
            assert_eq!(got["len"].u32().unwrap().get(0), Some(filtered.height() as u32));
            match filtered["b"].f64().unwrap().mean() {
                Some(m) => assert!((got["mean_b"].f64().unwrap().get(0).unwrap() - m).abs() < 1e-9),
                None => assert_eq!(got["mean_b"].null_count(), 1),
            }
        }
        let df = big(1000);
        // Outside the fused subset: the caller is told to take the general path.
        assert!(aggregate_filtered(&df, &col("a").lt(5), &[col("o").sum()]).unwrap().is_none());
        assert!(aggregate_filtered(&df, &col("a").lt(5), &[col("b").median()]).unwrap().is_none());
        assert!(aggregate_filtered(&df, &col("a").lt(5), &[(col("b") * 2).sum()]).unwrap().is_none());
    }

    #[test]
    fn deterministic_across_task_counts() {
        let df = big(300_000);
        let predicate = col("a").lt(700);
        let first = aggregate_filtered(&df, &predicate, &[col("b").sum()]).unwrap().unwrap();
        for _ in 0..3 {
            let again = aggregate_filtered(&df, &predicate, &[col("b").sum()]).unwrap().unwrap();
            assert_eq!(
                again["b"].f64().unwrap().get(0).unwrap().to_bits(),
                first["b"].f64().unwrap().get(0).unwrap().to_bits()
            );
        }
    }
}
