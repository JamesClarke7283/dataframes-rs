//! Block-wise filtering of several columns at once.
//!
//! A frame is filtered by cutting its rows into blocks and letting every core
//! take blocks through the ordered-commit pipeline
//! ([`kernels::pipeline`](crate::kernels::pipeline)). Each output column is a
//! [`Sink`]: a pre-allocated buffer that blocks write disjoint ranges of. A
//! block is written to all sinks either through its mask or through one list
//! of row indices shared by the columns, whichever is cheaper at its density
//! ([`emit_block`]).
//!
//! The eager `DataFrame::filter` uses this with a mask that already exists
//! ([`filter_by_words`]); the lazy executor uses the same sinks with a
//! predicate it evaluates block by block.

use std::ops::Range;

use super::DataFrame;
use crate::chunked_array::{ChunkedArray, StringChunked, View};
use crate::datatypes::PolarsNumericType;
use crate::kernels::pipeline::{self, FenceOnDrop};
use crate::kernels::{self, cache, par};
use crate::memory::{AlignedVec, Buffer, ColumnAlloc};
use crate::series::{Column, IntoSeries, Series, SeriesInner, for_numeric};

/// One output column of a fused filter.
pub(crate) trait Sink: Sync {
    /// Compact the block `rows` (mask in `mask`) to output rows `offset..offset + count`.
    fn emit(&self, rows: Range<usize>, mask: &[u64], offset: usize, count: usize, stream: bool);
    /// Copy rows `start + idx[j]` to output rows `offset + j`.
    fn emit_indices(&self, start: usize, idx: &[u32], offset: usize, stream: bool);
    /// Fix the output length and wrap it up.
    fn finish(self: Box<Self>, total: usize) -> Series;
}

struct PodSink<'a, T: crate::memory::Pod, F: Fn(AlignedVec<T>) -> Series + Sync> {
    src: &'a [T],
    dst: par::SendPtr<T>,
    out: AlignedVec<T>,
    wrap: F,
}

impl<T: crate::memory::Pod, F: Fn(AlignedVec<T>) -> Series + Sync> Sink for PodSink<'_, T, F> {
    #[inline]
    fn emit(&self, rows: Range<usize>, mask: &[u64], offset: usize, count: usize, stream: bool) {
        let src = &self.src[rows];
        // SAFETY: the pipeline hands every block a disjoint `offset..offset +
        // count` below the capacity of `out`; `compact` writes at most `count`
        // elements there, and `mask` covers exactly `src`.
        unsafe {
            let written = kernels::compact(src.as_ptr(), mask, src.len(), self.dst.0.add(offset), count, stream);
            debug_assert_eq!(written, count);
        }
    }

    #[inline]
    fn emit_indices(&self, start: usize, idx: &[u32], offset: usize, stream: bool) {
        // SAFETY: the indices are positions of set bits within the block that
        // starts at row `start`, so every source row is in bounds; the output
        // range is this block's own, as in `emit`.
        unsafe { kernels::gather(self.src.as_ptr().add(start), idx, self.dst.0.add(offset), stream) }
    }

    fn finish(mut self: Box<Self>, total: usize) -> Series {
        // SAFETY: the blocks tile `0..total` and each initialised its share.
        unsafe { self.out.set_len(total) };
        if self.out.capacity() > 2 * total + pipeline::MAX_BLOCK_ROWS {
            self.out.shrink_to_fit();
        }
        (self.wrap)(std::mem::replace(&mut self.out, Vec::new_in(ColumnAlloc)))
    }
}

pub(crate) fn sink_for<'a>(column: &'a Column, capacity: usize) -> Option<Box<dyn Sink + 'a>> {
    fn numeric<'a, T: PolarsNumericType>(ca: &'a ChunkedArray<T>, capacity: usize) -> Box<dyn Sink + 'a> {
        let mut out: AlignedVec<T::Native> = Vec::with_capacity_in(capacity.max(1), ColumnAlloc);
        let name = ca.name().clone();
        Box::new(PodSink {
            src: ca.values().as_slice(),
            dst: par::SendPtr(out.as_mut_ptr()),
            out,
            wrap: move |v| ChunkedArray::<T>::from_aligned(name.clone(), v).into_series(),
        })
    }
    if column.validity().is_some() {
        return None;
    }
    for_numeric!(column, ca => Some(numeric(ca, capacity)), other => match other {
        SeriesInner::String(ca) => {
            let mut out: AlignedVec<View> = Vec::with_capacity_in(capacity.max(1), ColumnAlloc);
            let (name, buffers) = (ca.name().clone(), ca.buffers.clone());
            Some(Box::new(PodSink {
                src: ca.views.as_slice(),
                dst: par::SendPtr(out.as_mut_ptr()),
                out,
                wrap: move |views: AlignedVec<View>| {
                    StringChunked { name: name.clone(), views: Buffer::from_aligned(views), buffers: buffers.clone(), validity: None }
                        .into_series()
                },
            }))
        }
        // Bit-packed values cannot be written by independent blocks.
        _ => None,
    })
}

/// Write one block of every output column.
///
/// A sparse block is emitted through a list of row indices when there are
/// several columns: the list is built once and each column then touches only
/// the rows it keeps, which is cheaper than decoding the mask again per
/// column. "Sparse" depends on what decoding costs: up to two rows in five
/// with the AVX2 permutation tables, one in eight with AVX-512 compression.
#[inline]
pub(crate) fn emit_block(
    sinks: &[Box<dyn Sink + '_>],
    rows: Range<usize>,
    mask: &[u64],
    offset: usize,
    count: usize,
    stream: bool,
    idx: &mut Vec<u32>,
) {
    let sparse = if kernels::level() >= kernels::Level::Avx512 { 8 * count <= rows.len() } else { 5 * count <= 2 * rows.len() };
    if sinks.len() >= 2 && sparse {
        kernels::mask_indices_into(mask, rows.len(), 0, idx);
        for sink in sinks {
            sink.emit_indices(rows.start, idx, offset, stream);
        }
    } else {
        for sink in sinks {
            sink.emit(rows.clone(), mask, offset, count, stream);
        }
    }
}

/// The rows of `df` whose bit is set in `words` (`count` of them), compacted
/// block by block over all columns at once. `None` if some column cannot be
/// written by independent blocks (nullable or bit-packed columns), or if
/// `words` turns out to hold more set bits than `count`.
pub(crate) fn filter_by_words(df: &DataFrame, words: &[u64], count: usize) -> Option<DataFrame> {
    let height = df.height();
    if df.width() == 0 || height == 0 {
        return None;
    }
    let row_bytes = df.row_bytes();
    let stream = count.saturating_mul(row_bytes) >= cache::stream_threshold();
    let sinks: Vec<Box<dyn Sink + '_>> = df.columns().iter().map(|c| sink_for(c, count)).collect::<Option<_>>()?;
    let tasks = par::tasks_for(height, row_bytes);
    let block = pipeline::frame_block_rows(height, tasks);
    let done = pipeline::scan_then_emit(
        height,
        block,
        tasks,
        count,
        || (Vec::new(), FenceOnDrop { words: Vec::new(), stream }),
        |rows, _| crate::memory::bitmap::count_ones_range(words, rows.start, rows.len()),
        |rows, scratch, offset, n| {
            let mask = &words[rows.start / 64..];
            emit_block(&sinks, rows, mask, offset, n, stream, &mut scratch.0);
        },
    );
    if !done.complete {
        // More set bits than `count` announced: the caller's count was wrong.
        // Only the blocks that fit were written; let the column-wise path,
        // which counts for itself, produce the answer.
        return None;
    }
    let columns = sinks.into_iter().map(|s| s.finish(done.total)).collect();
    Some(DataFrame::from_parts(done.total, columns))
}
