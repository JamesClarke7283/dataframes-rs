//! Block-wise filtering of several columns at once.
//!
//! A frame is filtered by cutting its rows into blocks and letting every core
//! take blocks through the ordered-commit pipeline
//! ([`kernels::pipeline`](crate::kernels::pipeline)). Each output column is a
//! [`Sink`]: a pre-allocated buffer that blocks write disjoint ranges of. A
//! block is written to all sinks either through its mask or through one list
//! of row indices shared by the columns, whichever is cheaper at its density
//! ([`emit_block`]) — and, for a frame of many blocks, whichever turns out to
//! be faster on this machine: the first blocks try both ([`EmitChoice`]).
//!
//! The eager `DataFrame::filter` uses this with a mask that already exists
//! ([`filter_by_words`]); the lazy executor uses the same sinks with a
//! predicate it evaluates block by block.

use std::ops::Range;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

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

/// A frame filter of at least this many blocks compares the two ways of
/// emitting a sparse block on its first ones.
const COMPARE_BLOCKS: usize = 64;

/// How the sparse blocks of one frame filter are emitted.
///
/// Through a shared list of row indices by default. Which way is faster for
/// data that is not in the cache depends on the machine, though — measured on
/// the benchmark runners with ten million rows and one row in ten kept, the
/// index lists were a third slower than plain compaction on a Zen 5 and a
/// third faster on an Emerald Rapids — so a filter with enough blocks to
/// afford it times two blocks each way and uses the faster one for the rest.
/// The result is the same either way.
pub(crate) struct EmitChoice {
    /// [`Self::LISTS`], [`Self::COMPARING`] or [`Self::DENSE`].
    state: AtomicU8,
    /// Sparse blocks handed out for timing.
    probes: AtomicU32,
    /// Timed blocks that have reported.
    reports: AtomicU32,
    /// The better of the two timed blocks each way, in nanoseconds per 65 536
    /// rows: `[index lists, dense]`. The better one, not the sum: a block
    /// that is the first to touch fresh output memory pays for the page
    /// faults, whichever way it is written.
    best: [AtomicU64; 2],
}

impl EmitChoice {
    const LISTS: u8 = 0;
    const COMPARING: u8 = 1;
    const DENSE: u8 = 2;
    /// Timed blocks: lists, dense, dense, lists — so that neither way has
    /// only the first, coldest blocks.
    const PROBES: u32 = 4;

    /// For a filter of `blocks` blocks into `sinks` columns.
    pub(crate) fn new(blocks: usize, sinks: usize) -> Self {
        let compare = blocks >= COMPARE_BLOCKS && sinks >= 2 && index_emit_permille() > 0;
        Self {
            state: AtomicU8::new(if compare { Self::COMPARING } else { Self::LISTS }),
            probes: AtomicU32::new(0),
            reports: AtomicU32::new(0),
            best: [AtomicU64::new(u64::MAX), AtomicU64::new(u64::MAX)],
        }
    }

    /// `Some(dense)` if the next sparse block is to be timed that way.
    fn probe(&self) -> Option<bool> {
        let probe = self.probes.fetch_add(1, Ordering::Relaxed);
        (probe < Self::PROBES).then_some(probe == 1 || probe == 2)
    }

    fn report(&self, dense: bool, nanos: u64, rows: usize) {
        let per_block = (u128::from(nanos) << 16) / rows.max(1) as u128;
        self.best[usize::from(dense)].fetch_min(per_block.min(u128::from(u64::MAX - 1)) as u64, Ordering::Relaxed);
        if self.reports.fetch_add(1, Ordering::AcqRel) + 1 == Self::PROBES {
            let [lists, dense] = [0, 1].map(|way| u128::from(self.best[way].load(Ordering::Relaxed)));
            // Dense only for a clear gain: an eighth less time per row.
            let faster = dense * 8 < lists * 7;
            self.state.store(if faster { Self::DENSE } else { Self::LISTS }, Ordering::Relaxed);
        }
    }
}

/// Write one block of every output column.
///
/// A sparse block is emitted through a list of row indices when there are
/// several columns: the list is built once and each column then touches only
/// the rows it keeps, which is cheaper than decoding the mask again per
/// column. "Sparse" depends on what decoding costs: up to two rows in five
/// with the AVX2 permutation tables, one in eight with AVX-512 compression.
/// `choice` may overrule the lists for the frame at hand. The block's rows
/// become the output rows `out`.
#[inline]
pub(crate) fn emit_block(
    sinks: &[Box<dyn Sink + '_>],
    rows: Range<usize>,
    mask: &[u64],
    out: Range<usize>,
    stream: bool,
    idx: &mut Vec<u32>,
    choice: &EmitChoice,
) {
    let (offset, count) = (out.start, out.len());
    let through_mask = || {
        for sink in sinks {
            sink.emit(rows.clone(), mask, offset, count, stream);
        }
    };
    let mut through_lists = || {
        kernels::mask_indices_into(mask, rows.len(), 0, idx);
        for sink in sinks {
            sink.emit_indices(rows.start, idx, offset, stream);
        }
    };
    if sinks.len() < 2 || 1000 * count > index_emit_permille() * rows.len() {
        return through_mask();
    }
    match choice.state.load(Ordering::Relaxed) {
        EmitChoice::DENSE => through_mask(),
        EmitChoice::COMPARING => match choice.probe() {
            Some(dense) => {
                let start = Instant::now();
                if dense { through_mask() } else { through_lists() }
                choice.report(dense, start.elapsed().as_nanos() as u64, rows.len());
            }
            None => through_lists(),
        },
        _ => through_lists(),
    }
}

/// Densest block, in selected rows per thousand, that is emitted through a
/// shared index list: 400 with the AVX2 permutation tables, 125 with AVX-512
/// compression. The `DATAFRAMES_INDEX_EMIT_PERMILLE` environment variable
/// overrides it (0 switches the index lists off).
fn index_emit_permille() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("DATAFRAMES_INDEX_EMIT_PERMILLE").ok().and_then(|v| v.parse().ok()).unwrap_or(
            if kernels::level() >= kernels::Level::Avx512 { 125 } else { 400 },
        )
    })
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
    let choice = EmitChoice::new(height.div_ceil(block), sinks.len());
    let done = pipeline::scan_then_emit(
        height,
        block,
        tasks,
        count,
        || (Vec::new(), FenceOnDrop { words: Vec::new(), stream }),
        |rows, _| crate::memory::bitmap::count_ones_range(words, rows.start, rows.len()),
        |rows, scratch, offset, n| {
            let mask = &words[rows.start / 64..];
            emit_block(&sinks, rows, mask, offset..offset + n, stream, &mut scratch.0, &choice);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_filter_of_many_blocks_times_both_ways_and_keeps_the_faster() {
        // Too few blocks, or a single column: lists, no questions asked.
        for choice in [EmitChoice::new(COMPARE_BLOCKS - 1, 4), EmitChoice::new(1000, 1)] {
            assert_eq!(choice.state.load(Ordering::Relaxed), EmitChoice::LISTS);
        }
        if index_emit_permille() == 0 {
            return;
        }
        // (nanoseconds of the two list blocks, of the two dense blocks) -> dense?
        // The better block of each way counts: the first block of all may
        // have paid for the page faults of fresh output memory.
        let cases =
            [([100, 100], [60, 60], true), ([100, 100], [95, 95], false), ([60, 70], [100, 90], false), ([900, 100], [95, 95], false)];
        for (lists, dense, want) in cases {
            let choice = EmitChoice::new(COMPARE_BLOCKS, 2);
            assert_eq!(choice.state.load(Ordering::Relaxed), EmitChoice::COMPARING);
            let order: Vec<Option<bool>> = (0..6).map(|_| choice.probe()).collect();
            assert_eq!(order, [Some(false), Some(true), Some(true), Some(false), None, None]);
            choice.report(false, lists[0], 1000);
            choice.report(true, dense[0], 1000);
            choice.report(true, dense[1], 1000);
            assert_eq!(choice.state.load(Ordering::Relaxed), EmitChoice::COMPARING, "three of four reports");
            choice.report(false, lists[1], 1000);
            let state = choice.state.load(Ordering::Relaxed);
            assert_eq!(state == EmitChoice::DENSE, want, "lists {lists:?} dense {dense:?}");
            assert_ne!(state, EmitChoice::COMPARING);
        }
    }
}
