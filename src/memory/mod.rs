//! Layer 0 — memory.
//!
//! Everything above this module sees column data through three types:
//!
//! | type | role |
//! |------|------|
//! | [`ColumnAlloc`] | the [`std::alloc::Allocator`] that fixes the physical layout (cache-line alignment, SIMD apron, huge pages, recycling) |
//! | [`AlignedVec<T>`] | `Vec<T, ColumnAlloc>` — a column while it is being built or mutated |
//! | [`Buffer<T>`] / [`Bitmap`] | the frozen, reference-counted, O(1)-sliceable form that is shared between frames, slices and threads |
//!
//! The module has no dependency on the rest of the crate.

pub mod alloc;
pub mod bitmap;
pub mod buffer;

pub use alloc::{APRON, AlignedVec, AllocStats, CACHE_LINE, ColumnAlloc, HUGE_BLOCK};
pub use bitmap::{Bitmap, BitmapIter, MutableBitmap, TrueIdxIter, tail_mask, words_for};
pub use buffer::{Buffer, Pod, Zeroable, aligned_with_capacity, zeroed_vec};
