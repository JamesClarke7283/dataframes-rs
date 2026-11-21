//! # dataframes
//!
//! Zero-copy columnar dataframes with a Polars- and pandas-style API.
//!
//! ```
//! use dataframes::prelude::*;
//!
//! let df = df!(
//!     "city" => ["Oslo", "Lima", "Pune", "Oslo"],
//!     "temp" => [4.5, 19.0, 31.5, 6.0],
//!     "year" => [2023, 2023, 2024, 2024],
//! )?;
//!
//! // Polars style: expressions and lazy queries.
//! let warm = df
//!     .clone()
//!     .lazy()
//!     .filter(col("temp").gt(lit(5.0)))
//!     .group_by([col("city")])
//!     .agg([col("temp").mean().alias("mean_temp"), len()])
//!     .collect()?;
//! assert_eq!(warm.shape(), (3, 3));
//!
//! // pandas style: the same data through `query`, `iloc`, `groupby`.
//! let recent = df.query("year == 2024 and temp > 5")?;
//! assert_eq!(recent.iloc(-1)?.column("city")?.str()?.get(0), Some("Oslo"));
//! assert_eq!(df.groupby(["year"])?.sum()?.column("temp")?.f64()?.get(0), Some(23.5));
//! # Ok::<(), PolarsError>(())
//! ```
//!
//! # Architecture
//!
//! The crate is a stack of layers; each one depends only on the layers below
//! it.
//!
//! | layer | module | responsibility |
//! |------:|--------|----------------|
//! | 7 | [`sync`] | concurrent mutation: snapshot-isolated [`sync::SharedFrame`], partitioned parallel updates |
//! | 6 | [`lazy`] | expressions, lazy plans, fused block-wise execution, pandas-style query strings |
//! | 5 | [`frame`] | [`frame::DataFrame`]: O(1) slicing and cloning, filter, sort, group-by, join |
//! | 4 | [`series`] | [`series::Series`]: a named column of any type, dispatched without a vtable |
//! | 3 | [`chunked_array`] | typed columns: numeric, boolean (bit-packed), string (16-byte views) |
//! | 2 | [`datatypes`] | logical types, schema, dynamically typed values |
//! | 1 | [`kernels`] | SIMD compare / compact / reduce kernels and the ordered-commit pipeline |
//! | 0 | [`memory`] | [`memory::ColumnAlloc`], [`memory::Buffer`], [`memory::Bitmap`] |
//!
//! Four ideas carry the design:
//!
//! * **The allocator owns the layout.** [`memory::ColumnAlloc`] implements
//!   [`std::alloc::Allocator`] (stable since Rust 1.100): 64-byte alignment, a
//!   writable 64-byte apron after every block, huge pages and recycling for
//!   large blocks. A column under construction is just a
//!   `Vec<T, ColumnAlloc>`; freezing it into a shared [`memory::Buffer`] is a
//!   pointer hand-over.
//! * **Slices are smart pointers.** A [`memory::Buffer`] is a pointer, a
//!   length and a reference-counted owner. Slicing bumps the count and moves
//!   the pointer; reading through a slice is reading a `&[T]`. A frame slices
//!   in O(1) regardless of its width.
//! * **Filters are bitmasks and vector kernels.** Predicates become 64-bit
//!   mask words; compaction is `vpcompress` (AVX-512) or a table-driven
//!   permute (AVX2), with optional non-temporal stores; the two are fused
//!   block by block so data is read from memory once, by all cores, into one
//!   contiguous result.
//! * **Mutation is proved, not locked.** Shared data is immutable; mutation
//!   needs unique ownership (checked by the borrow checker and one reference
//!   count) or goes through a snapshot-isolated [`sync::SharedFrame`]. Reading
//!   column data never takes a lock.
//!
//! # Safety
//!
//! `unsafe` is confined to [`memory`] (allocation and the buffer smart
//! pointer), [`kernels`] (vector intrinsics and raw output cursors) and a
//! handful of documented call sites that hand kernels their output slots.
//! Every block states the invariant it relies on. The vector kernels are
//! tested against portable reference implementations for every element type,
//! operator, length class and mask density.

#![warn(missing_docs)]
// The kernels' loop shapes are the ones LLVM vectorises best; keep them.
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod chunked_array;
pub mod datatypes;
pub mod error;
pub mod frame;
pub mod kernels;
pub mod lazy;
pub mod memory;
pub mod series;
pub mod sync;

/// Everything most programs need: `use dataframes::prelude::*;`
///
/// The names mirror `polars::prelude`, so code written against Polars' Rust
/// API ports by changing this one import.
pub mod prelude {
    pub use crate::chunked_array::*;
    pub use crate::datatypes::*;
    pub use crate::df;
    pub use crate::error::{PolarsError, PolarsResult};
    pub use crate::frame::{
        AggKind, DataFrame, FilterPredicate, GroupBy, JoinArgs, JoinType, Row, RowSelector, SortMultipleOptions,
        UniqueKeepStrategy, concat,
    };
    pub use crate::kernels::CmpOp;
    pub use crate::lazy::*;
    pub use crate::series::{ArithOp, Column, IntoColumn, IntoSeries, NamedFrom, Series};
    pub use crate::sync::{FrameReader, SharedFrame, Snapshot};
}

#[cfg(test)]
pub(crate) mod testutil;

/// Compiles and runs the examples of the README as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
