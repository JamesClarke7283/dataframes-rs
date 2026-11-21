# dataframes — architecture and specification

This document describes how the `dataframes` crate is put together and why:
the module structure, the columnar memory layout, the SIMD filter path, the
smart pointers behind zero-copy slicing, and the specification for concurrent
mutation. Section numbers are stable; the code refers to them.

1. [Goals and the governing constraint](#1-goals-and-the-governing-constraint)
2. [Module map](#2-module-map)
3. [Columnar memory layout](#3-columnar-memory-layout)
4. [SIMD-accelerated filtering](#4-simd-accelerated-filtering)
5. [Concurrency specification](#5-concurrency-specification)
6. [Ownership rules](#6-ownership-rules)
7. [The trade-off ledger](#7-the-trade-off-ledger)
8. [Safety and verification](#8-safety-and-verification)
9. [Benchmark methodology](#9-benchmark-methodology)

---

## 1. Goals and the governing constraint

**Goals**

- Retrieval — slicing, point access, filtering, scanning — at the speed the
  hardware allows: no indirection on the read path that the data model does
  not strictly require.
- Zero-copy by default: structural operations (slice, select, clone, share
  across threads) never touch column data.
- Mutation and concurrency that are safe by construction, with guarantees
  that are written down and tested rather than implied.
- The API of Polars and pandas, so existing knowledge and code carry over.
- Stable Rust only (1.100 or newer), two small dependencies (`rayon` for
  sorting, `libc` for `mmap`).

**Non-goals (for now)**: out-of-core execution, file formats, temporal and
nested types, a cost-based query optimiser.

**The governing constraint.** The implementation must balance extreme
retrieval speeds with the complexity of safe concurrency and ownership
management. The two pull in opposite directions: the fastest read is a bare
load from a raw pointer, and the safest shared structure checks, counts or
locks on every access. The design resolves the tension with one rule, applied
at every layer:

> **Pay for safety per operation, never per element.**
> An operation may take a reference count, read a version, check a bound or
> prove unique ownership *once*; after that it runs a plain loop over a native
> slice.

Everything in sections 3–6 is an application of that rule. Section 7 lists
what each application costs.

---

## 2. Module map

Eight layers. A layer builds on the layers below it, which is what lets each be
read, tested and replaced on its own. (One reference points upwards: the
marker types in `datatypes` name the column and series variants they belong
to, as they do in Polars.)

```text
 7  sync           SharedFrame, Snapshot, FrameReader; partitioned mutation
 6  lazy           Expr, LazyFrame, plan optimiser, block-wise executor, query-string parser
 5  frame          DataFrame, filter, sort, group-by, join, display, pandas façade
 4  series         Series (dynamically typed column), arithmetic, comparison, pandas façade
 3  chunked_array  ChunkedArray<T>, BooleanChunked, StringChunked, Selection
 2  datatypes      DataType, Field, Schema, AnyValue, native-type traits, PlSmallStr
 1  kernels        compare / compact / reduce / arithmetic kernels, pipeline, work splitting
 0  memory         ColumnAlloc, Buffer<T>, Bitmap, MutableBitmap
```

| module | files | what it owns | what it must not know |
|--------|-------|--------------|-----------------------|
| `memory` | `alloc.rs`, `buffer.rs`, `bitmap.rs` | the layout contract, lifetime of bytes, sharing, copy-on-write | columns, types, names |
| `kernels` | `x86.rs`, `scalar.rs`, `tables.rs`, `aggregate.rs`, `arith.rs`, `pipeline.rs`, `par.rs`, `team.rs`, `cache.rs` | every hot loop; CPU-feature dispatch; cache and thread policy; the fork-join team | nulls as a concept, names, frames |
| `datatypes` | `mod.rs`, `any_value.rs`, `small_str.rs` | the logical vocabulary | storage |
| `chunked_array` | `mod.rs`, `ops.rs`, `boolean.rs`, `string.rs` | typed columns: values + validity, and the typed operations on them | other columns |
| `series` | `mod.rs`, `arithmetic.rs`, `comparison.rs`, `pandas.rs` | run-time typing and dispatch | frames |
| `frame` | `mod.rs`, `filter.rs`, `sort.rs`, `group_by.rs`, `join.rs`, `display.rs`, `pandas.rs` | equal-length column sets, block-wise multi-column filtering, relational operations | query plans |
| `lazy` | `dsl.rs`, `eval.rs`, `exec.rs`, `frame.rs`, `query.rs` | expressions, plans, fused execution | — |
| `sync` | `mod.rs` | versions, snapshots, writer serialisation | how columns are stored |

Two consequences of the layering are worth stating:

- **Kernels take `&[T]` and `&[u64]`.** They can be tested exhaustively against
  portable reference implementations without constructing a column, and a new
  instruction set is a new file in `kernels`, invisible to the layers above.
- **`unsafe` is confined** to `memory` (allocation, the buffer smart pointer)
  and `kernels` (intrinsics, raw output cursors), plus a few call sites in the
  typed layers that hand a kernel its output slot. Every block carries a
  `SAFETY` comment naming the invariant it relies on.

---

## 3. Columnar memory layout

### 3.1 The allocator: `ColumnAlloc`

Rust 1.100 stabilised `std::alloc::Allocator` together with the
allocator-aware collection constructors (`Vec::new_in`, `with_capacity_in`,
`from_parts_in`, `into_parts_with_allocator`). The crate uses them to move the
memory layout out of the containers and into one allocator:

```rust
pub struct ColumnAlloc;                       // zero-sized, all instances equivalent
unsafe impl Allocator for ColumnAlloc { .. }
pub type AlignedVec<T> = Vec<T, ColumnAlloc>; // a column under construction
```

**Layout contract.** For every block handed out for a request of `size` bytes:

| # | guarantee | why |
|---|-----------|-----|
| L1 | the block starts on a 64-byte boundary | one cache line, and the widest x86 vector: a column never shares its first line, vector loads never split a line at the start |
| L2 | at least 64 bytes after `size` belong to the block and may be written (never read) | compaction kernels store whole vectors at the tail: no scalar epilogue, no bounds branch |
| L3 | a block of 2 MiB or more is mapped from the OS on a 2 MiB boundary and flagged `MADV_HUGEPAGE` | an 80 MB column is covered by 40 TLB entries instead of 20 480 |
| L4 | a freed block of 2 MiB or more is parked in a size-classed pool, not unmapped | re-use costs a mutex and a pointer pop instead of a page fault and a kernel `memset` per page |

Details:

- **Size classes** for large blocks are spaced a quarter of a power of two
  apart (2, 4, 6, 8, 10, 12, 14, 16, 20, 24 … MiB), bounding waste at 25 % of
  *virtual* memory while making it likely that a freed block fits the next
  request.
- **The pool is bounded and decays**: by default at most a quarter of physical
  memory is parked, and a block unused for 10 s is returned to the OS.
  `DATAFRAMES_POOL_LIMIT_MB`, `DATAFRAMES_NO_POOL` and
  `DATAFRAMES_NO_HUGEPAGES` override the defaults; `ColumnAlloc::stats()`
  exposes the counters.
- **Small blocks** go to the global allocator with the stricter alignment, so
  the application's `#[global_allocator]` is still honoured where
  general-purpose allocators are at their best.
- **Why an allocator and not a custom vector.** Because the guarantees are
  properties of the allocation, every `Vec` method keeps them — `push`,
  `extend`, `resize`, `reserve`, `shrink_to_fit` — and there is exactly one
  `unsafe impl` to audit. Before 1.100 the same layout needed a hand-written
  raw vector with its own growth, drop and panic-safety logic.

### 3.2 The smart pointer: `Buffer<T>`

```rust
pub struct Buffer<T: Pod> {
    ptr: NonNull<T>,              // first element of THIS VIEW
    len: usize,                   // elements in this view
    owner: Option<Arc<Owner>>,    // keeps the allocation alive; None for empty / 'static
}
```

Three machine words. `Deref<Target = [T]>`.

```text
 allocation (ColumnAlloc block)
 ┌───────────────────────────────────────────────────────────┬─────────┐
 │ v0  v1  v2  v3  v4  v5  v6  v7  v8  v9  …           spare │  apron  │
 └───────────────────────────────────────────────────────────┴─────────┘
   ▲                   ▲
   │ ptr,len=10        │ ptr,len=4          both hold an Arc to the same Owner
   Buffer A            Buffer B = A.slice(5, 4)
```

- **Slicing** clones the `Arc` and offsets `ptr`. The offset is applied once,
  at slicing time; reading through a slice is reading a `&[T]`, with nothing
  to add and no chunk list to search. A slice of a slice has the same shape
  and the same cost.
- **Freeze / thaw.** `Buffer::from_aligned(vec)` takes the vector apart with
  `Vec::into_parts_with_allocator` and records the layout it was allocated
  with; `try_into_aligned` reassembles it with `Vec::from_parts_in` when the
  handle is the sole owner. Neither touches an element.
- **Storage kinds.** The owner is one of: a `ColumnAlloc` block; a `Vec<T>`
  from the global allocator, adopted without copying (`from_vec`); or any
  `Send + Sync` object that keeps foreign memory alive (`from_arc`,
  `from_foreign` — never mutated).
- **Mutation** is described in section 6.

### 3.3 `Bitmap`

One bit per row in 64-bit words, least significant bit first (the Arrow layout
viewed eight bytes at a time), plus a bit offset below 64, a length and the
population count.

- The 64-bit word is the unit every kernel consumes: one word selects 64 rows,
  tests "all" or "none" with one compare and counts with one `popcnt`.
- Slicing shares the words. Bitmaps computed by this crate always start on a
  word boundary; `Bitmap::words()` borrows in that case and realigns into a
  small temporary otherwise.
- Bitmaps serve three roles with one representation: boolean values, validity
  (null) masks and filter predicates.
- The count is a plain field, filled in when the bitmap is built (the kernels
  that produce masks return it for free); only a partial slice longer than
  4096 bits leaves it open and counts on request. There is deliberately no
  interior mutability here — see 3.7.

### 3.4 Typed columns

```rust
#[repr(C)]
pub struct ChunkedArray<T: PolarsNumericType> {
    name: PlSmallStr,                 // 24 bytes, inline up to 22
    values: Buffer<T::Native>,        // one contiguous buffer
    validity: Option<Bitmap>,         // None = no nulls
}
```

- **One chunk, always.** The name `ChunkedArray` is kept for source
  compatibility with Polars, but a column never holds more than one buffer.
  Element access is a bounds check and a load; slicing is a pointer bump;
  kernels always see one slice.
- **`BooleanChunked`** stores values and validity as bitmaps; `&`, `|`, `!`
  work a word at a time with Kleene semantics for nulls.
- **`StringChunked`** stores one 16-byte `View` per row:

  ```text
  len ≤ 12:  | len: u32 | bytes: [u8; 12]                         |
  len > 12:  | len: u32 | prefix: [u8; 4] | buffer: u32 | offset: u32 |
  ```

  Long strings live in shared, immutable data buffers. Filtering, slicing and
  taking move views and share the data buffers, so no string byte is copied;
  most comparisons are decided by length and prefix without touching the data.
- All three are `#[repr(C)]` with the name first, so `column.name()` is at the
  same offset whatever the type.

### 3.5 `Series`

A `Series` wraps an enum with one variant per column type, stored inline.
Dispatch is a `match` on the discriminant: no heap box, no vtable, and the
compiler sees through it. Cloning or slicing a `Series` clones or slices the
buffers of the column inside. `Column` is an alias of `Series`.

### 3.6 `DataFrame`

```rust
pub struct DataFrame {
    height: usize,
    offset: usize,                     // row window into `base`
    windowed: bool,
    base: Arc<Vec<Column>>,            // shared by every slice and clone
    view: OnceLock<Vec<Column>>,       // `base` narrowed to the window, built on first use
}
```

- `slice`, `head`, `tail` and `clone` copy the `Arc` and set the window. Their
  cost does not depend on the number of columns.
- The first column access through a windowed frame narrows the columns once
  (each narrowing is an O(1) buffer slice) and caches them.
- Mutation resolves the window and calls `Arc::make_mut` on the column set:
  in place for a sole owner, one shallow copy of the handles otherwise. Column
  *data* is copy-on-write per column (section 6).

### 3.7 No interior mutability in a column

None of `Buffer`, `Bitmap`, the typed columns or the small string holds a
`Cell`, an atomic or a lock inline. Besides making them trivially `Sync`,
this has a measurable effect on plain element access: the compiler may treat
`&column` as read-only memory and keep the data pointer and the length in
registers across calls it cannot see into. With a cached counter in the
struct (an earlier design) every `get` in a loop reloaded both. Removing it
took the look-up benchmark from 10 % behind the same loop over an Arrow array
to about 20 % ahead on the benchmark runner.

---

## 4. SIMD-accelerated filtering

A filter is two kernels joined by a bitmask.

```text
 values ──► compare ──► mask words (u64 per 64 rows) ──► compact ──► selected values
                         │
                         └─ also: count (popcnt), combine (&, |, !), validity
```

### 4.1 Compare → bitmask

- AVX2 kernels for all ten numeric types produce one mask word per 64 rows.
- Six operators are covered by three vector predicates per type and an
  optional complement of the finished word.
- Unsigned integers are compared as signed after flipping the sign bit of both
  operands.
- Floats use a total order — NaN equals NaN and is greater than everything —
  so that filters, sorts and group-bys agree with each other.
- The kernels return the population count of what they wrote, so a filter
  knows its output size without a second pass.

### 4.2 Compaction

- **4- and 8-byte elements, dense words:** a table of `vpermd` index vectors,
  one entry per mask byte (or nibble). Stores are whole vectors. The output
  offset of every step of a 64-row word is derived from the word's population
  counts before the first step runs, so there is no data-dependent branch and
  no step waits for the previous one. The apron (L2) makes the last store
  legal.
- **Sparse stretches:** the density of each group of 16 mask words picks a
  fixed number of scalar steps per word (4, 8 or 12): take the lowest set
  bit, copy that element, clear the bit. Steps beyond a word's population
  count copy a harmless element past the cursor, so this loop is branch-free
  too.
- **1- and 2-byte elements:** table-driven `pshufb` steps; the cursor advances
  by a population count.
- **16-byte string views:** each mask bit is doubled and the pair goes through
  the 8-byte kernel.
- **Bitmaps** (boolean values, validity) are compacted with `pext`.
- All-ones words are copied with `memcpy`; all-zero words are skipped.

### 4.3 Fusion

When predicate and payload are the same column (`filter_cmp`, and every
lazy filter), compare and compact alternate over cache-sized blocks. The block
is still in L1/L2 when it is compacted, and the column crosses the memory bus
once instead of twice.

### 4.4 The ordered-commit pipeline

Filtering in parallel into *one* buffer has a chicken-and-egg problem: a
worker cannot know where its output goes until all earlier rows are counted.
Counting first reads the predicate column twice; per-worker output chunks
fragment the result. `kernels::pipeline::scan_then_emit` does neither:

1. **claim** the next block and **scan** it: evaluate the predicate into mask
   words and count the matches. The block is now in the worker's L2.
2. **commit**: wait until all earlier blocks have committed, read the running
   output offset, add this block's count, pass the turn on. This is the only
   serial section and it is two atomic operations long.
3. **emit**: compact the block — still cache-resident — at the offset just
   obtained. Emits of different blocks write disjoint ranges and overlap
   freely in time.

The result is byte-for-byte what a sequential filter produces. A `DropGuard`
(stable since 1.100) releases the waiters if a worker panics.

### 4.5 Non-temporal stores

A result larger than the last-level cache will be evicted before anyone reads
it. Writing it with ordinary stores first reads every output line into the
cache (read-for-ownership) — a third of the memory traffic of a 50 % filter.
For such results the compaction kernels stage output in an L1-resident buffer
and flush whole cache lines with non-temporal stores. Smaller results use
ordinary stores and stay hot for the next operator. The threshold comes from
the detected cache size (`DATAFRAMES_STREAM_BYTES` overrides it).

### 4.6 Sparse selections

A single column switches to the scalar steps of 4.2. A frame does better:
for every block in which at most two rows in five survive, the block's mask is
turned into a list of row indices once, and each output column is gathered
through the list — it touches only the rows it keeps instead of decoding the
mask again. Large results are gathered through the same non-temporal staging
buffer as 4.5.

### 4.7 Frames and expressions

- `DataFrame::filter(&mask)` runs the ordered-commit pipeline over blocks of
  rows of *all* columns: every core claims a block, takes its output offset,
  and writes that block of every column (4.2 or 4.6, by density). Blocks are
  large here (up to 65 536 rows) so that each column of a block is a long
  sequential run for the hardware prefetchers. The streaming decision is made
  for the frame as a whole. Nullable and bit-packed columns are filtered
  column by column instead.
- A lazy predicate (`col("a").lt(lit(5)).and(col("b").gt(lit(2.0)))`) is
  compiled into a small tree whose leaves are block kernels. The executor
  evaluates the tree for a block and immediately compacts that block of every
  output column through the ordered-commit pipeline.
- `filter(p).select([aggregates])` never materialises the filtered rows. The
  predicate is evaluated for a block; a sum then adds *every* row of the
  block, four at a time under a lane mask looked up from their mask bits, so
  rows that are not selected count as zero — one pass, no branch, no
  compaction. Minima and maxima compact the block into an L1-resident scratch
  buffer and reduce that.
- Projection pushdown ensures that only columns that reach the output are
  compacted.

### 4.8 Dispatch

The crate is compiled for the baseline of its target. `kernels::level()`
detects AVX2, BMI1/2, POPCNT, LZCNT and FMA once and caches the answer; kernel
entry points branch on it. Loops that LLVM vectorises well by itself are
written once and re-instantiated inside a `#[target_feature]` wrapper
(`simd_dispatch!`), so they too run with 256-bit vectors in a portable binary.
`DATAFRAMES_SIMD=scalar` forces the portable kernels.

### 4.9 Reductions and work splitting

- Sums, means, minima and maxima are reduced piece by piece (16 384 rows per
  piece) in a fixed association order: a floating-point sum is bit-identical
  whatever the number of threads or the vector width.
- Kernels use every hardware thread by default (`DATAFRAMES_THREADS`
  restricts them, e.g. to one per physical core; on the benchmark runner the
  second thread of a core never hurt and bought up to 30 % on multi-column
  filters). The input is cut into a few chunks per thread, claimed
  dynamically, so a slow or late thread does not hold the others up.
- **The team.** General-purpose pools put idle workers to sleep at once, and
  waking one costs tens of microseconds — longer than most operations on a
  column that fits in the cache, which is why such columns are normally
  processed by one core. `kernels::team` is a fork-join executor whose workers
  keep polling for 100 µs after a parallel region (`DATAFRAMES_SPIN_US`).
  While they poll, starting a region costs about half a microsecond, so
  inputs from 64 KiB upwards are split. When the workers are asleep, only
  inputs of 256 KiB or more — or a burst of smaller calls — wake them.
  `rayon` is used for sorting only.

---

## 5. Concurrency specification

### 5.1 Model

- Column data is **immutable while shared**. A `Buffer`, `Bitmap`, column,
  `Series` or `DataFrame` reachable from more than one handle is never written
  through any of them (the single exception, appending *past* every view, is
  specified in 5.4).
- All of these types are `Send + Sync`. Sharing one across threads needs no
  wrapper and no lock.
- Mutation requires `&mut` on a handle. Whether it happens in place is decided
  by a uniqueness check (section 6), never by a lock.

### 5.2 The four tiers

| tier | mechanism | proof of safety | read-path cost |
|-----:|-----------|-----------------|----------------|
| 0 | **Frozen sharing.** Handles are cloned and sent freely; all access is through `&self`. | the type system | none |
| 1 | **Exclusive mutation.** `&mut` methods update in place if the buffer is uniquely owned, else copy-on-write, per column. | the borrow checker + one reference count | none |
| 2 | **Partitioned mutation.** One `&mut [T]` is split into disjoint parts, or the columns of one `&mut DataFrame` are handed to different workers (`par_apply_columns`). | the borrow checker (disjoint `&mut`) | none |
| 3 | **Snapshot isolation.** `SharedFrame` publishes immutable versions; writers build the next one privately. | one atomic version swap | one atomic load per access |

Choose the lowest tier that fits: tiers 0–2 have no run-time cost at all.

### 5.3 `SharedFrame`

**State.** A current *version* — an immutable `DataFrame` and an epoch number
— held as `Arc<Version>`; an atomic copy of the current epoch; a writer mutex.

**Operations.**

| operation | effect |
|-----------|--------|
| `snapshot()` | pins the current version and returns it as a `Snapshot` (derefs to `DataFrame`) |
| `reader()` | a per-thread `FrameReader` that caches a snapshot; `get()` refreshes it only when the epoch has changed |
| `write(f)` / `try_write(f)` | runs `f` on a private frame initialised from the current version, then publishes the result as the next version; `try_write` publishes nothing if `f` fails |
| `update(f)` | functional form: the next version is `f(&current)` |
| `compare_and_set(epoch, frame)` | optimistic: publish only if the current epoch is still `epoch` |
| `append(batch)` | `try_write` with `vstack_mut` |

**Guarantees.**

- **G1 — Atomicity.** A version becomes visible in one step. No reader
  observes a state in which some of a write's effects are applied and others
  are not.
- **G2 — Isolation.** A `Snapshot` never changes. Nothing reachable from it is
  written after it was taken.
- **G3 — Non-blocking reads.** Taking a snapshot never waits for a writer's
  computation: the version pointer is locked only to clone or replace an
  `Arc`. Using a snapshot, or a `FrameReader` whose epoch is current, takes no
  lock at all.
- **G4 — Serialised writers.** Writes are totally ordered; version `n + 1` is
  computed from version `n`. `compare_and_set` lets a caller prepare a version
  outside the writer lock and detect having lost a race.
- **G5 — Monotonic reads.** The epochs observed by one reader never decrease.
- **G6 — Failure atomicity.** If a write closure panics or returns an error,
  the current version stays in place and the lock is released; lock poisoning
  is ignored because the protected state is replaced by a single assignment.
- **G7 — Proportional cost.** The private frame shares every buffer with the
  current version. Replacing or updating a column copies that column;
  appending rows copies only the new rows (5.4).
- **G8 — Reclamation.** A version's buffers are released, to the column pool,
  when the last snapshot of it is dropped. There are no epochs to advance and
  no hazard pointers to scan.

**Why the guarantees hold.** A version is immutable from construction (G2)
because the only `&mut DataFrame` that ever existed for it was the writer's
private one, which is moved into the `Arc` at publication. Publication
replaces one `Arc` under the pointer lock (G1). Readers hold that lock for the
duration of an `Arc` clone only (G3). Writers hold the writer mutex from
reading the base version to publishing (G4). The epoch is stored after the
pointer is replaced, and a `FrameReader` only moves to a version whose epoch
is greater than its own (G5).

### 5.4 Appending in place under readers

An append-only table should not copy itself on every append, yet older
snapshots must keep their view. `Buffer::append_in_place` reconciles the two:

- Each `ColumnAlloc` block records `tail`, the number of initialised bytes —
  the largest end of any view into it.
- A handle whose view ends exactly at `tail`, in a block with spare capacity,
  may claim the bytes after it by compare-and-swap on `tail`, write the new
  elements there and extend its own length.
- Every other handle views only bytes before the old `tail`, so the write
  touches memory no one else can observe. If the swap fails — the view does
  not end at the tail, or another handle won the race — the caller falls back
  to copying.

This is the one place where a shared allocation is written; the invariant
that makes it sound is *no view extends past `tail`, and `tail` only grows*.

### 5.5 Data parallelism inside kernels

- Workers receive disjoint ranges of the output; the ranges are derived from
  population counts (static partitioning) or from the commit order (4.4), so
  disjointness is by construction.
- Input is shared immutably.
- **The team's job slot.** A parallel region lends a closure on the caller's
  stack to the workers. A worker may use it only between announcing itself
  (`active += 1`) and leaving (`active -= 1`), and only if it saw the job
  open *after* announcing itself; the caller closes the job and then waits for
  `active == 0` before it returns. In the sequentially consistent order of
  those four operations either the worker sees the job closed and stays out,
  or the caller sees the worker and waits. Task indices are handed out by one
  `fetch_add`, so each runs exactly once whichever threads take part.
- Nested regions and regions started concurrently from several threads do not
  queue: they run on the calling thread.
- A panic in a task cancels the tasks not yet claimed and is re-raised on the
  caller after every worker has left the job.

### 5.6 Deliberately not offered

Concurrent in-place updates of the same column under readers. They would put
an atomic or a lock on every element read, which the governing constraint
rules out. Batch such updates into one `write` (tier 3), or partition the
column under a single writer (tier 2).

---

## 6. Ownership rules

| question | answer |
|----------|--------|
| Who may read a buffer? | any handle, from any thread, at any time |
| Who may write it? | a handle that is `&mut` **and** the sole owner (`Buffer::get_mut` checks the reference count) |
| What if it is shared? | `Buffer::make_mut` copies the view into a fresh block first (copy-on-write); other handles keep the old bytes |
| Granularity of copy-on-write | one column; the other columns of the frame stay shared |
| Can a frozen buffer become a `Vec` again? | yes, without copying, if it is uniquely owned (`try_into_aligned`) |
| Who frees the memory? | the last handle; large blocks go back to the pool |
| What about foreign memory? | adopted read-only; a mutation copies it |

In-place operations (`apply_mut`, `set`, `vstack_mut`, ...) follow one pattern:
check ownership once, then run a plain loop over `&mut [T]`.

---

## 7. The trade-off ledger

| decision | buys | costs |
|----------|------|-------|
| shared data is immutable | reads are plain loads; `Send + Sync` for free | writers copy what they change, or publish versions |
| offset applied at slicing time (`ptr` is pre-offset) | a slice reads as fast as the original | one more word per handle than a bare `Arc<[T]>` |
| one chunk per column | O(1) element access, kernels see one slice | `append` may reallocate (amortised; in place when uniquely owned or past every view) |
| layout guarantees in the allocator | one `unsafe impl`; every `Vec` method keeps them | small blocks are padded to 64 bytes plus the apron |
| huge-page blocks, pooled | no page-fault storm for large results; fewer TLB misses | parked memory (bounded, decaying); virtual size classes waste up to 25 % |
| `Series` as an inline enum | no vtable, no box, inlinable dispatch | `Series` is larger than a pointer; adding a type touches the enum |
| `DataFrame` as shared column set + window | slice and clone independent of width | first access through a window narrows the columns once |
| snapshot isolation for shared mutation | readers never block; atomic multi-column writes | one writer at a time; a writer copies the columns it changes |
| fixed association order for float reductions | bit-identical results across threads and CPUs | cannot use the fastest order on each machine |
| run-time CPU dispatch | one portable binary | one predictable branch per kernel call |
| workers poll for 100 µs after a parallel region | columns from 64 KiB use every core; a region starts in under a microsecond | a burst of idle CPU after each region |
| no interior mutability in columns | `&column` is read-only to the optimiser; pointer and length stay in registers | a bitmap cannot cache a count lazily: it is computed when the bitmap is built, and large partial slices count on request |
| frame filters block-wise over all columns | one pass, all cores, shared row-index lists | nullable and bit-packed columns take a column-wise path |

---

## 8. Safety and verification

- **`unsafe` inventory.** `memory::alloc` (the `Allocator` impl, `mmap`),
  `memory::buffer` (pointer arithmetic, freeze/thaw, in-place append),
  `memory::bitmap` (unchecked word access), `kernels` (intrinsics, raw output
  cursors, the pipeline's shared output), and call sites that pass
  uninitialised output capacity to a kernel.
- **Model checking of kernels.** Every vector kernel is compared with a
  portable reference for every element type, every operator, lengths around
  every block and word boundary, and mask densities from empty to full.
- **Both dispatch paths in CI.** The test suite runs with run-time dispatch,
  with `-C target-cpu=native`, with `DATAFRAMES_SIMD=scalar`, and on aarch64.
- **Whole-frame filters against a row-by-row model** at densities on both
  sides of every strategy switch, for heights below and above the block and
  parallel thresholds, eager and lazy, with and without nullable columns.
- **The team** is tested for exactly-once execution, borrowing from the
  caller's stack, nested and concurrent callers, and panic propagation.
- **Concurrency tests.** Snapshot isolation is exercised with concurrent
  readers and writers checking G1, G2 and G5; in-place append is tested
  against snapshots taken before it.
- **Examples are tests.** The examples in the README and in the API
  documentation are compiled and run as documentation tests.

---

## 9. Benchmark methodology

- **Separate processes.** One driver binary per library (`bench/ours`,
  `bench/polars`, `bench/arrow`) plus a Python driver for pandas and the
  Polars wheel, so allocators, thread pools and caches never interfere.
- **Same input.** All drivers build their tables from one deterministic
  generator (`bench/common`); the Python driver reads the same bytes from
  disk.
- **Same work.** Every case reports a check value (a row count, a sum); the
  report refuses to rank results whose check values differ.
- **Best configuration for everyone.** All Rust code is built with
  `-C target-cpu=native`, thin LTO, and the same allocator. Where a library
  has several spellings or engines for one operation, all are timed and the
  best is reported — for dataframes too.
- **Interleaving.** `scripts/bench.py run --per-case` runs one case at a time
  across all libraries, for several rounds; the reported number is the best
  median over rounds.
- **Verdicts.** More than 3 % faster than the fastest competitor is a win;
  within 3 % is level; anything else is a loss and fails the workflow.
- **Operations that can only be level.** A point look-up is a bounds check and
  a load in every library that stores a column contiguously; such cases are
  kept in the suite and reported as what they are.
