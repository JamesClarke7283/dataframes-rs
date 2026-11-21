//! [`ColumnAlloc`]: the allocator every column buffer lives in.
//!
//! Rust 1.100 stabilised [`std::alloc::Allocator`] together with the
//! `Vec::*_in` / `Vec::into_parts_with_allocator` constructors. This module is
//! where the crate cashes that in: instead of hand-rolling a raw vector, a
//! column under construction is simply a `Vec<T, ColumnAlloc>` and the memory
//! layout guarantees below are properties of the *allocator*, enforced in one
//! place and inherited by every buffer.
//!
//! # Layout contract
//!
//! For every block handed out for a request of `size` bytes:
//!
//! 1. **Alignment** — the block starts on a 64-byte boundary (one cache line,
//!    and the width of the widest x86 vector register), so a column never
//!    shares its first cache line with a neighbour and aligned vector loads
//!    never split a line.
//! 2. **Apron** — at least [`APRON`] (64) bytes after `size` are part of the
//!    block and may be *written* (never read) by compaction kernels, which lets
//!    them store whole vectors without a scalar epilogue or a bounds branch.
//! 3. **Huge pages** — blocks of [`HUGE_BLOCK`] (2 MiB) or more are mapped
//!    directly from the OS on a 2 MiB boundary and flagged `MADV_HUGEPAGE`, so
//!    a 80 MB column is covered by 40 TLB entries instead of 20 480.
//! 4. **Recycling** — huge blocks are returned to a size-classed pool instead
//!    of being unmapped. Re-using one costs a mutex and a pointer pop; a fresh
//!    one costs a page fault (and a kernel memset) per page. The pool is capped
//!    (default: a quarter of physical memory) and decays after
//!    [`POOL_DECAY_MS`] of disuse.
//!
//! Small blocks are delegated to the global allocator with the stricter
//! alignment, so the choice of `#[global_allocator]` is still honoured for the
//! many-small-allocations case where general-purpose allocators shine.

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Alignment of every column block, in bytes.
pub const CACHE_LINE: usize = 64;
/// Writable slack that follows every block, in bytes.
pub const APRON: usize = 64;
/// Blocks at least this large are huge-page aligned, OS-mapped and pooled.
pub const HUGE_BLOCK: usize = 2 * 1024 * 1024;
/// A pooled block that has not been reused for this long is released.
pub const POOL_DECAY_MS: u64 = 10_000;

/// Allocator for column data. See the [module documentation](self).
///
/// `ColumnAlloc` is a zero-sized handle: all instances are equivalent, so a
/// block allocated through one may be released through any other.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct ColumnAlloc;

/// A `Vec` whose storage obeys the [`ColumnAlloc`] layout contract.
///
/// This is the mutable half of a column: build it with the ordinary `Vec`
/// API, then freeze it into a shared [`Buffer`](super::Buffer) without copying.
pub type AlignedVec<T> = Vec<T, ColumnAlloc>;

/// Counters describing the huge-block pool. Obtained from [`ColumnAlloc::stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    /// Bytes currently parked in the pool, ready for reuse.
    pub pooled_bytes: usize,
    /// Huge blocks currently handed out.
    pub live_huge_bytes: usize,
    /// Huge allocations satisfied from the pool.
    pub pool_hits: u64,
    /// Huge allocations that had to map fresh memory.
    pub pool_misses: u64,
    /// Upper bound on `pooled_bytes`.
    pub pool_limit: usize,
}

/// How a request is satisfied.
#[derive(Clone, Copy)]
enum Class {
    /// Delegated to the global allocator with this (padded, 64-aligned) layout.
    Small(Layout),
    /// OS-mapped block of exactly this many bytes (a multiple of 2 MiB).
    Huge(usize),
}

/// Round `n` up to a multiple of the power of two `to`.
#[inline]
const fn round_up(n: usize, to: usize) -> Option<usize> {
    match n.checked_add(to - 1) {
        Some(v) => Some(v & !(to - 1)),
        None => None,
    }
}

/// Size of the huge block that serves a request of `need` bytes.
///
/// Classes are spaced a quarter of a power of two apart (2, 4, 6, 8, 10, 12,
/// 14, 16, 20, 24, ... MiB), which bounds internal waste at 25 % of *virtual*
/// memory while making it likely that a freed block fits the next request.
#[inline]
fn huge_class(need: usize) -> Option<usize> {
    let log = usize::BITS - 1 - need.leading_zeros();
    let step = (1usize << log.saturating_sub(2)).max(HUGE_BLOCK);
    round_up(need, step)
}

#[inline]
fn classify(layout: Layout) -> Option<Class> {
    let need = layout.size().checked_add(APRON)?;
    if need >= HUGE_BLOCK && layout.align() <= HUGE_BLOCK && pool().enabled {
        return Some(Class::Huge(huge_class(need)?));
    }
    let align = layout.align().max(CACHE_LINE);
    let size = round_up(need, CACHE_LINE)?;
    Layout::from_size_align(size, align).ok().map(Class::Small)
}

impl ColumnAlloc {
    /// Number of bytes that may be written starting at a block allocated for
    /// `layout`, apron included. Always at least `layout.size() + APRON`.
    #[inline]
    pub fn usable_size(layout: Layout) -> usize {
        match classify(layout) {
            Some(Class::Small(l)) => l.size(),
            Some(Class::Huge(s)) => s,
            None => layout.size(),
        }
    }

    /// Snapshot of the pool counters.
    pub fn stats() -> AllocStats {
        let p = pool();
        AllocStats {
            pooled_bytes: p.pooled.load(Ordering::Relaxed),
            live_huge_bytes: p.live.load(Ordering::Relaxed),
            pool_hits: p.hits.load(Ordering::Relaxed),
            pool_misses: p.misses.load(Ordering::Relaxed),
            pool_limit: p.limit.load(Ordering::Relaxed),
        }
    }

    /// Cap the number of bytes the pool may retain. `0` disables retention.
    pub fn set_pool_limit(bytes: usize) {
        pool().limit.store(bytes, Ordering::Relaxed);
        Self::trim_to(bytes);
    }

    /// Return every pooled block to the operating system.
    pub fn trim() {
        Self::trim_to(0);
    }

    fn trim_to(target: usize) {
        let p = pool();
        let mut victims = Vec::new();
        {
            let mut blocks = p.lock();
            while p.pooled.load(Ordering::Relaxed) > target {
                // Oldest first: the vector is in order of release.
                if blocks.is_empty() {
                    break;
                }
                let b = blocks.remove(0);
                p.pooled.fetch_sub(b.size, Ordering::Relaxed);
                victims.push(b);
            }
        }
        for b in victims {
            // SAFETY: the block was removed from the pool, so we own it.
            unsafe { os::unmap(b.addr as *mut u8, b.size) };
        }
    }
}

// ---------------------------------------------------------------------------
// The pool
// ---------------------------------------------------------------------------

struct Parked {
    addr: usize,
    size: usize,
    released_ms: u64,
}

struct Pool {
    blocks: Mutex<Vec<Parked>>,
    pooled: AtomicUsize,
    live: AtomicUsize,
    limit: AtomicUsize,
    hits: AtomicU64,
    misses: AtomicU64,
    last_decay_ms: AtomicU64,
    epoch: Instant,
    enabled: bool,
    huge_pages: bool,
}

impl Pool {
    #[inline]
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Parked>> {
        // A poisoned lock only means another thread panicked while holding it;
        // the vector itself is always in a consistent state.
        self.blocks.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[inline]
    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true") | Ok("yes"))
}

#[inline]
fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let limit = std::env::var("DATAFRAMES_POOL_LIMIT_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1 << 20))
            .unwrap_or_else(|| os::physical_memory() / 4);
        Pool {
            blocks: Mutex::new(Vec::new()),
            pooled: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
            limit: AtomicUsize::new(limit),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            last_decay_ms: AtomicU64::new(0),
            epoch: Instant::now(),
            // Miri cannot model `mmap`; route everything through `Global` there.
            enabled: !cfg!(miri) && !env_flag("DATAFRAMES_NO_POOL"),
            huge_pages: !env_flag("DATAFRAMES_NO_HUGEPAGES"),
        }
    })
}

/// Take a parked block of exactly `size` bytes, if there is one.
fn pool_take(size: usize) -> Option<NonNull<u8>> {
    let p = pool();
    if p.pooled.load(Ordering::Relaxed) < size {
        return None;
    }
    let mut blocks = p.lock();
    // Most recently released first: its pages are the likeliest to be cached.
    let i = blocks.iter().rposition(|b| b.size == size)?;
    let b = blocks.remove(i);
    p.pooled.fetch_sub(size, Ordering::Relaxed);
    NonNull::new(b.addr as *mut u8)
}

/// Park a block, or unmap it if the pool is full. Also expires stale blocks.
fn pool_put(ptr: NonNull<u8>, size: usize) {
    let p = pool();
    let now = p.now_ms();
    let mut expired = Vec::new();
    let keep = {
        let mut blocks = p.lock();
        let last = p.last_decay_ms.load(Ordering::Relaxed);
        if now.saturating_sub(last) >= 1_000 {
            p.last_decay_ms.store(now, Ordering::Relaxed);
            let mut i = 0;
            while i < blocks.len() {
                if now.saturating_sub(blocks[i].released_ms) > POOL_DECAY_MS {
                    let b = blocks.remove(i);
                    p.pooled.fetch_sub(b.size, Ordering::Relaxed);
                    expired.push(b);
                } else {
                    i += 1;
                }
            }
        }
        let limit = p.limit.load(Ordering::Relaxed);
        let keep = p.pooled.load(Ordering::Relaxed).saturating_add(size) <= limit;
        if keep {
            blocks.push(Parked { addr: ptr.as_ptr() as usize, size, released_ms: now });
            p.pooled.fetch_add(size, Ordering::Relaxed);
        }
        keep
    };
    for b in expired {
        // SAFETY: removed from the pool above, so nobody else can reach it.
        unsafe { os::unmap(b.addr as *mut u8, b.size) };
    }
    if !keep {
        // SAFETY: the caller relinquished the block and it was not parked.
        unsafe { os::unmap(ptr.as_ptr(), size) };
    }
}

/// Returns the block and whether its contents are known to be zero.
fn huge_alloc(size: usize) -> Result<(NonNull<u8>, bool), AllocError> {
    let p = pool();
    let (ptr, zeroed) = match pool_take(size) {
        Some(ptr) => {
            p.hits.fetch_add(1, Ordering::Relaxed);
            (ptr, false)
        }
        None => {
            p.misses.fetch_add(1, Ordering::Relaxed);
            (os::map(size, p.huge_pages).ok_or(AllocError)?, true)
        }
    };
    p.live.fetch_add(size, Ordering::Relaxed);
    Ok((ptr, zeroed))
}

fn huge_free(ptr: NonNull<u8>, size: usize) {
    pool().live.fetch_sub(size, Ordering::Relaxed);
    pool_put(ptr, size);
}

// ---------------------------------------------------------------------------
// The Allocator impl
// ---------------------------------------------------------------------------

// SAFETY:
// * Blocks stay valid until deallocated: small blocks belong to `Global`, huge
//   blocks are private anonymous mappings that are only unmapped after
//   `deallocate` (directly, or later by the pool, which owns them by then).
// * `deallocate`/`grow`/`shrink` recompute the block class from the layout the
//   caller passes. `classify` is a pure function of the layout (the pool's
//   `enabled` flag is fixed at first use), so a layout that *fits* a block maps
//   to the class the block was allocated with: `Layout::size` may only range
//   over `requested ..= returned`, and `grow`/`shrink` only return the same
//   block when the new layout has the same class.
// * No method unwinds: lock poisoning is tolerated and nothing else panics.
unsafe impl Allocator for ColumnAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match classify(layout).ok_or(AllocError)? {
            Class::Small(l) => {
                let block = Global.allocate(l)?;
                // Report exactly the padded size so that every layout that
                // "fits" this block classifies the same way on deallocation.
                Ok(NonNull::slice_from_raw_parts(block.cast(), layout.size()))
            }
            Class::Huge(size) => {
                let (ptr, _) = huge_alloc(size)?;
                Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
            }
        }
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match classify(layout).ok_or(AllocError)? {
            Class::Small(l) => {
                let block = Global.allocate_zeroed(l)?;
                Ok(NonNull::slice_from_raw_parts(block.cast(), layout.size()))
            }
            Class::Huge(size) => {
                let (ptr, zeroed) = huge_alloc(size)?;
                if !zeroed {
                    // SAFETY: the block is `size >= layout.size()` bytes long.
                    unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
                }
                Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
            }
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        match classify(layout) {
            // SAFETY: `ptr` was allocated by `Global` with this exact layout.
            Some(Class::Small(l)) => unsafe { Global.deallocate(ptr, l) },
            Some(Class::Huge(size)) => huge_free(ptr, size),
            // `allocate` would have failed for this layout, so no such block exists.
            None => {}
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.resize(ptr, old, new, false) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.resize(ptr, old, new, true) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.resize(ptr, old, new, false) }
    }
}

impl ColumnAlloc {
    /// Shared body of `grow`, `grow_zeroed` and `shrink`.
    ///
    /// # Safety
    /// Same contract as [`Allocator::grow`] / [`Allocator::shrink`].
    unsafe fn resize(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        zero_tail: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let same_block = match (classify(old), classify(new).ok_or(AllocError)?) {
            (Some(Class::Huge(a)), Class::Huge(b)) => a == b,
            (Some(Class::Small(a)), Class::Small(b)) => a == b,
            _ => false,
        };
        if same_block {
            // The block already has room (this is what makes `Vec` growth
            // inside a huge size class free of copies and system calls).
            if zero_tail && new.size() > old.size() {
                // SAFETY: both sizes are within the block.
                unsafe { ptr.as_ptr().add(old.size()).write_bytes(0, new.size() - old.size()) };
            }
            return Ok(NonNull::slice_from_raw_parts(ptr, new.size()));
        }
        let fresh = if zero_tail { self.allocate_zeroed(new)? } else { self.allocate(new)? };
        let keep = old.size().min(new.size());
        // SAFETY: `ptr` is valid for `old.size()` bytes, `fresh` for
        // `new.size()`, and distinct live blocks never overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), fresh.cast::<u8>().as_ptr(), keep);
            self.deallocate(ptr, old);
        }
        Ok(fresh)
    }
}

// ---------------------------------------------------------------------------
// OS layer
// ---------------------------------------------------------------------------

#[cfg(all(unix, not(miri)))]
mod os {
    use super::HUGE_BLOCK;
    use std::ptr::NonNull;

    /// Map `size` zeroed bytes on a 2 MiB boundary.
    pub fn map(size: usize, huge_pages: bool) -> Option<NonNull<u8>> {
        let span = size.checked_add(HUGE_BLOCK)?;
        // SAFETY: an anonymous private mapping at a kernel-chosen address
        // cannot alias any existing object.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                span,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return None;
        }
        let start = raw as usize;
        let aligned = (start + HUGE_BLOCK - 1) & !(HUGE_BLOCK - 1);
        let head = aligned - start;
        let tail = span - head - size;
        // SAFETY: both ranges lie inside the mapping created above and are
        // page-aligned (`start`, `aligned` and `size` all are).
        unsafe {
            if head > 0 {
                libc::munmap(raw, head);
            }
            if tail > 0 {
                libc::munmap((aligned + size) as *mut libc::c_void, tail);
            }
            #[cfg(any(target_os = "linux", target_os = "android"))]
            if huge_pages {
                // Advisory: failure (e.g. THP disabled) leaves 4 KiB pages.
                libc::madvise(aligned as *mut libc::c_void, size, libc::MADV_HUGEPAGE);
            }
        }
        let _ = huge_pages;
        NonNull::new(aligned as *mut u8)
    }

    /// # Safety
    /// `ptr..ptr+size` must be a mapping obtained from [`map`] that is no longer in use.
    pub unsafe fn unmap(ptr: *mut u8, size: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe { libc::munmap(ptr as *mut libc::c_void, size) };
    }

    pub fn physical_memory() -> usize {
        // SAFETY: `sysconf` has no preconditions.
        let (pages, page) =
            unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
        if pages <= 0 || page <= 0 {
            return 1 << 30;
        }
        (pages as usize).saturating_mul(page as usize)
    }
}

#[cfg(not(all(unix, not(miri))))]
mod os {
    use super::HUGE_BLOCK;
    use std::alloc::Layout;
    use std::ptr::NonNull;

    pub fn map(size: usize, _huge_pages: bool) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(size, HUGE_BLOCK).ok()?;
        // SAFETY: `size` is never zero (it is at least `HUGE_BLOCK`).
        NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
    }

    /// # Safety
    /// `ptr` must come from [`map`] with the same `size` and be unused.
    pub unsafe fn unmap(ptr: *mut u8, size: usize) {
        // SAFETY: the layout matches the one used in `map`.
        unsafe { std::alloc::dealloc(ptr, Layout::from_size_align_unchecked(size, HUGE_BLOCK)) };
    }

    pub fn physical_memory() -> usize {
        4 << 30
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_blocks_are_cache_aligned_with_apron() {
        for len in [1usize, 3, 63, 64, 65, 1000, 4097] {
            let v: AlignedVec<u8> = Vec::with_capacity_in(len, ColumnAlloc);
            assert_eq!(v.as_ptr() as usize % CACHE_LINE, 0, "len {len}");
            let layout = Layout::array::<u8>(v.capacity()).unwrap();
            assert!(ColumnAlloc::usable_size(layout) >= v.capacity() + APRON);
        }
    }

    #[test]
    fn huge_blocks_are_2mib_aligned_and_recycled() {
        let n = 3 * HUGE_BLOCK / 8;
        let first = {
            let mut v: AlignedVec<u64> = Vec::with_capacity_in(n, ColumnAlloc);
            v.resize(n, 7);
            assert_eq!(v.as_ptr() as usize % HUGE_BLOCK, 0);
            v.as_ptr() as usize
        };
        // The block went back to the pool and is handed out again.
        let again: AlignedVec<u64> = Vec::with_capacity_in(n, ColumnAlloc);
        if !cfg!(miri) {
            assert_eq!(again.as_ptr() as usize, first);
            assert!(ColumnAlloc::stats().pool_hits >= 1);
        }
    }

    #[test]
    fn zeroed_requests_are_zero_even_when_recycled() {
        let n = HUGE_BLOCK;
        {
            let mut v: AlignedVec<u8> = Vec::with_capacity_in(n, ColumnAlloc);
            v.resize(n, 0xAB);
        }
        let block = ColumnAlloc.allocate_zeroed(Layout::array::<u8>(n).unwrap()).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts(block.cast::<u8>().as_ptr(), n) };
        assert!(bytes.iter().all(|&b| b == 0));
        unsafe { ColumnAlloc.deallocate(block.cast(), Layout::array::<u8>(n).unwrap()) };
    }

    #[test]
    fn vec_growth_preserves_contents_across_classes() {
        let mut v: AlignedVec<u32> = Vec::new_in(ColumnAlloc);
        for i in 0..2_000_000u32 {
            v.push(i);
        }
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32));
        v.shrink_to_fit();
        v.truncate(10);
        v.shrink_to_fit();
        assert_eq!(&v[..], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn class_sizes_are_monotone_and_tight() {
        let mut last = 0;
        for need in (HUGE_BLOCK..(1 << 30)).step_by(777_777) {
            let c = huge_class(need).unwrap();
            assert!(c >= need && c % HUGE_BLOCK == 0);
            assert!(c <= need + need / 4 + HUGE_BLOCK, "need {need} class {c}");
            assert!(c >= last);
            last = c;
        }
    }
}
