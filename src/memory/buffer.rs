//! [`Buffer<T>`]: the shared, immutable, sliceable smart pointer behind every column.
//!
//! A `Buffer` is three words: a pointer to the first element *of this view*, a
//! length, and a reference-counted handle that keeps the allocation alive.
//!
//! * **Slicing is O(1) and leaves no residue.** `slice` bumps one reference
//!   count and offsets the pointer. The result has exactly the same shape as an
//!   unsliced buffer: there is no stored offset to add on every access and no
//!   chunk list to search, so reading through a slice of a slice of a slice
//!   costs the same as reading the original.
//! * **Reading is a plain `&[T]`.** `Deref` hands the kernels a native slice;
//!   nothing between the caller and the bytes is dynamic.
//! * **Mutation is gated by ownership, not by locks.** [`Buffer::get_mut`]
//!   succeeds only when this handle is the sole owner, which the borrow checker
//!   and the reference count prove together; [`Buffer::make_mut`] falls back to
//!   copy-on-write. Shared buffers are therefore immutable by construction and
//!   `Send + Sync` without synchronisation.
//! * **Freezing and thawing are free.** A [`AlignedVec`] becomes a `Buffer`
//!   via `Vec::into_parts_with_allocator`, and a uniquely owned `Buffer` turns
//!   back into the same `Vec` via `Vec::from_parts_in` — both stabilised in
//!   Rust 1.99/1.100, neither touches the elements.

use std::alloc::{Allocator, Global, Layout};
use std::any::Any;
use std::fmt;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::alloc::{AlignedVec, ColumnAlloc};

/// Element types that may be stored in a [`Buffer`].
///
/// # Safety
/// Implementors must be plain data: `Copy`, no padding-dependent invariants,
/// no drop glue, every bit pattern written by this crate valid, and not
/// zero-sized.
pub unsafe trait Pod: Copy + Send + Sync + 'static {}

macro_rules! impl_pod {
    ($($t:ty),*) => { $(unsafe impl Pod for $t {})* };
}
impl_pod!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, usize, isize, f32, f64);

/// What keeps the bytes alive, and how to give them back.
enum Storage {
    /// A block from [`ColumnAlloc`], described by the layout it was requested with.
    ///
    /// `tail` is the number of initialised bytes at the front of the block. It
    /// is the largest view end over all handles and only ever grows, by a
    /// compare-and-swap in [`Buffer::append_in_place`]; that makes the bytes
    /// past it writable by exactly one handle at a time even though the block
    /// is shared.
    Column { base: NonNull<u8>, layout: Layout, tail: AtomicUsize },
    /// A `Vec<T>` from the global allocator, adopted without copying.
    Global { base: NonNull<u8>, layout: Layout },
    /// Any other owner (memory map, `Arc<[T]>`, ...). Never mutated.
    Foreign(#[allow(dead_code)] Box<dyn Any + Send + Sync>),
}

struct Owner(Storage);

// SAFETY: the raw pointers are uniquely owned by this `Owner`; the memory they
// refer to is only reachable through `Buffer`s, whose API upholds the usual
// shared-xor-mutable discipline.
unsafe impl Send for Owner {}
unsafe impl Sync for Owner {}

impl Drop for Owner {
    fn drop(&mut self) {
        match self.0 {
            // SAFETY: `base`/`layout` are exactly what the allocator handed out
            // and was asked for; being in `drop` proves no `Buffer` is left.
            Storage::Column { base, layout, .. } => unsafe { ColumnAlloc.deallocate(base, layout) },
            Storage::Global { base, layout } => unsafe { Global.deallocate(base, layout) },
            Storage::Foreign(_) => {}
        }
    }
}

/// A cheaply cloneable, immutable, contiguous run of `T`. See the
/// [module documentation](self).
pub struct Buffer<T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    /// `None` for empty and `'static` buffers, which need no bookkeeping at all.
    owner: Option<Arc<Owner>>,
}

// SAFETY: a `Buffer` is a shared view of immutable plain data; the only way to
// obtain `&mut [T]` is through `&mut self` after proving unique ownership.
unsafe impl<T: Pod> Send for Buffer<T> {}
unsafe impl<T: Pod> Sync for Buffer<T> {}

impl<T: Pod> Buffer<T> {
    /// An empty buffer. Does not allocate.
    #[inline]
    pub const fn new() -> Self {
        Buffer { ptr: NonNull::dangling(), len: 0, owner: None }
    }

    /// View a `'static` slice. Does not allocate or count references.
    #[inline]
    pub const fn from_static(slice: &'static [T]) -> Self {
        // SAFETY: a slice pointer is never null.
        let ptr = unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) };
        Buffer { ptr, len: slice.len(), owner: None }
    }

    /// Freeze a column vector. O(1): the elements are not touched.
    #[inline]
    pub fn from_aligned(vec: AlignedVec<T>) -> Self {
        let (ptr, len, cap, _alloc) = vec.into_parts_with_allocator();
        if cap == 0 {
            return Buffer::new();
        }
        // The same computation `Vec` performed when it allocated, so it cannot fail.
        let layout = Layout::array::<T>(cap).expect("capacity was already allocated");
        let owner = Owner(Storage::Column { base: ptr.cast(), layout, tail: AtomicUsize::new(len * size_of::<T>()) });
        Buffer { ptr, len, owner: Some(Arc::new(owner)) }
    }

    /// Adopt a standard `Vec`. O(1): the elements are not touched.
    ///
    /// The storage keeps whatever alignment the global allocator gave it; the
    /// kernels accept that, they merely cannot assume a writable apron.
    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        let (ptr, len, cap) = vec.into_parts();
        if cap == 0 {
            return Buffer::new();
        }
        let layout = Layout::array::<T>(cap).expect("capacity was already allocated");
        let owner = Owner(Storage::Global { base: ptr.cast(), layout });
        Buffer { ptr, len, owner: Some(Arc::new(owner)) }
    }

    /// Copy a slice into a freshly allocated column block.
    pub fn from_slice(slice: &[T]) -> Self {
        let mut v: AlignedVec<T> = Vec::with_capacity_in(slice.len(), ColumnAlloc);
        v.extend_from_slice(slice);
        Buffer::from_aligned(v)
    }

    /// A buffer of `len` zeroed elements.
    pub fn zeroed(len: usize) -> Self
    where
        T: Zeroable,
    {
        Buffer::from_aligned(zeroed_vec(len))
    }

    /// Collect an iterator of known length into a column block.
    pub fn from_trusted_len_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let mut v: AlignedVec<T> = Vec::with_capacity_in(iter.len(), ColumnAlloc);
        v.extend(iter);
        Buffer::from_aligned(v)
    }

    /// View memory owned by an arbitrary object without copying it.
    ///
    /// # Safety
    /// `ptr..ptr+len` must be valid, initialised and immutable for as long as
    /// `owner` is alive, wherever `owner` is moved.
    #[inline]
    pub unsafe fn from_foreign(
        ptr: NonNull<T>,
        len: usize,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Buffer { ptr, len, owner: Some(Arc::new(Owner(Storage::Foreign(owner)))) }
    }

    /// Share an `Arc<[T]>` without copying.
    pub fn from_arc(arc: Arc<[T]>) -> Self {
        let len = arc.len();
        if len == 0 {
            return Buffer::new();
        }
        // SAFETY: the pointer targets the `Arc`'s heap block, which does not
        // move when the `Arc` handle does and lives as long as the handle.
        unsafe {
            let ptr = NonNull::new_unchecked(arc.as_ptr() as *mut T);
            Buffer::from_foreign(ptr, len, Box::new(arc))
        }
    }

    /// Number of elements in this view.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` if the view holds no elements.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The elements as a native slice.
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr..ptr+len` is initialised and kept alive by `owner`
        // (or is `'static` / empty-and-dangling).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Pointer to the first element of the view.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// A view of `len` elements starting at `offset`. O(1), no copy.
    ///
    /// # Panics
    /// If `offset + len` exceeds the length of this view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "slice {offset}..{offset}+{len} out of bounds for buffer of length {}",
            self.len
        );
        // SAFETY: bounds checked above.
        unsafe { self.slice_unchecked(offset, len) }
    }

    /// Like [`slice`](Self::slice) without the bounds check.
    ///
    /// # Safety
    /// `offset + len <= self.len()`.
    #[inline]
    pub unsafe fn slice_unchecked(&self, offset: usize, len: usize) -> Self {
        // SAFETY: the caller keeps the range inside this view.
        Buffer { ptr: unsafe { self.ptr.add(offset) }, len, owner: self.owner.clone() }
    }

    /// Consuming [`slice`](Self::slice): narrows the view in place, touching no
    /// reference count at all.
    #[inline]
    pub fn sliced(mut self, offset: usize, len: usize) -> Self {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "slice {offset}..{offset}+{len} out of bounds for buffer of length {}",
            self.len
        );
        // SAFETY: bounds checked above.
        self.ptr = unsafe { self.ptr.add(offset) };
        self.len = len;
        self
    }

    /// Split into `[0, mid)` and `[mid, len)`. O(1), no copy.
    #[inline]
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        let right = self.slice(mid, self.len - mid);
        (self.sliced(0, mid), right)
    }

    /// `true` if no other handle shares the allocation.
    #[inline]
    pub fn is_unique(&mut self) -> bool {
        match &mut self.owner {
            Some(owner) => Arc::get_mut(owner).is_some(),
            None => self.len == 0,
        }
    }

    /// Mutable access, granted only to the sole owner of writable storage.
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut [T]> {
        let writable = match &mut self.owner {
            Some(owner) => {
                matches!(Arc::get_mut(owner), Some(Owner(Storage::Column { .. } | Storage::Global { .. })))
            }
            None => self.len == 0,
        };
        // SAFETY: we hold `&mut self`, and the reference count proves no other
        // `Buffer` can observe these elements.
        writable.then(|| unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
    }

    /// Mutable access with copy-on-write: copies the view into a private
    /// block first if it is shared.
    #[inline]
    pub fn make_mut(&mut self) -> &mut [T] {
        if self.get_mut().is_none() {
            *self = Buffer::from_slice(self.as_slice());
        }
        self.get_mut().expect("a freshly copied buffer is uniquely owned")
    }

    /// Thaw into a growable vector.
    ///
    /// Free when this handle is the sole owner of a [`ColumnAlloc`] block (the
    /// view is moved to the front of the block if it was sliced); otherwise the
    /// view is copied.
    pub fn into_aligned(self) -> AlignedVec<T> {
        match self.try_into_aligned() {
            Ok(v) => v,
            Err(shared) => {
                let mut v: AlignedVec<T> = Vec::with_capacity_in(shared.len, ColumnAlloc);
                v.extend_from_slice(shared.as_slice());
                v
            }
        }
    }

    /// Thaw into a growable vector if that needs no allocation.
    pub fn try_into_aligned(mut self) -> Result<AlignedVec<T>, Self> {
        let (base, layout) = match self.owner.as_mut().map(Arc::get_mut) {
            Some(Some(Owner(Storage::Column { base, layout, .. }))) => (*base, *layout),
            _ => return Err(self),
        };
        let owner = Arc::into_inner(self.owner.take().expect("checked above"))
            .expect("uniqueness checked above");
        // The vector takes over the block; the owner must not free it.
        std::mem::forget(owner);
        let base = base.cast::<T>();
        // SAFETY: we are the sole owner. `base` is the block start, the view
        // lies within the block, and `copy` handles the overlap. The capacity
        // is recomputed from the layout the block was allocated with.
        unsafe {
            if self.ptr != base {
                std::ptr::copy(self.ptr.as_ptr(), base.as_ptr(), self.len);
            }
            let cap = layout.size() / size_of::<T>();
            Ok(Vec::from_parts_in(base, self.len, cap, ColumnAlloc))
        }
    }

    /// Extend this view by `extra` without copying what is already there,
    /// even if other handles share the allocation. Returns `false` (and does
    /// nothing) when that is not possible; the caller then copies.
    ///
    /// It is possible when the view ends exactly where the initialised part of
    /// its block ends and the block has spare capacity. Other handles view
    /// only bytes before that point, so the new elements are written into
    /// memory nobody else can observe: an append-only column can grow in
    /// place while older snapshots keep reading their prefix.
    pub fn append_in_place(&mut self, extra: &[T]) -> bool {
        let Some(owner) = &self.owner else { return false };
        let Storage::Column { base, layout, tail } = &owner.0 else { return false };
        let size = size_of::<T>();
        let end = self.ptr.as_ptr() as usize - base.as_ptr() as usize + self.len * size;
        let Some(new_end) = extra.len().checked_mul(size).and_then(|bytes| end.checked_add(bytes)) else {
            return false;
        };
        if new_end > layout.size() {
            return false;
        }
        // Claim `end..new_end`. This fails if the view does not end at the
        // tail, or if another handle that also ended there claimed it first.
        if tail.compare_exchange(end, new_end, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            return false;
        }
        // SAFETY: the bytes `end..new_end` are inside the block (checked
        // against the layout) and, by the invariant on `tail`, outside every
        // other handle's view; the successful exchange made this handle the
        // only one allowed to extend into them. `extra` cannot overlap them
        // because it is initialised memory reachable through a shared borrow.
        unsafe {
            std::ptr::copy_nonoverlapping(extra.as_ptr(), self.ptr.as_ptr().add(self.len), extra.len());
        }
        self.len += extra.len();
        true
    }

    /// Bytes of this view (not of the whole allocation).
    #[inline]
    pub const fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// `true` if both handles view the very same memory range.
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.len == other.len
    }
}

/// Types for which the all-zero bit pattern is a valid value.
///
/// # Safety
/// The all-zero bit pattern must be a valid `Self`.
pub unsafe trait Zeroable: Pod {}
macro_rules! impl_zeroable {
    ($($t:ty),*) => { $(unsafe impl Zeroable for $t {})* };
}
impl_zeroable!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, usize, isize, f32, f64);

/// A column vector of `len` zeroed elements.
pub fn zeroed_vec<T: Zeroable>(len: usize) -> AlignedVec<T> {
    if len == 0 {
        return Vec::new_in(ColumnAlloc);
    }
    let layout = Layout::array::<T>(len).expect("capacity overflow");
    let block = match ColumnAlloc.allocate_zeroed(layout) {
        Ok(b) => b,
        Err(_) => std::alloc::handle_alloc_error(layout),
    };
    // SAFETY: the block was allocated by `ColumnAlloc` for exactly `len`
    // elements and zero is a valid `T`.
    unsafe { Vec::from_parts_in(block.cast::<T>(), len, len, ColumnAlloc) }
}

/// A column vector with room for `cap` elements and length 0.
#[inline]
pub fn aligned_with_capacity<T: Pod>(cap: usize) -> AlignedVec<T> {
    Vec::with_capacity_in(cap, ColumnAlloc)
}

impl<T: Pod> Clone for Buffer<T> {
    #[inline]
    fn clone(&self) -> Self {
        Buffer { ptr: self.ptr, len: self.len, owner: self.owner.clone() }
    }
}

impl<T: Pod> Default for Buffer<T> {
    #[inline]
    fn default() -> Self {
        Buffer::new()
    }
}

impl<T: Pod> Deref for Buffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Pod> AsRef<[T]> for Buffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Pod> From<AlignedVec<T>> for Buffer<T> {
    #[inline]
    fn from(v: AlignedVec<T>) -> Self {
        Buffer::from_aligned(v)
    }
}

impl<T: Pod> From<Vec<T>> for Buffer<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        Buffer::from_vec(v)
    }
}

impl<T: Pod> From<&[T]> for Buffer<T> {
    #[inline]
    fn from(s: &[T]) -> Self {
        Buffer::from_slice(s)
    }
}

impl<T: Pod> FromIterator<T> for Buffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut v: AlignedVec<T> = Vec::with_capacity_in(iter.size_hint().0, ColumnAlloc);
        v.extend(iter);
        Buffer::from_aligned(v)
    }
}

impl<T: Pod + PartialEq> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Pod + fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::alloc::CACHE_LINE;

    #[test]
    fn freeze_slice_and_thaw_without_copying() {
        let mut v: AlignedVec<i64> = aligned_with_capacity(1000);
        v.extend(0..1000);
        let base = v.as_ptr();
        let buf = Buffer::from(v);
        assert_eq!(buf.as_ptr(), base);
        assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0);

        let mid = buf.slice(100, 50);
        assert_eq!(mid.as_ptr(), unsafe { base.add(100) });
        assert_eq!(mid[0], 100);
        assert_eq!(mid.len(), 50);
        let inner = mid.slice(10, 5);
        assert_eq!(&inner[..], &[110, 111, 112, 113, 114]);

        drop((mid, inner));
        let thawed = buf.try_into_aligned().expect("unique");
        assert_eq!(thawed.as_ptr(), base);
        assert_eq!(thawed.len(), 1000);
    }

    #[test]
    fn shared_buffers_refuse_mutation_then_copy_on_write() {
        let mut a = Buffer::from_slice(&[1u32, 2, 3, 4]);
        let b = a.clone();
        assert!(a.get_mut().is_none());
        a.make_mut()[0] = 99;
        assert_eq!(&a[..], &[99, 2, 3, 4]);
        assert_eq!(&b[..], &[1, 2, 3, 4]);
        // Now unique again: mutation is in place.
        let p = a.as_ptr();
        a.make_mut()[1] = 7;
        assert_eq!(a.as_ptr(), p);
    }

    #[test]
    fn sliced_unique_buffer_thaws_to_block_start() {
        let buf = Buffer::from_slice(&[10u16, 11, 12, 13, 14, 15]).sliced(2, 3);
        let v = buf.try_into_aligned().expect("unique");
        assert_eq!(&v[..], &[12, 13, 14]);
        assert_eq!(v.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[test]
    fn adopts_std_vec_and_arc_zero_copy() {
        let v = vec![1.5f64, 2.5, 3.5];
        let p = v.as_ptr();
        let mut buf = Buffer::from(v);
        assert_eq!(buf.as_ptr(), p);
        buf.make_mut()[0] = 0.5;
        assert_eq!(buf.as_ptr(), p, "unique global vec is mutated in place");

        let arc: Arc<[u8]> = Arc::from(&b"hello"[..]);
        let p = arc.as_ptr();
        let mut buf = Buffer::from_arc(arc);
        assert_eq!(buf.as_ptr(), p);
        assert!(buf.get_mut().is_none(), "foreign memory is read-only");
        buf.make_mut()[0] = b'j';
        assert_eq!(&buf[..], b"jello");
    }

    #[test]
    fn empty_and_static_buffers_have_no_owner() {
        let e: Buffer<i32> = Buffer::new();
        assert!(e.is_empty() && e.owner.is_none());
        static DATA: [i32; 3] = [1, 2, 3];
        let s = Buffer::from_static(&DATA);
        assert_eq!(&s.slice(1, 2)[..], &[2, 3]);
        assert!(s.owner.is_none());
        let z: Buffer<u64> = Buffer::zeroed(100);
        assert!(z.iter().all(|&x| x == 0));
    }

    #[test]
    fn shared_buffers_grow_in_place_at_the_tail() {
        let mut v: AlignedVec<u32> = aligned_with_capacity(8);
        v.extend([1, 2, 3]);
        let mut a = Buffer::from(v);
        let snapshot = a.clone();
        let base = a.as_ptr();

        // Shared, but `a` ends at the tail and there is room: no copy.
        assert!(a.append_in_place(&[4, 5]));
        assert_eq!(a.as_ptr(), base);
        assert_eq!(&a[..], &[1, 2, 3, 4, 5]);
        assert_eq!(&snapshot[..], &[1, 2, 3], "the older view is untouched");

        // The snapshot no longer ends at the tail, so it must not write there.
        let mut stale = snapshot.clone();
        assert!(!stale.append_in_place(&[9]));
        // A prefix slice does not end at the tail either.
        assert!(!a.slice(0, 2).append_in_place(&[9]));
        // Two handles ending at the tail: exactly one may claim the next slot.
        let mut b = a.clone();
        assert!(a.append_in_place(&[6]));
        assert!(!b.append_in_place(&[7]));
        assert_eq!((&a[..], &b[..]), (&[1, 2, 3, 4, 5, 6][..], &[1, 2, 3, 4, 5][..]));
        // Capacity is respected.
        assert!(a.append_in_place(&[7, 8]));
        assert!(!a.append_in_place(&[9]));
        assert!(a.append_in_place(&[]));
        // Storage that is not a column block never grows in place.
        assert!(!Buffer::from(vec![1u8, 2]).append_in_place(&[3]));
        assert!(!Buffer::<u8>::new().append_in_place(&[3]));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_out_of_bounds_panics() {
        Buffer::from_slice(&[1u8, 2, 3]).slice(2, 2);
    }
}
