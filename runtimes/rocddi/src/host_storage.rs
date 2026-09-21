//! Fallible host storage for internal owners and buffers.
//!
//! Callback state is copied into each allocation owner. No process-global or
//! thread-local allocator selection is involved. Constructors that will acquire
//! native resources should reserve their storage first with
//! [`Owned::try_new_uninit`] and publish the completed owner with
//! [`UninitOwned::write`], which performs no allocation.

#![allow(unsafe_code)]

use core::alloc::Layout;
use core::ffi::c_void;
use core::fmt;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit, align_of, offset_of, size_of};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::slice;
use core::sync::atomic::{AtomicUsize, Ordering, fence};
use std::alloc::GlobalAlloc;

/// C callback allocating a nonzero, suitably aligned uninitialized range.
pub type AllocateFn = unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void;
/// C callback resizing storage transactionally; NULL leaves the old range live.
pub type ResizeFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64, u64) -> *mut c_void;
/// Infallible C callback releasing storage returned by the same allocator.
pub type FreeFn = unsafe extern "C" fn(*mut c_void, *mut c_void);

/// Allocation failed or the requested capacity cannot be represented.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationError;

impl fmt::Display for AllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("host metadata allocation failed")
    }
}

impl std::error::Error for AllocationError {}

impl From<AllocationError> for crate::Error {
    fn from(_: AllocationError) -> Self {
        Self::Capacity {
            resource: "host metadata",
        }
    }
}

/// Callback fields do not describe either a complete allocator or all zeros.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidAllocator;

impl fmt::Display for InvalidAllocator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("host allocator requires allocate and free callbacks")
    }
}

impl std::error::Error for InvalidAllocator {}

/// Copied host allocator used for internal storage belonging to one core session.
///
/// The default value calls [`std::alloc::System`] directly. Selecting callbacks
/// changes only this value and the owners that retain it.
#[derive(Clone, Copy, Debug)]
pub struct Allocator {
    user_data: *mut c_void,
    allocate: Option<AllocateFn>,
    resize: Option<ResizeFn>,
    free: Option<FreeFn>,
}

// SAFETY: Callback construction requires thread-safe callbacks and shared state.
unsafe impl Send for Allocator {}
// SAFETY: The same construction contract permits concurrent calls with user_data.
unsafe impl Sync for Allocator {}

impl Default for Allocator {
    fn default() -> Self {
        Self::system()
    }
}

impl Allocator {
    /// Returns the default host allocator without performing any allocation.
    #[must_use]
    pub const fn system() -> Self {
        Self {
            user_data: ptr::null_mut(),
            allocate: None,
            resize: None,
            free: None,
        }
    }

    /// Copies callbacks implementing the rocddi host allocator contract.
    ///
    /// All-zero fields select the default allocator. Otherwise `allocate` and
    /// `free` are required; absent `resize` uses allocate-copy-free.
    ///
    /// # Errors
    /// Returns [`InvalidAllocator`] for an incomplete callback combination.
    ///
    /// # Safety
    /// The callbacks and `user_data` must remain valid until every copy of this
    /// allocator and every allocation made through it is released. Callbacks
    /// must be thread-safe and must not unwind. Allocate/resize must return NULL
    /// or writable, nonoverlapping storage satisfying the supplied size and
    /// alignment. Resize must preserve the lesser of the old/new byte lengths;
    /// NULL must preserve the original allocation unchanged. Free must accept
    /// allocations produced by these callbacks and release them infallibly.
    pub unsafe fn from_callbacks(
        user_data: *mut c_void,
        allocate: Option<AllocateFn>,
        resize: Option<ResizeFn>,
        free: Option<FreeFn>,
    ) -> Result<Self, InvalidAllocator> {
        if allocate.is_some() && free.is_some() {
            Ok(Self {
                user_data,
                allocate,
                resize,
                free,
            })
        } else if user_data.is_null() && allocate.is_none() && resize.is_none() && free.is_none() {
            Ok(Self::system())
        } else {
            Err(InvalidAllocator)
        }
    }

    /// Returns whether this value selects the default host allocator.
    #[must_use]
    pub const fn is_system(self) -> bool {
        self.allocate.is_none()
    }

    fn allocate(self, layout: Layout) -> Result<NonNull<u8>, AllocationError> {
        let pointer = if let Some(allocate) = self.allocate {
            // SAFETY: The constructor establishes callback validity; normalized
            // layouts have nonzero size and sufficient power-of-two alignment.
            unsafe { allocate(self.user_data, layout.size() as u64, layout.align() as u64).cast() }
        } else {
            // SAFETY: Normalized layouts always have a nonzero size.
            unsafe { std::alloc::System.alloc(layout) }
        };
        NonNull::new(pointer).ok_or(AllocationError)
    }

    unsafe fn deallocate(self, pointer: NonNull<u8>, layout: Layout) {
        if let Some(free) = self.free {
            // SAFETY: The caller supplies a live allocation from this allocator.
            unsafe { free(self.user_data, pointer.as_ptr().cast()) };
        } else {
            // SAFETY: The caller supplies the original pointer and layout.
            unsafe { std::alloc::System.dealloc(pointer.as_ptr(), layout) };
        }
    }

    unsafe fn reallocate(
        self,
        pointer: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<u8>, AllocationError> {
        if old.align() == new.align() {
            if let Some(resize) = self.resize {
                // SAFETY: This live allocation and its original size are passed
                // to its allocator; callback NULL preserves the old allocation.
                let replacement = unsafe {
                    resize(
                        self.user_data,
                        pointer.as_ptr().cast(),
                        old.size() as u64,
                        new.size() as u64,
                        new.align() as u64,
                    )
                };
                return NonNull::new(replacement.cast()).ok_or(AllocationError);
            }
            if self.is_system() {
                // SAFETY: realloc preserves alignment and leaves storage live on
                // failure; both sizes are nonzero and valid for this layout.
                let replacement =
                    unsafe { std::alloc::System.realloc(pointer.as_ptr(), old, new.size()) };
                return NonNull::new(replacement).ok_or(AllocationError);
            }
        }
        let replacement = self.allocate(new)?;
        // SAFETY: These live allocations do not overlap and both contain the
        // copied byte range. Byte copying permits uninitialized capacity bytes.
        unsafe {
            ptr::copy_nonoverlapping(
                pointer.as_ptr(),
                replacement.as_ptr(),
                old.size().min(new.size()),
            );
            self.deallocate(pointer, old);
        }
        Ok(replacement)
    }
}

// Callback allocations provide at least fundamental alignment even for byte
// buffers. Sixteen meets that bound on the supported x86, x86_64, and AArch64
// platforms; larger alignments required by T are preserved.
fn normalized_layout(layout: Layout) -> Result<Layout, AllocationError> {
    Layout::from_size_align(layout.size().max(1), layout.align().max(16))
        .map_err(|_| AllocationError)
}

struct RawBlock {
    pointer: NonNull<u8>,
    layout: Layout,
    allocator: Allocator,
}

impl RawBlock {
    fn new(layout: Layout, allocator: Allocator) -> Result<Self, AllocationError> {
        let layout = normalized_layout(layout)?;
        let pointer = allocator.allocate(layout)?;
        Ok(Self {
            pointer,
            layout,
            allocator,
        })
    }

    fn resize(&mut self, layout: Layout) -> Result<(), AllocationError> {
        let layout = normalized_layout(layout)?;
        // SAFETY: This block exclusively owns its live allocation and layout.
        let pointer = unsafe {
            self.allocator
                .reallocate(self.pointer, self.layout, layout)?
        };
        self.pointer = pointer;
        self.layout = layout;
        Ok(())
    }
}

impl Drop for RawBlock {
    fn drop(&mut self) {
        // SAFETY: The block exclusively owns this allocation and original layout.
        unsafe { self.allocator.deallocate(self.pointer, self.layout) };
    }
}

#[repr(C)]
struct OwnerSlot<T> {
    allocator: Allocator,
    value: MaybeUninit<T>,
}

/// Unique ownership of one value in callback-allocated metadata storage.
pub struct Owned<T> {
    slot: NonNull<OwnerSlot<T>>,
    owns: PhantomData<T>,
}

// SAFETY: Ownership transfers with T, and the allocator is thread-safe.
unsafe impl<T: Send> Send for Owned<T> {}
// SAFETY: Shared access exposes only &T; deallocation needs exclusive ownership.
unsafe impl<T: Sync> Sync for Owned<T> {}

impl<T> Owned<T> {
    /// Allocates storage and publishes `value` in it.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if storage is unavailable; `value` is dropped.
    pub fn new(value: T, allocator: Allocator) -> Result<Self, AllocationError> {
        Ok(Self::try_new_uninit(allocator)?.write(value))
    }

    /// Reserves owner storage before acquiring a native resource.
    ///
    /// Dropping the returned guard releases storage without dropping a T.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if storage is unavailable.
    pub fn try_new_uninit(allocator: Allocator) -> Result<UninitOwned<T>, AllocationError> {
        let block = RawBlock::new(Layout::new::<OwnerSlot<T>>(), allocator)?;
        let slot = block.pointer.cast::<OwnerSlot<T>>();
        // SAFETY: The allocation has sufficient size/alignment for OwnerSlot<T>.
        unsafe {
            slot.as_ptr().write(OwnerSlot {
                allocator,
                value: MaybeUninit::uninit(),
            });
        }
        Ok(UninitOwned { slot, block })
    }

    /// Returns the allocator retained by this owner.
    #[must_use]
    pub fn allocator(&self) -> Allocator {
        // SAFETY: The slot remains initialized and live throughout this borrow.
        unsafe { self.slot.as_ref().allocator }
    }

    /// Returns a borrowed pointer to the contained value.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        // SAFETY: addr_of does not create a reference or read the pointed value.
        unsafe { ptr::addr_of!((*self.slot.as_ptr()).value).cast() }
    }

    /// Returns an exclusive borrowed pointer to the contained value.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_ptr().cast_mut()
    }

    /// Transfers this owner into a raw C handle without allocating.
    ///
    /// The pointer must eventually be consumed once by [`Self::from_raw`].
    #[must_use]
    pub fn into_raw(self) -> *mut T {
        let owner = ManuallyDrop::new(self);
        owner.as_ptr().cast_mut()
    }

    /// Recovers an owner and its allocator from a previously published pointer.
    ///
    /// # Safety
    /// `pointer` must be an unconsumed result of [`Self::into_raw`] for this exact
    /// T. No other owner may free it, and outstanding references must obey the
    /// usual exclusive ownership rules.
    #[must_use]
    pub unsafe fn from_raw(pointer: *mut T) -> Self {
        // SAFETY: The matching into_raw points at this repr(C) slot's value.
        let slot = unsafe {
            NonNull::new_unchecked(
                pointer
                    .cast::<u8>()
                    .sub(offset_of!(OwnerSlot<T>, value))
                    .cast(),
            )
        };
        Self {
            slot,
            owns: PhantomData,
        }
    }

    /// Moves out the value and releases its metadata allocation.
    #[must_use]
    pub fn into_inner(self) -> T {
        let owner = ManuallyDrop::new(self);
        // SAFETY: ManuallyDrop suppresses destruction; the value is read once,
        // and the temporary guard releases only the allocation.
        unsafe {
            let _block = owner.storage_guard();
            owner.as_ptr().read()
        }
    }

    unsafe fn storage_guard(&self) -> RawBlock {
        RawBlock {
            pointer: self.slot.cast(),
            // OwnerSlot has nonzero size, and its alignment has already been
            // normalized successfully by the constructor.
            layout: unsafe {
                Layout::from_size_align_unchecked(
                    size_of::<OwnerSlot<T>>(),
                    align_of::<OwnerSlot<T>>().max(16),
                )
            },
            allocator: self.allocator(),
        }
    }
}

impl<T> Deref for Owned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: Published owners always contain a fully initialized value.
        unsafe { self.slot.as_ref().value.assume_init_ref() }
    }
}

impl<T> DerefMut for Owned<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: This unique owner exclusively borrows its initialized value.
        unsafe { self.slot.as_mut().value.assume_init_mut() }
    }
}

impl<T: fmt::Debug> fmt::Debug for Owned<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Owned").field(&**self).finish()
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        // SAFETY: This owner is unique and its initialized value is dropped once.
        // The guard releases storage even if the value's destructor unwinds.
        unsafe {
            let _block = self.storage_guard();
            self.slot.as_mut().value.assume_init_drop();
        }
    }
}

/// Preallocated unique-owner storage with no live contained value yet.
pub struct UninitOwned<T> {
    slot: NonNull<OwnerSlot<T>>,
    block: RawBlock,
}

// SAFETY: No T exists yet, and callback state permits transfer between threads.
unsafe impl<T: Send> Send for UninitOwned<T> {}
// SAFETY: Shared access cannot initialize or mutate the reserved value.
unsafe impl<T: Sync> Sync for UninitOwned<T> {}

impl<T> UninitOwned<T> {
    /// Initializes and publishes the owner without allocation or native calls.
    #[must_use]
    pub fn write(self, value: T) -> Owned<T> {
        let guard = ManuallyDrop::new(self);
        // SAFETY: The guard uniquely owns aligned, uninitialized value storage.
        unsafe { ptr::addr_of_mut!((*guard.slot.as_ptr()).value).write(MaybeUninit::new(value)) };
        Owned {
            slot: guard.slot,
            owns: PhantomData,
        }
    }

    /// Returns the allocator retained by the preallocated owner.
    #[must_use]
    pub fn allocator(&self) -> Allocator {
        self.block.allocator
    }
}

#[repr(C)]
struct SharedSlot<T> {
    strong: AtomicUsize,
    allocator: Allocator,
    value: T,
}

/// Atomic shared ownership of one callback-allocated metadata value.
///
/// Cloning increments a count without allocation. This type has no weak handles.
pub struct Shared<T> {
    slot: NonNull<SharedSlot<T>>,
    owns: PhantomData<T>,
}

// SAFETY: Any thread may become the final owner; both transfer and shared access
// to T must be safe. The retained allocator is thread-safe.
unsafe impl<T: Send + Sync> Send for Shared<T> {}
// SAFETY: Shared ownership exposes only &T and synchronizes final destruction.
unsafe impl<T: Send + Sync> Sync for Shared<T> {}

impl<T> Shared<T> {
    /// Allocates a shared owner with an initial strong reference count of one.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if storage is unavailable; `value` is dropped.
    pub fn new(value: T, allocator: Allocator) -> Result<Self, AllocationError> {
        Ok(Self::try_new_uninit(allocator)?.write(value))
    }

    /// Reserves the shared owner before acquiring a native resource.
    ///
    /// Dropping the returned guard releases storage without dropping a T.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if storage is unavailable.
    pub fn try_new_uninit(allocator: Allocator) -> Result<UninitShared<T>, AllocationError> {
        let block = RawBlock::new(Layout::new::<SharedSlot<T>>(), allocator)?;
        let slot = block.pointer.cast::<SharedSlot<T>>();
        Ok(UninitShared { slot, block })
    }

    /// Returns the allocator retained by this shared allocation.
    #[must_use]
    pub fn allocator(&self) -> Allocator {
        // SAFETY: The slot remains live while any strong owner exists.
        unsafe { self.slot.as_ref().allocator }
    }

    /// Returns whether two handles refer to the same allocation.
    #[must_use]
    pub fn ptr_eq(left: &Self, right: &Self) -> bool {
        left.slot == right.slot
    }

    /// Returns exclusive value access when this is the only strong owner.
    #[must_use]
    pub fn get_mut(owner: &mut Self) -> Option<&mut T> {
        // SAFETY: &mut Self excludes cloning this owner during the check. No
        // weak handles exist; count one therefore excludes all other handles.
        unsafe {
            if owner.slot.as_ref().strong.load(Ordering::Acquire) == 1 {
                Some(&mut owner.slot.as_mut().value)
            } else {
                None
            }
        }
    }

    /// Returns a snapshot of the current number of strong owners.
    #[must_use]
    pub fn strong_count(owner: &Self) -> usize {
        // SAFETY: The borrowed owner keeps the atomic counter alive.
        unsafe { owner.slot.as_ref().strong.load(Ordering::Relaxed) }
    }
}

/// Preallocated shared-owner storage with no live contained value yet.
pub struct UninitShared<T> {
    slot: NonNull<SharedSlot<T>>,
    block: RawBlock,
}

// SAFETY: No T exists yet, and callback state permits transfer between threads.
unsafe impl<T: Send> Send for UninitShared<T> {}
// SAFETY: Shared access cannot initialize or mutate the reserved value.
unsafe impl<T: Sync> Sync for UninitShared<T> {}

impl<T> UninitShared<T> {
    /// Initializes and publishes a shared owner without allocating.
    #[must_use]
    pub fn write(self, value: T) -> Shared<T> {
        let guard = ManuallyDrop::new(self);
        // SAFETY: The guard uniquely owns aligned storage for SharedSlot<T>.
        unsafe {
            guard.slot.as_ptr().write(SharedSlot {
                strong: AtomicUsize::new(1),
                allocator: guard.block.allocator,
                value,
            });
        }
        Shared {
            slot: guard.slot,
            owns: PhantomData,
        }
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        // SAFETY: This strong owner keeps the count alive and positive.
        let previous = unsafe { self.slot.as_ref().strong.fetch_add(1, Ordering::Relaxed) };
        if previous > isize::MAX as usize {
            // Match Arc's overflow protection: overflowing reference counts
            // could free live storage, so this is an unrecoverable invariant.
            std::process::abort();
        }
        Self {
            slot: self.slot,
            owns: PhantomData,
        }
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: A live strong owner retains the initialized value.
        unsafe { &self.slot.as_ref().value }
    }
}

impl<T: fmt::Debug> fmt::Debug for Shared<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Shared").field(&**self).finish()
    }
}

impl<T: PartialEq> PartialEq for Shared<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for Shared<T> {}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        // SAFETY: This owner accounts for one still-live reference.
        if unsafe { self.slot.as_ref().strong.fetch_sub(1, Ordering::Release) } != 1 {
            return;
        }
        // Pair with prior owners' release operations before destroying T.
        fence(Ordering::Acquire);
        // SAFETY: The final owner exclusively owns the value and allocation.
        // A stack guard frees the allocation even if T::drop unwinds.
        unsafe {
            let _block = RawBlock {
                pointer: self.slot.cast(),
                layout: Layout::from_size_align_unchecked(
                    size_of::<SharedSlot<T>>(),
                    align_of::<SharedSlot<T>>().max(16),
                ),
                allocator: self.slot.as_ref().allocator,
            };
            ptr::drop_in_place(ptr::addr_of_mut!((*self.slot.as_ptr()).value));
        }
    }
}

/// Growable contiguous host storage using one explicit allocator.
///
/// Growth is fallible and transactional. Zero-sized elements need no allocation.
pub struct Buffer<T> {
    pointer: NonNull<T>,
    length: usize,
    capacity: usize,
    allocator: Allocator,
    block: Option<RawBlock>,
    owns: PhantomData<T>,
}

// SAFETY: The buffer uniquely owns all initialized Ts and its thread-safe allocator.
unsafe impl<T: Send> Send for Buffer<T> {}
// SAFETY: Shared access exposes only &[T]; mutation requires &mut Buffer<T>.
unsafe impl<T: Sync> Sync for Buffer<T> {}

impl<T> Buffer<T> {
    /// Creates an empty buffer without allocating.
    #[must_use]
    pub const fn new(allocator: Allocator) -> Self {
        Self {
            pointer: NonNull::dangling(),
            length: 0,
            capacity: if size_of::<T>() == 0 { usize::MAX } else { 0 },
            allocator,
            block: None,
            owns: PhantomData,
        }
    }

    /// Reserves capacity before any elements or native resources are acquired.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if capacity overflows or allocation fails.
    pub fn try_with_capacity(
        capacity: usize,
        allocator: Allocator,
    ) -> Result<Self, AllocationError> {
        let mut result = Self::new(allocator);
        result.try_reserve_exact(capacity)?;
        Ok(result)
    }

    /// Returns the allocator retained by this buffer.
    #[must_use]
    pub const fn allocator(&self) -> Allocator {
        self.allocator
    }

    /// Returns the number of initialized elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the buffer contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the number of elements that fit before another allocation.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the initialized contiguous element range.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: The pointer is aligned/non-null even when empty or zero-sized;
        // length counts initialized elements and fits the allocated layout.
        unsafe { slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }

    /// Returns exclusive access to the initialized contiguous element range.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: The buffer exclusively owns all initialized elements.
        unsafe { slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }

    /// Iterates over initialized elements in insertion order.
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Iterates with exclusive access to each initialized element.
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Reserves room for at least `additional` elements with geometric growth.
    ///
    /// # Errors
    /// Overflow or allocation failure leaves elements and capacity unchanged.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocationError> {
        let needed = self.length.checked_add(additional).ok_or(AllocationError)?;
        if needed <= self.capacity {
            return Ok(());
        }
        let preferred = self
            .capacity
            .checked_mul(2)
            .unwrap_or(needed)
            .max(needed)
            .max(4);
        let capacity = if Layout::array::<T>(preferred).is_ok() {
            preferred
        } else {
            needed
        };
        self.grow(capacity)
    }

    /// Reserves exactly the required capacity, avoiding speculative growth.
    ///
    /// # Errors
    /// Overflow or allocation failure leaves elements and capacity unchanged.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocationError> {
        let needed = self.length.checked_add(additional).ok_or(AllocationError)?;
        if needed <= self.capacity {
            return Ok(());
        }
        self.grow(needed)
    }

    fn grow(&mut self, capacity: usize) -> Result<(), AllocationError> {
        let layout = Layout::array::<T>(capacity).map_err(|_| AllocationError)?;
        if let Some(block) = &mut self.block {
            block.resize(layout)?;
            self.pointer = block.pointer.cast();
        } else {
            let block = RawBlock::new(layout, self.allocator)?;
            self.pointer = block.pointer.cast();
            self.block = Some(block);
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Appends one element, growing storage when necessary.
    ///
    /// # Errors
    /// Allocation failure leaves the buffer unchanged and drops `value`.
    pub fn try_push(&mut self, value: T) -> Result<(), AllocationError> {
        self.try_reserve(1)?;
        // SAFETY: Reservation ensures an uninitialized, aligned element slot.
        unsafe { self.pointer.as_ptr().add(self.length).write(value) };
        self.length += 1;
        Ok(())
    }

    /// Appends a copy of a slice after reserving its entire additional range.
    ///
    /// # Errors
    /// Overflow or allocation failure leaves all existing elements unchanged.
    pub fn try_extend_from_slice(&mut self, values: &[T]) -> Result<(), AllocationError>
    where
        T: Copy,
    {
        self.try_reserve(values.len())?;
        // SAFETY: The reserved destination does not overlap a source slice that
        // is valid independently of this exclusive borrow.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.pointer.as_ptr().add(self.length),
                values.len(),
            );
        };
        self.length += values.len();
        Ok(())
    }

    /// Removes and returns the last element without reducing capacity.
    pub fn pop(&mut self) -> Option<T> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;
        // SAFETY: The previous last initialized element is now removed and read once.
        Some(unsafe { self.pointer.as_ptr().add(self.length).read() })
    }

    /// Drops elements after `length` while retaining capacity.
    pub fn truncate(&mut self, length: usize) {
        if length >= self.length {
            return;
        }
        let removed = self.length - length;
        self.length = length;
        // SAFETY: The removed tail was initialized and is no longer owned by
        // the retained prefix; slice drop handles every tail element once.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.pointer.as_ptr().add(length),
                removed,
            ));
        };
    }

    /// Drops every element while retaining the allocation for reuse.
    pub fn clear(&mut self) {
        self.truncate(0);
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(formatter)
    }
}

impl<T: PartialEq> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for Buffer<T> {}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        // SAFETY: All elements in this initialized prefix are owned by the
        // buffer. RawBlock is subsequently dropped even if an element unwinds.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.pointer.as_ptr(),
                self.length,
            ));
        };
    }
}

impl<'a, T> IntoIterator for &'a Buffer<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Buffer<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Consuming iterator retaining callback storage until its final element drops.
pub struct IntoIter<T> {
    buffer: Buffer<T>,
    next: usize,
    end: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.next == self.end {
            return None;
        }
        // SAFETY: This element is initialized and has not yet been yielded.
        let value = unsafe { self.buffer.pointer.as_ptr().add(self.next).read() };
        self.next += 1;
        Some(value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Some(remaining))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.next == self.end {
            return None;
        }
        self.end -= 1;
        // SAFETY: This tail element is initialized and has not been yielded.
        Some(unsafe { self.buffer.pointer.as_ptr().add(self.end).read() })
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}
impl<T> core::iter::FusedIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // SAFETY: The remaining range is initialized and belongs only to this
        // iterator. The buffer length is zero, preventing duplicate destruction.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.buffer.pointer.as_ptr().add(self.next),
                self.end - self.next,
            ));
        }
    }
}

impl<T> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(mut self) -> Self::IntoIter {
        let end = self.length;
        self.length = 0;
        IntoIter {
            buffer: self,
            next: 0,
            end,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    #[derive(Default)]
    struct Context {
        live: Mutex<HashMap<usize, Layout>>,
        allocations: AtomicUsize,
        frees: AtomicUsize,
        resizes: AtomicUsize,
        fail_allocate: AtomicBool,
        fail_resize: AtomicBool,
    }

    impl Context {
        fn allocator(&self, resize: bool) -> Allocator {
            // SAFETY: Each test keeps Context alive until all owners are gone;
            // the callbacks synchronize accounting and use the system allocator.
            unsafe {
                Allocator::from_callbacks(
                    ptr::from_ref(self).cast_mut().cast(),
                    Some(allocate),
                    resize.then_some(reallocate),
                    Some(free),
                )
                .unwrap()
            }
        }

        fn live_count(&self) -> usize {
            self.live.lock().unwrap().len()
        }
    }

    unsafe extern "C" fn allocate(data: *mut c_void, length: u64, alignment: u64) -> *mut c_void {
        // SAFETY: Tests pass a live Context as callback user data.
        let context = unsafe { &*data.cast::<Context>() };
        context.allocations.fetch_add(1, Ordering::Relaxed);
        assert!(length > 0);
        assert!(alignment >= 16 && alignment.is_power_of_two());
        if context.fail_allocate.load(Ordering::Relaxed) {
            return ptr::null_mut();
        }
        let layout = Layout::from_size_align(
            usize::try_from(length).unwrap(),
            usize::try_from(alignment).unwrap(),
        )
        .unwrap();
        // SAFETY: The validated callback layout has nonzero size.
        let pointer = unsafe { std::alloc::alloc(layout) };
        if !pointer.is_null() {
            context
                .live
                .lock()
                .unwrap()
                .insert(pointer as usize, layout);
        }
        pointer.cast()
    }

    unsafe extern "C" fn reallocate(
        data: *mut c_void,
        pointer: *mut c_void,
        old_length: u64,
        new_length: u64,
        alignment: u64,
    ) -> *mut c_void {
        // SAFETY: Tests pass a live Context as callback user data.
        let context = unsafe { &*data.cast::<Context>() };
        context.resizes.fetch_add(1, Ordering::Relaxed);
        let mut live = context.live.lock().unwrap();
        let old = live[&(pointer as usize)];
        assert_eq!(old.size() as u64, old_length);
        assert_eq!(old.align() as u64, alignment);
        if context.fail_resize.load(Ordering::Relaxed) {
            return ptr::null_mut();
        }
        let layout = Layout::from_size_align(
            usize::try_from(new_length).unwrap(),
            usize::try_from(alignment).unwrap(),
        )
        .unwrap();
        // SAFETY: The pointer's original layout was retained in live; the new
        // size is nonzero and preserves its alignment.
        let replacement = unsafe { std::alloc::realloc(pointer.cast(), old, layout.size()) };
        if !replacement.is_null() {
            live.remove(&(pointer as usize));
            live.insert(replacement as usize, layout);
        }
        replacement.cast()
    }

    unsafe extern "C" fn free(data: *mut c_void, pointer: *mut c_void) {
        // SAFETY: Tests pass a live Context as callback user data.
        let context = unsafe { &*data.cast::<Context>() };
        let layout = context
            .live
            .lock()
            .unwrap()
            .remove(&(pointer as usize))
            .unwrap();
        context.frees.fetch_add(1, Ordering::Relaxed);
        // SAFETY: Accounting removes exactly this live original allocation.
        unsafe { std::alloc::dealloc(pointer.cast(), layout) };
    }

    struct DropValue<'a>(&'a AtomicUsize);

    impl Drop for DropValue<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn callback_combinations_are_validated_without_allocation() {
        let context = Context::default();
        let data = ptr::from_ref(&context).cast_mut().cast();
        // SAFETY: Valid callback combinations reference this live test context;
        // invalid combinations are rejected without calling any callback.
        unsafe {
            assert!(
                Allocator::from_callbacks(ptr::null_mut(), None, None, None)
                    .unwrap()
                    .is_system()
            );
            assert!(Allocator::from_callbacks(data, None, None, None).is_err());
            assert!(Allocator::from_callbacks(data, Some(allocate), None, None).is_err());
            assert!(Allocator::from_callbacks(data, None, None, Some(free)).is_err());
            assert!(Allocator::from_callbacks(data, None, Some(reallocate), None).is_err());
            assert!(
                !Allocator::from_callbacks(data, Some(allocate), None, Some(free))
                    .unwrap()
                    .is_system()
            );
        }
        assert_eq!(context.allocations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn preallocated_owner_publishes_and_recovers_without_allocating() {
        let context = Context::default();
        let drops = AtomicUsize::new(0);
        let guard = Owned::<DropValue<'_>>::try_new_uninit(context.allocator(false)).unwrap();
        assert_eq!(context.live_count(), 1);
        context.fail_allocate.store(true, Ordering::Relaxed);
        let owner = guard.write(DropValue(&drops));
        assert!(!owner.allocator().is_system());
        let pointer = owner.into_raw();
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        // SAFETY: This is the unconsumed pointer returned just above for this T.
        let recovered = unsafe { Owned::<DropValue<'_>>::from_raw(pointer) };
        assert_eq!(context.allocations.load(Ordering::Relaxed), 1);
        drop(recovered);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(context.live_count(), 0);
        assert_eq!(context.frees.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn abandoned_preallocation_releases_only_storage() {
        let context = Context::default();
        let guard = Owned::<DropValue<'_>>::try_new_uninit(context.allocator(false)).unwrap();
        drop(guard);
        assert_eq!(context.live_count(), 0);
        assert_eq!(context.frees.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn owner_failure_and_extraction_drop_values_exactly_once() {
        let context = Context::default();
        let drops = AtomicUsize::new(0);
        context.fail_allocate.store(true, Ordering::Relaxed);
        assert!(Owned::new(DropValue(&drops), context.allocator(false)).is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(context.live_count(), 0);
        context.fail_allocate.store(false, Ordering::Relaxed);
        let owner = Owned::new(DropValue(&drops), context.allocator(false)).unwrap();
        let value = owner.into_inner();
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(context.live_count(), 0);
        drop(value);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        assert_eq!(
            crate::Error::from(AllocationError).kind(),
            crate::error::ErrorKind::ResourceExhausted
        );
    }

    #[test]
    fn overaligned_values_and_buffer_elements_keep_their_alignment() {
        #[repr(align(128))]
        struct Aligned(u8);
        let context = Context::default();
        let owner = Owned::new(Aligned(7), context.allocator(false)).unwrap();
        assert_eq!(owner.as_ptr() as usize % 128, 0);
        assert_eq!(owner.0, 7);
        let mut buffer = Buffer::new(context.allocator(false));
        buffer.try_push(Aligned(9)).unwrap();
        assert_eq!(buffer.as_ptr() as usize % 128, 0);
        assert_eq!(buffer[0].0, 9);
        drop(owner);
        drop(buffer);
        assert_eq!(context.live_count(), 0);
    }

    #[test]
    fn allocate_copy_free_fallback_preserves_initialized_data() {
        let context = Context::default();
        let mut buffer = Buffer::try_with_capacity(2, context.allocator(false)).unwrap();
        buffer.try_extend_from_slice(&[17_u32, 29]).unwrap();
        buffer.try_push(43).unwrap();
        assert_eq!(buffer.as_slice(), &[17, 29, 43]);
        assert_eq!(context.allocations.load(Ordering::Relaxed), 2);
        assert_eq!(context.frees.load(Ordering::Relaxed), 1);
        assert_eq!(context.resizes.load(Ordering::Relaxed), 0);
        drop(buffer);
        assert_eq!(context.live_count(), 0);
        assert_eq!(context.frees.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn failed_fallback_and_resize_preserve_pointer_capacity_and_contents() {
        for use_resize in [false, true] {
            let context = Context::default();
            let mut buffer = Buffer::try_with_capacity(2, context.allocator(use_resize)).unwrap();
            buffer.try_extend_from_slice(&[5_u64, 8]).unwrap();
            let pointer = buffer.as_ptr();
            context.fail_allocate.store(true, Ordering::Relaxed);
            context.fail_resize.store(true, Ordering::Relaxed);
            assert_eq!(buffer.try_push(13), Err(AllocationError));
            assert_eq!(buffer.as_ptr(), pointer);
            assert_eq!(buffer.capacity(), 2);
            assert_eq!(buffer.as_slice(), &[5, 8]);
            assert_eq!(context.live_count(), 1);
            assert_eq!(context.frees.load(Ordering::Relaxed), 0);
            context.fail_allocate.store(false, Ordering::Relaxed);
            context.fail_resize.store(false, Ordering::Relaxed);
            buffer.try_push(21).unwrap();
            assert_eq!(buffer.as_slice(), &[5, 8, 21]);
            drop(buffer);
            assert_eq!(context.live_count(), 0);
        }
    }

    #[test]
    fn buffer_iteration_and_truncation_drop_all_elements_once() {
        let context = Context::default();
        let drops = AtomicUsize::new(0);
        let mut buffer = Buffer::new(context.allocator(false));
        for _ in 0..5 {
            buffer.try_push(DropValue(&drops)).unwrap();
        }
        buffer.truncate(4);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        drop(buffer.pop());
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        let mut iter = buffer.into_iter();
        assert_eq!(iter.len(), 3);
        drop(iter.next());
        drop(iter.next_back());
        assert_eq!(drops.load(Ordering::Relaxed), 4);
        drop(iter);
        assert_eq!(drops.load(Ordering::Relaxed), 5);
        assert_eq!(context.live_count(), 0);
    }

    #[test]
    fn shared_clone_is_allocation_free_and_final_owner_destroys_the_value() {
        let context = Context::default();
        let drops = AtomicUsize::new(0);
        let mut shared = Shared::new(DropValue(&drops), context.allocator(false)).unwrap();
        let clone = shared.clone();
        assert!(Shared::ptr_eq(&shared, &clone));
        assert!(Shared::get_mut(&mut shared).is_none());
        assert_eq!(Shared::strong_count(&shared), 2);
        assert_eq!(context.allocations.load(Ordering::Relaxed), 1);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                let another = clone.clone();
                drop(clone);
                drop(another);
            });
        });
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        assert_eq!(Shared::strong_count(&shared), 1);
        assert!(Shared::get_mut(&mut shared).is_some());
        drop(shared);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(context.live_count(), 0);
    }

    #[test]
    fn preallocated_shared_owner_publishes_without_another_allocation() {
        let context = Context::default();
        let drops = AtomicUsize::new(0);
        let unused = Shared::<DropValue<'_>>::try_new_uninit(context.allocator(false)).unwrap();
        drop(unused);
        assert_eq!(context.live_count(), 0);
        assert_eq!(drops.load(Ordering::Relaxed), 0);

        let guard = Shared::try_new_uninit(context.allocator(false)).unwrap();
        context.fail_allocate.store(true, Ordering::Relaxed);
        let shared = guard.write(DropValue(&drops));
        assert_eq!(context.allocations.load(Ordering::Relaxed), 2);
        drop(shared);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(context.live_count(), 0);
    }

    #[test]
    fn zero_sized_buffers_avoid_allocations_and_capacity_overflow_is_fallible() {
        let context = Context::default();
        context.fail_allocate.store(true, Ordering::Relaxed);
        let mut buffer = Buffer::new(context.allocator(false));
        for _ in 0..10 {
            buffer.try_push(()).unwrap();
        }
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.try_reserve(usize::MAX), Err(AllocationError));
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.into_iter().count(), 10);
        assert_eq!(context.allocations.load(Ordering::Relaxed), 0);
        let mut bytes = Buffer::<u64>::new(context.allocator(false));
        assert_eq!(bytes.try_reserve_exact(usize::MAX), Err(AllocationError));
        assert!(bytes.is_empty());
        assert_eq!(context.allocations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn default_allocator_supports_all_ownership_primitives() {
        let allocator = Allocator::default();
        let mut value = Owned::new(41, allocator).unwrap();
        *value += 1;
        assert_eq!(*value, 42);
        let shared = Shared::new(77, allocator).unwrap();
        assert_eq!(*shared.clone(), 77);
        let mut buffer = Buffer::new(allocator);
        for index in 0..100 {
            buffer.try_push(index).unwrap();
        }
        assert_eq!(buffer.len(), 100);
        assert_eq!(buffer[99], 99);
        buffer.clear();
        assert!(buffer.is_empty());
    }
}
