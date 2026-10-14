// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Allocator-backed unique and shared value owners.

#[cfg(any(test, feature = "allocator-test-hooks"))]
use super::allocator::Allocator;
use super::allocator::{AllocationError, RawBlock};
use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit, align_of, offset_of, size_of};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicUsize, Ordering, fence};

#[repr(C)]
struct OwnerSlot<T> {
    #[cfg(any(test, feature = "allocator-test-hooks"))]
    allocator: Allocator,
    value: MaybeUninit<T>,
}

/// Unique ownership of one value in fallible metadata storage.
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
    pub fn new(value: T) -> Result<Self, AllocationError> {
        Ok(Self::try_new_uninit()?.write(value))
    }

    /// Reserves owner storage before acquiring a native resource.
    ///
    /// Dropping the returned guard releases storage without dropping a T.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if storage is unavailable.
    pub fn try_new_uninit() -> Result<UninitOwned<T>, AllocationError> {
        let block = RawBlock::new(Layout::new::<OwnerSlot<T>>())?;
        let slot = block.pointer.cast::<OwnerSlot<T>>();
        // SAFETY: The allocation has sufficient size/alignment for OwnerSlot<T>.
        unsafe {
            slot.as_ptr().write(OwnerSlot {
                #[cfg(any(test, feature = "allocator-test-hooks"))]
                allocator: block.allocator,
                value: MaybeUninit::uninit(),
            });
        }
        Ok(UninitOwned { slot, block })
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

    /// Recovers an owner from a previously published pointer.
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
            // Reconstruct the same normalized layout used by RawBlock::new.
            // OwnerSlot<()> has zero size outside allocator-hook builds.
            layout: unsafe {
                Layout::from_size_align_unchecked(
                    size_of::<OwnerSlot<T>>().max(1),
                    align_of::<OwnerSlot<T>>().max(16),
                )
            },
            #[cfg(any(test, feature = "allocator-test-hooks"))]
            allocator: unsafe { self.slot.as_ref().allocator },
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
    #[allow(dead_code, reason = "the guard owns storage until publication or drop")]
    block: RawBlock,
}

// SAFETY: No T exists yet, and the allocator permits transfer between threads.
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
}

#[repr(C)]
struct SharedSlot<T> {
    strong: AtomicUsize,
    #[cfg(any(test, feature = "allocator-test-hooks"))]
    allocator: Allocator,
    value: T,
}

/// Atomic shared ownership of one metadata value.
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
    pub fn new(value: T) -> Result<Self, AllocationError> {
        Ok(Self::try_new_uninit()?.write(value))
    }

    /// Reserves the shared owner before acquiring a native resource.
    ///
    /// Dropping the returned guard releases storage without dropping a T.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if storage is unavailable.
    pub fn try_new_uninit() -> Result<UninitShared<T>, AllocationError> {
        let block = RawBlock::new(Layout::new::<SharedSlot<T>>())?;
        let slot = block.pointer.cast::<SharedSlot<T>>();
        Ok(UninitShared { slot, block })
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
    #[allow(dead_code, reason = "the guard owns storage until publication or drop")]
    block: RawBlock,
}

// SAFETY: No T exists yet, and the allocator permits transfer between threads.
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
                #[cfg(any(test, feature = "allocator-test-hooks"))]
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
                #[cfg(any(test, feature = "allocator-test-hooks"))]
                allocator: self.slot.as_ref().allocator,
            };
            ptr::drop_in_place(ptr::addr_of_mut!((*self.slot.as_ptr()).value));
        }
    }
}
