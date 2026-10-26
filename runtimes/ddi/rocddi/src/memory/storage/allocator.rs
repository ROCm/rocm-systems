// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fallible owner allocation with an optional scoped callback policy.

use core::alloc::Layout;
use core::cell::Cell;
use core::ffi::c_void;
use core::fmt;
use core::marker::PhantomData;
use core::ptr;
use core::ptr::NonNull;
use std::alloc::GlobalAlloc;
use std::rc::Rc;

/// C callback allocating a nonzero, suitably aligned uninitialized range.
pub type AllocateFn = unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void;
/// C callback that preserves bytes while replacing a host metadata block.
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

impl From<std::collections::TryReserveError> for crate::Error {
    fn from(_: std::collections::TryReserveError) -> Self {
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

/// Scoped allocator for fallible owner metadata storage.
///
/// Owners retain the allocator that created them, including after the scope
/// exits or the owner moves to another thread. Standard Rust vectors still use
/// the system allocator; [`super::OwnerVec`] uses an explicit allocator value.
#[derive(Clone, Copy, Debug)]
pub struct Allocator {
    /// Opaque state passed back to allocation callbacks.
    user_data: *mut c_void,
    /// Optional callback replacing the system allocator.
    allocate: Option<AllocateFn>,
    /// Optional callback used when a vector grows.
    resize: Option<ResizeFn>,
    /// Callback paired with `allocate` for deallocation.
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

    pub(super) fn current() -> Self {
        CURRENT.with(Cell::get)
    }

    /// Installs this allocator for owner allocations on the current thread.
    /// Existing owners retain their own allocator for release.
    #[must_use]
    pub fn scope(self) -> AllocatorScope {
        AllocatorScope {
            previous: CURRENT.with(|current| current.replace(self)),
            _not_send: PhantomData,
        }
    }

    #[cfg(any(test, feature = "allocator-test-hooks"))]
    pub(super) fn probe(self, layout: Layout) -> Result<(), AllocationError> {
        if !self.is_system() && layout.size() != 0 {
            drop(RawBlock::new_with_allocator(layout, self)?);
        }
        Ok(())
    }

    /// Copies callbacks for owner metadata allocation.
    ///
    /// Null data and absent callbacks select the default allocator. Otherwise
    /// both `allocate` and `free` are required.
    ///
    /// # Errors
    /// Returns [`InvalidAllocator`] for an incomplete callback combination.
    ///
    /// # Safety
    /// The callbacks and `user_data` must remain valid until every copy of this
    /// allocator and every allocation made through it is released. Callbacks
    /// must be thread-safe and must not unwind. Allocate must return NULL
    /// or writable, nonoverlapping storage satisfying the supplied size and
    /// alignment. Free must accept allocations produced by these callbacks and
    /// release them infallibly.
    pub unsafe fn from_callbacks(
        user_data: *mut c_void,
        allocate: Option<AllocateFn>,
        free: Option<FreeFn>,
    ) -> Result<Self, InvalidAllocator> {
        // SAFETY: The same callback obligations apply without a resize hook.
        unsafe { Self::from_callbacks_with_resize(user_data, allocate, None, free) }
    }

    /// Copies allocate, resize, and release callbacks for frontend metadata.
    ///
    /// # Errors
    /// Rejects incomplete callback combinations.
    ///
    /// # Safety
    /// The callback state and code must remain valid through final owner
    /// release. Resize preserves the old allocation on failure, and all
    /// callbacks are thread-safe, aligned, and do not unwind.
    pub unsafe fn from_callbacks_with_resize(
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

    pub(super) fn resize(
        self,
        pointer: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<Option<NonNull<u8>>, AllocationError> {
        let Some(resize) = self.resize else {
            return Ok(None);
        };
        // SAFETY: The callback owns this block and promises to leave it live
        // and unchanged on null return.
        let resized = unsafe {
            resize(
                self.user_data,
                pointer.as_ptr().cast(),
                old_layout.size() as u64,
                new_layout.size() as u64,
                new_layout.align() as u64,
            )
        };
        NonNull::new(resized.cast())
            .map(Some)
            .ok_or(AllocationError)
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
        let pointer = NonNull::new(pointer).ok_or(AllocationError)?;
        if pointer.as_ptr().addr() % layout.align() != 0 {
            // SAFETY: Even a misaligned callback result belongs to the
            // callback allocator and must be returned to its matching free.
            unsafe { self.deallocate(pointer, layout) };
            return Err(AllocationError);
        }
        Ok(pointer)
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
}

thread_local! {
    static CURRENT: Cell<Allocator> = const { Cell::new(Allocator::system()) };
}

/// Restores the previous owner allocator when the current scope ends.
pub struct AllocatorScope {
    /// Prior thread-local policy restored when this guard drops.
    previous: Allocator,
    /// Keeps the guard on the thread whose allocator slot it replaced.
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for AllocatorScope {
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.previous));
    }
}

#[cfg(any(test, feature = "allocator-test-hooks"))]
pub(super) fn probe(layout: Layout) -> Result<(), AllocationError> {
    Allocator::current().probe(layout)
}

// Metadata storage has at least 16-byte alignment on supported targets.
// Larger alignments required by T are preserved.
fn normalized_layout(layout: Layout) -> Result<Layout, AllocationError> {
    Layout::from_size_align(layout.size().max(1), layout.align().max(16))
        .map_err(|_| AllocationError)
}

/// One fallibly allocated metadata block and the policy used to release it.
pub(super) struct RawBlock {
    /// Start of the aligned allocation.
    pub(super) pointer: NonNull<u8>,
    /// Exact layout required for deallocation.
    pub(super) layout: Layout,
    /// Allocator retained across changes to the thread-local scope.
    pub(super) allocator: Allocator,
}

impl RawBlock {
    pub(super) fn new(layout: Layout) -> Result<Self, AllocationError> {
        Self::new_with_allocator(layout, Allocator::current())
    }

    pub(super) fn new_with_allocator(
        layout: Layout,
        allocator: Allocator,
    ) -> Result<Self, AllocationError> {
        let layout = normalized_layout(layout)?;
        let pointer = allocator.allocate(layout)?;
        Ok(Self {
            pointer,
            layout,
            allocator,
        })
    }

    pub(super) fn try_resize(&mut self, requested: Layout) -> Result<bool, AllocationError> {
        let layout = normalized_layout(requested)?;
        let Some(pointer) = self.allocator.resize(self.pointer, self.layout, layout)? else {
            return Ok(false);
        };
        self.pointer = pointer;
        self.layout = layout;
        Ok(true)
    }
}

impl Drop for RawBlock {
    fn drop(&mut self) {
        // SAFETY: The block exclusively owns this allocation and original layout.
        unsafe {
            self.allocator.deallocate(self.pointer, self.layout);
        }
    }
}
