// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fallible system allocation and test-only callback injection.

use core::alloc::Layout;
#[cfg(any(test, feature = "allocator-test-hooks"))]
use core::cell::Cell;
#[cfg(any(test, feature = "allocator-test-hooks"))]
use core::ffi::c_void;
use core::fmt;
#[cfg(any(test, feature = "allocator-test-hooks"))]
use core::marker::PhantomData;
#[cfg(any(test, feature = "allocator-test-hooks"))]
use core::ptr;
use core::ptr::NonNull;
use std::alloc::GlobalAlloc;
#[cfg(any(test, feature = "allocator-test-hooks"))]
use std::rc::Rc;

/// C callback allocating a nonzero, suitably aligned uninitialized range.
#[cfg(any(test, feature = "allocator-test-hooks"))]
pub type AllocateFn = unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void;
/// Infallible C callback releasing storage returned by the same allocator.
#[cfg(any(test, feature = "allocator-test-hooks"))]
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

/// Callback fields do not describe either a complete test allocator or all zeros.
#[cfg(any(test, feature = "allocator-test-hooks"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidAllocator;

#[cfg(any(test, feature = "allocator-test-hooks"))]
impl fmt::Display for InvalidAllocator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("host allocator requires allocate and free callbacks")
    }
}

#[cfg(any(test, feature = "allocator-test-hooks"))]
impl std::error::Error for InvalidAllocator {}

/// Test-only allocator for fallible internal metadata storage.
///
/// Production builds use [`std::alloc::System`] directly. Test builds can
/// inject allocation callbacks to verify failure cleanup paths.
#[derive(Clone, Copy, Debug)]
#[cfg(any(test, feature = "allocator-test-hooks"))]
pub struct Allocator {
    user_data: *mut c_void,
    allocate: Option<AllocateFn>,
    free: Option<FreeFn>,
}

// SAFETY: Callback construction requires thread-safe callbacks and shared state.
#[cfg(any(test, feature = "allocator-test-hooks"))]
unsafe impl Send for Allocator {}
// SAFETY: The same construction contract permits concurrent calls with user_data.
#[cfg(any(test, feature = "allocator-test-hooks"))]
unsafe impl Sync for Allocator {}

#[cfg(any(test, feature = "allocator-test-hooks"))]
impl Default for Allocator {
    fn default() -> Self {
        Self::system()
    }
}

#[cfg(any(test, feature = "allocator-test-hooks"))]
impl Allocator {
    /// Returns the default host allocator without performing any allocation.
    #[must_use]
    pub const fn system() -> Self {
        Self {
            user_data: ptr::null_mut(),
            allocate: None,
            free: None,
        }
    }

    pub(super) fn current() -> Self {
        CURRENT.with(Cell::get)
    }

    /// Installs this test allocator for allocations on the current thread.
    /// Existing owners retain their own allocator for release.
    #[must_use]
    pub fn scope(self) -> AllocatorScope {
        AllocatorScope {
            previous: CURRENT.with(|current| current.replace(self)),
            _not_send: PhantomData,
        }
    }

    pub(super) fn probe(self, layout: Layout) -> Result<(), AllocationError> {
        if !self.is_system() && layout.size() != 0 {
            drop(RawBlock::new_with_allocator(layout, self)?);
        }
        Ok(())
    }

    /// Copies callbacks for metadata failure injection in tests.
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
        if allocate.is_some() && free.is_some() {
            Ok(Self {
                user_data,
                allocate,
                free,
            })
        } else if user_data.is_null() && allocate.is_none() && free.is_none() {
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
}

#[cfg(any(test, feature = "allocator-test-hooks"))]
thread_local! {
    static CURRENT: Cell<Allocator> = const { Cell::new(Allocator::system()) };
}

/// Restores the previous allocation fault hook when the current scope ends.
#[cfg(any(test, feature = "allocator-test-hooks"))]
pub struct AllocatorScope {
    previous: Allocator,
    // A scope must be dropped on the thread whose CURRENT slot it replaced.
    _not_send: PhantomData<Rc<()>>,
}

#[cfg(any(test, feature = "allocator-test-hooks"))]
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

pub(super) struct RawBlock {
    pub(super) pointer: NonNull<u8>,
    pub(super) layout: Layout,
    #[cfg(any(test, feature = "allocator-test-hooks"))]
    pub(super) allocator: Allocator,
}

impl RawBlock {
    pub(super) fn new(layout: Layout) -> Result<Self, AllocationError> {
        #[cfg(any(test, feature = "allocator-test-hooks"))]
        {
            Self::new_with_allocator(layout, Allocator::current())
        }
        #[cfg(not(any(test, feature = "allocator-test-hooks")))]
        {
            let layout = normalized_layout(layout)?;
            // SAFETY: Normalized layouts always have a nonzero size.
            let pointer =
                NonNull::new(unsafe { std::alloc::System.alloc(layout) }).ok_or(AllocationError)?;
            Ok(Self { pointer, layout })
        }
    }

    #[cfg(any(test, feature = "allocator-test-hooks"))]
    fn new_with_allocator(layout: Layout, allocator: Allocator) -> Result<Self, AllocationError> {
        let layout = normalized_layout(layout)?;
        let pointer = allocator.allocate(layout)?;
        Ok(Self {
            pointer,
            layout,
            allocator,
        })
    }
}

impl Drop for RawBlock {
    fn drop(&mut self) {
        // SAFETY: The block exclusively owns this allocation and original layout.
        #[cfg(any(test, feature = "allocator-test-hooks"))]
        unsafe {
            self.allocator.deallocate(self.pointer, self.layout);
        }
        #[cfg(not(any(test, feature = "allocator-test-hooks")))]
        unsafe {
            std::alloc::System.dealloc(self.pointer.as_ptr(), self.layout);
        }
    }
}
