// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fallible contiguous metadata storage retaining its callback allocator.

use super::TryPush;
use super::allocator::{AllocationError, Allocator, RawBlock};
use core::alloc::Layout;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};

/// A fallible vector for frontend metadata using one retained allocator.
pub struct OwnerVec<T> {
    block: Option<RawBlock>,
    len: usize,
    capacity: usize,
    allocator: Allocator,
    owns: PhantomData<T>,
}

// SAFETY: The allocator callbacks must be thread-safe and ownership moves
// only when T can be sent to another thread.
unsafe impl<T: Send> Send for OwnerVec<T> {}
// SAFETY: Immutable slices are safe to share when T is Sync.
unsafe impl<T: Sync> Sync for OwnerVec<T> {}

impl<T> OwnerVec<T> {
    /// Creates an empty vector with the supplied instance allocator.
    #[must_use]
    pub const fn with_allocator(allocator: Allocator) -> Self {
        Self {
            block: None,
            len: 0,
            capacity: if core::mem::size_of::<T>() == 0 {
                usize::MAX
            } else {
                0
            },
            allocator,
            owns: PhantomData,
        }
    }

    /// Reserves a requested capacity before acquiring native resources.
    ///
    /// # Errors
    /// Returns an allocation error on overflow or exhaustion.
    pub fn try_with_allocator(
        allocator: Allocator,
        capacity: usize,
    ) -> Result<Self, AllocationError> {
        let mut values = Self::with_allocator(allocator);
        values.try_reserve_exact(capacity)?;
        Ok(values)
    }

    fn pointer(&self) -> NonNull<T> {
        self.block
            .as_ref()
            .map_or(NonNull::dangling(), |block| block.pointer.cast::<T>())
    }

    /// Returns the number of initialized elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this vector is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the currently reserved element capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Borrows the initialized prefix.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Mutably borrows the initialized prefix.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Reserves at least the specified number of further elements.
    ///
    /// # Errors
    /// Returns an allocation error before mutation on overflow or exhaustion.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocationError> {
        let required = self.len.checked_add(additional).ok_or(AllocationError)?;
        if required <= self.capacity {
            return Ok(());
        }
        let capacity = required.max(self.capacity.saturating_mul(2)).max(4);
        self.grow(capacity)
    }

    /// Reserves exactly enough space for the requested additional elements.
    ///
    /// # Errors
    /// Returns an allocation error before mutation on overflow or exhaustion.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocationError> {
        let required = self.len.checked_add(additional).ok_or(AllocationError)?;
        if required <= self.capacity {
            return Ok(());
        }
        self.grow(required)
    }

    fn grow(&mut self, capacity: usize) -> Result<(), AllocationError> {
        let layout = Layout::array::<T>(capacity).map_err(|_| AllocationError)?;
        if let Some(block) = &mut self.block {
            if block.try_resize(layout)? {
                self.capacity = capacity;
                return Ok(());
            }
        }
        let new_block = RawBlock::new_with_allocator(layout, self.allocator)?;
        let new_pointer = new_block.pointer.cast::<T>().as_ptr();
        let old_pointer = self.pointer().as_ptr();
        for index in 0..self.len {
            // SAFETY: Both blocks cover index; read moves each live T exactly
            // once and the old block is then released without dropping T.
            unsafe { new_pointer.add(index).write(old_pointer.add(index).read()) };
        }
        self.block = Some(new_block);
        self.capacity = capacity;
        Ok(())
    }

    /// Removes and returns the final element, when present.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: The former final element was initialized and is now outside
        // the live prefix.
        Some(unsafe { self.pointer().as_ptr().add(self.len).read() })
    }
}

impl<T> TryPush<T> for OwnerVec<T> {
    fn try_push(&mut self, value: T) -> Result<(), AllocationError> {
        self.try_reserve(1)?;
        // SAFETY: The capacity covers the uninitialized slot at len.
        unsafe { self.pointer().as_ptr().add(self.len).write(value) };
        self.len += 1;
        Ok(())
    }
}

impl<T> Deref for OwnerVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // SAFETY: Exactly len elements form the initialized prefix.
        unsafe { core::slice::from_raw_parts(self.pointer().as_ptr(), self.len) }
    }
}

impl<T> DerefMut for OwnerVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: This exclusive borrow owns the initialized prefix.
        unsafe { core::slice::from_raw_parts_mut(self.pointer().as_ptr(), self.len) }
    }
}

impl<'a, T> IntoIterator for &'a OwnerVec<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut OwnerVec<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> Drop for OwnerVec<T> {
    fn drop(&mut self) {
        // SAFETY: The initialized prefix is uniquely owned and must be
        // dropped once before the allocation block is released.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.pointer().as_ptr(),
                self.len,
            ));
        }
    }
}
