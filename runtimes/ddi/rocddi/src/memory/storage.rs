// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fallible storage for runtime owners and callback-backed metadata vectors.
//!
//! Owner storage can use a scoped callback allocator. Standard vectors retain
//! the system allocator; test hooks can inject their allocation failures.
//! Constructors that will acquire
//! native resources should reserve their storage first with
//! [`Owned::try_new_uninit`] and publish the completed owner with
//! [`UninitOwned::write`], which performs no allocation. This storage manages
//! Rust ownership records; it does not establish agent access to their data.

#![allow(unsafe_code)]

mod allocator;
mod owners;
mod vector;

pub use allocator::AllocationError;
pub use allocator::{AllocateFn, Allocator, FreeFn, InvalidAllocator, ResizeFn};
pub use owners::{Owned, Shared, UninitOwned, UninitShared};
pub use vector::OwnerVec;

/// Creates a standard vector after fallibly reserving its initial capacity.
///
/// # Errors
/// Returns [`AllocationError`] if the requested capacity cannot be reserved.
pub fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, AllocationError> {
    let mut values = Vec::new();
    #[cfg(any(test, feature = "allocator-test-hooks"))]
    allocator::probe(core::alloc::Layout::array::<T>(capacity).map_err(|_| AllocationError)?)?;
    values
        .try_reserve_exact(capacity)
        .map_err(|_| AllocationError)?;
    Ok(values)
}

/// Reserves one slot before pushing, preserving a fallible append boundary.
///
/// # Errors
/// Returns [`AllocationError`] without modifying `values` if growth fails.
pub trait TryPush<T> {
    /// Reserves one element and appends it without aborting on capacity failure.
    ///
    /// # Errors
    /// Returns [`AllocationError`] if growth fails.
    fn try_push(&mut self, value: T) -> Result<(), AllocationError>;
}

impl<T> TryPush<T> for Vec<T> {
    fn try_push(&mut self, value: T) -> Result<(), AllocationError> {
        #[cfg(any(test, feature = "allocator-test-hooks"))]
        if self.len() == self.capacity() {
            allocator::probe(core::alloc::Layout::new::<T>())?;
        }
        self.try_reserve(1).map_err(|_| AllocationError)?;
        self.push(value);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
