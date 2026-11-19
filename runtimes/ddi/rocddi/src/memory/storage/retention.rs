// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Explicit release for owners whose native reachability may be uncertain.

/// Suppresses an owner's destructor until the caller explicitly releases it.
///
/// Normal scope exit and unwinding do not run `T`'s destructor, retaining the
/// resources whose release depends on it. After native cleanup proves release
/// is safe, [`Self::into_inner`] returns the owner for normal destruction.
/// The wrapper performs no cleanup and does not establish that release proof.
///
/// This does not pin `T`, preserve its inline storage after scope exit, or
/// extend borrowed lifetimes. Externally reachable backing must already have
/// suitable storage and ownership independent of this wrapper's address.
pub struct RetainOnDrop<T> {
    value: std::mem::ManuallyDrop<T>,
}

impl<T> RetainOnDrop<T> {
    /// Takes responsibility for retaining an owner through uncertain cleanup.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self {
            value: std::mem::ManuallyDrop::new(value),
        }
    }

    /// Returns the owner after the caller has established that it may be dropped.
    #[must_use]
    pub fn into_inner(self) -> T {
        std::mem::ManuallyDrop::into_inner(self.value)
    }
}

impl<T> std::ops::Deref for RetainOnDrop<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> std::ops::DerefMut for RetainOnDrop<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::RetainOnDrop;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn uncertain_owners_survive_normal_and_unwinding_scope_exit() {
        struct Owner<'a>(&'a AtomicUsize);
        impl Drop for Owner<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = AtomicUsize::new(0);
        drop(RetainOnDrop::new(Owner(&drops)));
        assert_eq!(drops.load(Ordering::Relaxed), 0);

        let result = std::panic::catch_unwind(|| {
            let _retained = RetainOnDrop::new(Owner(&drops));
            panic!("injected retirement unwind");
        });
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 0);

        drop(RetainOnDrop::new(Owner(&drops)).into_inner());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
