// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use core::alloc::Layout;
use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

#[derive(Default)]
struct Context {
    live: Mutex<HashMap<usize, Layout>>,
    allocations: AtomicUsize,
    frees: AtomicUsize,
    fail_allocate: AtomicBool,
}

impl Context {
    fn allocator(&self) -> Allocator {
        // SAFETY: Each test keeps Context alive until all owners are gone;
        // the callbacks synchronize accounting and use the system allocator.
        unsafe {
            Allocator::from_callbacks(
                ptr::from_ref(self).cast_mut().cast(),
                Some(allocate),
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

struct PanicValue<'a> {
    drops: &'a AtomicUsize,
    panic: bool,
}

impl Drop for PanicValue<'_> {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
        assert!(!self.panic, "intentional destructor panic");
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
            Allocator::from_callbacks(ptr::null_mut(), None, None)
                .unwrap()
                .is_system()
        );
        assert!(Allocator::from_callbacks(data, None, None).is_err());
        assert!(Allocator::from_callbacks(data, Some(allocate), None).is_err());
        assert!(Allocator::from_callbacks(data, None, Some(free)).is_err());
        assert!(
            !Allocator::from_callbacks(data, Some(allocate), Some(free))
                .unwrap()
                .is_system()
        );
    }
    assert_eq!(context.allocations.load(Ordering::Relaxed), 0);
}

#[test]
fn preallocated_owner_publishes_and_recovers_without_allocating() {
    let context = Context::default();
    let _scope = context.allocator().scope();
    let drops = AtomicUsize::new(0);
    let guard = Owned::<DropValue<'_>>::try_new_uninit().unwrap();
    assert_eq!(context.live_count(), 1);
    context.fail_allocate.store(true, Ordering::Relaxed);
    let owner = guard.write(DropValue(&drops));
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
    let _scope = context.allocator().scope();
    let guard = Owned::<DropValue<'_>>::try_new_uninit().unwrap();
    drop(guard);
    assert_eq!(context.live_count(), 0);
    assert_eq!(context.frees.load(Ordering::Relaxed), 1);
}

#[test]
fn owner_failure_and_extraction_drop_values_exactly_once() {
    let context = Context::default();
    let _scope = context.allocator().scope();
    let drops = AtomicUsize::new(0);
    context.fail_allocate.store(true, Ordering::Relaxed);
    assert!(Owned::new(DropValue(&drops)).is_err());
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(context.live_count(), 0);
    context.fail_allocate.store(false, Ordering::Relaxed);
    let owner = Owned::new(DropValue(&drops)).unwrap();
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
fn panicking_destructors_release_storage_without_double_drop() {
    let context = Context::default();
    let _scope = context.allocator().scope();
    let drops = AtomicUsize::new(0);
    let value = |panic| PanicValue {
        drops: &drops,
        panic,
    };

    let owner = Owned::new(value(true)).unwrap();
    assert!(std::panic::catch_unwind(|| drop(owner)).is_err());
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(context.live_count(), 0);

    let shared = Shared::new(value(true)).unwrap();
    let clone = shared.clone();
    drop(clone);
    assert!(std::panic::catch_unwind(|| drop(shared)).is_err());
    assert_eq!(drops.load(Ordering::Relaxed), 2);
    assert_eq!(context.live_count(), 0);
}

#[test]
fn overaligned_owner_keeps_its_alignment() {
    #[repr(align(128))]
    struct Aligned(u8);
    let context = Context::default();
    let _scope = context.allocator().scope();
    let owner = Owned::new(Aligned(7)).unwrap();
    assert_eq!(owner.as_ptr() as usize % 128, 0);
    assert_eq!(owner.0, 7);
    drop(owner);
    assert_eq!(context.live_count(), 0);
}

#[test]
fn shared_clone_is_allocation_free_and_final_owner_destroys_the_value() {
    let context = Context::default();
    let _scope = context.allocator().scope();
    let drops = AtomicUsize::new(0);
    let mut shared = Shared::new(DropValue(&drops)).unwrap();
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
fn shared_callbacks_release_once_after_concurrent_clone_and_drop() {
    let context = Context::default();
    let _scope = context.allocator().scope();
    let drops = AtomicUsize::new(0);
    let mut shared = Shared::new(DropValue(&drops)).unwrap();
    let barrier = std::sync::Barrier::new(9);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let worker = shared.clone();
            let ready = &barrier;
            scope.spawn(move || {
                ready.wait();
                for _ in 0..128 {
                    let transient = worker.clone();
                    assert!(Shared::ptr_eq(&worker, &transient));
                    drop(transient);
                }
            });
        }
        barrier.wait();
        for _ in 0..128 {
            drop(shared.clone());
        }
    });
    assert_eq!(Shared::strong_count(&shared), 1);
    assert_eq!(drops.load(Ordering::Acquire), 0);
    assert!(Shared::get_mut(&mut shared).is_some());

    // All workers hold a strong reference before the original owner is
    // dropped. The final release may now happen on any worker thread.
    let final_release = std::sync::Barrier::new(9);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let worker = shared.clone();
            let ready = &final_release;
            scope.spawn(move || {
                ready.wait();
                for _ in 0..32 {
                    drop(worker.clone());
                }
            });
        }
        final_release.wait();
        drop(shared);
    });
    assert_eq!(drops.load(Ordering::Acquire), 1);
    assert_eq!(context.live_count(), 0);
}

#[test]
fn preallocated_shared_owner_publishes_without_another_allocation() {
    let context = Context::default();
    let _scope = context.allocator().scope();
    let drops = AtomicUsize::new(0);
    let unused = Shared::<DropValue<'_>>::try_new_uninit().unwrap();
    drop(unused);
    assert_eq!(context.live_count(), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 0);

    let guard = Shared::try_new_uninit().unwrap();
    context.fail_allocate.store(true, Ordering::Relaxed);
    let shared = guard.write(DropValue(&drops));
    assert_eq!(context.allocations.load(Ordering::Relaxed), 2);
    drop(shared);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(context.live_count(), 0);
}

#[test]
fn system_storage_supports_owners_and_vectors() {
    let mut value = Owned::new(41).unwrap();
    *value += 1;
    assert_eq!(*value, 42);
    let shared = Shared::new(77).unwrap();
    assert_eq!(*shared.clone(), 77);
    let mut buffer = try_vec_with_capacity(100).unwrap();
    for index in 0..100 {
        buffer.try_push(index).unwrap();
    }
    assert_eq!(buffer.len(), 100);
    assert_eq!(buffer[99], 99);
    buffer.clear();
    assert!(buffer.is_empty());
}

#[test]
fn vector_growth_failure_preserves_initialized_elements() {
    let context = Context::default();
    let _scope = context.allocator().scope();
    let mut values = try_vec_with_capacity(1).unwrap();
    values.try_push(7_u64).unwrap();
    context.fail_allocate.store(true, Ordering::Relaxed);
    assert_eq!(values.try_push(9), Err(AllocationError));
    assert_eq!(values.as_slice(), &[7]);
    assert_eq!(values.len(), 1);
    assert_eq!(context.live_count(), 0);
}
