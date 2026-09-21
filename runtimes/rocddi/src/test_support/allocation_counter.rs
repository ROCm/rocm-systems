//! Counts allocations made by one test thread. Tracking is scoped to a warmed
//! operation so parallel tests, fixture construction, and diagnostics cannot
//! mask a regression in a supposedly allocation-free query.

#![allow(
    unsafe_code,
    reason = "test-only allocator forwards unchanged to System and observes calls"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counter;

thread_local! {
    static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record() {
    let _ = COUNT.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}

// SAFETY: Every allocation and deallocation is delegated with its original
// layout and pointer. The counter owns no memory and cannot change alignment,
// identity, initialization, or the allocator's failure behavior.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: Forward the caller's valid allocator request unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: Forward the caller's valid allocator request unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        // SAFETY: This allocator obtained the pointer from System, and forwards
        // the original layout and requested size without modifying either.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout came from the same delegated allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counter = Counter;

pub(crate) fn allocations(action: impl FnOnce()) -> usize {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            COUNT.set(None);
        }
    }
    assert!(COUNT.get().is_none(), "allocation measurements cannot nest");
    COUNT.set(Some(0));
    let stop = Stop;
    action();
    let count = COUNT.get().unwrap();
    drop(stop);
    count
}
