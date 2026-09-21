//! Test callbacks deliberately bypass the Rust global allocator so metadata
//! provenance and incidental standard-library allocations can be measured apart.
#![allow(unsafe_code, clippy::unwrap_used)]
use crate::host_storage::Allocator;
use core::ffi::c_void;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Clone, Copy)]
struct AllocationHeader {
    base: *mut u8,
    layout: Layout,
}

#[derive(Default)]
pub(crate) struct State {
    pub allocations: AtomicUsize,
    pub frees: AtomicUsize,
    pub fail: AtomicBool,
}
impl State {
    /// The returned allocator must not outlive this stable-address test state.
    pub(crate) unsafe fn allocator(&self) -> Allocator {
        // SAFETY: Test caller retains this state through all owner destruction.
        unsafe {
            Allocator::from_callbacks(
                std::ptr::from_ref(self).cast_mut().cast(),
                Some(allocate),
                None,
                Some(release),
            )
            .unwrap()
        }
    }
}

fn allocation_layout(size: u64, alignment: u64) -> Option<(Layout, usize)> {
    let size = usize::try_from(size).ok()?;
    let requested_alignment = usize::try_from(alignment).ok()?;
    if size == 0 || !requested_alignment.is_power_of_two() {
        return None;
    }
    let allocation_alignment = requested_alignment.max(align_of::<AllocationHeader>());
    let prefix = size_of::<AllocationHeader>().checked_add(allocation_alignment - 1)?
        & !(allocation_alignment - 1);
    let total = prefix.checked_add(size)?;
    let layout = Layout::from_size_align(total, allocation_alignment).ok()?;
    Some((layout, prefix))
}

unsafe extern "C" fn allocate(data: *mut c_void, size: u64, alignment: u64) -> *mut c_void {
    // SAFETY: All test callbacks receive the stable State supplied above.
    let state = unsafe { &*data.cast::<State>() };
    state.allocations.fetch_add(1, Ordering::Relaxed);
    if state.fail.load(Ordering::Relaxed) {
        return std::ptr::null_mut();
    }
    let Some((layout, prefix)) = allocation_layout(size, alignment) else {
        return std::ptr::null_mut();
    };
    // SAFETY: The validated layout is nonzero and comes from Layout.
    let base = unsafe { System.alloc(layout) };
    if base.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: prefix is within the allocation and leaves `size` payload bytes.
    // It is a multiple of the requested alignment. The header is immediately
    // before the payload and has sufficient alignment for AllocationHeader.
    let pointer = unsafe { base.add(prefix) };
    unsafe {
        std::ptr::write_unaligned(
            pointer
                .sub(size_of::<AllocationHeader>())
                .cast::<AllocationHeader>(),
            AllocationHeader { base, layout },
        );
    }
    pointer.cast()
}
unsafe extern "C" fn release(data: *mut c_void, pointer: *mut c_void) {
    // SAFETY: All test callbacks receive the stable State supplied above.
    let state = unsafe { &*data.cast::<State>() };
    state.frees.fetch_add(1, Ordering::Relaxed);
    if pointer.is_null() {
        return;
    }
    let pointer = pointer.cast::<u8>();
    // SAFETY: Non-null pointers accepted here were returned by allocate. The
    // adjacent header retains the exact base pointer and Layout used by System.
    let header = unsafe {
        std::ptr::read_unaligned(
            pointer
                .sub(size_of::<AllocationHeader>())
                .cast::<AllocationHeader>(),
        )
    };
    // SAFETY: The header records this live allocation's original System layout.
    unsafe { System.dealloc(header.base, header.layout) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(state: &State) -> *mut c_void {
        std::ptr::from_ref(state).cast_mut().cast()
    }

    #[test]
    fn allocations_satisfy_requested_alignment_without_the_global_allocator() {
        for alignment in [16, 64, 4096] {
            let state = State::default();
            let mut pointer = std::ptr::null_mut();
            let global_allocations = crate::test_support::allocation_counter::allocations(|| {
                // SAFETY: State remains live through release below.
                pointer = unsafe { allocate(data(&state), 257, alignment) };
            });
            assert_eq!(global_allocations, 0);
            assert!(!pointer.is_null());
            assert_eq!(pointer.addr() % usize::try_from(alignment).unwrap(), 0);
            // SAFETY: The allocation contains 257 writable payload bytes.
            unsafe {
                pointer.cast::<u8>().write(0xa5);
                pointer.cast::<u8>().add(256).write(0x5a);
                release(data(&state), pointer);
            }
            assert_eq!(state.allocations.load(Ordering::Relaxed), 1);
            assert_eq!(state.frees.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn malformed_or_unrepresentable_requests_fail_without_unwinding() {
        let state = State::default();
        for (size, alignment) in [(0, 16), (1, 0), (1, 3), (u64::MAX, 16)] {
            // SAFETY: State remains live; failed requests return no allocation.
            assert!(unsafe { allocate(data(&state), size, alignment) }.is_null());
        }
        assert_eq!(state.allocations.load(Ordering::Relaxed), 4);
        assert_eq!(state.frees.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn failure_injection_and_null_release_are_supported() {
        let state = State::default();
        state.fail.store(true, Ordering::Relaxed);
        // SAFETY: State remains live; failure injection returns no allocation.
        assert!(unsafe { allocate(data(&state), 64, 16) }.is_null());
        // SAFETY: The callback contract permits releasing a null pointer.
        unsafe { release(data(&state), std::ptr::null_mut()) };
        assert_eq!(state.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(state.frees.load(Ordering::Relaxed), 1);
    }
}
