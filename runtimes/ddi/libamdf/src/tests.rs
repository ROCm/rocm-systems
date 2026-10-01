//! Cross-module AMDF negotiation, allocator, and release-callback tests.
//!
//! These fixtures exercise contracts that span more than one implementation
//! module: ABI-version selection, extensible-record preservation, callback
//! allocation failure, and exactly-once release. The allocator deliberately
//! uses explicit aligned system allocation so tests do not depend on the Rust
//! global allocator or a platform-specific C allocation API.
#![allow(clippy::cast_possible_truncation, clippy::cast_ptr_alignment)]
#![allow(clippy::unwrap_used, clippy::panic)]

use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

#[test]
fn reset_epoch_advances_once_for_observed_native_loss() {
    let epoch = AtomicU64::new(7);
    instance::advance_reset_epoch(&epoch, 7);
    instance::advance_reset_epoch(&epoch, 7);
    assert_eq!(epoch.load(Ordering::Acquire), 8);
    instance::advance_reset_epoch(&epoch, u64::MAX);
    assert_eq!(epoch.load(Ordering::Acquire), u64::MAX);
}

#[test]
fn sdma_format_features_follow_native_encoding_rules() {
    assert_eq!(instance::sdma_format_features(11, 5, false), 0);
    assert_eq!(
        instance::sdma_format_features(11, 5, true),
        AMDF_GPU_SDMA_FORMAT_FEATURE_GCR
    );
    assert_eq!(
        instance::sdma_format_features(12, 0, true),
        AMDF_GPU_SDMA_FORMAT_FEATURE_GCR | AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM
    );
    assert_eq!(
        instance::sdma_format_features(12, 5, false),
        AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM | AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE
    );
}

#[repr(C, align(8))]
struct Short {
    kind: u32,
    size: u32,
}

#[test]
fn negotiation_and_short_records_preserve_outputs() {
    // SAFETY: Deliberately invalid values are never dereferenced by admission.
    unsafe {
        let sentinel = std::ptr::dangling::<amdf_api_t>();
        let mut api = sentinel;
        assert_eq!(amdf_query_api(2, 1, &raw mut api), INVALID);
        assert_eq!(api, sentinel);
        assert_eq!(amdf_query_api(2, 2, &raw mut api), VERSION);
        assert_eq!(api, sentinel);
        assert_eq!(amdf_query_api(1, 3, &raw mut api), 0);
        assert_eq!(api, &raw const API);
        assert_eq!((*api).abi_version, AMDF_ABI_VERSION_3);
        let mut extension = sentinel.cast::<c_void>();
        assert_eq!(
            query_extension(AMDF_EXTENSION_XDNA, 1, 1, &raw mut extension),
            UNSUPPORTED
        );
        assert_eq!(extension, sentinel.cast());
        let short = Short {
            kind: AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            size: 8,
        };
        let mut owner = std::ptr::dangling_mut::<amdf_instance_t>();
        assert_eq!(
            instance::create((&raw const short).cast(), &raw mut owner),
            INVALID
        );
        assert_eq!(owner, std::ptr::dangling_mut());
    }
}

struct Allocations {
    attempts: AtomicUsize,
    live: AtomicUsize,
    fail_at: usize,
}

#[derive(Clone, Copy)]
struct AllocationHeader {
    base: *mut u8,
    layout: Layout,
}

fn allocation_layout(length: u64, alignment: u64) -> Option<(Layout, usize)> {
    let length = usize::try_from(length).ok()?;
    let requested_alignment = usize::try_from(alignment).ok()?;
    if length == 0 || !requested_alignment.is_power_of_two() {
        return None;
    }
    let allocation_alignment = requested_alignment.max(align_of::<AllocationHeader>());
    let prefix = size_of::<AllocationHeader>().checked_add(allocation_alignment - 1)?
        & !(allocation_alignment - 1);
    let total = prefix.checked_add(length)?;
    let layout = Layout::from_size_align(total, allocation_alignment).ok()?;
    Some((layout, prefix))
}

unsafe extern "C" fn allocate(state: *mut c_void, length: u64, alignment: u64) -> *mut c_void {
    // SAFETY: Each test retains the state until all provider owners are gone.
    let state = unsafe { &*state.cast::<Allocations>() };
    if state.attempts.fetch_add(1, Ordering::SeqCst) == state.fail_at {
        return std::ptr::null_mut();
    }
    let Some((layout, prefix)) = allocation_layout(length, alignment) else {
        return std::ptr::null_mut();
    };
    // SAFETY: The validated layout is nonzero and comes from Layout.
    let base = unsafe { System.alloc(layout) };
    if base.is_null() {
        return base.cast();
    }
    // SAFETY: prefix is within the allocation and leaves `length` payload bytes.
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
    };
    state.live.fetch_add(1, Ordering::SeqCst);
    pointer.cast()
}

unsafe extern "C" fn free(state: *mut c_void, allocation: *mut c_void) {
    let state = unsafe { &*state.cast::<Allocations>() };
    if allocation.is_null() {
        return;
    }
    let pointer = allocation.cast::<u8>();
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
    state.live.fetch_sub(1, Ordering::SeqCst);
}

fn parameters() -> amdf_instance_create_info_t {
    amdf_instance_create_info_t {
        r#type: AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        structure_size: size_of::<amdf_instance_create_info_t>() as u32,
        ..Default::default()
    }
}

#[test]
fn constructor_failures_release_only_private_metadata() {
    for fail_at in 0..3 {
        let state = Allocations {
            attempts: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
            fail_at,
        };
        let mut info = parameters();
        info.host_allocator = amdf_allocator_t {
            user_data: (&raw const state).cast_mut().cast(),
            allocate: Some(allocate),
            resize: None,
            free: Some(free),
        };
        let sentinel = std::ptr::dangling_mut::<amdf_instance_t>();
        let mut instance = sentinel;
        unsafe {
            let status = instance::create(&raw const info, &raw mut instance);
            if status == 0 {
                assert_eq!(instance::destroy(instance), 0);
            } else {
                assert_eq!(status, support::EXHAUSTED);
                assert_eq!(instance, sentinel);
            }
        }
        assert_eq!(state.live.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn larger_records_keep_their_tail_and_unknown_extensions_are_rejected() {
    #[repr(C)]
    struct Extended {
        info: amdf_memory_scope_info_t,
        tail: [u64; 3],
    }
    unsafe {
        let mut instance = std::ptr::null_mut();
        assert_eq!(instance::create(&parameters(), &raw mut instance), 0);
        let mut scope = std::ptr::null_mut();
        let mut count = 0;
        assert_eq!(
            memory::instance_scopes(instance, 1, &raw mut scope, &raw mut count),
            0
        );
        let mut extended = Extended {
            info: amdf_memory_scope_info_t {
                r#type: AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
                structure_size: size_of::<Extended>() as u32,
                ..Default::default()
            },
            tail: [0xface; 3],
        };
        assert_eq!(memory::scope_info(scope, &raw mut extended.info), 0);
        assert_eq!(extended.info.structure_size, size_of::<Extended>() as u32);
        assert_eq!(extended.tail, [0xface; 3]);
        extended.info.next = std::ptr::dangling_mut::<c_void>();
        extended.info.memory_profile_count = 99;
        assert_eq!(
            memory::scope_info(scope, &raw mut extended.info),
            UNSUPPORTED
        );
        assert_eq!(extended.info.memory_profile_count, 99);
        assert_eq!(extended.tail, [0xface; 3]);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn callback_external_values_are_released_exactly_once() {
    struct ReleaseObservation {
        calls: AtomicUsize,
        value: AtomicPtr<amdf_external_memory_t>,
    }
    unsafe extern "C" fn release(
        state: *mut c_void,
        kind: u32,
        value: amdf_external_memory_payload_t,
    ) {
        assert_eq!(kind, AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD);
        assert_eq!(unsafe { value.file_descriptor }, 41);
        let state = unsafe { &*state.cast::<ReleaseObservation>() };
        let cleared = unsafe { &*state.value.load(Ordering::SeqCst) };
        assert_eq!(cleared.r#type, AMDF_EXTERNAL_MEMORY_TYPE_NONE);
        assert_eq!(cleared.byte_length, 0);
        assert!(cleared.release.is_none());
        state.calls.fetch_add(1, Ordering::SeqCst);
    }
    let state = ReleaseObservation {
        calls: AtomicUsize::new(0),
        value: AtomicPtr::new(std::ptr::null_mut()),
    };
    let mut value = amdf_external_memory_t {
        r#type: AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD,
        payload: amdf_external_memory_payload_t {
            file_descriptor: 41,
        },
        release: Some(release),
        release_user_data: (&raw const state).cast_mut().cast(),
        ..Default::default()
    };
    state.value.store(&raw mut value, Ordering::SeqCst);
    unsafe {
        memory::external_release(&raw mut value);
        memory::external_release(&raw mut value);
    }
    assert_eq!(state.calls.load(Ordering::SeqCst), 1);
    assert_eq!(value.r#type, 0);
    assert!(value.release.is_none());
}
