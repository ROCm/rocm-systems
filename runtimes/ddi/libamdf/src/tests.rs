// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Cross-module AMDF negotiation and release-callback tests.
//!
//! These fixtures exercise contracts that span more than one implementation
//! module: ABI-version selection, extensible-record preservation, and
//! exactly-once release.
#![allow(clippy::cast_possible_truncation, clippy::cast_ptr_alignment)]
#![allow(clippy::unwrap_used, clippy::panic)]

use super::*;
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
fn reset_epoch_policy_uses_reset_evidence_and_device_identity() {
    use rocddi::device::event::{
        DeviceEvent, GpuHardwareException, GpuMemoryFault, GpuMemoryFaultCause, GpuResetCause,
        GpuResetScope,
    };

    let endpoint_id = [1; 16];
    let memory_fault = DeviceEvent::GpuMemoryFault(GpuMemoryFault {
        endpoint_id: Some(endpoint_id),
        virtual_address: 0x1000,
        page_not_present: true,
        read_only: false,
        no_execute: false,
        imprecise: false,
        cause: GpuMemoryFaultCause::None,
    });
    assert!(!instance::event_resets_device(memory_fault, endpoint_id));

    let exception = GpuHardwareException {
        endpoint_id: Some(endpoint_id),
        scope: GpuResetScope::Other,
        memory_lost: false,
        cause: GpuResetCause::Hang,
    };
    assert!(instance::event_resets_device(
        DeviceEvent::GpuHardwareException(exception),
        endpoint_id
    ));
    assert!(!instance::event_resets_device(
        DeviceEvent::GpuHardwareException(exception),
        [2; 16]
    ));
    assert!(instance::event_resets_device(
        DeviceEvent::GpuHardwareException(GpuHardwareException {
            memory_lost: true,
            ..exception
        }),
        [2; 16]
    ));
}

#[test]
fn sdma_format_features_follow_native_encoding_rules() {
    assert_eq!(instance::sdma_format_features(11, 5, false), 0);
    assert_eq!(
        instance::sdma_format_features(11, 5, true),
        AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR
    );
    assert_eq!(
        instance::sdma_format_features(12, 0, true),
        AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR | AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM
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
        assert_eq!(amdf_query_api(1, 4, &raw mut api), VERSION);
        assert_eq!(api, sentinel);
        assert_eq!(amdf_query_api(1, 5, &raw mut api), 0);
        assert_eq!(api, &raw const API);
        assert_eq!((*api).abi_version, AMDF_ABI_VERSION_5);
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

fn parameters() -> amdf_instance_create_info_t {
    amdf_instance_create_info_t {
        r#type: AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        structure_size: size_of::<amdf_instance_create_info_t>() as u32,
        ..Default::default()
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
