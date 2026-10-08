// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Scripted DRM failures exercise the native VM owner transitions without a GPU.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::super::{memory, sysfs};
use super::*;
use crate::host_storage::Allocator;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static FILE_ID: AtomicUsize = AtomicUsize::new(0);

fn page() -> u64 {
    util::page_size().unwrap() as u64
}

fn backing_file() -> File {
    let path = std::env::temp_dir().join(format!(
        "rocddi-drm-test-{}-{}",
        std::process::id(),
        FILE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(path).unwrap();
    file.set_len(page()).unwrap();
    file
}

fn fixture() -> (
    Shared<DeviceVm>,
    Owned<KfdVirtualAddress>,
    Owned<KfdVirtualMemory>,
) {
    let allocator = Allocator::default();
    let kfd = Shared::new(
        sys::Kfd::with_hook(
            File::open("/dev/null").unwrap(),
            Arc::new(|call| {
                if let sys::Call::Wait(args, _) = call {
                    args.result = uapi::WAIT_TIMEOUT;
                }
                Ok(())
            }),
        ),
        allocator,
    )
    .unwrap();
    let node = sysfs::NativeNode {
        node: 1,
        gpu_id: 42,
        render_minor: Some(128),
        unique_id: Some(123),
        identity: [0; 16],
        queues: sysfs::NativeQueueProperties {
            gfx_target: 120_001,
            xcc_count: 1,
            ..sysfs::NativeQueueProperties::default()
        },
        local_memory_bytes: 1 << 30,
        public_memory_bytes: 0,
    };
    let vm = memory::queue_fixture(kfd, backing_file(), node);
    let reservation =
        KfdVirtualAddress::reserve((0x10000, isize::MAX as u64), page(), page(), 0, allocator)
            .unwrap();
    let source = backing_file();
    let physical = KfdVirtualMemory::import(source.as_raw_fd(), allocator).unwrap();
    (vm, reservation, physical)
}

#[test]
fn ambiguous_map_retains_address_backing_gem_and_pending_point() {
    let (vm, mut address, mut physical) = fixture();
    drm::with_script(
        [
            drm::TestCall::ImportGem(11),
            drm::TestCall::CreateSyncobj(7),
            drm::TestCall::Map(Err(14)),
        ],
        || {
            let result = KfdVirtualDeviceMapping::create(
                &physical,
                &address,
                vm.clone(),
                address.address(),
                0,
                page(),
                DeviceAccess::READ,
            );
            assert!(result.is_err());
            assert_eq!(
                address.free().err().map(|e| e.kind()),
                Some(ErrorKind::DriverContract)
            );
            assert_eq!(
                physical.free().err().map(|e| e.kind()),
                Some(ErrorKind::DriverContract)
            );
            let state = vm.vmem.lock().unwrap();
            assert_eq!(state.pending_maps.len(), 1);
            assert_eq!(state.gems.len(), 1);
            assert_eq!(state.gems[0].handle, 11);
        },
    );
    // Their destructors intentionally quarantine the reservation and VM.
}

#[test]
fn failed_unmap_wait_retries_only_the_wait_and_keeps_gem_until_completion() {
    let (vm, mut address, mut physical) = fixture();
    drm::with_script(
        [
            drm::TestCall::ImportGem(11),
            drm::TestCall::CreateSyncobj(7),
            drm::TestCall::Map(Ok(())),
            drm::TestCall::Wait(Ok(())),
            drm::TestCall::Unmap(Ok(())),
            drm::TestCall::Wait(Err(110)),
            drm::TestCall::Wait(Ok(())),
            drm::TestCall::CloseGem(Ok(())),
            drm::TestCall::DestroySyncobj,
        ],
        || {
            let mut mapping = KfdVirtualDeviceMapping::create(
                &physical,
                &address,
                vm.clone(),
                address.address(),
                0,
                page(),
                DeviceAccess::READ,
            )
            .unwrap();
            assert!(mapping.free().is_err());
            assert!(mapping.mapped);
            assert!(mapping.owns_gem);
            assert!(mapping.pending_unmap.is_some());
            assert_eq!(vm.vmem.lock().unwrap().gems.len(), 1);
            mapping.free().unwrap();
            assert!(!mapping.mapped);
            assert!(!mapping.owns_gem);
            assert!(mapping.pending_unmap.is_none());
            drop(mapping);
            address.free().unwrap();
            physical.free().unwrap();
            vm.vmem.lock().unwrap().close(vm.render().unwrap()).unwrap();
        },
    );
}

#[test]
fn failed_map_wait_quarantines_the_possible_native_mapping() {
    let (vm, mut address, mut physical) = fixture();
    drm::with_script(
        [
            drm::TestCall::ImportGem(11),
            drm::TestCall::CreateSyncobj(7),
            drm::TestCall::Map(Ok(())),
            drm::TestCall::Wait(Err(110)),
        ],
        || {
            let result = KfdVirtualDeviceMapping::create(
                &physical,
                &address,
                vm.clone(),
                address.address(),
                0,
                page(),
                DeviceAccess::READ,
            );
            assert!(result.is_err());
            assert_eq!(
                address.free().err().map(|e| e.kind()),
                Some(ErrorKind::DriverContract)
            );
            assert_eq!(
                physical.free().err().map(|e| e.kind()),
                Some(ErrorKind::DriverContract)
            );
            let state = vm.vmem.lock().unwrap();
            assert_eq!(state.pending_maps.len(), 1);
            assert_eq!(state.gems.len(), 1);
        },
    );
}

#[test]
fn ambiguous_unmap_is_not_replayed_or_treated_as_retired() {
    let (vm, address, physical) = fixture();
    drm::with_script(
        [
            drm::TestCall::ImportGem(11),
            drm::TestCall::CreateSyncobj(7),
            drm::TestCall::Map(Ok(())),
            drm::TestCall::Wait(Ok(())),
            drm::TestCall::Unmap(Err(14)),
        ],
        || {
            let mut mapping = KfdVirtualDeviceMapping::create(
                &physical,
                &address,
                vm.clone(),
                address.address(),
                0,
                page(),
                DeviceAccess::READ,
            )
            .unwrap();
            assert!(mapping.free().is_err());
            assert!(mapping.uncertain);
            assert!(mapping.mapped);
            assert!(mapping.owns_gem);
            assert_eq!(
                mapping.free().err().map(|e| e.kind()),
                Some(ErrorKind::DriverContract)
            );
            assert_eq!(vm.vmem.lock().unwrap().gems.len(), 1);
        },
    );
    // The core mapping owner retains these dependencies after failed teardown.
    std::mem::forget(address);
    std::mem::forget(physical);
}

#[test]
fn failed_gem_close_retries_without_repeating_native_unmap() {
    let (vm, mut address, mut physical) = fixture();
    drm::with_script(
        [
            drm::TestCall::ImportGem(11),
            drm::TestCall::CreateSyncobj(7),
            drm::TestCall::Map(Ok(())),
            drm::TestCall::Wait(Ok(())),
            drm::TestCall::Unmap(Ok(())),
            drm::TestCall::Wait(Ok(())),
            drm::TestCall::CloseGem(Err(5)),
            drm::TestCall::CloseGem(Ok(())),
            drm::TestCall::DestroySyncobj,
        ],
        || {
            let mut mapping = KfdVirtualDeviceMapping::create(
                &physical,
                &address,
                vm.clone(),
                address.address(),
                0,
                page(),
                DeviceAccess::READ,
            )
            .unwrap();
            assert!(mapping.free().is_err());
            assert!(!mapping.mapped);
            assert!(mapping.owns_gem);
            assert_eq!(vm.vmem.lock().unwrap().gems.len(), 1);
            mapping.free().unwrap();
            assert!(!mapping.owns_gem);
            drop(mapping);
            address.free().unwrap();
            physical.free().unwrap();
            vm.vmem.lock().unwrap().close(vm.render().unwrap()).unwrap();
        },
    );
}

#[test]
fn ambiguous_syncobj_destroy_quarantines_the_id_and_render_vm() {
    let (vm, _address, _physical) = fixture();
    let render = vm.render().unwrap();
    drm::with_script(
        [
            drm::TestCall::CreateSyncobj(7),
            drm::TestCall::FailDestroySyncobj(14),
            drm::TestCall::CreateSyncobj(7),
            drm::TestCall::DestroySyncobj,
        ],
        || {
            let mut state = vm.vmem.lock().unwrap();
            assert_eq!(state.ensure_syncobj(render).unwrap(), 7);
            assert_eq!(
                state.close(render).unwrap_err().native_error_code(),
                Some(14)
            );
            let replacement = drm::create_syncobj(render).unwrap();
            assert_eq!(replacement, 7);
            assert_eq!(
                state.close(render).unwrap_err().kind(),
                ErrorKind::DriverContract
            );
            assert!(state.ensure_syncobj(render).is_err());
            drm::destroy_syncobj(render, replacement).unwrap();
        },
    );
}
