// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Memory, SVM, interop, and virtual-memory regression tests.

use super::allocation::*;
use super::interop::*;
use super::virtual_memory::*;
use super::*;

#[test]
/// Verifies that execute access permission matches the native bitmask.
fn execute_access_permission_matches_the_native_bitmask() {
    assert_eq!(ACCESS_PERMISSION_EX, DeviceAccess::EXECUTE.bits());
    assert_eq!(
        ACCESS_PERMISSION_RW,
        (DeviceAccess::READ | DeviceAccess::WRITE).bits()
    );
    for access in 0..=7 {
        assert!(valid_access_permissions(access));
    }
    for access in [8, 12, u32::MAX] {
        assert!(!valid_access_permissions(access));
    }
}

#[test]
/// Verifies that completed memory workers are reaped without joining active
/// work.
fn completed_memory_workers_are_reaped_without_joining_active_work() {
    let mut state = MemoryState::new();
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = stop.clone();
    let active = thread::spawn(move || {
        while !worker_stop.load(Ordering::Acquire) {
            thread::yield_now();
        }
    });
    let active_id = active.thread().id();
    let complete = thread::spawn(|| {});
    while !complete.is_finished() {
        thread::yield_now();
    }
    state.workers.extend([active, complete]);

    state.reap_finished_workers();
    assert_eq!(state.workers.len(), 1);
    assert_eq!(state.workers[0].thread().id(), active_id);

    stop.store(true, Ordering::Release);
    state.join_workers();
    assert!(state.workers.is_empty());
}

#[test]
/// Verifies that pointer info matches the public x86 64 ABI.
fn pointer_info_matches_the_public_x86_64_abi() {
    assert_eq!(size_of::<HsaAmdPointerInfo>(), 64);
    assert_eq!(std::mem::align_of::<HsaAmdPointerInfo>(), 8);
    assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, pointer_type), 4);
    assert_eq!(
        std::mem::offset_of!(HsaAmdPointerInfo, agent_base_address),
        8
    );
    assert_eq!(
        std::mem::offset_of!(HsaAmdPointerInfo, host_base_address),
        16
    );
    assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, size_in_bytes), 24);
    assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, agent_owner), 40);
    assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, global_flags), 48);
    assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, registered), 52);
    assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, alloc_flags), 56);
    let _: unsafe extern "C" fn(*const c_void, *mut c_void) -> Status =
        hsa_amd_pointer_info_set_userdata;
}

#[test]
/// Verifies that IPC attach does not clobber an aliased input on failure.
fn ipc_attach_does_not_clobber_an_aliased_input_on_failure() {
    #[repr(align(8))]
    /// Aligns a shared IPC handle for an aliasing test.
    struct AlignedHandle(
        /// IPC handle whose input and output pointers alias in this test.
        HsaAmdIpcMemory,
    );

    let original = [0xfeed_beef_u32; 8];
    let mut handle = AlignedHandle(HsaAmdIpcMemory { handle: original });
    // SAFETY: Both raw pointers refer to the same aligned writable storage.
    // The input is read before any successful output can be published.
    let status = unsafe {
        hsa_amd_ipc_memory_attach(
            &raw const handle.0,
            1,
            0,
            std::ptr::null(),
            (&raw mut handle.0).cast::<*mut c_void>(),
        )
    };
    assert_ne!(status, SUCCESS);
    assert_eq!(handle.0.handle, original);
}

#[test]
/// Verifies that allow access rejects reserved flags before runtime lookup.
fn allow_access_rejects_reserved_flags_before_runtime_lookup() {
    let agent = HsaAgent {
        handle: GPU_AGENT_BASE,
    };
    let flags = 0_u32;
    // SAFETY: The scalar inputs remain readable for the duration of the call.
    assert_eq!(
        unsafe {
            hsa_amd_agents_allow_access(
                1,
                &raw const agent,
                &raw const flags,
                std::ptr::dangling::<c_void>(),
            )
        },
        INVALID_ARGUMENT
    );
}

#[test]
/// Verifies that registered range aligns native pages and preserves host offset.
fn registered_range_aligns_native_pages_and_preserves_host_offset() {
    assert_eq!(
        registered_range(0x10_000, 1, 4096),
        Some((0x10_000, 4096, 0))
    );
    assert_eq!(
        registered_range(0x10_345, 1, 4096),
        Some((0x10_000, 4096, 0x345))
    );
    assert_eq!(
        registered_range(0x10_345, 4096, 4096),
        Some((0x10_000, 8192, 0x345))
    );
    assert_eq!(registered_range(usize::MAX, 2, 4096), None);
    assert_eq!(
        registered_range(0x10_123, 1, 65_536),
        Some((0x10_000, 65_536, 0x123))
    );
    assert_eq!(
        registered_range(0x10_123, 65_536, 65_536),
        Some((0x10_000, 131_072, 0x123))
    );
}

#[test]
/// Verifies that registered GPU alias translates to host backing.
fn registered_gpu_alias_translates_to_host_backing() {
    let description = PointerDescription {
        pointer_type: POINTER_TYPE_LOCKED,
        agent_base: 0x20_000,
        host_base: Some(0x10_000),
        size: 4096,
        owner: HsaAgent { handle: CPU_AGENT },
        global_flags: POOL_FLAG_COARSE,
        registered: true,
        alloc_flags: POINTER_ALLOC_NONPAGED | POINTER_ALLOC_HOST_ACCESS,
        user_data: 0x1234,
        accessible: Vec::new(),
    };

    assert_eq!(description.host_address(0x20_000), Some(0x10_000));
    assert_eq!(description.host_address(0x20_123), Some(0x10_123));
    assert_eq!(description.host_address(0x10_123), Some(0x10_123));
    assert_eq!(description.host_address(0x21_000), None);
    assert_eq!(description.host_range(0x20_123, 100), Some(0x10_123));
    assert_eq!(description.host_range(0x20_ff0, 16), Some(0x10_ff0));
    assert_eq!(description.host_range(0x20_ff0, 17), None);
    assert_eq!(description.offset_range(0x20_123, 100), Some(0x123));
    assert_eq!(description.offset_range(0x10_123, 100), Some(0x123));
    let mut info = std::mem::MaybeUninit::<HsaAmdPointerInfo>::uninit();
    // SAFETY: write_pointer_info initializes the complete public structure.
    unsafe {
        write_pointer_info(
            info.as_mut_ptr(),
            size_of::<HsaAmdPointerInfo>() as u32,
            Some(&description),
        );
        assert_eq!(info.assume_init().user_data as usize, 0x1234);
    }
}

#[test]
/// Verifies that private graphics addresses do not claim a CPU mapping.
fn private_graphics_addresses_do_not_claim_a_cpu_mapping() {
    let description = PointerDescription {
        pointer_type: POINTER_TYPE_GRAPHICS,
        agent_base: 0x20_000,
        host_base: None,
        size: 4096,
        owner: HsaAgent { handle: 0 },
        global_flags: POOL_FLAG_COARSE,
        registered: true,
        alloc_flags: POINTER_ALLOC_NONPAGED,
        user_data: 0,
        accessible: Vec::new(),
    };

    assert!(description.contains(0x20_123));
    assert_eq!(description.host_address(0x20_123), None);
    assert_eq!(description.host_range(0x20_123, 1), None);
    assert_eq!(description.offset_range(0x20_123, 100), Some(0x123));
}

#[test]
/// Verifies that external memory entry points match the linux public ABI.
fn external_memory_entry_points_match_the_linux_public_abi() {
    assert_eq!(size_of::<HsaAmdIpcMemory>(), 32);
    assert_eq!(align_of::<HsaAmdIpcMemory>(), 4);
    assert_eq!(size_of::<HsaAmdExternalSemaphore>(), 8);
    assert_eq!(align_of::<HsaAmdExternalSemaphore>(), 8);
    assert_eq!(size_of::<HsaAmdExternalSemaphoreHandleDescriptor>(), 16);
    assert_eq!(align_of::<HsaAmdExternalSemaphoreHandleDescriptor>(), 8);
    let _: unsafe extern "C" fn(*mut c_void, usize, *mut HsaAmdIpcMemory) -> Status =
        hsa_amd_ipc_memory_create;
    let _: unsafe extern "C" fn(
        *const HsaAmdIpcMemory,
        usize,
        u32,
        *const HsaAgent,
        *mut *mut c_void,
    ) -> Status = hsa_amd_ipc_memory_attach;
    let _: extern "C" fn(*mut c_void) -> Status = hsa_amd_ipc_memory_detach;
    let _: unsafe extern "C" fn(
        u32,
        *mut HsaAgent,
        HsaHandle,
        u32,
        *mut usize,
        *mut *mut c_void,
        *mut usize,
        *mut *const c_void,
    ) -> Status = hsa_amd_interop_map_buffer;
    let _: unsafe extern "C" fn(
        u32,
        *mut HsaAgent,
        HsaHandle,
        u32,
        usize,
        *mut usize,
        *mut *mut c_void,
        *mut usize,
        *mut *const c_void,
    ) -> Status = hsa_amd_interop_map_buffer_with_size;
    let _: extern "C" fn(*mut c_void) -> Status = hsa_amd_interop_unmap_buffer;
    let _: unsafe extern "C" fn(
        HsaAgent,
        *const HsaAmdExternalSemaphoreHandleDescriptor,
        *mut HsaAmdExternalSemaphore,
    ) -> Status = hsa_amd_external_semaphore_handle_open;
    let _: extern "C" fn(HsaAmdExternalSemaphore) -> Status =
        hsa_amd_external_semaphore_handle_close;
    let _: unsafe extern "C" fn(
        HsaAmdAisFileHandle,
        *mut c_void,
        u64,
        i64,
        *mut u64,
        *mut i32,
    ) -> Status = hsa_amd_ais_file_read;
    let _: unsafe extern "C" fn(
        HsaAmdAisFileHandle,
        *mut c_void,
        u64,
        i64,
        *mut u64,
        *mut i32,
    ) -> Status = hsa_amd_ais_file_write;
    let _: unsafe extern "C" fn(*const c_void, usize, *mut i32, *mut u64) -> Status =
        hsa_amd_portable_export_dmabuf;
    let _: unsafe extern "C" fn(*const c_void, usize, *mut i32, *mut u64, u64) -> Status =
        hsa_amd_portable_export_dmabuf_v2;
    let _: extern "C" fn(i32) -> Status = hsa_amd_portable_close_dmabuf;
}

#[test]
/// Verifies that virtual memory entry points match the public ABI.
fn virtual_memory_entry_points_match_the_public_abi() {
    assert_eq!(size_of::<HsaAmdVmemAllocHandle>(), 8);
    assert_eq!(align_of::<HsaAmdVmemAllocHandle>(), 8);
    assert_eq!(size_of::<HsaAmdMemoryAccessDesc>(), 16);
    assert_eq!(align_of::<HsaAmdMemoryAccessDesc>(), 8);
    assert_eq!(size_of::<HsaFabricHandle>(), 16);
    let _: unsafe extern "C" fn(*mut *mut c_void, usize, u64, u64) -> Status =
        hsa_amd_vmem_address_reserve;
    let _: unsafe extern "C" fn(*mut *mut c_void, usize, u64, u64, u64) -> Status =
        hsa_amd_vmem_address_reserve_align;
    let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_amd_vmem_address_free;
    let _: unsafe extern "C" fn(
        HsaMemoryPool,
        usize,
        u32,
        u64,
        *mut HsaAmdVmemAllocHandle,
    ) -> Status = hsa_amd_vmem_handle_create;
    let _: extern "C" fn(HsaAmdVmemAllocHandle) -> Status = hsa_amd_vmem_handle_release;
    let _: extern "C" fn(*mut c_void, usize, usize, HsaAmdVmemAllocHandle, u64) -> Status =
        hsa_amd_vmem_map;
    let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_amd_vmem_unmap;
    let _: unsafe extern "C" fn(
        *mut c_void,
        usize,
        *const HsaAmdMemoryAccessDesc,
        usize,
    ) -> Status = hsa_amd_vmem_set_access;
    let _: unsafe extern "C" fn(*mut c_void, *mut u32, HsaAgent) -> Status =
        hsa_amd_vmem_get_access;
    let _: unsafe extern "C" fn(*mut i32, HsaAmdVmemAllocHandle, u64) -> Status =
        hsa_amd_vmem_export_shareable_handle;
    let _: unsafe extern "C" fn(i32, *mut HsaAmdVmemAllocHandle) -> Status =
        hsa_amd_vmem_import_shareable_handle;
    let _: unsafe extern "C" fn(*mut HsaAmdVmemAllocHandle, *mut c_void) -> Status =
        hsa_amd_vmem_retain_alloc_handle;
    let _: unsafe extern "C" fn(HsaAmdVmemAllocHandle, *mut HsaMemoryPool, *mut u32) -> Status =
        hsa_amd_vmem_get_alloc_properties_from_handle;
    let _: unsafe extern "C" fn(*mut HsaFabricHandle, HsaAmdVmemAllocHandle, u64) -> Status =
        hsa_amd_vmem_export_fabric_handle;
    let _: unsafe extern "C" fn(HsaFabricHandle, *mut HsaAmdVmemAllocHandle) -> Status =
        hsa_amd_vmem_import_fabric_handle;
}

#[test]
/// Verifies that virtual mapping contains only its half open range.
fn virtual_mapping_contains_only_its_half_open_range() {
    let mapping = VmemMapping {
        handle: 1,
        reservation: 0x1_0000,
        address: 0x1_2000,
        offset: 0,
        size: 0x2000,
        access: HashMap::new(),
    };
    assert!(mapping.contains(0x1_2000));
    assert!(mapping.contains(0x1_3fff));
    assert!(!mapping.contains(0x1_4000));
}

#[test]
/// Verifies that virtual map rejects unaligned addresses before runtime lookup.
fn virtual_map_rejects_unaligned_addresses_before_runtime_lookup() {
    let page = current_host_page().unwrap();
    assert_eq!(
        hsa_amd_vmem_map(
            (page + 1) as *mut c_void,
            page,
            0,
            HsaAmdVmemAllocHandle { handle: 0 },
            0,
        ),
        INVALID_ARGUMENT
    );
}

#[test]
/// Verifies that AIS transfer preserves offsets partial reads and status.
fn ais_transfer_preserves_offsets_partial_reads_and_status()
-> Result<(), Box<dyn std::error::Error>> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;

    let path = std::env::temp_dir().join(format!(
        "rocddi-ais-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    file.write_all(b"prefix........suffix")?;

    let source = *b"rocddi";
    let mut copied = u64::MAX;
    let mut status = i32::MIN;
    // SAFETY: source is readable and file remains live for the transfer.
    assert_eq!(
        unsafe {
            ais_transfer(
                file.as_raw_fd(),
                source.as_ptr() as usize,
                source.len(),
                6,
                &raw mut copied,
                &raw mut status,
                AisOperation::Write,
            )
        },
        SUCCESS
    );
    assert_eq!(copied, source.len() as u64);
    assert_eq!(status, 0);

    let mut destination = [0u8; 9];
    copied = u64::MAX;
    status = i32::MIN;
    // SAFETY: destination is writable and file remains live for the transfer.
    assert_eq!(
        unsafe {
            ais_transfer(
                file.as_raw_fd(),
                destination.as_mut_ptr() as usize,
                destination.len(),
                6,
                &raw mut copied,
                &raw mut status,
                AisOperation::Read,
            )
        },
        SUCCESS
    );
    assert_eq!(copied, destination.len() as u64);
    assert_eq!(status, 0);
    assert_eq!(&destination, b"rocddi..s");

    file.seek(SeekFrom::Start(0))?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;
    assert_eq!(&contents, b"prefixrocddi..suffix");

    copied = u64::MAX;
    status = i32::MIN;
    // SAFETY: A zero-byte transfer borrows no host storage.
    assert_eq!(
        unsafe {
            ais_transfer(
                file.as_raw_fd(),
                0,
                0,
                0,
                &raw mut copied,
                &raw mut status,
                AisOperation::Read,
            )
        },
        SUCCESS
    );
    assert_eq!((copied, status), (0, 0));

    copied = u64::MAX;
    status = i32::MIN;
    // SAFETY: The source is readable; Linux reports the invalid descriptor.
    assert_eq!(
        unsafe {
            ais_transfer(
                -1,
                source.as_ptr() as usize,
                source.len(),
                0,
                &raw mut copied,
                &raw mut status,
                AisOperation::Write,
            )
        },
        ERROR
    );
    assert_eq!((copied, status), (0, -9));
    drop(file);
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
/// Verifies that deallocation callback entry points match the public ABI.
fn deallocation_callback_entry_points_match_the_public_abi() {
    let _: unsafe extern "C" fn(*mut c_void, DeallocationCallbackFn, *mut c_void) -> Status =
        hsa_amd_register_deallocation_callback;
    let _: unsafe extern "C" fn(*mut c_void, DeallocationCallbackFn) -> Status =
        hsa_amd_deregister_deallocation_callback;
}

#[test]
/// Verifies that migration entry points match the public ABI.
fn migration_entry_points_match_the_public_abi() {
    let _: unsafe extern "C" fn(HsaMemoryPool, HsaMemoryPool, *mut bool) -> Status =
        hsa_amd_memory_pool_can_migrate;
    let _: unsafe extern "C" fn(*const c_void, HsaMemoryPool, u32) -> Status =
        hsa_amd_memory_migrate;
}

#[test]
/// Verifies that base memory entry points match the public ABI.
fn base_memory_entry_points_match_the_public_abi() {
    let _: unsafe extern "C" fn(HsaRegion, usize, *mut *mut c_void) -> Status = hsa_memory_allocate;
    let _: unsafe extern "C" fn(*mut c_void) -> Status = hsa_memory_free;
    let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_memory_register;
    let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_memory_deregister;
    let _: extern "C" fn(*mut c_void, HsaAgent, u32) -> Status = hsa_memory_assign_agent;
    let _: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> Status = hsa_memory_copy;
}

#[test]
/// Verifies that zero length memory operations skip address resolution.
fn zero_length_memory_operations_skip_address_resolution() {
    let aligned = align_of::<u32>() as *mut c_void;
    assert_eq!(memory_copy_has_work(aligned, aligned, 0), Ok(false));
    assert_eq!(
        memory_copy_has_work(aligned, aligned, isize::MAX as usize + 1),
        Err(INVALID_ARGUMENT)
    );
    assert_eq!(memory_fill_size(aligned, 0), Ok(None));

    assert_eq!(
        memory_copy_has_work(std::ptr::null_mut(), aligned, 0),
        Err(INVALID_ARGUMENT)
    );
    assert_eq!(
        memory_fill_size(std::ptr::null_mut(), 0),
        Err(INVALID_ARGUMENT)
    );
    assert_eq!(
        memory_fill_size((align_of::<u32>() + 1) as *mut c_void, 0),
        Err(INVALID_ARGUMENT)
    );
    assert_eq!(
        memory_fill_size(aligned, isize::MAX as usize / size_of::<u32>() + 1),
        Err(INVALID_ARGUMENT)
    );
}

#[test]
/// Verifies that rectangular copy entry point matches the public ABI.
fn rectangular_copy_entry_point_matches_the_public_abi() {
    let _: unsafe extern "C" fn(
        *const HsaPitchedPtr,
        *const HsaDim3,
        *const HsaPitchedPtr,
        *const HsaDim3,
        *const HsaDim3,
        HsaAgent,
        u32,
        u32,
        *const HsaSignal,
        HsaSignal,
    ) -> Status = hsa_amd_memory_async_copy_rect;
    assert_eq!(size_of::<HsaPitchedPtr>(), 24);
    assert_eq!(align_of::<HsaPitchedPtr>(), 8);
    assert_eq!(std::mem::offset_of!(HsaPitchedPtr, base), 0);
    assert_eq!(std::mem::offset_of!(HsaPitchedPtr, pitch), 8);
    assert_eq!(std::mem::offset_of!(HsaPitchedPtr, slice), 16);
}

#[test]
/// Verifies that batch copy entry point and descriptor match the public ABI.
fn batch_copy_entry_point_and_descriptor_match_the_public_abi() {
    let _: unsafe extern "C" fn(*const HsaAmdMemoryCopyOp, u32, u32, *const HsaSignal) -> Status =
        hsa_amd_memory_async_batch_copy;
    assert_eq!(size_of::<HsaAmdMemoryCopyWait>(), 32);
    assert_eq!(align_of::<HsaAmdMemoryCopyWait>(), 8);
    assert_eq!(size_of::<HsaAmdMemoryCopySignal>(), 24);
    assert_eq!(align_of::<HsaAmdMemoryCopySignal>(), 8);
    assert_eq!(size_of::<HsaAmdMemoryCopyOp>(), 128);
    assert_eq!(align_of::<HsaAmdMemoryCopyOp>(), 8);
    assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, source), 16);
    assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, size), 48);
    assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, wait), 64);
    assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, signal), 96);
    assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, reserved), 120);
}

#[test]
/// Verifies that async copy entry points preserve no init validation.
fn async_copy_entry_points_preserve_no_init_validation() {
    let agent = HsaAgent { handle: 0 };
    let completion = HsaSignal { handle: 1 };
    let dependency = HsaSignal { handle: 2 };
    let mut destination = 0x5a_u8;
    let source = 0xa5_u8;
    assert_eq!(
        unsafe {
            hsa_amd_memory_async_copy(
                (&raw mut destination).cast(),
                agent,
                (&raw const source).cast(),
                agent,
                1,
                1,
                &raw const dependency,
                completion,
            )
        },
        NOT_INITIALIZED
    );
    assert_eq!(
        unsafe {
            hsa_amd_memory_async_copy_on_engine(
                (&raw mut destination).cast(),
                agent,
                (&raw const source).cast(),
                agent,
                1,
                1,
                &raw const dependency,
                completion,
                1,
                true,
            )
        },
        NOT_INITIALIZED
    );
    assert_eq!(
        unsafe { hsa_amd_memory_async_batch_copy(std::ptr::null(), 1, 1, &raw const dependency) },
        NOT_INITIALIZED
    );
    assert_eq!(
        unsafe {
            hsa_amd_memory_async_copy_rect(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                agent,
                1,
                1,
                &raw const dependency,
                completion,
            )
        },
        NOT_INITIALIZED
    );
    assert_eq!(destination, 0x5a);
}

#[test]
/// Verifies that SVM entry points match the public ABI.
fn svm_entry_points_match_the_public_abi() {
    assert_eq!(size_of::<HsaAmdSvmAttributePair>(), 16);
    assert_eq!(align_of::<HsaAmdSvmAttributePair>(), 8);
    let _: unsafe extern "C" fn(*mut c_void, usize, *mut HsaAmdSvmAttributePair, usize) -> Status =
        hsa_amd_svm_attributes_set;
    let _: unsafe extern "C" fn(*mut c_void, usize, *mut HsaAmdSvmAttributePair, usize) -> Status =
        hsa_amd_svm_attributes_get;
    let _: unsafe extern "C" fn(
        *mut c_void,
        usize,
        HsaAgent,
        u32,
        *const HsaSignal,
        HsaSignal,
    ) -> Status = hsa_amd_svm_prefetch_async;
    let _: unsafe extern "C" fn(
        *mut *mut c_void,
        *mut usize,
        u32,
        u32,
        *const HsaSignal,
        HsaSignal,
    ) -> Status = hsa_amd_svm_discard_batch_async;
    let _: unsafe extern "C" fn(
        *mut *mut c_void,
        *mut usize,
        u32,
        *const HsaAgent,
        u32,
        u32,
        *const HsaSignal,
        HsaSignal,
    ) -> Status = hsa_amd_svm_discard_and_prefetch_batch_async;
}

#[test]
/// Verifies that SVM ranges cover each touched page.
fn svm_ranges_cover_each_touched_page() {
    let page = current_host_page().unwrap();
    let page_address = page as u64;
    let interior = (page + 0x123) as *mut c_void;
    assert_eq!(svm_range(interior, 1), Ok((page_address, page_address)));
    assert_eq!(
        svm_range(interior, page),
        Ok((page_address, page_address * 2))
    );
    assert_eq!(svm_range(std::ptr::null_mut(), 1), Err(INVALID_ARGUMENT));
    assert_eq!(svm_range(page as *mut c_void, 0), Err(INVALID_ARGUMENT));
}

#[test]
/// Verifies that portable close works without an initialized runtime.
fn portable_close_works_without_an_initialized_runtime() {
    use std::os::fd::IntoRawFd;

    let descriptor = std::fs::File::open("/dev/null").map_or(-1, IntoRawFd::into_raw_fd);
    assert_eq!(hsa_amd_portable_close_dmabuf(descriptor), SUCCESS);
    assert_eq!(hsa_amd_portable_close_dmabuf(-1), RESOURCE_FREE);
}

#[test]
/// Verifies that portable export checks initialization before arguments.
fn portable_export_checks_initialization_before_arguments() {
    assert!(lock().is_ok_and(|runtime| runtime.as_ref().is_none()));

    let mut descriptor = -1;
    let mut offset = u64::MAX;
    // SAFETY: The runtime is uninitialized, so ROCr-compatible validation
    // returns before dereferencing any of the deliberately invalid inputs.
    assert_eq!(
        unsafe {
            hsa_amd_portable_export_dmabuf(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        NOT_INITIALIZED
    );
    // ROCr only consumes the PCIe bit and ignores the remaining bits.
    assert_eq!(
        unsafe {
            hsa_amd_portable_export_dmabuf_v2(
                std::ptr::dangling::<c_void>(),
                1,
                &raw mut descriptor,
                &raw mut offset,
                u64::MAX,
            )
        },
        NOT_INITIALIZED
    );
    assert_eq!(descriptor, -1);
    assert_eq!(offset, u64::MAX);
}

#[test]
/// Verifies that memory lock checks initialization before arguments.
fn memory_lock_checks_initialization_before_arguments() {
    assert!(lock().is_ok_and(|runtime| runtime.as_ref().is_none()));

    let mut agent_ptr = usize::MAX as *mut c_void;
    // SAFETY: An uninitialized runtime returns before inspecting the
    // deliberately invalid input pointers.
    assert_eq!(
        unsafe {
            hsa_amd_memory_lock(
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                &raw mut agent_ptr,
            )
        },
        NOT_INITIALIZED
    );
    assert_eq!(agent_ptr, usize::MAX as *mut c_void);
    // The uncached pool-allocation flag is accepted by ROCr and reaches
    // its host-memory registration path.
    assert_eq!(
        unsafe {
            hsa_amd_memory_lock_to_pool(
                std::ptr::dangling_mut::<c_void>(),
                1,
                std::ptr::null(),
                0,
                HsaMemoryPool {
                    handle: CPU_POOL_COARSE,
                },
                ALLOC_UNCACHED,
                &raw mut agent_ptr,
            )
        },
        NOT_INITIALIZED
    );
    assert_eq!(agent_ptr, usize::MAX as *mut c_void);
}

#[test]
/// Verifies that CPU pool allocations preserve their cache policy.
fn cpu_pool_allocations_preserve_their_cache_policy() {
    for (pool, cache) in [
        (CPU_POOL_FINE, HostMappingPolicy::Fine),
        (CPU_POOL_EXTENDED, HostMappingPolicy::Extended),
        (CPU_POOL_COARSE, HostMappingPolicy::Coarse),
    ] {
        assert_eq!(
            cpu_pool_memory_kind(
                HsaMemoryPool { handle: pool },
                DriverContextLifetime::Process,
                0
            ),
            MemoryKind::OwnedHost { policy: cache }
        );
    }
    assert_eq!(
        cpu_pool_memory_kind(
            HsaMemoryPool {
                handle: CPU_POOL_KERNARG,
            },
            DriverContextLifetime::Process,
            0
        ),
        MemoryKind::System
    );
    for pool in [CPU_POOL_FINE, CPU_POOL_EXTENDED, CPU_POOL_COARSE] {
        assert_eq!(
            cpu_pool_memory_kind(
                HsaMemoryPool { handle: pool },
                DriverContextLifetime::Session,
                0
            ),
            MemoryKind::System
        );
    }
}

#[test]
/// Verifies that GFX12 GPU pool and region enumeration matches `ROCr`.
fn gfx12_gpu_pool_and_region_enumeration_matches_rocr() {
    let index = 2;
    let pools = gpu_agent_pools(index, false, false)
        .into_iter()
        .map(|pool| pool.handle)
        .collect::<Vec<_>>();
    assert_eq!(
        pools,
        [
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
        ]
    );
    let fine_pools = gpu_agent_pools(index, true, false)
        .into_iter()
        .map(|pool| pool.handle)
        .collect::<Vec<_>>();
    assert_eq!(
        fine_pools,
        [
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_FINE,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
        ]
    );

    let extended_pools = gpu_agent_pools(index, true, true)
        .into_iter()
        .map(|pool| pool.handle)
        .collect::<Vec<_>>();
    assert_eq!(
        extended_pools,
        [
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_EXTENDED,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_FINE,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
        ]
    );

    let regions = gpu_agent_regions(index, true, false)
        .into_iter()
        .map(|pool| pool.handle)
        .collect::<Vec<_>>();
    assert_eq!(
        regions,
        [
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_FINE,
            GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
            CPU_POOL_FINE,
            CPU_POOL_KERNARG,
            CPU_POOL_EXTENDED,
            CPU_POOL_COARSE,
        ]
    );
}

#[test]
/// Verifies that agent pool access matches `ROCr` relationships.
fn agent_pool_access_matches_rocr_relationships() {
    assert_eq!(
        pool_access(true, PoolStorage::System, false, false),
        POOL_ACCESS_DEFAULT
    );
    assert_eq!(
        pool_access(true, PoolStorage::LocalCoarse, false, false),
        POOL_ACCESS_DEFAULT
    );
    assert_eq!(
        pool_access(true, PoolStorage::Group, false, false),
        POOL_ACCESS_DEFAULT
    );
    assert_eq!(
        pool_access(false, PoolStorage::System, true, false),
        POOL_ACCESS_DISALLOWED
    );
    assert_eq!(
        pool_access(false, PoolStorage::LocalCoarse, true, false),
        POOL_ACCESS_DISALLOWED
    );
    assert_eq!(
        pool_access(false, PoolStorage::LocalCoarse, false, false),
        POOL_ACCESS_NEVER
    );
    assert_eq!(
        pool_access(false, PoolStorage::LocalFine, true, true),
        POOL_ACCESS_DISALLOWED
    );
    assert_eq!(
        pool_access(false, PoolStorage::LocalFine, true, false),
        POOL_ACCESS_NEVER
    );
    assert_eq!(
        pool_access(false, PoolStorage::Group, true, true),
        POOL_ACCESS_NEVER
    );
}

#[test]
/// Verifies that topology link types use the HSA AMD encoding.
fn topology_link_types_use_the_hsa_amd_encoding() {
    assert_eq!(hsa_link_type(MemoryLinkType::HyperTransport), 0);
    assert_eq!(hsa_link_type(MemoryLinkType::Qpi), 1);
    assert_eq!(hsa_link_type(MemoryLinkType::Pcie), 2);
    assert_eq!(hsa_link_type(MemoryLinkType::Infiniband), 3);
    assert_eq!(hsa_link_type(MemoryLinkType::Xgmi), 4);
    assert_eq!(hsa_link_type(MemoryLinkType::Unknown), 0);
}
