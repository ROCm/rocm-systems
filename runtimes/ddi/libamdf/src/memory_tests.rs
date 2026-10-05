//! Regression coverage for host transition recipes, independent of GPU access.
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
use super::*;
use crate::instance;
use rocddi::host_storage::{Allocator, Buffer};
use std::mem::size_of;

#[test]
fn dma_buf_export_remains_available_without_native_import_capability() {
    assert!(supports_dma_buf_export(false, false, true));
    for (local, registered, endpoint) in [
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ] {
        assert!(!supports_dma_buf_export(local, registered, endpoint));
    }
}

#[test]
fn dma_buf_support_enforces_offsets_lengths_and_operations() {
    let mut profile = amdf_memory_profile_t {
        external_memory_support_count: 1,
        ..Default::default()
    };
    profile.external_memory_support[0] = amdf_external_memory_support_t {
        r#type: AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        flags: AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT
            | AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT
            | AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET,
        source_offset_alignment: 4096,
        byte_length_alignment: 1,
        maximum_byte_length: 65536,
        ..Default::default()
    };
    assert!(
        external_support(
            &profile,
            AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT,
            4096,
            4097,
        )
        .is_ok()
    );
    for (kind, operation, offset, length) in [
        (
            AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT,
            4096,
            4097,
        ),
        (
            AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS,
            4096,
            4097,
        ),
        (
            AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT,
            1,
            4097,
        ),
        (
            AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT,
            4096,
            65537,
        ),
    ] {
        assert_eq!(
            external_support(&profile, kind, operation, offset, length)
                .err()
                .unwrap(),
            UNSUPPORTED
        );
    }
}

#[test]
fn multi_device_admission_requires_unique_devices_and_uniform_requirements() {
    let first = 0x1000_usize as *mut amdf_device_t;
    let second = 0x2000_usize as *mut amdf_device_t;
    let requirements = amdf_memory_access_requirements_t {
        access: AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        flags: AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        address_kinds: 1 << AMDF_MEMORY_ADDRESS_GPU,
        ..Default::default()
    };
    assert!(unique_devices(&[
        amdf_memory_device_access_t {
            device: first,
            requirements,
        },
        amdf_memory_device_access_t {
            device: second,
            requirements,
        },
    ]));
    assert!(!unique_devices(&[
        amdf_memory_device_access_t {
            device: first,
            requirements,
        },
        amdf_memory_device_access_t {
            device: first,
            requirements,
        },
    ]));
    assert!(uniform_requirements([requirements, requirements]));
    assert!(!uniform_requirements([
        requirements,
        amdf_memory_access_requirements_t {
            access: AMDF_MEMORY_ACCESS_READ,
            ..requirements
        },
    ]));
}

#[test]
fn local_backing_uses_the_scope_owner_even_when_it_is_not_first() {
    let owner = [2; 16];
    assert_eq!(
        physical_owner_index(true, Some(owner), [[1; 16], owner, [3; 16]]),
        Ok(1)
    );
    assert_eq!(physical_owner_index(false, None, [[1; 16], owner]), Ok(0));
    assert_eq!(
        physical_owner_index(true, Some(owner), [[1; 16], [3; 16]]),
        Err(INTERNAL)
    );
}

unsafe extern "C" fn reject_metadata_allocation(
    _: *mut std::ffi::c_void,
    _: u64,
    _: u64,
) -> *mut std::ffi::c_void {
    std::ptr::null_mut()
}

unsafe extern "C" fn ignore_metadata_free(_: *mut std::ffi::c_void, _: *mut std::ffi::c_void) {}

#[test]
fn duplicate_device_handles_are_invalid_before_metadata_allocation() {
    unsafe {
        let (instance, memory) = host_resource();
        assert_eq!(destroy(memory), 0);
        (*instance.cast::<Instance>()).allocator = Allocator::from_callbacks(
            std::ptr::null_mut(),
            Some(reject_metadata_allocation),
            None,
            Some(ignore_metadata_free),
        )
        .unwrap();
        let scope = (&raw mut (*instance.cast::<Instance>()).scope).cast();
        let request = amdf_memory_device_access_t {
            device: std::ptr::null_mut(),
            requirements: amdf_memory_access_requirements_t::default(),
        };
        let accesses = [request, request];
        let mut profile = amdf_memory_profile_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
            structure_size: size_of::<amdf_memory_profile_t>() as u32,
            ..Default::default()
        };
        let mut capabilities = [amdf_memory_access_capabilities_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
            structure_size: size_of::<amdf_memory_access_capabilities_t>() as u32,
            ..Default::default()
        }; 2];
        assert_eq!(
            device_profile(
                scope,
                0,
                2,
                accesses.as_ptr(),
                &raw mut profile,
                capabilities.as_mut_ptr(),
            ),
            INVALID
        );
        let create_info = amdf_memory_create_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            structure_size: size_of::<amdf_memory_create_info_t>() as u32,
            access_count: 2,
            byte_length: 4096,
            accesses: accesses.as_ptr(),
            ..Default::default()
        };
        let sentinel = std::ptr::NonNull::<amdf_memory_t>::dangling().as_ptr();
        let mut output = sentinel;
        assert_eq!(
            create(scope, &raw const create_info, &raw mut output),
            INVALID
        );
        assert_eq!(output, sentinel);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn unsupported_import_preserves_the_external_value_and_output() {
    unsafe {
        let (instance, memory) = host_resource();
        let scope = (&raw mut (*instance.cast::<Instance>()).scope).cast();
        let import_info = amdf_memory_import_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
            structure_size: size_of::<amdf_memory_import_info_t>() as u32,
            ..Default::default()
        };
        let mut external = amdf_external_memory_t {
            r#type: AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
            payload: amdf_external_memory_payload_t {
                file_descriptor: 41,
            },
            byte_length: 4096,
            ..Default::default()
        };
        let sentinel = std::ptr::dangling_mut::<amdf_memory_t>();
        let mut output = sentinel;
        assert_eq!(
            import(
                scope,
                &raw const import_info,
                &raw mut external,
                &raw mut output,
            ),
            UNSUPPORTED
        );
        assert_eq!(output, sentinel);
        assert_eq!(external.r#type, AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
        assert_eq!(external.payload.file_descriptor, 41);
        assert_eq!(external.byte_length, 4096);
        assert_eq!(destroy(memory), 0);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn export_range_errors_are_invalid_and_preserve_output() {
    unsafe {
        let (instance, memory) = host_resource();
        let export_info = amdf_memory_export_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
            structure_size: size_of::<amdf_memory_export_info_t>() as u32,
            external_memory_type: AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
            byte_offset: 4096,
            byte_length: 2,
            ..Default::default()
        };
        let mut output = amdf_external_memory_t {
            r#type: AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD,
            byte_length: 77,
            ..Default::default()
        };
        assert_eq!(
            export(memory, &raw const export_info, &raw mut output),
            INVALID
        );
        assert_eq!(output.r#type, AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD);
        assert_eq!(output.byte_length, 77);
        assert_eq!(destroy(memory), 0);
        assert_eq!(instance::destroy(instance), 0);
    }
}

unsafe fn host_resource() -> (*mut amdf_instance_t, *mut amdf_memory_t) {
    let parameters = amdf_instance_create_info_t {
        r#type: AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        structure_size: size_of::<amdf_instance_create_info_t>() as u32,
        ..Default::default()
    };
    let mut instance = std::ptr::null_mut();
    unsafe {
        assert_eq!(
            instance::create(&raw const parameters, &raw mut instance),
            0
        );
    };
    let mut scope = std::ptr::null_mut();
    let mut count = 0;
    unsafe {
        assert_eq!(
            instance_scopes(instance, 1, &raw mut scope, &raw mut count),
            0
        );
    };
    let parameters = amdf_memory_create_info_t {
        r#type: AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        structure_size: size_of::<amdf_memory_create_info_t>() as u32,
        byte_length: 4097,
        ..Default::default()
    };
    let mut memory = std::ptr::null_mut();
    unsafe { assert_eq!(create(scope, &raw const parameters, &raw mut memory), 0) };
    (instance, memory)
}

#[test]
fn mapping_overrun_is_invalid_and_preserves_output() {
    unsafe {
        let (instance, memory) = host_resource();
        let parameters = amdf_memory_map_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
            structure_size: size_of::<amdf_memory_map_info_t>() as u32,
            byte_offset: 4096,
            byte_length: 2,
            flags: AMDF_MEMORY_MAP_FLAG_READ,
            ..Default::default()
        };
        let sentinel = std::ptr::dangling_mut::<amdf_host_mapping_t>();
        let mut mapping = sentinel;
        assert_eq!(
            map(memory, &raw const parameters, &raw mut mapping),
            INVALID
        );
        assert_eq!(mapping, sentinel);
        assert_eq!(destroy(memory), 0);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn mapping_info_preserves_extended_output_header_and_tail() {
    #[repr(C)]
    struct Extended {
        info: amdf_host_mapping_info_t,
        tail: [u8; 16],
    }

    unsafe {
        let (instance, memory) = host_resource();
        let parameters = amdf_memory_map_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
            structure_size: size_of::<amdf_memory_map_info_t>() as u32,
            byte_offset: 17,
            byte_length: 257,
            flags: AMDF_MEMORY_MAP_FLAG_READ,
            ..Default::default()
        };
        let mut mapping = std::ptr::null_mut();
        assert_eq!(map(memory, &raw const parameters, &raw mut mapping), 0);
        let mut extended = Extended {
            info: amdf_host_mapping_info_t {
                r#type: AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
                structure_size: size_of::<Extended>() as u32,
                ..Default::default()
            },
            tail: [0xa5; 16],
        };
        assert_eq!(mapping_info(mapping, &raw mut extended.info), 0);
        assert_eq!(extended.info.structure_size, size_of::<Extended>() as u32);
        assert!(extended.info.next.is_null());
        assert_eq!(extended.info.memory_byte_offset, 17);
        assert_eq!(extended.info.byte_length, 257);
        assert_eq!(extended.tail, [0xa5; 16]);
        assert_eq!(unmap(mapping), 0);
        assert_eq!(destroy(memory), 0);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn write_combined_views_require_fences_even_for_host_pairs() {
    // A real CPU allocation supplies valid lifetimes. Changing only the cached
    // cacheability exercises WC recipe selection without requiring public VRAM.
    unsafe {
        let (instance, memory) = host_resource();
        (*memory.cast::<Memory>()).cacheability = AMDF_HOST_CACHEABILITY_WRITE_COMBINED;
        let parameters = amdf_memory_map_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
            structure_size: size_of::<amdf_memory_map_info_t>() as u32,
            byte_offset: 17,
            byte_length: 257,
            flags: AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
            ..Default::default()
        };
        let mut mapping = std::ptr::null_mut();
        assert_eq!(map(memory, &raw const parameters, &raw mut mapping), 0);
        let mut info = amdf_host_mapping_info_t {
            r#type: AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
            structure_size: size_of::<amdf_host_mapping_info_t>() as u32,
            ..Default::default()
        };
        assert_eq!(mapping_info(mapping, &raw mut info), 0);
        assert_eq!(info.cache_line_size, 0);
        assert_eq!(info.flush.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
        assert_eq!(
            info.flush.host_fence_after,
            AMDF_HOST_CACHE_FENCE_X86_MFENCE
        );
        assert_eq!(
            info.flush.host_instruction,
            AMDF_HOST_CACHE_INSTRUCTION_NONE
        );
        assert_eq!(
            cache_control(mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 257),
            0
        );
        assert_eq!(
            cache_control(mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 256, 2),
            INVALID
        );
        let site = amdf_memory_site_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_SITE,
            structure_size: size_of::<amdf_memory_site_t>() as u32,
            kind: AMDF_MEMORY_SITE_KIND_HOST,
            value: amdf_memory_site_t__value {
                host_mapping: mapping,
            },
            ..Default::default()
        };
        let mut pair = amdf_memory_pair_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
            structure_size: size_of::<amdf_memory_pair_info_t>() as u32,
            ..Default::default()
        };
        assert_eq!(
            pair_info(&raw const site, &raw const site, &raw mut pair),
            0
        );
        assert_eq!(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
        assert_eq!(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
        assert_eq!(pair.flags & AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN, 0);
        assert_eq!(destroy(memory), BUSY);
        assert_eq!(unmap(mapping), 0);
        assert_eq!(destroy(memory), 0);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn unavailable_host_recipe_does_not_publish_a_host_visible_profile() {
    unsafe {
        let (instance, memory) = host_resource();
        assert_eq!(destroy(memory), 0);
        (*instance.cast::<Instance>()).host_cache_line = None;
        let scope = (&raw mut (*instance.cast::<Instance>()).scope).cast();
        let mut profile = amdf_memory_profile_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
            structure_size: size_of::<amdf_memory_profile_t>() as u32,
            roles: u64::MAX,
            ..Default::default()
        };
        assert_eq!(
            device_profile(
                scope,
                0,
                0,
                std::ptr::null(),
                &raw mut profile,
                std::ptr::null_mut()
            ),
            UNSUPPORTED
        );
        assert_eq!(profile.roles, u64::MAX);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn registration_requires_the_declared_host_cacheability() {
    unsafe {
        let (instance, owned) = host_resource();
        assert_eq!(destroy(owned), 0);
        let scope = (&raw mut (*instance.cast::<Instance>()).scope).cast();
        let mut profile = amdf_memory_profile_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
            structure_size: size_of::<amdf_memory_profile_t>() as u32,
            ..Default::default()
        };
        assert_eq!(
            device_profile(
                scope,
                1,
                0,
                std::ptr::null(),
                &raw mut profile,
                std::ptr::null_mut()
            ),
            0
        );
        assert_ne!(profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER, 0);
        assert_eq!(
            profile.registration.registered_host_cacheability,
            AMDF_HOST_CACHEABILITY_WRITE_BACK
        );
        let mut bytes = [0_u8; 32];
        let mut parameters = amdf_memory_create_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            structure_size: size_of::<amdf_memory_create_info_t>() as u32,
            memory_profile_ordinal: profile.ordinal,
            registered_host_pointer: bytes.as_mut_ptr().cast(),
            byte_length: bytes.len() as u64,
            ..Default::default()
        };
        let mut memory = std::ptr::null_mut();
        assert_eq!(
            create(scope, &raw const parameters, &raw mut memory),
            UNSUPPORTED
        );
        assert!(memory.is_null());
        parameters.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
        assert_eq!(create(scope, &raw const parameters, &raw mut memory), 0);
        let page = memory::host_page_size().unwrap();
        let address = bytes.as_ptr() as u64;
        let source_byte_offset = address & (page - 1);
        let mut memory_info = amdf_memory_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_INFO,
            structure_size: size_of::<amdf_memory_info_t>() as u32,
            ..Default::default()
        };
        assert_eq!(info(memory, &raw mut memory_info), 0);
        assert_eq!(memory_info.source_byte_offset, source_byte_offset);
        assert_eq!(
            memory_info.native_allocation_byte_length,
            rounded(source_byte_offset + bytes.len() as u64, page).unwrap()
        );
        assert_eq!(memory_info.alignment, 1u64 << address.trailing_zeros());
        assert_eq!(memory_info.physical_backing_id.words, [0; 2]);
        let parameters = amdf_memory_map_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
            structure_size: size_of::<amdf_memory_map_info_t>() as u32,
            byte_offset: 3,
            byte_length: 7,
            flags: AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
            ..Default::default()
        };
        let mut mapping = std::ptr::null_mut();
        assert_eq!(map(memory, &raw const parameters, &raw mut mapping), 0);
        let mut mapping_info = amdf_host_mapping_info_t {
            r#type: AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
            structure_size: size_of::<amdf_host_mapping_info_t>() as u32,
            ..Default::default()
        };
        assert_eq!(super::mapping_info(mapping, &raw mut mapping_info), 0);
        assert_eq!(mapping_info.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
        mapping_info.pointer.cast::<u8>().write(0x5a);
        assert_eq!(bytes[3], 0x5a);
        assert_eq!(destroy(memory), BUSY);
        assert_eq!(unmap(mapping), 0);
        assert_eq!(destroy(memory), 0);
        bytes[3] = 0xa5;
        assert_eq!(bytes[3], 0xa5);
        assert_eq!(instance::destroy(instance), 0);
    }
}

#[test]
fn queue_scratch_resolves_one_device_range_and_blocks_early_free() {
    unsafe {
        let allocator = Allocator::system();
        let device = std::ptr::NonNull::<Device>::dangling().as_ptr();
        let other_device = device.wrapping_add(1);
        let mut accesses = Buffer::try_with_capacity(1, allocator).unwrap();
        accesses
            .try_push(Access {
                device,
                info: amdf_memory_access_info_t {
                    access: AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                    flags: AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                    address_kinds: 1 << AMDF_MEMORY_ADDRESS_GPU,
                    reset_epoch: 7,
                    ..Default::default()
                },
                address: 0x0080_0000,
            })
            .unwrap();
        let owner = Owned::new(
            Memory {
                scope: std::ptr::null_mut(),
                backing: Backing::Registered,
                info: amdf_memory_info_t {
                    access_count: 1,
                    byte_length: 16 * 1024,
                    ..Default::default()
                },
                access: accesses,
                host: None,
                cacheability: AMDF_HOST_CACHEABILITY_UNKNOWN,
                children: AtomicU64::new(0),
                freeing: false,
                host_backing_next: AtomicUsize::new(0),
                listed_host_backing: false,
            },
            allocator,
        )
        .unwrap();
        let pointer = owner.into_raw();
        let scratch = queue_scratch(pointer.cast(), 0, device, 7, 4096, 8192).unwrap();
        assert_eq!(scratch.memory, pointer);
        assert_eq!(scratch.device_address, 0x0080_1000);
        assert_eq!(
            queue_scratch(pointer.cast(), 1, device, 7, 0, 1),
            Err(RANGE)
        );
        assert_eq!(
            queue_scratch(pointer.cast(), 0, other_device, 7, 0, 1),
            Err(INVALID)
        );
        assert_eq!(
            queue_scratch(pointer.cast(), 0, device, 8, 0, 1),
            Err(PRECONDITION)
        );
        assert_eq!(
            queue_scratch(pointer.cast(), 0, device, 7, 16 * 1024, 1),
            Err(RANGE)
        );
        let borrow = borrow_for_queue(pointer).unwrap();
        assert_eq!(destroy(pointer.cast()), BUSY);
        drop(borrow);
        assert_eq!(destroy(pointer.cast()), 0);
    }
}

#[test]
fn kernel_commands_require_exact_executable_device_access_and_logical_range() {
    unsafe {
        let allocator = Allocator::system();
        let device = std::ptr::NonNull::<Device>::dangling().as_ptr();
        let mut accesses = Buffer::try_with_capacity(1, allocator).unwrap();
        accesses
            .try_push(Access {
                device,
                info: amdf_memory_access_info_t {
                    access: AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                    flags: AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                    address_kinds: 1 << AMDF_MEMORY_ADDRESS_GPU,
                    reset_epoch: 7,
                    ..Default::default()
                },
                address: 0x0080_0000,
            })
            .unwrap();
        let owner = Owned::new(
            Memory {
                scope: std::ptr::null_mut(),
                backing: Backing::Registered,
                info: amdf_memory_info_t {
                    access_count: 1,
                    byte_length: 4096,
                    ..Default::default()
                },
                access: accesses,
                host: None,
                cacheability: AMDF_HOST_CACHEABILITY_UNKNOWN,
                children: AtomicU64::new(0),
                freeing: false,
                host_backing_next: AtomicUsize::new(0),
                listed_host_backing: false,
            },
            allocator,
        )
        .unwrap();
        let pointer = owner.into_raw();
        let command = kernel_command(pointer.cast(), 0, device, 7, 512, 24).unwrap();
        assert_eq!(command.device_address, 0x0080_0200);
        assert_eq!(command.byte_length, 24);
        assert_eq!(
            kernel_command(pointer.cast(), 0, device, 7, 0, 0),
            Err(INVALID)
        );
        assert_eq!(
            kernel_command(pointer.cast(), 0, device, 7, 2, 24),
            Err(INVALID)
        );
        assert_eq!(
            kernel_command(pointer.cast(), 1, device, 7, 0, 24),
            Err(RANGE)
        );
        assert_eq!(
            kernel_command(pointer.cast(), 0, device, 8, 0, 24),
            Err(PRECONDITION)
        );
        assert_eq!(
            kernel_command(pointer.cast(), 0, device, 7, 4092, 24),
            Err(RANGE)
        );
        (&mut (*pointer).access)[0].info.access = AMDF_MEMORY_ACCESS_READ;
        assert_eq!(
            kernel_command(pointer.cast(), 0, device, 7, 0, 24),
            Err(UNSUPPORTED)
        );
        assert_eq!(destroy(pointer.cast()), 0);
    }
}

#[test]
fn multi_device_access_queries_and_scratch_preserve_request_order() {
    unsafe {
        let allocator = Allocator::system();
        let first = std::ptr::NonNull::<Device>::dangling().as_ptr();
        let second = first.wrapping_add(1);
        let mut accesses = Buffer::try_with_capacity(2, allocator).unwrap();
        for (ordinal, (device, address)) in [(first, 0x0080_0000), (second, 0x0090_0000)]
            .into_iter()
            .enumerate()
        {
            accesses
                .try_push(Access {
                    device,
                    info: amdf_memory_access_info_t {
                        ordinal: ordinal as u32,
                        access: AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                        device_id: amdf_device_id_t {
                            words: [ordinal as u64 + 1, 0],
                        },
                        flags: AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                        address_kinds: 1 << AMDF_MEMORY_ADDRESS_GPU,
                        reset_epoch: 7,
                        ..Default::default()
                    },
                    address,
                })
                .unwrap();
        }
        let owner = Owned::new(
            Memory {
                scope: std::ptr::null_mut(),
                backing: Backing::Registered,
                info: amdf_memory_info_t {
                    access_count: 2,
                    byte_length: 16 * 1024,
                    ..Default::default()
                },
                access: accesses,
                host: None,
                cacheability: AMDF_HOST_CACHEABILITY_UNKNOWN,
                children: AtomicU64::new(0),
                freeing: false,
                host_backing_next: AtomicUsize::new(0),
                listed_host_backing: false,
            },
            allocator,
        )
        .unwrap();
        let pointer = owner.into_raw();
        for (ordinal, (device, expected_address)) in [(first, 0x0080_0000), (second, 0x0090_0000)]
            .into_iter()
            .enumerate()
        {
            let mut access = amdf_memory_access_info_t {
                r#type: AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
                structure_size: size_of::<amdf_memory_access_info_t>() as u32,
                ..Default::default()
            };
            assert_eq!(
                access_info(pointer.cast(), ordinal as u32, &raw mut access),
                0
            );
            assert_eq!(access.ordinal, ordinal as u32);
            assert_eq!(access.device_id.words[0], ordinal as u64 + 1);
            let mut address_output = u64::MAX;
            assert_eq!(
                address(
                    pointer.cast(),
                    ordinal as u32,
                    AMDF_MEMORY_ADDRESS_GPU,
                    &raw mut address_output,
                ),
                0
            );
            assert_eq!(address_output, expected_address);
            let scratch =
                queue_scratch(pointer.cast(), ordinal as u32, device, 7, 4096, 4096).unwrap();
            assert_eq!(scratch.device_address, expected_address + 4096);
        }
        let mut unchanged = amdf_memory_access_info_t {
            r#type: AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
            structure_size: size_of::<amdf_memory_access_info_t>() as u32,
            ordinal: 99,
            ..Default::default()
        };
        assert_eq!(access_info(pointer.cast(), 2, &raw mut unchanged), RANGE);
        assert_eq!(unchanged.ordinal, 99);
        assert_eq!(
            queue_scratch(pointer.cast(), 1, first, 7, 0, 1),
            Err(INVALID)
        );
        assert_eq!(destroy(pointer.cast()), 0);
    }
}

fn test_site(host: bool, readable: bool, writable: bool) -> Site {
    Site {
        memory: std::ptr::null_mut(),
        host,
        access: if readable { AMDF_MEMORY_ACCESS_READ } else { 0 }
            | if writable {
                AMDF_MEMORY_ACCESS_WRITE
            } else {
                0
            },
        host_coherent: !host,
        cacheability: if host {
            AMDF_HOST_CACHEABILITY_WRITE_BACK
        } else {
            AMDF_HOST_CACHEABILITY_UNKNOWN
        },
        release: if host {
            amdf_cache_transition_t::default()
        } else {
            amdf_cache_transition_t {
                kind: AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                executor: AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
                operation: AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
                ..Default::default()
            }
        },
        acquire: if host {
            amdf_cache_transition_t::default()
        } else {
            amdf_cache_transition_t {
                kind: AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                executor: AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
                operation: AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
                ..Default::default()
            }
        },
    }
}

fn aql_family(cache_operations: u64) -> amdf_queue_family_info_t {
    amdf_queue_family_info_t {
        command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
        roles: AMDF_QUEUE_ROLE_COMPUTE
            | if cache_operations == 0 {
                0
            } else {
                AMDF_QUEUE_ROLE_CACHE_CONTROL
            },
        cache_operations,
        cache_transition_kinds: if cache_operations == 0 {
            0
        } else {
            AMDF_CACHE_TRANSITION_KINDS_GLOBAL
        },
        ..Default::default()
    }
}

#[test]
fn qualified_aql_system_pairs_publish_directional_gpu_transitions() {
    let host = test_site(true, true, true);
    let device = device_site(
        std::ptr::null_mut(),
        AMDF_MEMORY_CLASS_SYSTEM,
        AMDF_MEMORY_FLAG_HOST_COHERENT,
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        &aql_family(
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM | AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
        ),
        false,
    );
    let to_device = describe_pair(&host, &device).unwrap();
    assert_eq!(to_device.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    assert_eq!(to_device.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    assert_eq!(
        to_device.acquire.executor,
        AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE
    );
    assert_eq!(
        to_device.acquire.operation,
        AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
    );
    assert_eq!(to_device.flags & AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN, 0);

    let to_host = describe_pair(&device, &host).unwrap();
    assert_eq!(to_host.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    assert_eq!(
        to_host.release.operation,
        AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
    );
    assert_eq!(to_host.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    assert_eq!(to_host.flags & AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN, 0);
}

#[test]
fn qualified_system_device_pairs_publish_release_then_acquire() {
    let producer = test_site(false, true, true);
    let consumer = test_site(false, true, true);
    let pair = describe_pair(&producer, &consumer).unwrap();
    assert_eq!(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    assert_eq!(
        pair.release.operation,
        AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
    );
    assert_eq!(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    assert_eq!(
        pair.acquire.operation,
        AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
    );
    assert_eq!(pair.flags & AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN, 0);
}

#[test]
fn incomplete_aql_system_cache_control_rejects_the_affected_pair_direction() {
    let host = test_site(true, true, true);
    for (operations, to_device_status, to_host_status) in [
        (0, UNSUPPORTED, UNSUPPORTED),
        (AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM, UNSUPPORTED, 0),
        (AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM, 0, UNSUPPORTED),
    ] {
        let device = device_site(
            std::ptr::null_mut(),
            AMDF_MEMORY_CLASS_SYSTEM,
            AMDF_MEMORY_FLAG_HOST_COHERENT,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            &aql_family(operations),
            false,
        );
        assert_eq!(
            describe_pair(&host, &device).err().unwrap_or(0),
            to_device_status
        );
        assert_eq!(
            describe_pair(&device, &host).err().unwrap_or(0),
            to_host_status
        );
    }
}

#[test]
fn pair_direction_requires_producer_write_and_consumer_read_access() {
    let host = test_site(true, true, true);
    let device = test_site(false, true, true);
    for (producer, consumer) in [
        (test_site(true, true, false), device),
        (host, test_site(false, false, true)),
        (test_site(false, true, false), host),
        (device, test_site(true, false, true)),
    ] {
        assert_eq!(
            describe_pair(&producer, &consumer).err().unwrap(),
            UNSUPPORTED
        );
    }
    let mut unknown_host = host;
    unknown_host.cacheability = AMDF_HOST_CACHEABILITY_UNKNOWN;
    assert_eq!(
        describe_pair(&unknown_host, &device).err().unwrap(),
        UNSUPPORTED
    );
}

fn wc_local_host_site() -> Site {
    let mut host = test_site(true, true, true);
    host.cacheability = AMDF_HOST_CACHEABILITY_WRITE_COMBINED;
    host.release = amdf_cache_transition_t {
        kind: AMDF_CACHE_TRANSITION_KIND_GLOBAL,
        executor: AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
        host_operation: AMDF_HOST_CACHE_OPERATION_FLUSH,
        host_fence_after: AMDF_HOST_CACHE_FENCE_X86_MFENCE,
        ..Default::default()
    };
    host.acquire = amdf_cache_transition_t {
        host_operation: AMDF_HOST_CACHE_OPERATION_INVALIDATE,
        ..host.release
    };
    host
}

#[test]
fn single_owner_wc_local_sdma_and_aql_pairs_are_qualified() {
    let host = wc_local_host_site();
    let family = amdf_queue_family_info_t {
        command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
        format_version: AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
        format_features: AMDF_GPU_SDMA_FORMAT_FEATURE_GCR
            | AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM,
        cache_operations: AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM
            | AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
        cache_transition_kinds: AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
        ..Default::default()
    };
    let device = device_site(
        std::ptr::null_mut(),
        AMDF_MEMORY_CLASS_LOCAL,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE,
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        &family,
        true,
    );
    let to_device = describe_pair(&host, &device).unwrap();
    assert_eq!(to_device.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    assert_eq!(
        to_device.release.executor,
        AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT
    );
    assert_eq!(
        to_device.acquire.operation,
        AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
    );
    let to_host = describe_pair(&device, &host).unwrap();
    assert_eq!(
        to_host.release.operation,
        AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
    );
    assert_eq!(
        to_host.acquire.host_operation,
        AMDF_HOST_CACHE_OPERATION_INVALIDATE
    );

    let aql_family = amdf_queue_family_info_t {
        command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
        format_version: 1,
        format_features: 0,
        ..family
    };
    let aql_device = device_site(
        std::ptr::null_mut(),
        AMDF_MEMORY_CLASS_LOCAL,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE,
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        &aql_family,
        true,
    );
    assert!(describe_pair(&host, &aql_device).is_ok());
    assert!(describe_pair(&aql_device, &host).is_ok());

    for (flags, single_owner, family) in [
        (AMDF_MEMORY_FLAG_DEVICE_LOCAL, true, family),
        (
            AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE,
            false,
            family,
        ),
    ] {
        let unqualified = device_site(
            std::ptr::null_mut(),
            AMDF_MEMORY_CLASS_LOCAL,
            flags,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            &family,
            single_owner,
        );
        assert_eq!(describe_pair(&host, &unqualified).err(), Some(UNSUPPORTED));
        assert_eq!(describe_pair(&unqualified, &host).err(), Some(UNSUPPORTED));
    }
}

#[test]
fn single_owner_wc_local_pm4_pairs_require_the_qualified_format() {
    let host = wc_local_host_site();
    let pm4_family = amdf_queue_family_info_t {
        command_type: AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
        format_version: AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
        format_features: AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
        cache_operations: AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM
            | AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
        cache_transition_kinds: AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
        ..Default::default()
    };
    let pm4_device = device_site(
        std::ptr::null_mut(),
        AMDF_MEMORY_CLASS_LOCAL,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE,
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        &pm4_family,
        true,
    );
    let to_pm4 = describe_pair(&host, &pm4_device).unwrap();
    assert_eq!(
        to_pm4.release.host_operation,
        AMDF_HOST_CACHE_OPERATION_FLUSH
    );
    assert_eq!(
        to_pm4.acquire.operation,
        AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
    );
    let from_pm4 = describe_pair(&pm4_device, &host).unwrap();
    assert_eq!(
        from_pm4.release.operation,
        AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
    );
    assert_eq!(
        from_pm4.acquire.host_operation,
        AMDF_HOST_CACHE_OPERATION_INVALIDATE
    );

    for family in [
        amdf_queue_family_info_t {
            format_version: 2,
            ..pm4_family
        },
        amdf_queue_family_info_t {
            format_features: 0,
            ..pm4_family
        },
    ] {
        let unqualified = device_site(
            std::ptr::null_mut(),
            AMDF_MEMORY_CLASS_LOCAL,
            AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            &family,
            true,
        );
        assert_eq!(describe_pair(&host, &unqualified).err(), Some(UNSUPPORTED));
        assert_eq!(describe_pair(&unqualified, &host).err(), Some(UNSUPPORTED));
    }
}
