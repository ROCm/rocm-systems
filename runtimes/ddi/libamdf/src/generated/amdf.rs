// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Snapshot generated from the pinned AMDF headers. Do not edit by hand.
// Source: api-headers/include/amdf/ at upstream 4aa34130de44c45d68a48575cebfd0ff0610c461.
//! AMDF C ABI declarations. See api-headers/README.md.
//!
//! Zero defaults initialize storage only. Callers must set structure tags,
//! structure sizes, and all other required inputs before calling the API.
//! C inline helpers and preprocessor controls are defined only in the headers.

#![allow(
    dead_code,
    missing_docs,
    non_camel_case_types,
    non_snake_case,
    clippy::all,
    clippy::pedantic
)]

pub type amdf_abi_version_t = u32;

pub type amdf_status_t = u64;

pub type amdf_status_domain_t = u32;

pub type amdf_status_domain_e = ::core::ffi::c_uint;

pub const AMDF_STATUS_DOMAIN_API: amdf_status_domain_t = 0;

pub const AMDF_STATUS_DOMAIN_NTSTATUS: amdf_status_domain_t = 1;

pub const AMDF_STATUS_DOMAIN_FIRMWARE: amdf_status_domain_t = 2;

pub const AMDF_STATUS_DOMAIN_ERRNO: amdf_status_domain_t = 3;

pub const AMDF_STATUS_DOMAIN_WIN32: amdf_status_domain_t = 4;

pub const AMDF_STATUS_DOMAIN_HRESULT: amdf_status_domain_t = 5;

pub type amdf_status_code_t = u32;

pub type amdf_status_code_e = ::core::ffi::c_uint;

pub const AMDF_STATUS_CODE_OK: amdf_status_code_t = 0;

pub const AMDF_STATUS_CODE_INVALID_ARGUMENT: amdf_status_code_t = 1;

pub const AMDF_STATUS_CODE_OUT_OF_RANGE: amdf_status_code_t = 2;

pub const AMDF_STATUS_CODE_UNSUPPORTED: amdf_status_code_t = 3;

pub const AMDF_STATUS_CODE_NOT_FOUND: amdf_status_code_t = 4;

pub const AMDF_STATUS_CODE_RESOURCE_EXHAUSTED: amdf_status_code_t = 5;

pub const AMDF_STATUS_CODE_BUSY: amdf_status_code_t = 6;

pub const AMDF_STATUS_CODE_DEADLINE_EXCEEDED: amdf_status_code_t = 7;

pub const AMDF_STATUS_CODE_PERMISSION_DENIED: amdf_status_code_t = 8;

pub const AMDF_STATUS_CODE_DEVICE_LOST: amdf_status_code_t = 9;

pub const AMDF_STATUS_CODE_VERSION_MISMATCH: amdf_status_code_t = 10;

pub const AMDF_STATUS_CODE_BUFFER_TOO_SMALL: amdf_status_code_t = 11;

pub const AMDF_STATUS_CODE_FAILED_PRECONDITION: amdf_status_code_t = 12;

pub const AMDF_STATUS_CODE_INTERNAL: amdf_status_code_t = 13;

pub type amdf_structure_type_t = u32;

pub type amdf_structure_type_e = ::core::ffi::c_uint;

pub const AMDF_STRUCTURE_TYPE_NONE: amdf_structure_type_t = 0;

pub const AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: amdf_structure_type_t = 1;

pub const AMDF_STRUCTURE_TYPE_ENDPOINT_INFO: amdf_structure_type_t = 2;

pub const AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO: amdf_structure_type_t = 3;

pub const AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO: amdf_structure_type_t = 4;

pub const AMDF_STRUCTURE_TYPE_MEMORY_INFO: amdf_structure_type_t = 5;

pub const AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO: amdf_structure_type_t = 6;

pub const AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO: amdf_structure_type_t = 7;

pub const AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO: amdf_structure_type_t = 8;

pub const AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS: amdf_structure_type_t = 9;

pub const AMDF_STRUCTURE_TYPE_MEMORY_PROFILE: amdf_structure_type_t = 10;

pub const AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO: amdf_structure_type_t = 11;

pub const AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO: amdf_structure_type_t = 12;

pub const AMDF_STRUCTURE_TYPE_MEMORY_SITE: amdf_structure_type_t = 13;

pub const AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO: amdf_structure_type_t = 14;

pub const AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO: amdf_structure_type_t = 15;

pub const AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO: amdf_structure_type_t = 16;

pub const AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS: amdf_structure_type_t = 17;

pub const AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO: amdf_structure_type_t = 18;

pub const AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO: amdf_structure_type_t = 19;

pub const AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES: amdf_structure_type_t = 20;

pub const AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY: amdf_structure_type_t = 21;

pub type amdf_extension_id_t = u32;

pub type amdf_extension_id_e = ::core::ffi::c_uint;

pub const AMDF_EXTENSION_XDNA: amdf_extension_id_t = 1;

pub const AMDF_EXTENSION_GPU: amdf_extension_id_t = 2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_input_structure_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
}
impl Default for amdf_input_structure_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_output_structure_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
}
impl Default for amdf_output_structure_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_instance_t {
    _opaque: [u8; 0],
}
impl Default for amdf_instance_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_endpoint_t {
    _opaque: [u8; 0],
}
impl Default for amdf_endpoint_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_device_t {
    _opaque: [u8; 0],
}
impl Default for amdf_device_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_scope_t {
    _opaque: [u8; 0],
}
impl Default for amdf_memory_scope_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_t {
    _opaque: [u8; 0],
}
impl Default for amdf_memory_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_host_mapping_t {
    _opaque: [u8; 0],
}
impl Default for amdf_host_mapping_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_user_queue_t {
    _opaque: [u8; 0],
}
impl Default for amdf_user_queue_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_user_queue_mapping_t {
    _opaque: [u8; 0],
}
impl Default for amdf_user_queue_mapping_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_kernel_queue_t {
    _opaque: [u8; 0],
}
impl Default for amdf_kernel_queue_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_allocator_allocate_fn_t =
    Option<unsafe extern "C" fn(*mut ::core::ffi::c_void, u64, u64) -> *mut ::core::ffi::c_void>;

pub type amdf_allocator_resize_fn_t = Option<
    unsafe extern "C" fn(
        *mut ::core::ffi::c_void,
        *mut ::core::ffi::c_void,
        u64,
        u64,
        u64,
    ) -> *mut ::core::ffi::c_void,
>;

pub type amdf_allocator_free_fn_t =
    Option<unsafe extern "C" fn(*mut ::core::ffi::c_void, *mut ::core::ffi::c_void)>;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_allocator_t {
    pub user_data: *mut ::core::ffi::c_void,
    pub allocate: amdf_allocator_allocate_fn_t,
    pub resize: amdf_allocator_resize_fn_t,
    pub free: amdf_allocator_free_fn_t,
}
impl Default for amdf_allocator_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_native_lifetime_t = u32;

pub type amdf_native_lifetime_e = ::core::ffi::c_uint;

pub const AMDF_NATIVE_LIFETIME_PROCESS: amdf_native_lifetime_t = 0;

pub const AMDF_NATIVE_LIFETIME_INSTANCE: amdf_native_lifetime_t = 1;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_instance_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub native_lifetime: amdf_native_lifetime_t,
    pub reserved: u32,
    pub host_allocator: amdf_allocator_t,
}
impl Default for amdf_instance_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_endpoint_id_t {
    pub words: [u64; 2],
}
impl Default for amdf_endpoint_id_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_device_id_t {
    pub words: [u64; 2],
}
impl Default for amdf_device_id_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_queue_id_t {
    pub words: [u64; 2],
}
impl Default for amdf_queue_id_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_physical_memory_id_t {
    pub words: [u64; 2],
}
impl Default for amdf_physical_memory_id_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_engine_kind_t = u32;

pub type amdf_engine_kind_e = ::core::ffi::c_uint;

pub const AMDF_ENGINE_KIND_UNKNOWN: amdf_engine_kind_t = 0;

pub const AMDF_ENGINE_KIND_GPU: amdf_engine_kind_t = 1;

pub const AMDF_ENGINE_KIND_XDNA: amdf_engine_kind_t = 2;

pub type amdf_endpoint_type_flags_t = u32;

pub type amdf_endpoint_type_flag_bits_e = ::core::ffi::c_uint;

pub const AMDF_ENDPOINT_TYPE_FLAG_DISPLAY_SUPPORTED: amdf_endpoint_type_flags_t = 1;

pub const AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED: amdf_endpoint_type_flags_t = 2;

pub const AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY: amdf_endpoint_type_flags_t = 4;

pub const AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE: amdf_endpoint_type_flags_t = 8;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_pci_info_t {
    pub vendor_id: u32,
    pub device_id: u32,
    pub subsystem_vendor_id: u32,
    pub subsystem_device_id: u32,
    pub revision_id: u32,
}
impl Default for amdf_pci_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_endpoint_native_identity_type_t = u32;

pub type amdf_endpoint_native_identity_type_e = ::core::ffi::c_uint;

pub const AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_NONE: amdf_endpoint_native_identity_type_t = 0;

pub const AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE: amdf_endpoint_native_identity_type_t = 1;

pub const AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER: amdf_endpoint_native_identity_type_t =
    2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_endpoint_native_identity_t__value__linux_device {
    pub major: u32,
    pub minor: u32,
}
impl Default for amdf_endpoint_native_identity_t__value__linux_device {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_endpoint_native_identity_t__value__windows_adapter {
    pub luid: u64,
    pub physical_adapter_index: u32,
}
impl Default for amdf_endpoint_native_identity_t__value__windows_adapter {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union amdf_endpoint_native_identity_t__value {
    pub linux_device: amdf_endpoint_native_identity_t__value__linux_device,
    pub windows_adapter: amdf_endpoint_native_identity_t__value__windows_adapter,
}
impl Default for amdf_endpoint_native_identity_t__value {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_endpoint_native_identity_t {
    pub r#type: amdf_endpoint_native_identity_type_t,
    pub value: amdf_endpoint_native_identity_t__value,
}
impl Default for amdf_endpoint_native_identity_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_endpoint_summary_t {
    pub id: amdf_endpoint_id_t,
    pub engine_kind: amdf_engine_kind_t,
    pub type_flags: amdf_endpoint_type_flags_t,
    pub name: [::core::ffi::c_char; 128],
}
impl Default for amdf_endpoint_summary_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_endpoint_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub id: amdf_endpoint_id_t,
    pub engine_kind: amdf_engine_kind_t,
    pub type_flags: amdf_endpoint_type_flags_t,
    pub pci: amdf_pci_info_t,
    pub name: [::core::ffi::c_char; 128],
    pub queue_family_count: u32,
    pub native_identity: amdf_endpoint_native_identity_t,
}
impl Default for amdf_endpoint_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_memory_class_t = u32;

pub type amdf_memory_class_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_CLASS_UNKNOWN: amdf_memory_class_t = 0;

pub const AMDF_MEMORY_CLASS_SYSTEM: amdf_memory_class_t = 1;

pub const AMDF_MEMORY_CLASS_LOCAL: amdf_memory_class_t = 2;

pub const AMDF_MEMORY_CLASS_PRIVATE: amdf_memory_class_t = 3;

pub type amdf_memory_flags_t = u64;

pub type amdf_memory_flag_bits_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_FLAG_HOST_VISIBLE: amdf_memory_flags_t = 1;

pub const AMDF_MEMORY_FLAG_DEVICE_LOCAL: amdf_memory_flags_t = 2;

pub const AMDF_MEMORY_FLAG_SHAREABLE: amdf_memory_flags_t = 4;

pub const AMDF_MEMORY_FLAG_QUEUE_STORAGE: amdf_memory_flags_t = 8;

pub const AMDF_MEMORY_FLAG_HOST_COHERENT: amdf_memory_flags_t = 16;

pub const AMDF_MEMORY_FLAG_DEVICE_ADDRESS: amdf_memory_flags_t = 32;

pub type amdf_memory_access_t = u32;

pub type amdf_memory_access_bit_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_ACCESS_READ: amdf_memory_access_t = 1;

pub const AMDF_MEMORY_ACCESS_WRITE: amdf_memory_access_t = 2;

pub const AMDF_MEMORY_ACCESS_EXECUTE: amdf_memory_access_t = 4;

pub type amdf_memory_address_kind_t = u32;

pub type amdf_memory_address_kind_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_ADDRESS_GPU: amdf_memory_address_kind_t = 0;

pub const AMDF_MEMORY_ADDRESS_XDNA_DMA: amdf_memory_address_kind_t = 1;

pub const AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE: amdf_memory_address_kind_t = 2;

pub type amdf_memory_address_kinds_t = u64;

pub type amdf_memory_map_flags_t = u32;

pub type amdf_memory_map_flag_bits_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_MAP_FLAG_READ: amdf_memory_map_flags_t = 1;

pub const AMDF_MEMORY_MAP_FLAG_WRITE: amdf_memory_map_flags_t = 2;

pub type amdf_atomic_operations_t = u64;

pub type amdf_atomic_operation_bits_e = ::core::ffi::c_uint;

pub const AMDF_ATOMIC_OPERATION_WAIT: amdf_atomic_operations_t = 1;

pub const AMDF_ATOMIC_OPERATION_STORE: amdf_atomic_operations_t = 2;

pub const AMDF_ATOMIC_OPERATION_ADD: amdf_atomic_operations_t = 4;

pub const AMDF_ATOMIC_OPERATION_SUBTRACT: amdf_atomic_operations_t = 8;

pub const AMDF_ATOMIC_OPERATION_AND: amdf_atomic_operations_t = 16;

pub const AMDF_ATOMIC_OPERATION_OR: amdf_atomic_operations_t = 32;

pub const AMDF_ATOMIC_OPERATION_XOR: amdf_atomic_operations_t = 64;

pub type amdf_atomic_wait_conditions_t = u32;

pub type amdf_atomic_wait_condition_bits_e = ::core::ffi::c_uint;

pub const AMDF_ATOMIC_WAIT_CONDITION_EQUAL: amdf_atomic_wait_conditions_t = 1;

pub const AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL: amdf_atomic_wait_conditions_t = 2;

pub const AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL: amdf_atomic_wait_conditions_t = 4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_atomic_capabilities_t {
    pub operations_32: amdf_atomic_operations_t,
    pub operations_64: amdf_atomic_operations_t,
    pub wait_conditions_32: amdf_atomic_wait_conditions_t,
    pub wait_conditions_64: amdf_atomic_wait_conditions_t,
    pub operations_without_dispatch_32: amdf_atomic_operations_t,
    pub operations_without_dispatch_64: amdf_atomic_operations_t,
}
impl Default for amdf_atomic_capabilities_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_atomic_scope_t = u32;

pub type amdf_atomic_scope_e = ::core::ffi::c_uint;

pub const AMDF_ATOMIC_SCOPE_NONE: amdf_atomic_scope_t = 0;

pub const AMDF_ATOMIC_SCOPE_DEVICE: amdf_atomic_scope_t = 1;

pub const AMDF_ATOMIC_SCOPE_FABRIC: amdf_atomic_scope_t = 2;

pub const AMDF_ATOMIC_SCOPE_SYSTEM: amdf_atomic_scope_t = 3;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_atomic_reach_t {
    pub scope_32: amdf_atomic_scope_t,
    pub scope_64: amdf_atomic_scope_t,
}
impl Default for amdf_atomic_reach_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_external_memory_type_t = u32;

pub type amdf_external_memory_type_e = ::core::ffi::c_uint;

pub const AMDF_EXTERNAL_MEMORY_TYPE_NONE: amdf_external_memory_type_t = 0;

pub const AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD: amdf_external_memory_type_t = 1;

pub const AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD: amdf_external_memory_type_t = 2;

pub const AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE: amdf_external_memory_type_t = 3;

pub const AMDF_EXTERNAL_MEMORY_TYPE_HOST_POINTER: amdf_external_memory_type_t = 4;

pub const AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS: amdf_external_memory_type_t = 5;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_external_memory_provenance_t {
    pub words: [u64; 2],
}
impl Default for amdf_external_memory_provenance_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union amdf_external_memory_payload_t {
    pub file_descriptor: i64,
    pub native_handle: *mut ::core::ffi::c_void,
    pub host_pointer: *mut ::core::ffi::c_void,
    pub device_address: u64,
}
impl Default for amdf_external_memory_payload_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_external_memory_release_fn_t = Option<
    unsafe extern "C" fn(
        *mut ::core::ffi::c_void,
        amdf_external_memory_type_t,
        amdf_external_memory_payload_t,
    ),
>;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_external_memory_t {
    pub r#type: amdf_external_memory_type_t,
    pub reserved: u32,
    pub payload: amdf_external_memory_payload_t,
    pub provenance: amdf_external_memory_provenance_t,
    pub source_byte_offset: u64,
    pub byte_length: u64,
    pub physical_backing_id: amdf_physical_memory_id_t,
    pub release: amdf_external_memory_release_fn_t,
    pub release_user_data: *mut ::core::ffi::c_void,
}
impl Default for amdf_external_memory_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_memory_profile_roles_t = u64;

pub type amdf_memory_profile_role_bits_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_PROFILE_ROLE_CREATE: amdf_memory_profile_roles_t = 1;

pub const AMDF_MEMORY_PROFILE_ROLE_REGISTER: amdf_memory_profile_roles_t = 2;

pub const AMDF_MEMORY_PROFILE_ROLE_IMPORT: amdf_memory_profile_roles_t = 4;

pub const AMDF_MEMORY_PROFILE_ROLE_EXPORT: amdf_memory_profile_roles_t = 8;

pub const AMDF_MEMORY_PROFILE_ROLE_HOST_MAP: amdf_memory_profile_roles_t = 16;

pub const AMDF_MEMORY_PROFILE_ROLE_MAPPING_SOURCE: amdf_memory_profile_roles_t = 32;

pub const AMDF_MEMORY_PROFILE_ROLE_MAPPING_TARGET: amdf_memory_profile_roles_t = 64;

pub type amdf_external_memory_support_flags_t = u32;

pub type amdf_external_memory_support_flag_bits_e = ::core::ffi::c_uint;

pub const AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT: amdf_external_memory_support_flags_t = 1;

pub const AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT: amdf_external_memory_support_flags_t = 2;

pub const AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET: amdf_external_memory_support_flags_t = 4;

pub const AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS: amdf_external_memory_support_flags_t = 8;

pub const AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API: amdf_external_memory_support_flags_t = 16;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_external_memory_support_t {
    pub r#type: amdf_external_memory_type_t,
    pub flags: amdf_external_memory_support_flags_t,
    pub provenance: amdf_external_memory_provenance_t,
    pub source_offset_alignment: u64,
    pub byte_length_alignment: u64,
    pub maximum_byte_length: u64,
}
impl Default for amdf_external_memory_support_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_host_cacheability_t = u32;

pub type amdf_host_cacheability_e = ::core::ffi::c_uint;

pub const AMDF_HOST_CACHEABILITY_UNKNOWN: amdf_host_cacheability_t = 0;

pub const AMDF_HOST_CACHEABILITY_WRITE_BACK: amdf_host_cacheability_t = 2;

pub const AMDF_HOST_CACHEABILITY_WRITE_COMBINED: amdf_host_cacheability_t = 3;

pub const AMDF_HOST_CACHEABILITY_UNCACHED: amdf_host_cacheability_t = 4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_construction_capabilities_t {
    pub maximum_byte_length: u64,
    pub byte_length_granularity: u64,
    pub registered_host_pointer_alignment: u64,
    pub registered_host_cacheability: amdf_host_cacheability_t,
    pub reserved: u32,
    pub minimum_alignment: u64,
    pub maximum_alignment: u64,
    pub native_byte_length_granularity: u64,
    pub native_byte_length_prefix: u64,
}
impl Default for amdf_memory_construction_capabilities_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_address_capabilities_t {
    pub address_domain_ordinal: u32,
    pub address_bit_count: u32,
    pub minimum_address: u64,
    pub maximum_address: u64,
    pub minimum_alignment: u64,
}
impl Default for amdf_memory_address_capabilities_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_access_requirements_t {
    pub access: amdf_memory_access_t,
    pub reserved: u32,
    pub flags: amdf_memory_flags_t,
    pub address_kinds: amdf_memory_address_kinds_t,
}
impl Default for amdf_memory_access_requirements_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_device_access_t {
    pub device: *mut amdf_device_t,
    pub requirements: amdf_memory_access_requirements_t,
}
impl Default for amdf_memory_device_access_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_access_capabilities_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub guaranteed_access: amdf_memory_access_t,
    pub supported_access: amdf_memory_access_t,
    pub guaranteed_flags: amdf_memory_flags_t,
    pub supported_flags: amdf_memory_flags_t,
    pub atomic_operations_32: amdf_atomic_operations_t,
    pub atomic_operations_64: amdf_atomic_operations_t,
    pub device_address: amdf_memory_address_capabilities_t,
    pub address_kinds: amdf_memory_address_kinds_t,
}
impl Default for amdf_memory_access_capabilities_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_host_mapping_capabilities_t {
    pub maximum_byte_length: u64,
    pub byte_offset_granularity: u64,
    pub byte_length_granularity: u64,
    pub supported_access: amdf_memory_map_flags_t,
    pub reserved: u32,
}
impl Default for amdf_host_mapping_capabilities_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_profile_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub ordinal: u32,
    pub memory_class: amdf_memory_class_t,
    pub roles: amdf_memory_profile_roles_t,
    pub guaranteed_flags: amdf_memory_flags_t,
    pub supported_flags: amdf_memory_flags_t,
    pub allocation: amdf_memory_construction_capabilities_t,
    pub registration: amdf_memory_construction_capabilities_t,
    pub import: amdf_memory_construction_capabilities_t,
    pub host_mapping: amdf_host_mapping_capabilities_t,
    pub external_memory_support_count: u32,
    pub reserved: u32,
    pub external_memory_support: [amdf_external_memory_support_t; 5],
}
impl Default for amdf_memory_profile_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_memory_scope_kind_t = u32;

pub type amdf_memory_scope_kind_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_SCOPE_KIND_SYSTEM: amdf_memory_scope_kind_t = 0;

pub const AMDF_MEMORY_SCOPE_KIND_LOCAL: amdf_memory_scope_kind_t = 1;

pub const AMDF_MEMORY_SCOPE_KIND_PRIVATE: amdf_memory_scope_kind_t = 2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_scope_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub kind: amdf_memory_scope_kind_t,
    pub memory_profile_count: u32,
    pub physical_endpoint_id: amdf_endpoint_id_t,
}
impl Default for amdf_memory_scope_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub memory_profile_ordinal: u32,
    pub access_count: u32,
    pub required_flags: amdf_memory_flags_t,
    pub byte_length: u64,
    pub minimum_alignment: u64,
    pub registered_host_pointer: *mut ::core::ffi::c_void,
    pub accesses: *const amdf_memory_device_access_t,
    pub registered_host_cacheability: amdf_host_cacheability_t,
    pub reserved: u32,
}
impl Default for amdf_memory_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub memory_profile_ordinal: u32,
    pub memory_class: amdf_memory_class_t,
    pub access_count: u32,
    pub reserved: u32,
    pub flags: amdf_memory_flags_t,
    pub source_byte_offset: u64,
    pub byte_length: u64,
    pub alignment: u64,
    pub native_allocation_byte_length: u64,
    pub native_allocation_granularity: u64,
    pub physical_backing_id: amdf_physical_memory_id_t,
}
impl Default for amdf_memory_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_access_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub ordinal: u32,
    pub access: amdf_memory_access_t,
    pub device_id: amdf_device_id_t,
    pub flags: amdf_memory_flags_t,
    pub atomic_operations_32: amdf_atomic_operations_t,
    pub atomic_operations_64: amdf_atomic_operations_t,
    pub address_domain_ordinal: u32,
    pub reserved: u32,
    pub address_kinds: amdf_memory_address_kinds_t,
    pub reset_epoch: u64,
}
impl Default for amdf_memory_access_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_import_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub memory_profile_ordinal: u32,
    pub access_count: u32,
    pub required_flags: amdf_memory_flags_t,
    pub minimum_alignment: u64,
    pub accesses: *const amdf_memory_device_access_t,
}
impl Default for amdf_memory_import_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_export_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub external_memory_type: amdf_external_memory_type_t,
    pub reserved: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}
impl Default for amdf_memory_export_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_map_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub flags: amdf_memory_map_flags_t,
}
impl Default for amdf_memory_map_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_host_cache_operation_t = u32;

pub type amdf_host_cache_operation_e = ::core::ffi::c_uint;

pub const AMDF_HOST_CACHE_OPERATION_NONE: amdf_host_cache_operation_t = 0;

pub const AMDF_HOST_CACHE_OPERATION_FLUSH: amdf_host_cache_operation_t = 1;

pub const AMDF_HOST_CACHE_OPERATION_INVALIDATE: amdf_host_cache_operation_t = 2;

pub type amdf_cache_operation_t = u32;

pub type amdf_cache_operation_e = ::core::ffi::c_uint;

pub const AMDF_CACHE_OPERATION_NONE: amdf_cache_operation_t = 0;

pub const AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM: amdf_cache_operation_t = 1;

pub const AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM: amdf_cache_operation_t = 2;

pub type amdf_cache_operations_t = u64;

pub type amdf_cache_operation_bits_e = ::core::ffi::c_uint;

pub const AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM: amdf_cache_operations_t = 2;

pub const AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM: amdf_cache_operations_t = 4;

pub type amdf_cache_transition_kind_t = u32;

pub type amdf_cache_transition_kind_e = ::core::ffi::c_uint;

pub const AMDF_CACHE_TRANSITION_KIND_UNKNOWN: amdf_cache_transition_kind_t = 0;

pub const AMDF_CACHE_TRANSITION_KIND_NONE: amdf_cache_transition_kind_t = 1;

pub const AMDF_CACHE_TRANSITION_KIND_RANGE: amdf_cache_transition_kind_t = 2;

pub const AMDF_CACHE_TRANSITION_KIND_GLOBAL: amdf_cache_transition_kind_t = 3;

pub type amdf_cache_transition_kinds_t = u32;

pub type amdf_cache_transition_kind_bits_e = ::core::ffi::c_uint;

pub const AMDF_CACHE_TRANSITION_KINDS_RANGE: amdf_cache_transition_kinds_t = 4;

pub const AMDF_CACHE_TRANSITION_KINDS_GLOBAL: amdf_cache_transition_kinds_t = 8;

pub type amdf_cache_transition_executor_t = u32;

pub type amdf_cache_transition_executor_e = ::core::ffi::c_uint;

pub const AMDF_CACHE_TRANSITION_EXECUTOR_NONE: amdf_cache_transition_executor_t = 0;

pub const AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE: amdf_cache_transition_executor_t = 1;

pub const AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM: amdf_cache_transition_executor_t = 2;

pub const AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT: amdf_cache_transition_executor_t = 3;

pub const AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API: amdf_cache_transition_executor_t = 4;

pub type amdf_host_cache_instruction_t = u32;

pub type amdf_host_cache_instruction_e = ::core::ffi::c_uint;

pub const AMDF_HOST_CACHE_INSTRUCTION_NONE: amdf_host_cache_instruction_t = 0;

pub const AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH: amdf_host_cache_instruction_t = 1;

pub const AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT: amdf_host_cache_instruction_t = 2;

pub const AMDF_HOST_CACHE_INSTRUCTION_X86_CLWB: amdf_host_cache_instruction_t = 3;

pub type amdf_host_cache_fence_t = u32;

pub type amdf_host_cache_fence_e = ::core::ffi::c_uint;

pub const AMDF_HOST_CACHE_FENCE_NONE: amdf_host_cache_fence_t = 0;

pub const AMDF_HOST_CACHE_FENCE_X86_SFENCE: amdf_host_cache_fence_t = 1;

pub const AMDF_HOST_CACHE_FENCE_X86_MFENCE: amdf_host_cache_fence_t = 2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_cache_transition_t {
    pub kind: amdf_cache_transition_kind_t,
    pub executor: amdf_cache_transition_executor_t,
    pub operation: amdf_cache_operation_t,
    pub host_operation: amdf_host_cache_operation_t,
    pub host_instruction: amdf_host_cache_instruction_t,
    pub host_fence_before: amdf_host_cache_fence_t,
    pub host_fence_after: amdf_host_cache_fence_t,
    pub range_granularity: u64,
}
impl Default for amdf_cache_transition_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_host_mapping_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub flags: amdf_memory_map_flags_t,
    pub cacheability: amdf_host_cacheability_t,
    pub pointer: *mut ::core::ffi::c_void,
    pub memory_byte_offset: u64,
    pub byte_length: u64,
    pub byte_offset_granularity: u64,
    pub byte_length_granularity: u64,
    pub cache_line_size: u32,
    pub reserved: u32,
    pub flush: amdf_cache_transition_t,
    pub invalidate: amdf_cache_transition_t,
}
impl Default for amdf_host_mapping_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_memory_site_kind_t = u32;

pub type amdf_memory_site_kind_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_SITE_KIND_DEVICE: amdf_memory_site_kind_t = 0;

pub const AMDF_MEMORY_SITE_KIND_HOST: amdf_memory_site_kind_t = 1;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_site_t__value__device {
    pub memory: *mut amdf_memory_t,
    pub access_ordinal: u32,
    pub queue_family_ordinal: u32,
}
impl Default for amdf_memory_site_t__value__device {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union amdf_memory_site_t__value {
    pub device: amdf_memory_site_t__value__device,
    pub host_mapping: *mut amdf_host_mapping_t,
}
impl Default for amdf_memory_site_t__value {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_site_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub kind: amdf_memory_site_kind_t,
    pub reserved: u32,
    pub value: amdf_memory_site_t__value,
}
impl Default for amdf_memory_site_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_profile_site_t__value__device {
    pub access_ordinal: u32,
    pub queue_family_ordinal: u32,
}
impl Default for amdf_memory_profile_site_t__value__device {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union amdf_memory_profile_site_t__value {
    pub device: amdf_memory_profile_site_t__value__device,
    pub host_access: amdf_memory_map_flags_t,
}
impl Default for amdf_memory_profile_site_t__value {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_profile_site_t {
    pub kind: amdf_memory_site_kind_t,
    pub reserved: u32,
    pub value: amdf_memory_profile_site_t__value,
}
impl Default for amdf_memory_profile_site_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_profile_pair_query_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub memory_profile_ordinal: u32,
    pub access_count: u32,
    pub required_flags: amdf_memory_flags_t,
    pub accesses: *const amdf_memory_device_access_t,
    pub registered_host_cacheability: amdf_host_cacheability_t,
    pub reserved: u32,
    pub external_memory_type: amdf_external_memory_type_t,
    pub external_memory_reserved: u32,
    pub external_memory_provenance: amdf_external_memory_provenance_t,
    pub producer: amdf_memory_profile_site_t,
    pub consumer: amdf_memory_profile_site_t,
}
impl Default for amdf_memory_profile_pair_query_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_memory_pair_flags_t = u64;

pub type amdf_memory_pair_flag_bits_e = ::core::ffi::c_uint;

pub const AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE: amdf_memory_pair_flags_t = 1;

pub const AMDF_MEMORY_PAIR_FLAG_MAPPING_SOURCE: amdf_memory_pair_flags_t = 2;

pub const AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN: amdf_memory_pair_flags_t = 4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_memory_pair_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub flags: amdf_memory_pair_flags_t,
    pub release: amdf_cache_transition_t,
    pub acquire: amdf_cache_transition_t,
    pub atomic_reach: amdf_atomic_reach_t,
    pub estimated_fixed_cost_nanoseconds: u64,
}
impl Default for amdf_memory_pair_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_queue_command_type_t = u32;

pub type amdf_queue_command_type_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_COMMAND_TYPE_UNKNOWN: amdf_queue_command_type_t = 0;

pub const AMDF_QUEUE_COMMAND_TYPE_GPU_PM4: amdf_queue_command_type_t = 1;

pub const AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA: amdf_queue_command_type_t = 2;

pub const AMDF_QUEUE_COMMAND_TYPE_GPU_AQL: amdf_queue_command_type_t = 3;

pub const AMDF_QUEUE_COMMAND_TYPE_XDNA: amdf_queue_command_type_t = 4;

pub const AMDF_QUEUE_COMMAND_TYPE_GPU_AQL_METADATA: amdf_queue_command_type_t = 5;

pub type amdf_queue_format_features_t = u64;

pub type amdf_queue_publication_modes_t = u32;

pub type amdf_queue_publication_mode_bits_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_PUBLICATION_MODE_USER: amdf_queue_publication_modes_t = 1;

pub const AMDF_QUEUE_PUBLICATION_MODE_KERNEL: amdf_queue_publication_modes_t = 2;

pub type amdf_queue_roles_t = u64;

pub type amdf_queue_role_bits_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_ROLE_COMPUTE: amdf_queue_roles_t = 1;

pub const AMDF_QUEUE_ROLE_TRANSFER: amdf_queue_roles_t = 2;

pub const AMDF_QUEUE_ROLE_ATOMIC: amdf_queue_roles_t = 4;

pub const AMDF_QUEUE_ROLE_CACHE_CONTROL: amdf_queue_roles_t = 8;

pub const AMDF_QUEUE_ROLE_VIRTUAL_MEMORY: amdf_queue_roles_t = 16;

pub type amdf_queue_producer_mode_t = u32;

pub type amdf_queue_producer_mode_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_PRODUCER_MODE_SINGLE: amdf_queue_producer_mode_t = 1;

pub const AMDF_QUEUE_PRODUCER_MODE_MULTI: amdf_queue_producer_mode_t = 2;

pub type amdf_queue_producer_modes_t = u32;

pub type amdf_queue_producer_mode_bits_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE: amdf_queue_producer_modes_t = 2;

pub const AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI: amdf_queue_producer_modes_t = 4;

pub type amdf_queue_priority_t = u32;

pub type amdf_queue_priority_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_PRIORITY_LOW: amdf_queue_priority_t = 1;

pub const AMDF_QUEUE_PRIORITY_NORMAL: amdf_queue_priority_t = 2;

pub const AMDF_QUEUE_PRIORITY_HIGH: amdf_queue_priority_t = 3;

pub type amdf_queue_priority_capabilities_t = u32;

pub type amdf_queue_priority_capability_bits_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_PRIORITY_CAPABILITY_LOW: amdf_queue_priority_capabilities_t = 2;

pub const AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL: amdf_queue_priority_capabilities_t = 4;

pub const AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH: amdf_queue_priority_capabilities_t = 8;

pub type amdf_user_queue_capabilities_t = u64;

pub type amdf_user_queue_capability_bits_e = ::core::ffi::c_uint;

pub const AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER: amdf_user_queue_capabilities_t = 1;

pub const AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER: amdf_user_queue_capabilities_t = 2;

pub type amdf_kernel_queue_capabilities_t = u64;

pub type amdf_kernel_queue_capability_bits_e = ::core::ffi::c_uint;

pub const AMDF_KERNEL_QUEUE_CAPABILITY_VECTOR_SUBMIT: amdf_kernel_queue_capabilities_t = 1;

pub const AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY: amdf_kernel_queue_capabilities_t = 2;

pub const AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION: amdf_kernel_queue_capabilities_t =
    4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_queue_metadata_format_t {
    pub command_type: amdf_queue_command_type_t,
    pub dispatch_version: u32,
    pub barrier_version: u32,
    pub reserved: u32,
}
impl Default for amdf_queue_metadata_format_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_queue_family_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub ordinal: u32,
    pub command_type: amdf_queue_command_type_t,
    pub publication_modes: amdf_queue_publication_modes_t,
    pub format_version: u32,
    pub format_features: amdf_queue_format_features_t,
    pub roles: amdf_queue_roles_t,
    pub cache_operations: amdf_cache_operations_t,
    pub cache_transition_kinds: amdf_cache_transition_kinds_t,
    pub reserved: u32,
    pub atomic_capabilities: amdf_atomic_capabilities_t,
    pub user_queue_capabilities: amdf_user_queue_capabilities_t,
    pub kernel_queue_capabilities: amdf_kernel_queue_capabilities_t,
    pub producer_modes: amdf_queue_producer_modes_t,
    pub priority_capabilities: amdf_queue_priority_capabilities_t,
    pub metadata: amdf_queue_metadata_format_t,
    pub minimum_ring_byte_length: u64,
    pub maximum_ring_byte_length: u64,
    pub ring_byte_length_alignment: u64,
}
impl Default for amdf_queue_family_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_queue_state_t = u32;

pub type amdf_queue_state_e = ::core::ffi::c_uint;

pub const AMDF_QUEUE_STATE_ACTIVE: amdf_queue_state_t = 1;

pub const AMDF_QUEUE_STATE_FAILED: amdf_queue_state_t = 2;

pub const AMDF_QUEUE_STATE_DEVICE_LOST: amdf_queue_state_t = 3;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_user_queue_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub device_id: amdf_device_id_t,
    pub queue_id: amdf_queue_id_t,
    pub reset_epoch: u64,
    pub queue_family_ordinal: u32,
    pub command_type: amdf_queue_command_type_t,
    pub format_version: u32,
    pub format_features: amdf_queue_format_features_t,
    pub producer_mode: amdf_queue_producer_mode_t,
    pub priority: amdf_queue_priority_t,
    pub capabilities: amdf_user_queue_capabilities_t,
    pub roles: amdf_queue_roles_t,
    pub metadata: amdf_queue_metadata_format_t,
    pub ring_byte_length: u64,
    pub metadata_ring_byte_length: u64,
}
impl Default for amdf_user_queue_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_user_queue_mapping_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub producer_device_id: amdf_device_id_t,
    pub queue_id: amdf_queue_id_t,
    pub queue_reset_epoch: u64,
    pub producer_reset_epoch: u64,
    pub command_type: amdf_queue_command_type_t,
    pub format_version: u32,
    pub format_features: amdf_queue_format_features_t,
    pub ring_address: u64,
    pub ring_byte_length: u64,
    pub read_index_address: u64,
    pub write_index_address: u64,
    pub doorbell_address: u64,
    pub index_bits: u32,
    pub doorbell_bits: u32,
    pub metadata: amdf_queue_metadata_format_t,
    pub metadata_ring_address: u64,
    pub metadata_ring_byte_length: u64,
}
impl Default for amdf_user_queue_mapping_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_user_queue_status_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub state: amdf_queue_state_t,
    pub reserved: u32,
    pub reset_epoch: u64,
    pub producer_index: u64,
    pub consumed_index: u64,
    pub terminal_status: amdf_status_t,
}
impl Default for amdf_user_queue_status_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_kernel_queue_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub device_id: amdf_device_id_t,
    pub reset_epoch: u64,
    pub queue_family_ordinal: u32,
    pub command_type: amdf_queue_command_type_t,
    pub maximum_pending_submission_count: u32,
    pub maximum_command_count: u32,
}
impl Default for amdf_kernel_queue_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_kernel_queue_status_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub retired_submission: u64,
    pub state: amdf_queue_state_t,
    pub reserved: u32,
    pub terminal_status: amdf_status_t,
}
impl Default for amdf_kernel_queue_status_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_api_t {
    pub structure_size: u32,
    pub abi_version: amdf_abi_version_t,
    pub instance_create: Option<
        unsafe extern "C" fn(
            *const amdf_instance_create_info_t,
            *mut *mut amdf_instance_t,
        ) -> amdf_status_t,
    >,
    pub instance_destroy: Option<unsafe extern "C" fn(*mut amdf_instance_t) -> amdf_status_t>,
    pub endpoint_enumerate: Option<
        unsafe extern "C" fn(
            *mut amdf_instance_t,
            u32,
            *mut amdf_endpoint_summary_t,
            *mut u32,
        ) -> amdf_status_t,
    >,
    pub endpoint_open: Option<
        unsafe extern "C" fn(
            *mut amdf_instance_t,
            *const amdf_endpoint_id_t,
            *mut *mut amdf_endpoint_t,
        ) -> amdf_status_t,
    >,
    pub endpoint_query_info: Option<
        unsafe extern "C" fn(*mut amdf_endpoint_t, *mut amdf_endpoint_info_t) -> amdf_status_t,
    >,
    pub endpoint_close: Option<unsafe extern "C" fn(*mut amdf_endpoint_t) -> amdf_status_t>,
    pub query_extension: Option<
        unsafe extern "C" fn(
            amdf_extension_id_t,
            u32,
            u32,
            *mut *const ::core::ffi::c_void,
        ) -> amdf_status_t,
    >,
    pub endpoint_query_queue_family_info: Option<
        unsafe extern "C" fn(
            *mut amdf_endpoint_t,
            u32,
            *mut amdf_queue_family_info_t,
        ) -> amdf_status_t,
    >,
    pub device_destroy: Option<unsafe extern "C" fn(*mut amdf_device_t) -> amdf_status_t>,
    pub instance_enumerate_memory_scopes: Option<
        unsafe extern "C" fn(
            *mut amdf_instance_t,
            u32,
            *mut *mut amdf_memory_scope_t,
            *mut u32,
        ) -> amdf_status_t,
    >,
    pub device_enumerate_memory_scopes: Option<
        unsafe extern "C" fn(
            *mut amdf_device_t,
            u32,
            *mut *mut amdf_memory_scope_t,
            *mut u32,
        ) -> amdf_status_t,
    >,
    pub memory_scope_query_info: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_scope_t,
            *mut amdf_memory_scope_info_t,
        ) -> amdf_status_t,
    >,
    pub memory_scope_query_device_profile: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_scope_t,
            u32,
            u32,
            *const amdf_memory_device_access_t,
            *mut amdf_memory_profile_t,
            *mut amdf_memory_access_capabilities_t,
        ) -> amdf_status_t,
    >,
    pub memory_create: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_scope_t,
            *const amdf_memory_create_info_t,
            *mut *mut amdf_memory_t,
        ) -> amdf_status_t,
    >,
    pub memory_import: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_scope_t,
            *const amdf_memory_import_info_t,
            *mut amdf_external_memory_t,
            *mut *mut amdf_memory_t,
        ) -> amdf_status_t,
    >,
    pub memory_query_info:
        Option<unsafe extern "C" fn(*mut amdf_memory_t, *mut amdf_memory_info_t) -> amdf_status_t>,
    pub memory_query_access_info: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_t,
            u32,
            *mut amdf_memory_access_info_t,
        ) -> amdf_status_t,
    >,
    pub memory_export: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_t,
            *const amdf_memory_export_info_t,
            *mut amdf_external_memory_t,
        ) -> amdf_status_t,
    >,
    pub external_memory_release: Option<unsafe extern "C" fn(*mut amdf_external_memory_t)>,
    pub memory_query_pair_info: Option<
        unsafe extern "C" fn(
            *const amdf_memory_site_t,
            *const amdf_memory_site_t,
            *mut amdf_memory_pair_info_t,
        ) -> amdf_status_t,
    >,
    pub memory_map: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_t,
            *const amdf_memory_map_info_t,
            *mut *mut amdf_host_mapping_t,
        ) -> amdf_status_t,
    >,
    pub host_mapping_query_info: Option<
        unsafe extern "C" fn(
            *mut amdf_host_mapping_t,
            *mut amdf_host_mapping_info_t,
        ) -> amdf_status_t,
    >,
    pub host_mapping_cache_control: Option<
        unsafe extern "C" fn(
            *mut amdf_host_mapping_t,
            amdf_host_cache_operation_t,
            u64,
            u64,
        ) -> amdf_status_t,
    >,
    pub host_mapping_destroy:
        Option<unsafe extern "C" fn(*mut amdf_host_mapping_t) -> amdf_status_t>,
    pub memory_destroy: Option<unsafe extern "C" fn(*mut amdf_memory_t) -> amdf_status_t>,
    pub kernel_queue_query_info: Option<
        unsafe extern "C" fn(
            *mut amdf_kernel_queue_t,
            *mut amdf_kernel_queue_info_t,
        ) -> amdf_status_t,
    >,
    pub kernel_queue_query_status: Option<
        unsafe extern "C" fn(
            *mut amdf_kernel_queue_t,
            *mut amdf_kernel_queue_status_t,
        ) -> amdf_status_t,
    >,
    pub kernel_queue_wait:
        Option<unsafe extern "C" fn(*mut amdf_kernel_queue_t, u64, u64, u64) -> amdf_status_t>,
    pub kernel_queue_destroy:
        Option<unsafe extern "C" fn(*mut amdf_kernel_queue_t) -> amdf_status_t>,
    pub user_queue_query_info: Option<
        unsafe extern "C" fn(*mut amdf_user_queue_t, *mut amdf_user_queue_info_t) -> amdf_status_t,
    >,
    pub user_queue_map: Option<
        unsafe extern "C" fn(
            *mut amdf_user_queue_t,
            *mut amdf_device_t,
            *mut *mut amdf_user_queue_mapping_t,
        ) -> amdf_status_t,
    >,
    pub user_queue_mapping_query_info: Option<
        unsafe extern "C" fn(
            *mut amdf_user_queue_mapping_t,
            *mut amdf_user_queue_mapping_info_t,
        ) -> amdf_status_t,
    >,
    pub user_queue_mapping_destroy:
        Option<unsafe extern "C" fn(*mut amdf_user_queue_mapping_t) -> amdf_status_t>,
    pub user_queue_query_status: Option<
        unsafe extern "C" fn(
            *mut amdf_user_queue_t,
            *mut amdf_user_queue_status_t,
        ) -> amdf_status_t,
    >,
    pub user_queue_wait_consumed:
        Option<unsafe extern "C" fn(*mut amdf_user_queue_t, u64, u64, u64) -> amdf_status_t>,
    pub user_queue_destroy: Option<unsafe extern "C" fn(*mut amdf_user_queue_t) -> amdf_status_t>,
    pub memory_query_address: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_t,
            u32,
            amdf_memory_address_kind_t,
            *mut u64,
        ) -> amdf_status_t,
    >,
    pub memory_scope_query_pair_info: Option<
        unsafe extern "C" fn(
            *mut amdf_memory_scope_t,
            *const amdf_memory_profile_pair_query_t,
            *mut amdf_memory_pair_info_t,
        ) -> amdf_status_t,
    >,
}
impl Default for amdf_api_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_query_api_fn_t = Option<
    unsafe extern "C" fn(
        amdf_abi_version_t,
        amdf_abi_version_t,
        *mut *const amdf_api_t,
    ) -> amdf_status_t,
>;

unsafe extern "C" {
    pub fn amdf_query_api(
        minimum_version: amdf_abi_version_t,
        maximum_version: amdf_abi_version_t,
        out_api: *mut *const amdf_api_t,
    ) -> amdf_status_t;
}

pub type amdf_gpu_device_features_t = u64;

pub type amdf_gpu_device_feature_bits_e = ::core::ffi::c_uint;

pub const AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION: amdf_gpu_device_features_t = 1;

pub const AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION: amdf_gpu_device_features_t = 2;

pub const AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY: amdf_gpu_device_features_t = 4;

pub const AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY: amdf_gpu_device_features_t = 8;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_endpoint_info_t__gfx_ip {
    pub major: u32,
    pub minor: u32,
    pub stepping: u32,
}
impl Default for amdf_gpu_endpoint_info_t__gfx_ip {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_endpoint_info_t__compute {
    pub wavefront_size: u32,
    pub compute_unit_count: u32,
    pub maximum_wave_count_per_compute_unit: u32,
    pub maximum_scratch_wave_count_per_compute_unit: u32,
    pub local_data_share_byte_length: u64,
}
impl Default for amdf_gpu_endpoint_info_t__compute {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_endpoint_info_t__topology {
    pub xcc_count: u32,
    pub shader_engine_count_per_xcc: u32,
}
impl Default for amdf_gpu_endpoint_info_t__topology {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_endpoint_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub gfx_ip: amdf_gpu_endpoint_info_t__gfx_ip,
    pub asic_revision: u32,
    pub compute: amdf_gpu_endpoint_info_t__compute,
    pub topology: amdf_gpu_endpoint_info_t__topology,
}
impl Default for amdf_gpu_endpoint_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_device_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub reserved: u64,
}
impl Default for amdf_gpu_device_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_device_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub id: amdf_device_id_t,
    pub reset_epoch: u64,
    pub features: amdf_gpu_device_features_t,
}
impl Default for amdf_gpu_device_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_gpu_pm4_format_feature_bits_e = ::core::ffi::c_uint;

pub const AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR: amdf_queue_format_features_t = 1;

pub type amdf_gpu_sdma_format_feature_bits_e = ::core::ffi::c_uint;

pub const AMDF_GPU_SDMA_FORMAT_FEATURE_GCR: amdf_queue_format_features_t = 1;

pub const AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM: amdf_queue_format_features_t = 2;

pub const AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE: amdf_queue_format_features_t = 4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_queue_scratch_t {
    pub memory: *mut amdf_memory_t,
    pub access_ordinal: u32,
    pub reserved: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub maximum_private_segment_byte_length: u32,
    pub maximum_wave_count: u32,
}
impl Default for amdf_gpu_queue_scratch_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_user_queue_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub queue_family_ordinal: u32,
    pub priority: amdf_queue_priority_t,
    pub producer_mode: amdf_queue_producer_mode_t,
    pub reserved: u32,
    pub required_capabilities: amdf_user_queue_capabilities_t,
    pub ring_byte_length: u64,
    pub scratch: amdf_gpu_queue_scratch_t,
}
impl Default for amdf_gpu_user_queue_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_kernel_command_t {
    pub memory: *mut amdf_memory_t,
    pub access_ordinal: u32,
    pub reserved: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}
impl Default for amdf_gpu_kernel_command_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_kernel_queue_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub queue_family_ordinal: u32,
    pub reserved: u32,
}
impl Default for amdf_gpu_kernel_queue_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_kernel_queue_submission_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub command_count: u32,
    pub reserved: u32,
    pub commands: *const amdf_gpu_kernel_command_t,
}
impl Default for amdf_gpu_kernel_queue_submission_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_gpu_api_t {
    pub structure_size: u32,
    pub extension_version: u32,
    pub endpoint_query_info: Option<
        unsafe extern "C" fn(*mut amdf_endpoint_t, *mut amdf_gpu_endpoint_info_t) -> amdf_status_t,
    >,
    pub device_create: Option<
        unsafe extern "C" fn(
            *mut amdf_endpoint_t,
            *const amdf_gpu_device_create_info_t,
            *mut *mut amdf_device_t,
        ) -> amdf_status_t,
    >,
    pub device_query_info: Option<
        unsafe extern "C" fn(*mut amdf_device_t, *mut amdf_gpu_device_info_t) -> amdf_status_t,
    >,
    pub kernel_queue_create: Option<
        unsafe extern "C" fn(
            *mut amdf_device_t,
            *const amdf_gpu_kernel_queue_create_info_t,
            *mut *mut amdf_kernel_queue_t,
        ) -> amdf_status_t,
    >,
    pub kernel_queue_submit: Option<
        unsafe extern "C" fn(
            *mut amdf_kernel_queue_t,
            *const amdf_gpu_kernel_queue_submission_info_t,
            *mut u64,
        ) -> amdf_status_t,
    >,
    pub user_queue_create: Option<
        unsafe extern "C" fn(
            *mut amdf_device_t,
            *const amdf_gpu_user_queue_create_info_t,
            *mut *mut amdf_user_queue_t,
        ) -> amdf_status_t,
    >,
}
impl Default for amdf_gpu_api_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_context_t {
    _opaque: [u8; 0],
}
impl Default for amdf_xdna_context_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub type amdf_xdna_architecture_t = u32;

pub type amdf_xdna_architecture_e = ::core::ffi::c_uint;

pub const AMDF_XDNA_ARCHITECTURE_UNKNOWN: amdf_xdna_architecture_t = 0;

pub const AMDF_XDNA_ARCHITECTURE_AIE2: amdf_xdna_architecture_t = 1;

pub const AMDF_XDNA_ARCHITECTURE_AIE2P: amdf_xdna_architecture_t = 2;

pub const AMDF_XDNA_ARCHITECTURE_AIE4: amdf_xdna_architecture_t = 3;

pub type amdf_xdna_scheduling_modes_t = u32;

pub type amdf_xdna_scheduling_mode_bits_e = ::core::ffi::c_uint;

pub const AMDF_XDNA_SCHEDULING_MODE_EXCLUSIVE: amdf_xdna_scheduling_modes_t = 1;

pub const AMDF_XDNA_SCHEDULING_MODE_SPATIAL: amdf_xdna_scheduling_modes_t = 2;

pub const AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED: amdf_xdna_scheduling_modes_t = 4;

pub type amdf_xdna_placement_modes_t = u32;

pub type amdf_xdna_placement_mode_bits_e = ::core::ffi::c_uint;

pub const AMDF_XDNA_PLACEMENT_MODE_FIXED_FULL_ARRAY: amdf_xdna_placement_modes_t = 1;

pub type amdf_xdna_binary_format_t = u32;

pub type amdf_xdna_binary_format_e = ::core::ffi::c_uint;

pub const AMDF_XDNA_BINARY_FORMAT_UNKNOWN: amdf_xdna_binary_format_t = 0;

pub const AMDF_XDNA_BINARY_FORMAT_TRANSACTION: amdf_xdna_binary_format_t = 2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_binary_format_info_t {
    pub format: amdf_xdna_binary_format_t,
    pub version: u32,
}
impl Default for amdf_xdna_binary_format_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_endpoint_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub architecture: amdf_xdna_architecture_t,
    pub target_id: [::core::ffi::c_char; 64],
}
impl Default for amdf_xdna_endpoint_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_context_id_t {
    pub words: [u64; 2],
}
impl Default for amdf_xdna_context_id_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_device_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
}
impl Default for amdf_xdna_device_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_device_info_t__array {
    pub column_origin: u32,
    pub column_count: u32,
    pub row_count: u32,
    pub column_stride: u64,
}
impl Default for amdf_xdna_device_info_t__array {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_device_info_t__context {
    pub scheduling_modes: amdf_xdna_scheduling_modes_t,
    pub minimum_column_count: u32,
    pub maximum_column_count: u32,
    pub column_count_granularity: u32,
}
impl Default for amdf_xdna_device_info_t__context {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_device_info_t__instruction {
    pub maximum_byte_length: u64,
    pub address_alignment: u32,
    pub byte_length_granularity: u32,
    pub format: amdf_xdna_binary_format_info_t,
}
impl Default for amdf_xdna_device_info_t__instruction {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_device_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub id: amdf_device_id_t,
    pub reset_epoch: u64,
    pub placement_modes: amdf_xdna_placement_modes_t,
    pub array: amdf_xdna_device_info_t__array,
    pub context: amdf_xdna_device_info_t__context,
    pub instruction: amdf_xdna_device_info_t__instruction,
}
impl Default for amdf_xdna_device_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_context_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub logical_column_count: u32,
    pub physical_column_origin: u32,
    pub acceptable_scheduling_modes: amdf_xdna_scheduling_modes_t,
}
impl Default for amdf_xdna_context_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_context_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub id: amdf_xdna_context_id_t,
    pub device_id: amdf_device_id_t,
    pub reset_epoch: u64,
    pub scheduling_mode: amdf_xdna_scheduling_modes_t,
    pub logical_column_count: u32,
    pub row_count: u32,
}
impl Default for amdf_xdna_context_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_context_placement_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *mut ::core::ffi::c_void,
    pub column_origin: u32,
    pub column_count: u32,
}
impl Default for amdf_xdna_context_placement_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_kernel_queue_create_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub queue_family_ordinal: u32,
    pub reserved: u32,
}
impl Default for amdf_xdna_kernel_queue_create_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_kernel_command_t {
    pub memory: *mut amdf_memory_t,
    pub access_ordinal: u32,
    pub reserved: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}
impl Default for amdf_xdna_kernel_command_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_kernel_queue_submission_info_t {
    pub r#type: amdf_structure_type_t,
    pub structure_size: u32,
    pub next: *const ::core::ffi::c_void,
    pub command_count: u32,
    pub reserved: u32,
    pub commands: *const amdf_xdna_kernel_command_t,
}
impl Default for amdf_xdna_kernel_queue_submission_info_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct amdf_xdna_api_t {
    pub structure_size: u32,
    pub extension_version: u32,
    pub endpoint_query_info: Option<
        unsafe extern "C" fn(*mut amdf_endpoint_t, *mut amdf_xdna_endpoint_info_t) -> amdf_status_t,
    >,
    pub device_create: Option<
        unsafe extern "C" fn(
            *mut amdf_endpoint_t,
            *const amdf_xdna_device_create_info_t,
            *mut *mut amdf_device_t,
        ) -> amdf_status_t,
    >,
    pub device_query_info: Option<
        unsafe extern "C" fn(*mut amdf_device_t, *mut amdf_xdna_device_info_t) -> amdf_status_t,
    >,
    pub kernel_queue_create: Option<
        unsafe extern "C" fn(
            *mut amdf_xdna_context_t,
            *const amdf_xdna_kernel_queue_create_info_t,
            *mut *mut amdf_kernel_queue_t,
        ) -> amdf_status_t,
    >,
    pub kernel_queue_submit: Option<
        unsafe extern "C" fn(
            *mut amdf_kernel_queue_t,
            *const amdf_xdna_kernel_queue_submission_info_t,
            *mut u64,
        ) -> amdf_status_t,
    >,
    pub context_create: Option<
        unsafe extern "C" fn(
            *mut amdf_device_t,
            *const amdf_xdna_context_create_info_t,
            *mut *mut amdf_xdna_context_t,
        ) -> amdf_status_t,
    >,
    pub context_query_info: Option<
        unsafe extern "C" fn(
            *mut amdf_xdna_context_t,
            *mut amdf_xdna_context_info_t,
        ) -> amdf_status_t,
    >,
    pub context_query_placement_info: Option<
        unsafe extern "C" fn(
            *mut amdf_xdna_context_t,
            *mut amdf_xdna_context_placement_info_t,
        ) -> amdf_status_t,
    >,
    pub context_enumerate_memory_scopes: Option<
        unsafe extern "C" fn(
            *mut amdf_xdna_context_t,
            u32,
            *mut *mut amdf_memory_scope_t,
            *mut u32,
        ) -> amdf_status_t,
    >,
    pub context_destroy: Option<unsafe extern "C" fn(*mut amdf_xdna_context_t) -> amdf_status_t>,
}
impl Default for amdf_xdna_api_t {
    fn default() -> Self {
        // SAFETY: ABI records contain only integer values, raw pointers,
        // nullable function pointers, and aggregates of those types.
        unsafe { ::core::mem::zeroed() }
    }
}

pub const AMDF_QUERY_API_SYMBOL: &[u8] = b"amdf_query_api\0";

pub const AMDF_ABI_VERSION_1: amdf_abi_version_t = 1;

pub const AMDF_ABI_VERSION_2: amdf_abi_version_t = 2;

pub const AMDF_ABI_VERSION_3: amdf_abi_version_t = 3;

pub const AMDF_ABI_VERSION_LATEST: amdf_abi_version_t = 3;

pub const AMDF_ADDRESS_DOMAIN_ORDINAL_NONE: u32 = 4294967295;

pub const AMDF_ENABLE_ASSERTS: ::core::ffi::c_int = 1;

pub const AMDF_ENDPOINT_NAME_CAPACITY: ::core::ffi::c_uint = 128;

pub const AMDF_EXTERNAL_MEMORY_TYPE_COUNT: ::core::ffi::c_uint = 5;

pub const AMDF_GPU_EXTENSION_VERSION_1: ::core::ffi::c_uint = 1;

pub const AMDF_GPU_EXTENSION_VERSION_LATEST: ::core::ffi::c_uint = 1;

pub const AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1: ::core::ffi::c_uint = 1;

pub const AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1: ::core::ffi::c_uint = 1;

pub const AMDF_MEMORY_PROFILE_EXTERNAL_SUPPORT_CAPACITY: ::core::ffi::c_uint = 5;

pub const AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN: u32 = 4294967295;

pub const AMDF_STATUS_OK: amdf_status_t = 0;

pub const AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO: amdf_structure_type_t = 131074;

pub const AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO: amdf_structure_type_t = 131075;

pub const AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO: amdf_structure_type_t = 131073;

pub const AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO: amdf_structure_type_t = 131076;

pub const AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO: amdf_structure_type_t = 131077;

pub const AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO: amdf_structure_type_t = 131079;

pub const AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO: amdf_structure_type_t = 65546;

pub const AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO: amdf_structure_type_t = 65547;

pub const AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_PLACEMENT_INFO: amdf_structure_type_t = 65549;

pub const AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO: amdf_structure_type_t = 65538;

pub const AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO: amdf_structure_type_t = 65539;

pub const AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO: amdf_structure_type_t = 65537;

pub const AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO: amdf_structure_type_t = 65544;

pub const AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO: amdf_structure_type_t = 65545;

pub const AMDF_TIMEOUT_INFINITE: u64 = 18446744073709551615;

pub const AMDF_XDNA_EXTENSION_VERSION_1: ::core::ffi::c_uint = 1;

pub const AMDF_XDNA_EXTENSION_VERSION_LATEST: ::core::ffi::c_uint = 1;

pub const AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY: u32 = 4294967295;

pub const AMDF_XDNA_QUEUE_FORMAT_VERSION_1: ::core::ffi::c_uint = 1;

pub const AMDF_XDNA_TARGET_ID_CAPACITY: ::core::ffi::c_uint = 64;

pub const AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1: ::core::ffi::c_uint = 1;
