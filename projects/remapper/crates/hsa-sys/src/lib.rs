//! The HSA runtime functions the remapper calls, declared by hand from `hsa/hsa.h` and `hsa/hsa_ext_amd.h`.
//!
//! The remapper runs inside a process whose HSA runtime is already initialized (it is driven from a rocprofiler-sdk
//! tool), so it never calls `hsa_init` or `hsa_shut_down`.
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};

pub type hsa_status_t = u32;
pub const HSA_STATUS_SUCCESS: hsa_status_t = 0x0;
pub const HSA_STATUS_INFO_BREAK: hsa_status_t = 0x1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct hsa_agent_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct hsa_amd_memory_pool_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct hsa_code_object_reader_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct hsa_executable_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct hsa_executable_symbol_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct hsa_loaded_code_object_t {
    pub handle: u64,
}

pub type hsa_agent_info_t = u32;
pub const HSA_AGENT_INFO_PROFILE: hsa_agent_info_t = 4;
pub const HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: hsa_agent_info_t = 5;
pub const HSA_AGENT_INFO_DEVICE: hsa_agent_info_t = 17;
pub const HSA_AMD_AGENT_INFO_DRIVER_NODE_ID: hsa_agent_info_t = 0xA004;
pub const HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES: hsa_agent_info_t = 0xA00C;

pub type hsa_device_type_t = u32;
pub const HSA_DEVICE_TYPE_GPU: hsa_device_type_t = 1;

pub type hsa_profile_t = u32;
pub type hsa_default_float_rounding_mode_t = u32;

pub type hsa_executable_symbol_info_t = u32;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: hsa_executable_symbol_info_t = 11;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: hsa_executable_symbol_info_t = 13;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: hsa_executable_symbol_info_t = 14;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: hsa_executable_symbol_info_t = 22;

pub type hsa_amd_segment_t = u32;
pub const HSA_AMD_SEGMENT_GLOBAL: hsa_amd_segment_t = 0;

pub type hsa_amd_memory_pool_info_t = u32;
pub const HSA_AMD_MEMORY_POOL_INFO_SEGMENT: hsa_amd_memory_pool_info_t = 0;
pub const HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: hsa_amd_memory_pool_info_t = 1;
pub const HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: hsa_amd_memory_pool_info_t = 5;

pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: u32 = 1;
pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED: u32 = 2;
pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED: u32 = 4;

pub type hsa_agent_callback_t =
    unsafe extern "C" fn(agent: hsa_agent_t, data: *mut c_void) -> hsa_status_t;
pub type hsa_amd_memory_pool_callback_t =
    unsafe extern "C" fn(pool: hsa_amd_memory_pool_t, data: *mut c_void) -> hsa_status_t;

extern "C" {
    pub fn hsa_status_string(
        status: hsa_status_t,
        status_string: *mut *const c_char,
    ) -> hsa_status_t;

    pub fn hsa_iterate_agents(callback: hsa_agent_callback_t, data: *mut c_void) -> hsa_status_t;
    pub fn hsa_agent_get_info(
        agent: hsa_agent_t,
        attribute: hsa_agent_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;

    pub fn hsa_amd_agent_iterate_memory_pools(
        agent: hsa_agent_t,
        callback: hsa_amd_memory_pool_callback_t,
        data: *mut c_void,
    ) -> hsa_status_t;
    pub fn hsa_amd_memory_pool_get_info(
        pool: hsa_amd_memory_pool_t,
        attribute: hsa_amd_memory_pool_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;
    pub fn hsa_amd_memory_pool_allocate(
        pool: hsa_amd_memory_pool_t,
        size: usize,
        flags: u32,
        ptr: *mut *mut c_void,
    ) -> hsa_status_t;
    pub fn hsa_amd_memory_pool_free(ptr: *mut c_void) -> hsa_status_t;
    pub fn hsa_amd_agents_allow_access(
        num_agents: u32,
        agents: *const hsa_agent_t,
        flags: *const u32,
        ptr: *const c_void,
    ) -> hsa_status_t;
    pub fn hsa_memory_copy(dst: *mut c_void, src: *const c_void, size: usize) -> hsa_status_t;

    pub fn hsa_code_object_reader_create_from_memory(
        code_object: *const c_void,
        size: usize,
        code_object_reader: *mut hsa_code_object_reader_t,
    ) -> hsa_status_t;
    pub fn hsa_code_object_reader_destroy(
        code_object_reader: hsa_code_object_reader_t,
    ) -> hsa_status_t;
    pub fn hsa_executable_create_alt(
        profile: hsa_profile_t,
        default_float_rounding_mode: hsa_default_float_rounding_mode_t,
        options: *const c_char,
        executable: *mut hsa_executable_t,
    ) -> hsa_status_t;
    pub fn hsa_executable_load_agent_code_object(
        executable: hsa_executable_t,
        agent: hsa_agent_t,
        code_object_reader: hsa_code_object_reader_t,
        options: *const c_char,
        loaded_code_object: *mut hsa_loaded_code_object_t,
    ) -> hsa_status_t;
    pub fn hsa_executable_freeze(
        executable: hsa_executable_t,
        options: *const c_char,
    ) -> hsa_status_t;
    pub fn hsa_executable_get_symbol_by_name(
        executable: hsa_executable_t,
        symbol_name: *const c_char,
        agent: *const hsa_agent_t,
        symbol: *mut hsa_executable_symbol_t,
    ) -> hsa_status_t;
    pub fn hsa_executable_symbol_get_info(
        executable_symbol: hsa_executable_symbol_t,
        attribute: hsa_executable_symbol_info_t,
        value: *mut c_void,
    ) -> hsa_status_t;
    pub fn hsa_executable_destroy(executable: hsa_executable_t) -> hsa_status_t;
}
