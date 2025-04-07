// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_api_trace_version.h>
#include <hsa/hsa_ext_amd.h>

namespace rocprofiler
{
namespace counters
{
namespace test_constants
{
AmdExtTable&
get_ext_table()
{
    static auto _v = []() {
        auto val                                      = AmdExtTable{};
        val.version.major_id                          = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
        val.version.minor_id                          = sizeof(AmdExtTable);
        val.version.step_id                           = HSA_AMD_EXT_API_TABLE_STEP_VERSION;
        val.hsa_amd_coherency_get_type_fn             = hsa_amd_coherency_get_type;
        val.hsa_amd_coherency_set_type_fn             = hsa_amd_coherency_set_type;
        val.hsa_amd_profiling_set_profiler_enabled_fn = hsa_amd_profiling_set_profiler_enabled;
        val.hsa_amd_profiling_async_copy_enable_fn    = hsa_amd_profiling_async_copy_enable;
        val.hsa_amd_profiling_get_dispatch_time_fn    = hsa_amd_profiling_get_dispatch_time;
        val.hsa_amd_profiling_get_async_copy_time_fn  = hsa_amd_profiling_get_async_copy_time;
        val.hsa_amd_profiling_convert_tick_to_system_domain_fn =
            hsa_amd_profiling_convert_tick_to_system_domain;
        val.hsa_amd_signal_async_handler_fn          = hsa_amd_signal_async_handler;
        val.hsa_amd_async_function_fn                = hsa_amd_async_function;
        val.hsa_amd_signal_wait_any_fn               = hsa_amd_signal_wait_any;
        val.hsa_amd_queue_cu_set_mask_fn             = hsa_amd_queue_cu_set_mask;
        val.hsa_amd_memory_pool_get_info_fn          = hsa_amd_memory_pool_get_info;
        val.hsa_amd_agent_iterate_memory_pools_fn    = hsa_amd_agent_iterate_memory_pools;
        val.hsa_amd_memory_pool_allocate_fn          = hsa_amd_memory_pool_allocate;
        val.hsa_amd_memory_pool_free_fn              = hsa_amd_memory_pool_free;
        val.hsa_amd_memory_async_copy_fn             = hsa_amd_memory_async_copy;
        val.hsa_amd_memory_async_copy_on_engine_fn   = hsa_amd_memory_async_copy_on_engine;
        val.hsa_amd_memory_copy_engine_status_fn     = hsa_amd_memory_copy_engine_status;
        val.hsa_amd_agent_memory_pool_get_info_fn    = hsa_amd_agent_memory_pool_get_info;
        val.hsa_amd_agents_allow_access_fn           = hsa_amd_agents_allow_access;
        val.hsa_amd_memory_pool_can_migrate_fn       = hsa_amd_memory_pool_can_migrate;
        val.hsa_amd_memory_migrate_fn                = hsa_amd_memory_migrate;
        val.hsa_amd_memory_lock_fn                   = hsa_amd_memory_lock;
        val.hsa_amd_memory_unlock_fn                 = hsa_amd_memory_unlock;
        val.hsa_amd_memory_fill_fn                   = hsa_amd_memory_fill;
        val.hsa_amd_interop_map_buffer_fn            = hsa_amd_interop_map_buffer;
        val.hsa_amd_interop_unmap_buffer_fn          = hsa_amd_interop_unmap_buffer;
        val.hsa_amd_image_create_fn                  = hsa_amd_image_create;
        val.hsa_amd_pointer_info_fn                  = hsa_amd_pointer_info;
        val.hsa_amd_pointer_info_set_userdata_fn     = hsa_amd_pointer_info_set_userdata;
        val.hsa_amd_ipc_memory_create_fn             = hsa_amd_ipc_memory_create;
        val.hsa_amd_ipc_memory_attach_fn             = hsa_amd_ipc_memory_attach;
        val.hsa_amd_ipc_memory_detach_fn             = hsa_amd_ipc_memory_detach;
        val.hsa_amd_signal_create_fn                 = hsa_amd_signal_create;
        val.hsa_amd_ipc_signal_create_fn             = hsa_amd_ipc_signal_create;
        val.hsa_amd_ipc_signal_attach_fn             = hsa_amd_ipc_signal_attach;
        val.hsa_amd_register_system_event_handler_fn = hsa_amd_register_system_event_handler;
        // Cannot be set, no visable public symbols
        // val.hsa_amd_queue_intercept_create_fn = hsa_amd_queue_intercept_create;
        // val.hsa_amd_queue_intercept_register_fn = hsa_amd_queue_intercept_register;
        val.hsa_amd_queue_set_priority_fn     = hsa_amd_queue_set_priority;
        val.hsa_amd_memory_async_copy_rect_fn = hsa_amd_memory_async_copy_rect;
        // val.hsa_amd_runtime_queue_create_register_fn = hsa_amd_runtime_queue_create_register;
        val.hsa_amd_memory_lock_to_pool_fn              = hsa_amd_memory_lock_to_pool;
        val.hsa_amd_register_deallocation_callback_fn   = hsa_amd_register_deallocation_callback;
        val.hsa_amd_deregister_deallocation_callback_fn = hsa_amd_deregister_deallocation_callback;
        val.hsa_amd_signal_value_pointer_fn             = hsa_amd_signal_value_pointer;
        val.hsa_amd_svm_attributes_set_fn               = hsa_amd_svm_attributes_set;
        val.hsa_amd_svm_attributes_get_fn               = hsa_amd_svm_attributes_get;
        val.hsa_amd_svm_prefetch_async_fn               = hsa_amd_svm_prefetch_async;
        val.hsa_amd_spm_acquire_fn                      = hsa_amd_spm_acquire;
        val.hsa_amd_spm_release_fn                      = hsa_amd_spm_release;
        val.hsa_amd_spm_set_dest_buffer_fn              = hsa_amd_spm_set_dest_buffer;
        val.hsa_amd_queue_cu_get_mask_fn                = hsa_amd_queue_cu_get_mask;
        val.hsa_amd_portable_export_dmabuf_fn           = hsa_amd_portable_export_dmabuf;
        val.hsa_amd_portable_close_dmabuf_fn            = hsa_amd_portable_close_dmabuf;
        val.hsa_amd_vmem_address_reserve_fn             = hsa_amd_vmem_address_reserve;
        val.hsa_amd_vmem_address_free_fn                = hsa_amd_vmem_address_free;
        val.hsa_amd_vmem_handle_create_fn               = hsa_amd_vmem_handle_create;
        val.hsa_amd_vmem_handle_release_fn              = hsa_amd_vmem_handle_release;
        val.hsa_amd_vmem_map_fn                         = hsa_amd_vmem_map;
        val.hsa_amd_vmem_unmap_fn                       = hsa_amd_vmem_unmap;
        val.hsa_amd_vmem_set_access_fn                  = hsa_amd_vmem_set_access;
        val.hsa_amd_vmem_get_access_fn                  = hsa_amd_vmem_get_access;
        val.hsa_amd_vmem_export_shareable_handle_fn     = hsa_amd_vmem_export_shareable_handle;
        val.hsa_amd_vmem_import_shareable_handle_fn     = hsa_amd_vmem_import_shareable_handle;
        val.hsa_amd_vmem_retain_alloc_handle_fn         = hsa_amd_vmem_retain_alloc_handle;
        val.hsa_amd_vmem_get_alloc_properties_from_handle_fn =
            hsa_amd_vmem_get_alloc_properties_from_handle;
        val.hsa_amd_agent_set_async_scratch_limit_fn = hsa_amd_agent_set_async_scratch_limit;
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x02
        val.hsa_amd_queue_get_info_fn = hsa_amd_queue_get_info;
#endif
        return val;
    }();
    return _v;
}

CoreApiTable&
get_api_table()
{
    static auto _v = []() {
        auto val                                     = CoreApiTable{};
        val.version.major_id                         = HSA_CORE_API_TABLE_MAJOR_VERSION;
        val.version.minor_id                         = sizeof(CoreApiTable);
        val.version.step_id                          = HSA_CORE_API_TABLE_STEP_VERSION;
        val.hsa_init_fn                              = hsa_init;
        val.hsa_shut_down_fn                         = hsa_shut_down;
        val.hsa_system_get_info_fn                   = hsa_system_get_info;
        val.hsa_system_extension_supported_fn        = hsa_system_extension_supported;
        val.hsa_system_get_extension_table_fn        = hsa_system_get_extension_table;
        val.hsa_iterate_agents_fn                    = hsa_iterate_agents;
        val.hsa_agent_get_info_fn                    = hsa_agent_get_info;
        val.hsa_queue_create_fn                      = hsa_queue_create;
        val.hsa_soft_queue_create_fn                 = hsa_soft_queue_create;
        val.hsa_queue_destroy_fn                     = hsa_queue_destroy;
        val.hsa_queue_inactivate_fn                  = hsa_queue_inactivate;
        val.hsa_queue_load_read_index_scacquire_fn   = hsa_queue_load_read_index_scacquire;
        val.hsa_queue_load_read_index_relaxed_fn     = hsa_queue_load_read_index_relaxed;
        val.hsa_queue_load_write_index_scacquire_fn  = hsa_queue_load_write_index_scacquire;
        val.hsa_queue_load_write_index_relaxed_fn    = hsa_queue_load_write_index_relaxed;
        val.hsa_queue_store_write_index_relaxed_fn   = hsa_queue_store_write_index_relaxed;
        val.hsa_queue_store_write_index_screlease_fn = hsa_queue_store_write_index_screlease;
        val.hsa_queue_cas_write_index_scacq_screl_fn = hsa_queue_cas_write_index_scacq_screl;
        val.hsa_queue_cas_write_index_scacquire_fn   = hsa_queue_cas_write_index_scacquire;
        val.hsa_queue_cas_write_index_relaxed_fn     = hsa_queue_cas_write_index_relaxed;
        val.hsa_queue_cas_write_index_screlease_fn   = hsa_queue_cas_write_index_screlease;
        val.hsa_queue_add_write_index_scacq_screl_fn = hsa_queue_add_write_index_scacq_screl;
        val.hsa_queue_add_write_index_scacquire_fn   = hsa_queue_add_write_index_scacquire;
        val.hsa_queue_add_write_index_relaxed_fn     = hsa_queue_add_write_index_relaxed;
        val.hsa_queue_add_write_index_screlease_fn   = hsa_queue_add_write_index_screlease;
        val.hsa_queue_store_read_index_relaxed_fn    = hsa_queue_store_read_index_relaxed;
        val.hsa_queue_store_read_index_screlease_fn  = hsa_queue_store_read_index_screlease;
        val.hsa_agent_iterate_regions_fn             = hsa_agent_iterate_regions;
        val.hsa_region_get_info_fn                   = hsa_region_get_info;
        val.hsa_agent_get_exception_policies_fn      = hsa_agent_get_exception_policies;
        val.hsa_agent_extension_supported_fn         = hsa_agent_extension_supported;
        val.hsa_memory_register_fn                   = hsa_memory_register;
        val.hsa_memory_deregister_fn                 = hsa_memory_deregister;
        val.hsa_memory_allocate_fn                   = hsa_memory_allocate;
        val.hsa_memory_free_fn                       = hsa_memory_free;
        val.hsa_memory_copy_fn                       = hsa_memory_copy;
        val.hsa_memory_assign_agent_fn               = hsa_memory_assign_agent;
        val.hsa_signal_create_fn                     = hsa_signal_create;
        val.hsa_signal_destroy_fn                    = hsa_signal_destroy;
        val.hsa_signal_load_relaxed_fn               = hsa_signal_load_relaxed;
        val.hsa_signal_load_scacquire_fn             = hsa_signal_load_scacquire;
        val.hsa_signal_store_relaxed_fn              = hsa_signal_store_relaxed;
        val.hsa_signal_store_screlease_fn            = hsa_signal_store_screlease;
        val.hsa_signal_wait_relaxed_fn               = hsa_signal_wait_relaxed;
        val.hsa_signal_wait_scacquire_fn             = hsa_signal_wait_scacquire;
        val.hsa_signal_and_relaxed_fn                = hsa_signal_and_relaxed;
        val.hsa_signal_and_scacquire_fn              = hsa_signal_and_scacquire;
        val.hsa_signal_and_screlease_fn              = hsa_signal_and_screlease;
        val.hsa_signal_and_scacq_screl_fn            = hsa_signal_and_scacq_screl;
        val.hsa_signal_or_relaxed_fn                 = hsa_signal_or_relaxed;
        val.hsa_signal_or_scacquire_fn               = hsa_signal_or_scacquire;
        val.hsa_signal_or_screlease_fn               = hsa_signal_or_screlease;
        val.hsa_signal_or_scacq_screl_fn             = hsa_signal_or_scacq_screl;
        val.hsa_signal_xor_relaxed_fn                = hsa_signal_xor_relaxed;
        val.hsa_signal_xor_scacquire_fn              = hsa_signal_xor_scacquire;
        val.hsa_signal_xor_screlease_fn              = hsa_signal_xor_screlease;
        val.hsa_signal_xor_scacq_screl_fn            = hsa_signal_xor_scacq_screl;
        val.hsa_signal_exchange_relaxed_fn           = hsa_signal_exchange_relaxed;
        val.hsa_signal_exchange_scacquire_fn         = hsa_signal_exchange_scacquire;
        val.hsa_signal_exchange_screlease_fn         = hsa_signal_exchange_screlease;
        val.hsa_signal_exchange_scacq_screl_fn       = hsa_signal_exchange_scacq_screl;
        val.hsa_signal_add_relaxed_fn                = hsa_signal_add_relaxed;
        val.hsa_signal_add_scacquire_fn              = hsa_signal_add_scacquire;
        val.hsa_signal_add_screlease_fn              = hsa_signal_add_screlease;
        val.hsa_signal_add_scacq_screl_fn            = hsa_signal_add_scacq_screl;
        val.hsa_signal_subtract_relaxed_fn           = hsa_signal_subtract_relaxed;
        val.hsa_signal_subtract_scacquire_fn         = hsa_signal_subtract_scacquire;
        val.hsa_signal_subtract_screlease_fn         = hsa_signal_subtract_screlease;
        val.hsa_signal_subtract_scacq_screl_fn       = hsa_signal_subtract_scacq_screl;
        val.hsa_signal_cas_relaxed_fn                = hsa_signal_cas_relaxed;
        val.hsa_signal_cas_scacquire_fn              = hsa_signal_cas_scacquire;
        val.hsa_signal_cas_screlease_fn              = hsa_signal_cas_screlease;
        val.hsa_signal_cas_scacq_screl_fn            = hsa_signal_cas_scacq_screl;
        val.hsa_isa_from_name_fn                     = hsa_isa_from_name;
        val.hsa_isa_get_info_fn                      = hsa_isa_get_info;
        val.hsa_isa_compatible_fn                    = hsa_isa_compatible;
        val.hsa_code_object_serialize_fn             = hsa_code_object_serialize;
        val.hsa_code_object_deserialize_fn           = hsa_code_object_deserialize;
        val.hsa_code_object_destroy_fn               = hsa_code_object_destroy;
        val.hsa_code_object_get_info_fn              = hsa_code_object_get_info;
        val.hsa_code_object_get_symbol_fn            = hsa_code_object_get_symbol;
        val.hsa_code_symbol_get_info_fn              = hsa_code_symbol_get_info;
        val.hsa_code_object_iterate_symbols_fn       = hsa_code_object_iterate_symbols;
        val.hsa_executable_create_fn                 = hsa_executable_create;
        val.hsa_executable_destroy_fn                = hsa_executable_destroy;
        val.hsa_executable_load_code_object_fn       = hsa_executable_load_code_object;
        val.hsa_executable_freeze_fn                 = hsa_executable_freeze;
        val.hsa_executable_get_info_fn               = hsa_executable_get_info;
        val.hsa_executable_global_variable_define_fn = hsa_executable_global_variable_define;
        val.hsa_executable_agent_global_variable_define_fn =
            hsa_executable_agent_global_variable_define;
        val.hsa_executable_readonly_variable_define_fn = hsa_executable_readonly_variable_define;
        val.hsa_executable_validate_fn                 = hsa_executable_validate;
        val.hsa_executable_get_symbol_fn               = hsa_executable_get_symbol;
        val.hsa_executable_symbol_get_info_fn          = hsa_executable_symbol_get_info;
        val.hsa_executable_iterate_symbols_fn          = hsa_executable_iterate_symbols;
        val.hsa_status_string_fn                       = hsa_status_string;
        val.hsa_extension_get_name_fn                  = hsa_extension_get_name;
        val.hsa_system_major_extension_supported_fn    = hsa_system_major_extension_supported;
        val.hsa_system_get_major_extension_table_fn    = hsa_system_get_major_extension_table;
        val.hsa_agent_major_extension_supported_fn     = hsa_agent_major_extension_supported;
        val.hsa_cache_get_info_fn                      = hsa_cache_get_info;
        val.hsa_agent_iterate_caches_fn                = hsa_agent_iterate_caches;
        val.hsa_signal_silent_store_relaxed_fn         = hsa_signal_silent_store_relaxed;
        val.hsa_signal_silent_store_screlease_fn       = hsa_signal_silent_store_screlease;
        val.hsa_signal_group_create_fn                 = hsa_signal_group_create;
        val.hsa_signal_group_destroy_fn                = hsa_signal_group_destroy;
        val.hsa_signal_group_wait_any_scacquire_fn     = hsa_signal_group_wait_any_scacquire;
        val.hsa_signal_group_wait_any_relaxed_fn       = hsa_signal_group_wait_any_relaxed;
        val.hsa_agent_iterate_isas_fn                  = hsa_agent_iterate_isas;
        val.hsa_isa_get_info_alt_fn                    = hsa_isa_get_info_alt;
        val.hsa_isa_get_exception_policies_fn          = hsa_isa_get_exception_policies;
        val.hsa_isa_get_round_method_fn                = hsa_isa_get_round_method;
        val.hsa_wavefront_get_info_fn                  = hsa_wavefront_get_info;
        val.hsa_isa_iterate_wavefronts_fn              = hsa_isa_iterate_wavefronts;
        val.hsa_code_object_get_symbol_from_name_fn    = hsa_code_object_get_symbol_from_name;
        val.hsa_code_object_reader_create_from_file_fn = hsa_code_object_reader_create_from_file;
        val.hsa_code_object_reader_create_from_memory_fn =
            hsa_code_object_reader_create_from_memory;
        val.hsa_code_object_reader_destroy_fn          = hsa_code_object_reader_destroy;
        val.hsa_executable_create_alt_fn               = hsa_executable_create_alt;
        val.hsa_executable_load_program_code_object_fn = hsa_executable_load_program_code_object;
        val.hsa_executable_load_agent_code_object_fn   = hsa_executable_load_agent_code_object;
        val.hsa_executable_validate_alt_fn             = hsa_executable_validate_alt;
        val.hsa_executable_get_symbol_by_name_fn       = hsa_executable_get_symbol_by_name;
        val.hsa_executable_iterate_agent_symbols_fn    = hsa_executable_iterate_agent_symbols;
        val.hsa_executable_iterate_program_symbols_fn  = hsa_executable_iterate_program_symbols;
        return val;
    }();
    return _v;
}
}  // namespace test_constants
}  // namespace counters
}  // namespace rocprofiler
