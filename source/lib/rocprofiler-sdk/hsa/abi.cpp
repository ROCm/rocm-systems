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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/hsa.h>

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

namespace rocprofiler
{
namespace hsa
{
static_assert(HSA_CORE_API_TABLE_MAJOR_VERSION == 0x02, "Major version updated");
static_assert(HSA_AMD_EXT_API_TABLE_MAJOR_VERSION == 0x02, "Major version updated");
static_assert(HSA_IMAGE_API_TABLE_MAJOR_VERSION == 0x02, "Major version updated");
static_assert(HSA_FINALIZER_API_TABLE_MAJOR_VERSION == 0x02, "Major version updated");
static_assert(HSA_TOOLS_API_TABLE_MAJOR_VERSION == 0x01, "Major version updated");
static_assert(HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION == 0x01, "Major version updated");

#if HSA_CORE_API_TABLE_STEP_VERSION == 0x00
ROCP_SDK_ENFORCE_ABI_VERSIONING(::CoreApiTable, 126)
#endif

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x00
ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 68);
#elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x01
ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 69);
#elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x02
ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 70);
#elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x03
ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 71);
#elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x04
ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 72);
#elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x05
ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 73);
#elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x06
ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 74);
#else
INTERNAL_CI_ROCP_SDK_ENFORCE_ABI_VERSIONING(::AmdExtTable, 0);
#endif

#if HSA_IMAGE_EXT_API_TABLE_STEP_VERSION == 0x00
ROCP_SDK_ENFORCE_ABI_VERSIONING(::ImageExtTable, 14);
#endif

#if HSA_FINALIZER_EXT_API_TABLE_STEP_VERSION == 0x00
ROCP_SDK_ENFORCE_ABI_VERSIONING(::FinalizerExtTable, 7);
#endif

#if HSA_TOOLS_API_TABLE_STEP_VERSION == 0x00
ROCP_SDK_ENFORCE_ABI_VERSIONING(::ToolsApiTable, 7);
#endif

#if HSA_PC_SAMPLING_API_TABLE_STEP_VERSION == 0x00
ROCP_SDK_ENFORCE_ABI_VERSIONING(::PcSamplingExtTable, 8);
#endif

// These ensure that function pointers are not re-ordered
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_init_fn, 1);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_shut_down_fn, 2);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_system_get_info_fn, 3);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_system_extension_supported_fn, 4);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_system_get_extension_table_fn, 5);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_iterate_agents_fn, 6);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_agent_get_info_fn, 7);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_create_fn, 8);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_soft_queue_create_fn, 9);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_destroy_fn, 10);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_inactivate_fn, 11);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_load_read_index_scacquire_fn, 12);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_load_read_index_relaxed_fn, 13);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_load_write_index_scacquire_fn, 14);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_load_write_index_relaxed_fn, 15);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_store_write_index_relaxed_fn, 16);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_store_write_index_screlease_fn, 17);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_cas_write_index_scacq_screl_fn, 18);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_cas_write_index_scacquire_fn, 19);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_cas_write_index_relaxed_fn, 20);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_cas_write_index_screlease_fn, 21);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_add_write_index_scacq_screl_fn, 22);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_add_write_index_scacquire_fn, 23);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_add_write_index_relaxed_fn, 24);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_add_write_index_screlease_fn, 25);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_store_read_index_relaxed_fn, 26);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_queue_store_read_index_screlease_fn, 27);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_agent_iterate_regions_fn, 28);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_region_get_info_fn, 29);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_agent_get_exception_policies_fn, 30);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_agent_extension_supported_fn, 31);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_memory_register_fn, 32);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_memory_deregister_fn, 33);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_memory_allocate_fn, 34);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_memory_free_fn, 35);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_memory_copy_fn, 36);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_memory_assign_agent_fn, 37);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_create_fn, 38);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_destroy_fn, 39);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_load_relaxed_fn, 40);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_load_scacquire_fn, 41);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_store_relaxed_fn, 42);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_store_screlease_fn, 43);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_wait_relaxed_fn, 44);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_wait_scacquire_fn, 45);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_and_relaxed_fn, 46);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_and_scacquire_fn, 47);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_and_screlease_fn, 48);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_and_scacq_screl_fn, 49);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_or_relaxed_fn, 50);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_or_scacquire_fn, 51);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_or_screlease_fn, 52);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_or_scacq_screl_fn, 53);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_xor_relaxed_fn, 54);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_xor_scacquire_fn, 55);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_xor_screlease_fn, 56);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_xor_scacq_screl_fn, 57);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_exchange_relaxed_fn, 58);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_exchange_scacquire_fn, 59);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_exchange_screlease_fn, 60);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_exchange_scacq_screl_fn, 61);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_add_relaxed_fn, 62);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_add_scacquire_fn, 63);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_add_screlease_fn, 64);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_add_scacq_screl_fn, 65);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_subtract_relaxed_fn, 66);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_subtract_scacquire_fn, 67);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_subtract_screlease_fn, 68);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_subtract_scacq_screl_fn, 69);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_cas_relaxed_fn, 70);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_cas_scacquire_fn, 71);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_cas_screlease_fn, 72);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_cas_scacq_screl_fn, 73);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_isa_from_name_fn, 74);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_isa_get_info_fn, 75);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_isa_compatible_fn, 76);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_serialize_fn, 77);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_deserialize_fn, 78);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_destroy_fn, 79);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_get_info_fn, 80);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_get_symbol_fn, 81);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_symbol_get_info_fn, 82);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_iterate_symbols_fn, 83);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_create_fn, 84);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_destroy_fn, 85);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_load_code_object_fn, 86);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_freeze_fn, 87);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_get_info_fn, 88);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_global_variable_define_fn, 89);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_agent_global_variable_define_fn, 90);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_readonly_variable_define_fn, 91);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_validate_fn, 92);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_get_symbol_fn, 93);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_symbol_get_info_fn, 94);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_iterate_symbols_fn, 95);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_status_string_fn, 96);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_extension_get_name_fn, 97);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_system_major_extension_supported_fn, 98);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_system_get_major_extension_table_fn, 99);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_agent_major_extension_supported_fn, 100);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_cache_get_info_fn, 101);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_agent_iterate_caches_fn, 102);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_silent_store_relaxed_fn, 103);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_silent_store_screlease_fn, 104);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_group_create_fn, 105);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_group_destroy_fn, 106);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_group_wait_any_scacquire_fn, 107);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_signal_group_wait_any_relaxed_fn, 108);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_agent_iterate_isas_fn, 109);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_isa_get_info_alt_fn, 110);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_isa_get_exception_policies_fn, 111);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_isa_get_round_method_fn, 112);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_wavefront_get_info_fn, 113);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_isa_iterate_wavefronts_fn, 114);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_get_symbol_from_name_fn, 115);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_reader_create_from_file_fn, 116);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_reader_create_from_memory_fn, 117);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_code_object_reader_destroy_fn, 118);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_create_alt_fn, 119);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_load_program_code_object_fn, 120);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_load_agent_code_object_fn, 121);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_validate_alt_fn, 122);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_get_symbol_by_name_fn, 123);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_iterate_agent_symbols_fn, 124);
ROCP_SDK_ENFORCE_ABI(::CoreApiTable, hsa_executable_iterate_program_symbols_fn, 125);

// These ensure that function pointers are not re-ordered
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_coherency_get_type_fn, 1);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_coherency_set_type_fn, 2);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_profiling_set_profiler_enabled_fn, 3);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_profiling_async_copy_enable_fn, 4);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_profiling_get_dispatch_time_fn, 5);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_profiling_get_async_copy_time_fn, 6);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_profiling_convert_tick_to_system_domain_fn, 7);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_signal_async_handler_fn, 8);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_async_function_fn, 9);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_signal_wait_any_fn, 10);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_queue_cu_set_mask_fn, 11);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_pool_get_info_fn, 12);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_agent_iterate_memory_pools_fn, 13);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_pool_allocate_fn, 14);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_pool_free_fn, 15);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_async_copy_fn, 16);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_async_copy_on_engine_fn, 17);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_copy_engine_status_fn, 18);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_agent_memory_pool_get_info_fn, 19);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_agents_allow_access_fn, 20);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_pool_can_migrate_fn, 21);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_migrate_fn, 22);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_lock_fn, 23);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_unlock_fn, 24);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_fill_fn, 25);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_interop_map_buffer_fn, 26);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_interop_unmap_buffer_fn, 27);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_image_create_fn, 28);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_pointer_info_fn, 29);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_pointer_info_set_userdata_fn, 30);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_ipc_memory_create_fn, 31);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_ipc_memory_attach_fn, 32);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_ipc_memory_detach_fn, 33);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_signal_create_fn, 34);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_ipc_signal_create_fn, 35);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_ipc_signal_attach_fn, 36);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_register_system_event_handler_fn, 37);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_queue_intercept_create_fn, 38);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_queue_intercept_register_fn, 39);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_queue_set_priority_fn, 40);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_async_copy_rect_fn, 41);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_runtime_queue_create_register_fn, 42);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_lock_to_pool_fn, 43);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_register_deallocation_callback_fn, 44);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_deregister_deallocation_callback_fn, 45);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_signal_value_pointer_fn, 46);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_svm_attributes_set_fn, 47);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_svm_attributes_get_fn, 48);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_svm_prefetch_async_fn, 49);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_spm_acquire_fn, 50);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_spm_release_fn, 51);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_spm_set_dest_buffer_fn, 52);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_queue_cu_get_mask_fn, 53);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_portable_export_dmabuf_fn, 54);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_portable_close_dmabuf_fn, 55);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_address_reserve_fn, 56);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_address_free_fn, 57);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_handle_create_fn, 58);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_handle_release_fn, 59);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_map_fn, 60);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_unmap_fn, 61);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_set_access_fn, 62);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_get_access_fn, 63);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_export_shareable_handle_fn, 64);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_import_shareable_handle_fn, 65);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_retain_alloc_handle_fn, 66);
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_get_alloc_properties_from_handle_fn, 67);
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x01
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_agent_set_async_scratch_limit_fn, 68);
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x02
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_queue_get_info_fn, 69);
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x03
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_vmem_address_reserve_align_fn, 70);
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x04
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_enable_logging_fn, 71);
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x05
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_signal_wait_all_fn, 72);
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x06
ROCP_SDK_ENFORCE_ABI(::AmdExtTable, hsa_amd_memory_get_preferred_copy_engine_fn, 73);
#endif

ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_get_capability_fn, 1);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_data_get_info_fn, 2);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_create_fn, 3);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_import_fn, 4);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_export_fn, 5);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_copy_fn, 6);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_clear_fn, 7);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_destroy_fn, 8);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_sampler_create_fn, 9);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_sampler_destroy_fn, 10);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_get_capability_with_layout_fn, 11);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_data_get_info_with_layout_fn, 12);
ROCP_SDK_ENFORCE_ABI(::ImageExtTable, hsa_ext_image_create_with_layout_fn, 13);

ROCP_SDK_ENFORCE_ABI(::FinalizerExtTable, hsa_ext_program_create_fn, 1);
ROCP_SDK_ENFORCE_ABI(::FinalizerExtTable, hsa_ext_program_destroy_fn, 2);
ROCP_SDK_ENFORCE_ABI(::FinalizerExtTable, hsa_ext_program_add_module_fn, 3);
ROCP_SDK_ENFORCE_ABI(::FinalizerExtTable, hsa_ext_program_iterate_modules_fn, 4);
ROCP_SDK_ENFORCE_ABI(::FinalizerExtTable, hsa_ext_program_get_info_fn, 5);
ROCP_SDK_ENFORCE_ABI(::FinalizerExtTable, hsa_ext_program_finalize_fn, 6);

ROCP_SDK_ENFORCE_ABI(::ToolsApiTable, hsa_amd_tool_scratch_event_alloc_start_fn, 1);
ROCP_SDK_ENFORCE_ABI(::ToolsApiTable, hsa_amd_tool_scratch_event_alloc_end_fn, 2);
ROCP_SDK_ENFORCE_ABI(::ToolsApiTable, hsa_amd_tool_scratch_event_free_start_fn, 3);
ROCP_SDK_ENFORCE_ABI(::ToolsApiTable, hsa_amd_tool_scratch_event_free_end_fn, 4);
ROCP_SDK_ENFORCE_ABI(::ToolsApiTable, hsa_amd_tool_scratch_event_async_reclaim_start_fn, 5);
ROCP_SDK_ENFORCE_ABI(::ToolsApiTable, hsa_amd_tool_scratch_event_async_reclaim_end_fn, 6);

ROCP_SDK_ENFORCE_ABI(::PcSamplingExtTable, hsa_ven_amd_pcs_iterate_configuration_fn, 1);
ROCP_SDK_ENFORCE_ABI(::PcSamplingExtTable, hsa_ven_amd_pcs_create_fn, 2);
ROCP_SDK_ENFORCE_ABI(::PcSamplingExtTable, hsa_ven_amd_pcs_create_from_id_fn, 3);
ROCP_SDK_ENFORCE_ABI(::PcSamplingExtTable, hsa_ven_amd_pcs_destroy_fn, 4);
ROCP_SDK_ENFORCE_ABI(::PcSamplingExtTable, hsa_ven_amd_pcs_start_fn, 5);
ROCP_SDK_ENFORCE_ABI(::PcSamplingExtTable, hsa_ven_amd_pcs_stop_fn, 6);
ROCP_SDK_ENFORCE_ABI(::PcSamplingExtTable, hsa_ven_amd_pcs_flush_fn, 7);
}  // namespace hsa
}  // namespace rocprofiler
