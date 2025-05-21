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

#pragma once

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/hsa/api_trace_version.h>

/**
 * @brief ROCProfiler enumeration of HSA AMD Extended API tracing operations
 */
typedef enum rocprofiler_hsa_amd_ext_api_id_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_HSA_AMD_EXT_API_ID_NONE = -1,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_coherency_get_type,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_coherency_set_type,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_set_profiler_enabled,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_async_copy_enable,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_get_dispatch_time,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_get_async_copy_time,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_convert_tick_to_system_domain,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_async_handler,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_async_function,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_wait_any,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_cu_set_mask,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_get_info,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agent_iterate_memory_pools,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_allocate,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_free,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_on_engine,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_copy_engine_status,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agent_memory_pool_get_info,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agents_allow_access,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_can_migrate,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_migrate,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_lock,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_unlock,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_fill,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_interop_map_buffer,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_interop_unmap_buffer,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_image_create,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_pointer_info,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_pointer_info_set_userdata,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_memory_create,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_memory_attach,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_memory_detach,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_create,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_signal_create,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_signal_attach,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_register_system_event_handler,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_intercept_create,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_intercept_register,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_set_priority,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_rect,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_runtime_queue_create_register,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_lock_to_pool,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_register_deallocation_callback,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_deregister_deallocation_callback,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_value_pointer,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_svm_attributes_set,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_svm_attributes_get,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_svm_prefetch_async,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_spm_acquire,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_spm_release,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_spm_set_dest_buffer,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_cu_get_mask,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_portable_export_dmabuf,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_portable_close_dmabuf,

#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION >= 0x02
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_address_reserve,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_address_free,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_handle_create,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_handle_release,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_map,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_unmap,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_set_access,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_get_access,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_export_shareable_handle,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_import_shareable_handle,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_retain_alloc_handle,
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_get_alloc_properties_from_handle,
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x01
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agent_set_async_scratch_limit,
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x02
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_get_info,
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x03
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_address_reserve_align,
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x04
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_enable_logging,
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x05
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_wait_all,
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x06
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_get_preferred_copy_engine,
#    endif
#endif

    ROCPROFILER_HSA_AMD_EXT_API_ID_LAST,
} rocprofiler_hsa_amd_ext_api_id_t;
