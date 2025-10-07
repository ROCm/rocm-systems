/* Copyright (c) 2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "os/os.hpp"
#include "utils/flags.hpp"
#include "rocrctx.hpp"

namespace amd {
namespace roc {

std::once_flag Hsa::initialized;
RocrEntryPoints Hsa::cep_;
bool Hsa::is_ready_ = false;

bool Hsa::LoadLib() {
#if defined(ROCR_DYN_DLL)
  static const char* rocr_lib_name = WINDOWS_SWITCH("hsa-runtime64.dll", "hsa-runtime64.so.1");
  cep_.handle = Os::loadLibrary(rocr_lib_name);
  if (nullptr == cep_.handle) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Failed to load COMGR library.");
    return false;
  }
#endif
  GET_ROCR_SYMBOL(hsa_init)
  GET_ROCR_SYMBOL(hsa_shut_down)
  GET_ROCR_SYMBOL(hsa_system_get_info)
  GET_ROCR_SYMBOL(hsa_iterate_agents)
  GET_ROCR_SYMBOL(hsa_agent_get_info)
  GET_ROCR_SYMBOL(hsa_queue_create)
  GET_ROCR_SYMBOL(hsa_queue_destroy)
  GET_ROCR_SYMBOL(hsa_queue_load_read_index_scacquire)
  GET_ROCR_SYMBOL(hsa_queue_load_read_index_relaxed)
  GET_ROCR_SYMBOL(hsa_queue_load_write_index_relaxed)
  GET_ROCR_SYMBOL(hsa_queue_add_write_index_screlease)
  GET_ROCR_SYMBOL(hsa_memory_register)
  GET_ROCR_SYMBOL(hsa_memory_deregister)
  GET_ROCR_SYMBOL(hsa_memory_copy)
  GET_ROCR_SYMBOL(hsa_signal_create)
  GET_ROCR_SYMBOL(hsa_signal_destroy)
  GET_ROCR_SYMBOL(hsa_signal_load_relaxed)
  GET_ROCR_SYMBOL(hsa_signal_store_relaxed)
  GET_ROCR_SYMBOL(hsa_signal_silent_store_relaxed)
  GET_ROCR_SYMBOL(hsa_signal_store_screlease)
  GET_ROCR_SYMBOL(hsa_signal_wait_scacquire)
  GET_ROCR_SYMBOL(hsa_signal_add_relaxed)
  GET_ROCR_SYMBOL(hsa_signal_subtract_relaxed)
  GET_ROCR_SYMBOL(hsa_isa_get_info_alt)
  GET_ROCR_SYMBOL(hsa_agent_iterate_isas)
  GET_ROCR_SYMBOL(hsa_system_get_major_extension_table)
  GET_ROCR_SYMBOL(hsa_status_string)
  GET_ROCR_SYMBOL(hsa_executable_create_alt)
  GET_ROCR_SYMBOL(hsa_executable_destroy)
  GET_ROCR_SYMBOL(hsa_executable_get_info)
  GET_ROCR_SYMBOL(hsa_code_object_reader_destroy)
  GET_ROCR_SYMBOL(hsa_code_object_reader_create_from_memory)
  GET_ROCR_SYMBOL(hsa_executable_load_agent_code_object)
  GET_ROCR_SYMBOL(hsa_executable_agent_global_variable_define)
  GET_ROCR_SYMBOL(hsa_executable_get_symbol_by_name)
  GET_ROCR_SYMBOL(hsa_executable_symbol_get_info)
  GET_ROCR_SYMBOL(hsa_executable_freeze)
  // AMD extensions
  GET_ROCR_SYMBOL(hsa_amd_coherency_set_type)
  GET_ROCR_SYMBOL(hsa_amd_profiling_set_profiler_enabled)
  GET_ROCR_SYMBOL(hsa_amd_profiling_async_copy_enable)
  GET_ROCR_SYMBOL(hsa_amd_profiling_get_dispatch_time)
  GET_ROCR_SYMBOL(hsa_amd_profiling_get_async_copy_time)
  GET_ROCR_SYMBOL(hsa_amd_signal_async_handler)
  GET_ROCR_SYMBOL(hsa_amd_queue_cu_set_mask)
  GET_ROCR_SYMBOL(hsa_amd_memory_pool_get_info)
  GET_ROCR_SYMBOL(hsa_amd_agent_iterate_memory_pools)
  GET_ROCR_SYMBOL(hsa_amd_memory_pool_allocate)
  GET_ROCR_SYMBOL(hsa_amd_memory_pool_free)
  GET_ROCR_SYMBOL(hsa_amd_memory_async_copy)
  GET_ROCR_SYMBOL(hsa_amd_memory_async_copy_on_engine)
  GET_ROCR_SYMBOL(hsa_amd_memory_copy_engine_status)
  GET_ROCR_SYMBOL(hsa_amd_agent_memory_pool_get_info)
  GET_ROCR_SYMBOL(hsa_amd_agents_allow_access)
  GET_ROCR_SYMBOL(hsa_amd_memory_unlock)
  GET_ROCR_SYMBOL(hsa_amd_interop_map_buffer)
  GET_ROCR_SYMBOL(hsa_amd_interop_unmap_buffer)
  GET_ROCR_SYMBOL(hsa_amd_image_create)
  GET_ROCR_SYMBOL(hsa_amd_pointer_info)
  GET_ROCR_SYMBOL(hsa_amd_ipc_memory_create)
  GET_ROCR_SYMBOL(hsa_amd_ipc_memory_attach)
  GET_ROCR_SYMBOL(hsa_amd_ipc_memory_detach)
  GET_ROCR_SYMBOL(hsa_amd_signal_create)
  GET_ROCR_SYMBOL(hsa_amd_register_system_event_handler)
  GET_ROCR_SYMBOL(hsa_amd_queue_set_priority)
  GET_ROCR_SYMBOL(hsa_amd_memory_async_copy_rect)
  GET_ROCR_SYMBOL(hsa_amd_memory_lock_to_pool)
  GET_ROCR_SYMBOL(hsa_amd_signal_value_pointer)
  GET_ROCR_SYMBOL(hsa_amd_svm_attributes_set)
  GET_ROCR_SYMBOL(hsa_amd_svm_attributes_get)
  GET_ROCR_SYMBOL(hsa_amd_svm_prefetch_async)
  GET_ROCR_SYMBOL(hsa_amd_portable_export_dmabuf)
  GET_ROCR_SYMBOL(hsa_amd_portable_close_dmabuf)
  GET_ROCR_SYMBOL(hsa_amd_vmem_address_reserve)
  GET_ROCR_SYMBOL(hsa_amd_vmem_address_free)
  GET_ROCR_SYMBOL(hsa_amd_vmem_handle_create)
  GET_ROCR_SYMBOL(hsa_amd_vmem_handle_release)
  GET_ROCR_SYMBOL(hsa_amd_vmem_map)
  GET_ROCR_SYMBOL(hsa_amd_vmem_unmap)
  GET_ROCR_SYMBOL(hsa_amd_vmem_set_access)
  GET_ROCR_SYMBOL(hsa_amd_vmem_get_access)
  GET_ROCR_SYMBOL(hsa_amd_vmem_export_shareable_handle)
  GET_ROCR_SYMBOL(hsa_amd_vmem_import_shareable_handle)
  GET_ROCR_SYMBOL(hsa_amd_vmem_retain_alloc_handle)
  GET_ROCR_SYMBOL(hsa_amd_agent_set_async_scratch_limit)
  GET_ROCR_SYMBOL(hsa_amd_vmem_address_reserve_align)
  GET_ROCR_SYMBOL(hsa_amd_enable_logging)
  GET_ROCR_SYMBOL(hsa_amd_memory_get_preferred_copy_engine)
  GET_ROCR_SYMBOL(hsa_amd_ais_file_read)
  GET_ROCR_SYMBOL(hsa_amd_ais_file_write)

  // Image extensions
  GET_ROCR_SYMBOL(hsa_ext_image_data_get_info)
  GET_ROCR_SYMBOL(hsa_ext_image_create)
  GET_ROCR_SYMBOL(hsa_ext_image_import)
  GET_ROCR_SYMBOL(hsa_ext_image_export)
  GET_ROCR_SYMBOL(hsa_ext_image_destroy)
  GET_ROCR_SYMBOL(hsa_ext_sampler_create_v2)
  GET_ROCR_SYMBOL(hsa_ext_sampler_destroy)
  GET_ROCR_SYMBOL(hsa_ext_image_create_with_layout)
  is_ready_ = true;
  return true;
}
}  // namespace roc
}  // namespace amd
