////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// HSA AMD extension.

#ifndef HSA_RUNTIME_CORE_INC_EXT_AMD_H_
#define HSA_RUNTIME_CORE_INC_EXT_AMD_H_

#include "inc/hsa.h"
#include "inc/hsa_ext_image.h"
#include "inc/hsa_ext_amd.h"

// Wrap internal implementation inside AMD namespace
namespace rocr {
namespace AMD {

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_coherency_get_type(hsa_agent_t agent,
                                                hsa_amd_coherency_type_t* type);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_coherency_set_type(hsa_agent_t agent,
                                                hsa_amd_coherency_type_t type);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_profiling_async_copy_enable(bool enable);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_profiling_get_dispatch_time(
    hsa_agent_t agent, hsa_signal_t signal,
    hsa_amd_profiling_dispatch_time_t* time);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_profiling_get_async_copy_time(
    hsa_signal_t signal, hsa_amd_profiling_async_copy_time_t* time);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent,
                                                    uint64_t agent_tick,
                                                    uint64_t* system_tick);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_signal_async_handler(hsa_signal_t signal,
                                 hsa_signal_condition_t cond,
                                 hsa_signal_value_t value,
                                 hsa_amd_signal_handler handler, void* arg);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_async_function(void (*callback)(void* arg), void* arg);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                                           const hsa_agent_t* consumers, uint64_t attributes,
                                           hsa_signal_t* signal);

// Mirrors Amd Extension Apis
uint32_t hsa_amd_signal_wait_all(uint32_t signal_count, hsa_signal_t* signals,
                                 hsa_signal_condition_t* conds, hsa_signal_value_t* values,
                                 uint64_t timeout_hint, hsa_wait_state_t wait_hint,
                                 hsa_signal_value_t* satisfying_values);

// Mirrors Amd Extension Apis
uint32_t
    hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* signals,
                            hsa_signal_condition_t* conds,
                            hsa_signal_value_t* values, uint64_t timeout_hint,
                            hsa_wait_state_t wait_hint,
                            hsa_signal_value_t* satisfying_value);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue,
                                               uint32_t num_cu_mask_count,
                                               const uint32_t* cu_mask);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_cu_get_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count,
                                               uint32_t* cu_mask);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                 hsa_amd_memory_pool_info_t attribute,
                                 void* value);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    void* data);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                 uint32_t flags, void** ptr);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_pool_free(void* ptr);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src,
                              hsa_agent_t src_agent, size_t size,
                              uint32_t num_dep_signals,
                              const hsa_signal_t* dep_signals,
                              hsa_signal_t completion_signal);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_async_copy_on_engine(void* dst, hsa_agent_t dst_agent, const void* src,
                              hsa_agent_t src_agent, size_t size,
                              uint32_t num_dep_signals,
                              const hsa_signal_t* dep_signals,
                              hsa_signal_t completion_signal,
                              hsa_amd_sdma_engine_id_t engine_id,
                              bool force_copy_on_sdma);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                      uint32_t *engine_ids_mask);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                             uint32_t* recommended_ids_mask);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_agent_memory_pool_get_info(
    hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool,
    hsa_amd_agent_memory_pool_info_t attribute, void* value);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents,
                                const uint32_t* flags, const void* ptr);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src_memory_pool,
                                    hsa_amd_memory_pool_t dst_memory_pool,
                                    bool* result);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_migrate(const void* ptr,
                                            hsa_amd_memory_pool_t memory_pool,
                                            uint32_t flags);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_lock(void* host_ptr, size_t size,
                                         hsa_agent_t* agents, int num_agent,
                                         void** agent_ptr);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents,
                                                 int num_agent, hsa_amd_memory_pool_t pool,
                                                 uint32_t flags, void** agent_ptr);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_unlock(void* host_ptr);

// Mirrors Amd Extension Apis
hsa_status_t
    hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_interop_map_buffer(uint32_t num_agents,
                                        hsa_agent_t* agents,
                                        int interop_handle,
                                        uint32_t flags,
                                        size_t* size,
                                        void** ptr,
                                        size_t* metadata_size,
                                        const void** metadata);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_interop_unmap_buffer(void* ptr);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_pointer_info(const void* ptr, hsa_amd_pointer_info_t* info,
                                          void* (*alloc)(size_t), uint32_t* num_agents_accessible,
                                          hsa_agent_t** accessible);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_pointer_info_set_userdata(const void* ptr, void* userdata);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* handle, size_t len,
                                               uint32_t num_agents,
                                               const hsa_agent_t* mapping_agents,
                                               void** mapped_ptr);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_detach(void* mapped_ptr);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_signal_create(hsa_signal_t signal, hsa_amd_ipc_signal_t* handle);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_signal_attach(const hsa_amd_ipc_signal_t* handle,
                                               hsa_signal_t* signal);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback,
                                                           void* data);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_set_priority(hsa_queue_t* queue,
                                                hsa_amd_queue_priority_t priority);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_register_deallocation_callback(
    void* ptr, hsa_amd_deallocation_callback_t callback, void* user_data);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_deregister_deallocation_callback(
    void* ptr, hsa_amd_deallocation_callback_t callback);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_signal_value_pointer(hsa_signal_t signal,
                                          volatile hsa_signal_value_t** value_ptr);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_svm_attributes_set(void* ptr, size_t size,
                                        hsa_amd_svm_attribute_pair_t* attribute_list,
                                        size_t attribute_count);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_svm_attributes_get(void* ptr, size_t size,
                                        hsa_amd_svm_attribute_pair_t* attribute_list,
                                        size_t attribute_count);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_svm_prefetch_async(void* ptr, size_t size, hsa_agent_t agent,
                                        uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                        hsa_signal_t completion_signal);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_spm_acquire(hsa_agent_t agent);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_spm_release(hsa_agent_t agent);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_spm_set_dest_buffer(hsa_agent_t agent, size_t size, uint32_t* timeout,
                                                 uint32_t* size_copied, void* dest,
                                                 bool* is_data_loss);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf,
                                                    uint64_t* offset);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_portable_close_dmabuf(int dmabuf);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_address_reserve(void** ptr, size_t size, uint64_t address,
                                          uint64_t flags);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_address_reserve_align(void** ptr, size_t size, uint64_t address,
                                          uint64_t alignment, uint64_t flags);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_address_free(void* ptr, size_t size);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
                                        hsa_amd_memory_type_t type, uint64_t flags,
                                        hsa_amd_vmem_alloc_handle_t* memory_handle);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t memory_handle);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                              hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_unmap(void* va, size_t size);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_set_access(void* va, size_t size,
                                     const hsa_amd_memory_access_desc_t* desc,
                                     const size_t desc_cnt);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_get_access(void* va, hsa_access_permission_t* flags,
                                     const hsa_agent_t agent_handle);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_export_shareable_handle(int* dmabuf_fd,
                                                  hsa_amd_vmem_alloc_handle_t handle,
                                                  uint64_t flags);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_import_shareable_handle(int dmabuf_fd,
                                                  hsa_amd_vmem_alloc_handle_t* handle);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_retain_alloc_handle(hsa_amd_vmem_alloc_handle_t* allocHandle, void* addr);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_vmem_get_alloc_properties_from_handle(hsa_amd_vmem_alloc_handle_t allocHandle,
                                                           hsa_amd_memory_pool_t* pool,
                                                           hsa_amd_memory_type_t* type);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold);

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_get_info(hsa_queue_t* queue, hsa_queue_info_attribute_t attribute,
                                    void* value);

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_enable_logging(uint8_t* flags, void* file);

}  // namespace amd
}  // namespace rocr

#endif  // header guard
