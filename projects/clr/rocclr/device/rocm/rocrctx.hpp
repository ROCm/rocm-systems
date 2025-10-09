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

#pragma once

#include <mutex>
#include "top.hpp"

#ifdef ROCR_DYN_DLL
#include "hsa.h"
#include "hsa_ext_image.h"
#include "hsa_ext_amd.h"
#include "amd_hsa_signal.h"
#include "hsa_ven_amd_loader.h"
#include "hsa_ven_amd_aqlprofile.h"
#else
#include "hsa/hsa.h"
#include "hsa/hsa_ext_image.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/amd_hsa_signal.h"
#include "hsa/hsa_ven_amd_loader.h"
#include "hsa/hsa_ven_amd_aqlprofile.h"
#endif

namespace amd {
namespace roc {

struct RocrEntryPoints {
  void* handle;

  // Core functionality
  decltype(hsa_init)* hsa_init_;
  decltype(hsa_shut_down)* hsa_shut_down_;
  decltype(hsa_system_get_info)* hsa_system_get_info_;
  decltype(hsa_iterate_agents)* hsa_iterate_agents_;
  decltype(hsa_agent_get_info)* hsa_agent_get_info_;
  decltype(hsa_queue_create)* hsa_queue_create_;
  decltype(hsa_queue_destroy)* hsa_queue_destroy_;
  decltype(hsa_queue_load_read_index_scacquire)* hsa_queue_load_read_index_scacquire_;
  decltype(hsa_queue_load_read_index_relaxed)* hsa_queue_load_read_index_relaxed_;
  decltype(hsa_queue_load_write_index_relaxed)* hsa_queue_load_write_index_relaxed_;
  decltype(hsa_queue_add_write_index_screlease)* hsa_queue_add_write_index_screlease_;
  decltype(hsa_memory_register)* hsa_memory_register_;
  decltype(hsa_memory_deregister)* hsa_memory_deregister_;
  decltype(hsa_memory_copy)* hsa_memory_copy_;
  decltype(hsa_signal_create)* hsa_signal_create_;
  decltype(hsa_signal_destroy)* hsa_signal_destroy_;
  decltype(hsa_signal_load_relaxed)* hsa_signal_load_relaxed_;
  decltype(hsa_signal_store_relaxed)* hsa_signal_store_relaxed_;
  decltype(hsa_signal_silent_store_relaxed)* hsa_signal_silent_store_relaxed_;
  decltype(hsa_signal_store_screlease)* hsa_signal_store_screlease_;
  decltype(hsa_signal_wait_scacquire)* hsa_signal_wait_scacquire_;
  decltype(hsa_signal_add_relaxed)* hsa_signal_add_relaxed_;
  decltype(hsa_signal_subtract_relaxed)* hsa_signal_subtract_relaxed_;
  decltype(hsa_isa_get_info_alt)* hsa_isa_get_info_alt_;
  decltype(hsa_agent_iterate_isas)* hsa_agent_iterate_isas_;
  decltype(hsa_system_get_major_extension_table)* hsa_system_get_major_extension_table_;
  decltype(hsa_status_string)* hsa_status_string_;
  decltype(hsa_executable_create_alt)* hsa_executable_create_alt_;
  decltype(hsa_executable_destroy)* hsa_executable_destroy_;
  decltype(hsa_executable_get_info)* hsa_executable_get_info_;
  decltype(hsa_code_object_reader_destroy)* hsa_code_object_reader_destroy_;
  decltype(hsa_code_object_reader_create_from_memory)* hsa_code_object_reader_create_from_memory_;
  decltype(hsa_executable_load_agent_code_object)* hsa_executable_load_agent_code_object_;
  decltype(hsa_executable_agent_global_variable_define)*
      hsa_executable_agent_global_variable_define_;
  decltype(hsa_executable_get_symbol_by_name)* hsa_executable_get_symbol_by_name_;
  decltype(hsa_executable_symbol_get_info)* hsa_executable_symbol_get_info_;
  decltype(hsa_executable_freeze)* hsa_executable_freeze_;
  // AMD extensions
  decltype(hsa_amd_coherency_set_type)* hsa_amd_coherency_set_type_;
  decltype(hsa_amd_profiling_set_profiler_enabled)* hsa_amd_profiling_set_profiler_enabled_;
  decltype(hsa_amd_profiling_async_copy_enable)* hsa_amd_profiling_async_copy_enable_;
  decltype(hsa_amd_profiling_get_dispatch_time)* hsa_amd_profiling_get_dispatch_time_;
  decltype(hsa_amd_profiling_get_async_copy_time)* hsa_amd_profiling_get_async_copy_time_;
  decltype(hsa_amd_signal_async_handler)* hsa_amd_signal_async_handler_;
  decltype(hsa_amd_queue_cu_set_mask)* hsa_amd_queue_cu_set_mask_;
  decltype(hsa_amd_memory_pool_get_info)* hsa_amd_memory_pool_get_info_;
  decltype(hsa_amd_agent_iterate_memory_pools)* hsa_amd_agent_iterate_memory_pools_;
  decltype(hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate_;
  decltype(hsa_amd_memory_pool_free)* hsa_amd_memory_pool_free_;
  decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_;
  decltype(hsa_amd_memory_async_copy_on_engine)* hsa_amd_memory_async_copy_on_engine_;
  decltype(hsa_amd_memory_copy_engine_status)* hsa_amd_memory_copy_engine_status_;
  decltype(hsa_amd_agent_memory_pool_get_info)* hsa_amd_agent_memory_pool_get_info_;
  decltype(hsa_amd_agents_allow_access)* hsa_amd_agents_allow_access_;
  decltype(hsa_amd_memory_unlock)* hsa_amd_memory_unlock_;
  decltype(hsa_amd_interop_map_buffer)* hsa_amd_interop_map_buffer_;
  decltype(hsa_amd_interop_unmap_buffer)* hsa_amd_interop_unmap_buffer_;
  decltype(hsa_amd_image_create)* hsa_amd_image_create_;
  decltype(hsa_amd_pointer_info)* hsa_amd_pointer_info_;
  decltype(hsa_amd_ipc_memory_create)* hsa_amd_ipc_memory_create_;
  decltype(hsa_amd_ipc_memory_attach)* hsa_amd_ipc_memory_attach_;
  decltype(hsa_amd_ipc_memory_detach)* hsa_amd_ipc_memory_detach_;
  decltype(hsa_amd_signal_create)* hsa_amd_signal_create_;
  decltype(hsa_amd_register_system_event_handler)* hsa_amd_register_system_event_handler_;
  decltype(hsa_amd_queue_set_priority)* hsa_amd_queue_set_priority_;
  decltype(hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_;
  decltype(hsa_amd_memory_lock_to_pool)* hsa_amd_memory_lock_to_pool_;
  decltype(hsa_amd_signal_value_pointer)* hsa_amd_signal_value_pointer_;
  decltype(hsa_amd_svm_attributes_set)* hsa_amd_svm_attributes_set_;
  decltype(hsa_amd_svm_attributes_get)* hsa_amd_svm_attributes_get_;
  decltype(hsa_amd_svm_prefetch_async)* hsa_amd_svm_prefetch_async_;
  decltype(hsa_amd_portable_export_dmabuf)* hsa_amd_portable_export_dmabuf_;
  decltype(hsa_amd_portable_close_dmabuf)* hsa_amd_portable_close_dmabuf_;  // CLR doesn't use it?
  decltype(hsa_amd_vmem_address_reserve)* hsa_amd_vmem_address_reserve_;
  decltype(hsa_amd_vmem_address_free)* hsa_amd_vmem_address_free_;
  decltype(hsa_amd_vmem_handle_create)* hsa_amd_vmem_handle_create_;
  decltype(hsa_amd_vmem_handle_release)* hsa_amd_vmem_handle_release_;
  decltype(hsa_amd_vmem_map)* hsa_amd_vmem_map_;
  decltype(hsa_amd_vmem_unmap)* hsa_amd_vmem_unmap_;
  decltype(hsa_amd_vmem_set_access)* hsa_amd_vmem_set_access_;
  decltype(hsa_amd_vmem_get_access)* hsa_amd_vmem_get_access_;
  decltype(hsa_amd_vmem_export_shareable_handle)* hsa_amd_vmem_export_shareable_handle_;
  decltype(hsa_amd_vmem_import_shareable_handle)* hsa_amd_vmem_import_shareable_handle_;
  decltype(hsa_amd_vmem_retain_alloc_handle)* hsa_amd_vmem_retain_alloc_handle_;
  decltype(hsa_amd_agent_set_async_scratch_limit)* hsa_amd_agent_set_async_scratch_limit_;
  decltype(hsa_amd_vmem_address_reserve_align)* hsa_amd_vmem_address_reserve_align_;
  decltype(hsa_amd_enable_logging)* hsa_amd_enable_logging_;
  decltype(hsa_amd_memory_get_preferred_copy_engine)* hsa_amd_memory_get_preferred_copy_engine_;
  decltype(hsa_amd_ais_file_read)* hsa_amd_ais_file_read_;
  decltype(hsa_amd_ais_file_write)* hsa_amd_ais_file_write_;
  // Image extensions
  decltype(hsa_ext_image_data_get_info)* hsa_ext_image_data_get_info_;
  decltype(hsa_ext_image_create)* hsa_ext_image_create_;
  decltype(hsa_ext_image_import)* hsa_ext_image_import_;
  decltype(hsa_ext_image_export)* hsa_ext_image_export_;
  decltype(hsa_ext_image_destroy)* hsa_ext_image_destroy_;
  decltype(hsa_ext_sampler_create_v2)* hsa_ext_sampler_create_v2_;
  decltype(hsa_ext_sampler_destroy)* hsa_ext_sampler_destroy_;
  decltype(hsa_ext_image_create_with_layout)* hsa_ext_image_create_with_layout_;
};

#ifdef ROCR_DYN_DLL
#define ROCR_DYN(NAME) cep_.NAME##_
#define GET_ROCR_SYMBOL(NAME)                                                                     \
  cep_.NAME##_ = reinterpret_cast<decltype(NAME)*>(Os::getSymbol(cep_.handle, #NAME));             \
  if (nullptr == cep_.NAME##_) {                                                                  \
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Failed to load ROCR function %s", #NAME);            \
    return false;                                                                                 \
  }
#define GET_ROCR_OPTIONAL_SYMBOL(NAME)                                                            \
  cep_.NAME = reinterpret_cast<t_##NAME>(Os::getSymbol(cep_.handle, #NAME));
#else
#define ROCR_DYN(NAME) NAME
#define GET_ROCR_SYMBOL(NAME)
#define GET_ROCR_OPTIONAL_SYMBOL(NAME)
#endif

class Hsa : public amd::AllStatic {
 public:
  static std::once_flag initialized;

  static bool LoadLib();

  static bool IsReady() { return is_ready_; }

  static hsa_status_t init() { return ROCR_DYN(hsa_init)(); }
  static hsa_status_t shut_down() { return ROCR_DYN(hsa_shut_down)(); }
  static hsa_status_t system_get_info(hsa_system_info_t attribute, void* value) {
    return ROCR_DYN(hsa_system_get_info)(attribute, value);
  }
  static hsa_status_t iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data),
                                     void* data) {
    return ROCR_DYN(hsa_iterate_agents)(callback, data);
  }
  static hsa_status_t agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value) {
    return ROCR_DYN(hsa_agent_get_info)(agent, attribute, value);
  }
  static hsa_status_t queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                   void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                    void* data),
                                   void* data, uint32_t private_segment_size,
                                   uint32_t group_segment_size, hsa_queue_t** queue) {
    return ROCR_DYN(hsa_queue_create)(agent, size, type, callback, data, private_segment_size,
                                      group_segment_size, queue);
  }
  static hsa_status_t queue_destroy(hsa_queue_t* queue) {
    return ROCR_DYN(hsa_queue_destroy)(queue);
  }
  static uint64_t queue_load_read_index_scacquire(const hsa_queue_t* queue) {
    return ROCR_DYN(hsa_queue_load_read_index_scacquire)(queue);
  }
  static uint64_t queue_load_read_index_relaxed(const hsa_queue_t* queue) {
    return ROCR_DYN(hsa_queue_load_read_index_relaxed)(queue);
  }
  static uint64_t queue_load_write_index_relaxed(const hsa_queue_t* queue) {
    return ROCR_DYN(hsa_queue_load_write_index_relaxed)(queue);
  }
  static uint64_t queue_add_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
    return ROCR_DYN(hsa_queue_add_write_index_screlease)(queue, value);
  }
  static hsa_status_t memory_register(void* ptr, size_t size) {
    return ROCR_DYN(hsa_memory_register)(ptr, size);
  }
  static hsa_status_t memory_deregister(void* ptr, size_t size) {
    return ROCR_DYN(hsa_memory_deregister)(ptr, size);
  }
  static hsa_status_t memory_copy(void* dst, const void* src, size_t size) {
    return ROCR_DYN(hsa_memory_copy)(dst, src, size);
  }
  static hsa_status_t signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                                    const hsa_agent_t* consumers, hsa_signal_t* signal) {
    return ROCR_DYN(hsa_signal_create)(initial_value, num_consumers, consumers, signal);
  }
  static hsa_status_t signal_destroy(hsa_signal_t signal) {
    return ROCR_DYN(hsa_signal_destroy)(signal);
  }
  static hsa_signal_value_t signal_load_relaxed(hsa_signal_t signal) {
    return ROCR_DYN(hsa_signal_load_relaxed)(signal);
  }
  static void signal_silent_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    ROCR_DYN(hsa_signal_silent_store_relaxed)(signal, value);
  }
  static void signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    ROCR_DYN(hsa_signal_store_relaxed)(signal, value);
  }
  static void signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    ROCR_DYN(hsa_signal_store_screlease)(signal, value);
  }
  static hsa_signal_value_t signal_wait_scacquire(hsa_signal_t signal,
                                                  hsa_signal_condition_t condition,
                                                  hsa_signal_value_t compare_value,
                                                  uint64_t timeout_hint,
                                                  hsa_wait_state_t wait_state_hint) {
    return ROCR_DYN(hsa_signal_wait_scacquire)(signal, condition, compare_value, timeout_hint,
                                               wait_state_hint);
  }
  static void signal_add_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    ROCR_DYN(hsa_signal_add_relaxed)(signal, value);
  }
  static void signal_subtract_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    ROCR_DYN(hsa_signal_subtract_relaxed)(signal, value);
  }
  static hsa_status_t isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void* value) {
    return ROCR_DYN(hsa_isa_get_info_alt)(isa, attribute, value);
  }
  static hsa_status_t agent_iterate_isas(hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_isa_t isa, void* data), void* data) {
    return ROCR_DYN(hsa_agent_iterate_isas)(agent, callback, data);
  }
  static hsa_status_t system_get_major_extension_table(uint16_t extension, uint16_t version_major,
    size_t table_length, void* table) {
    return ROCR_DYN(hsa_system_get_major_extension_table)(extension, version_major,
                                                          table_length, table);
  }
  static hsa_status_t status_string(hsa_status_t status, const char** status_string) {
    return ROCR_DYN(hsa_status_string)(status, status_string);
  }
  static hsa_status_t executable_create_alt(
      hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode,
      const char* options, hsa_executable_t* executable) {
    return ROCR_DYN(hsa_executable_create_alt)(profile, default_float_rounding_mode, options,
                                               executable);
  }
  static hsa_status_t executable_destroy(hsa_executable_t executable) {
    return ROCR_DYN(hsa_executable_destroy)(executable);
  }
  static hsa_status_t executable_get_info(hsa_executable_t executable,
                                          hsa_executable_info_t attribute, void* value) {
    return ROCR_DYN(hsa_executable_get_info)(executable, attribute, value);
  }
  static hsa_status_t code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
    return ROCR_DYN(hsa_code_object_reader_destroy)(code_object_reader);
  }
  static hsa_status_t code_object_reader_create_from_memory(
      const void* code_object, size_t size, hsa_code_object_reader_t* code_object_reader) {
    return ROCR_DYN(hsa_code_object_reader_create_from_memory)(code_object, size,
                                                               code_object_reader);
  }
  static hsa_status_t executable_load_agent_code_object(
      hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
      const char* options, hsa_loaded_code_object_t* loaded_code_object) {
    return ROCR_DYN(hsa_executable_load_agent_code_object)(executable, agent, code_object_reader,
                                                           options, loaded_code_object);
  }
  static hsa_status_t executable_agent_global_variable_define(hsa_executable_t executable,
    hsa_agent_t agent, const char* variable_name, void* address) {
    return ROCR_DYN(hsa_executable_agent_global_variable_define)(executable, agent,
                                                                 variable_name, address);
  }
  static hsa_status_t executable_get_symbol_by_name(hsa_executable_t executable,
    const char* symbol_name, const hsa_agent_t* agent, hsa_executable_symbol_t* symbol) {
    return ROCR_DYN(hsa_executable_get_symbol_by_name)(executable, symbol_name, agent, symbol);
  }
  static hsa_status_t executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute, void* value) {
    return ROCR_DYN(hsa_executable_symbol_get_info)(executable_symbol, attribute, value);
  }
  static hsa_status_t executable_freeze(hsa_executable_t executable, const char* options) {
    return ROCR_DYN(hsa_executable_freeze)(executable, options);
  }
  // AMD extensions
  static hsa_status_t coherency_set_type(hsa_agent_t agent, hsa_amd_coherency_type_t type) {
    return ROCR_DYN(hsa_amd_coherency_set_type)(agent, type);
  }
  static hsa_status_t profiling_set_profiler_enabled(hsa_queue_t* queue, int enable) {
    return ROCR_DYN(hsa_amd_profiling_set_profiler_enabled)(queue, enable);
  }
  static hsa_status_t profiling_async_copy_enable(bool enable) {
    return ROCR_DYN(hsa_amd_profiling_async_copy_enable)(enable);
  }
  static hsa_status_t profiling_get_dispatch_time(hsa_agent_t agent, hsa_signal_t signal,
                                                  hsa_amd_profiling_dispatch_time_t* time) {
    return ROCR_DYN(hsa_amd_profiling_get_dispatch_time)(agent, signal, time);
  }
  static hsa_status_t profiling_get_async_copy_time(hsa_signal_t signal,
                                                    hsa_amd_profiling_async_copy_time_t* time) {
    return ROCR_DYN(hsa_amd_profiling_get_async_copy_time)(signal, time);
  }
  static hsa_status_t signal_async_handler(hsa_signal_t signal, hsa_signal_condition_t cond,
                                           hsa_signal_value_t value, hsa_amd_signal_handler handler,
                                           void* arg) {
    return ROCR_DYN(hsa_amd_signal_async_handler)(signal, cond, value, handler, arg);
  }
  static hsa_status_t queue_cu_set_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count,
                                        const uint32_t* cu_mask) {
    return ROCR_DYN(hsa_amd_queue_cu_set_mask)(queue, num_cu_mask_count, cu_mask);
  }
  static hsa_status_t memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                           hsa_amd_memory_pool_info_t attribute, void* value) {
    return ROCR_DYN(hsa_amd_memory_pool_get_info)(memory_pool, attribute, value);
  }
  static hsa_status_t agent_iterate_memory_pools(
      hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
      void* data) {
    return ROCR_DYN(hsa_amd_agent_iterate_memory_pools)(agent, callback, data);
  }
  static hsa_status_t memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                           uint32_t flags, void** ptr) {
    return ROCR_DYN(hsa_amd_memory_pool_allocate)(memory_pool, size, flags, ptr);
  }
  static hsa_status_t memory_pool_free(void* ptr) {
    return ROCR_DYN(hsa_amd_memory_pool_free)(ptr);
  }
  static hsa_status_t memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src,
                                        hsa_agent_t src_agent, size_t size,
                                        uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                        hsa_signal_t completion_signal) {
    return ROCR_DYN(hsa_amd_memory_async_copy)(dst, dst_agent, src, src_agent, size,
                                               num_dep_signals, dep_signals, completion_signal);
  }
  static hsa_status_t memory_async_copy_on_engine(
      void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size,
      uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal,
      hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
    return ROCR_DYN(hsa_amd_memory_async_copy_on_engine)(
        dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal,
        engine_id, force_copy_on_sdma);
  }
  static hsa_status_t memory_copy_engine_status(hsa_agent_t dst_agent,
    hsa_agent_t src_agent, uint32_t* engine_ids_mask) {
    return ROCR_DYN(hsa_amd_memory_copy_engine_status)(dst_agent, src_agent, engine_ids_mask);
  }
  static hsa_status_t agent_memory_pool_get_info(hsa_agent_t agent,
    hsa_amd_memory_pool_t memory_pool, hsa_amd_agent_memory_pool_info_t attribute, void* value) {
    return ROCR_DYN(hsa_amd_agent_memory_pool_get_info)(agent, memory_pool, attribute, value);
  }
  static hsa_status_t agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents,
    const uint32_t* flags, const void* ptr) {
    return ROCR_DYN(hsa_amd_agents_allow_access)(num_agents, agents, flags, ptr);
  }
  static hsa_status_t memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents,
    int num_agent, hsa_amd_memory_pool_t pool, uint32_t flags, void** agent_ptr) {
    return ROCR_DYN(hsa_amd_memory_lock_to_pool)(host_ptr, size, agents, num_agent, pool, flags,
                                                 agent_ptr);
  }
  static hsa_status_t memory_unlock(void* host_ptr) {
    return ROCR_DYN(hsa_amd_memory_unlock)(host_ptr);
  }
  static hsa_status_t interop_map_buffer(uint32_t num_agents, hsa_agent_t* agents,
    int interop_handle, uint32_t flags, size_t* size,
    void** ptr, size_t* metadata_size, const void** metadata) {
    return ROCR_DYN(hsa_amd_interop_map_buffer)(num_agents, agents, interop_handle, flags, size,
                                                ptr, metadata_size, metadata);
  }
  static hsa_status_t interop_unmap_buffer(void* ptr) {
    return ROCR_DYN(hsa_amd_interop_unmap_buffer)(ptr);
  }
  static hsa_status_t pointer_info(const void* ptr, hsa_amd_pointer_info_t* info,
    void* (*alloc)(size_t), uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
    return ROCR_DYN(hsa_amd_pointer_info)(ptr, info, alloc, num_agents_accessible, accessible);
  }
  static hsa_status_t ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
    return ROCR_DYN(hsa_amd_ipc_memory_create)(ptr, len, handle);
  }
  static hsa_status_t ipc_memory_attach(const hsa_amd_ipc_memory_t* handle, size_t len,
    uint32_t num_agents, const hsa_agent_t* mapping_agents, void** mapped_ptr) {
    return ROCR_DYN(hsa_amd_ipc_memory_attach)(
      handle, len, num_agents, mapping_agents, mapped_ptr);
  }
  static hsa_status_t ipc_memory_detach(void* mapped_ptr) {
    return ROCR_DYN(hsa_amd_ipc_memory_detach)(mapped_ptr);
  }
  static hsa_status_t signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
    const hsa_agent_t* consumers, uint64_t attributes, hsa_signal_t* signal) {
    return ROCR_DYN(hsa_amd_signal_create)(initial_value, num_consumers, consumers, attributes,
                                           signal);
  }
  static hsa_status_t register_system_event_handler(hsa_amd_system_event_callback_t callback,
    void* data) {
    return ROCR_DYN(hsa_amd_register_system_event_handler)(callback, data);
  }
  static hsa_status_t queue_set_priority(hsa_queue_t* queue, hsa_amd_queue_priority_t priority) {
    return ROCR_DYN(hsa_amd_queue_set_priority)(queue, priority);
  }
  static hsa_status_t memory_async_copy_rect(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal) {
    return ROCR_DYN(hsa_amd_memory_async_copy_rect)(dst, dst_offset, src, src_offset, range,
                                                    copy_agent, dir, num_dep_signals, dep_signals,
                                                    completion_signal);
  }
  static hsa_status_t signal_value_pointer(hsa_signal_t signal,
    volatile hsa_signal_value_t** value_ptr) {
    return ROCR_DYN(hsa_amd_signal_value_pointer)(signal, value_ptr);
  }
  static hsa_status_t svm_attributes_set(void* ptr, size_t size,
    hsa_amd_svm_attribute_pair_t* attribute_list, size_t attribute_count) {
    return ROCR_DYN(hsa_amd_svm_attributes_set)(ptr, size, attribute_list, attribute_count);
  }
  static hsa_status_t svm_attributes_get(void* ptr, size_t size,
    hsa_amd_svm_attribute_pair_t* attribute_list, size_t attribute_count) {
    return ROCR_DYN(hsa_amd_svm_attributes_get)(ptr, size, attribute_list, attribute_count);
  }
  static hsa_status_t svm_prefetch_async(void* ptr, size_t size, hsa_agent_t agent,
    uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
    return ROCR_DYN(hsa_amd_svm_prefetch_async)(ptr, size, agent, num_dep_signals,
        dep_signals, completion_signal);
  }
  static hsa_status_t portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf,
    uint64_t* offset) {
    return ROCR_DYN(hsa_amd_portable_export_dmabuf)(ptr, size, dmabuf, offset);
  }
  static hsa_status_t vmem_address_reserve(void** ptr, size_t size, uint64_t address,
    uint64_t flags) {
    return ROCR_DYN(hsa_amd_vmem_address_reserve)(ptr, size, address, flags);
  }
  static hsa_status_t vmem_address_free(void* ptr, size_t size) {
    return ROCR_DYN(hsa_amd_vmem_address_free)(ptr, size);
  }
  static hsa_status_t vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
    hsa_amd_memory_type_t type, uint64_t flags, hsa_amd_vmem_alloc_handle_t* memory_handle) {
    return ROCR_DYN(hsa_amd_vmem_handle_create)(pool, size, type, flags, memory_handle);
  }
  static hsa_status_t vmem_handle_release(hsa_amd_vmem_alloc_handle_t memory_handle) {
    return ROCR_DYN(hsa_amd_vmem_handle_release)(memory_handle);
  }
  static hsa_status_t vmem_map(void* va, size_t size, size_t in_offset,
    hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags) {
    return ROCR_DYN(hsa_amd_vmem_map)(va, size, in_offset, memory_handle, flags);
  }
  static hsa_status_t vmem_unmap(void* va, size_t size) {
    return ROCR_DYN(hsa_amd_vmem_unmap)(va, size);
  }
  static hsa_status_t vmem_set_access(void* va, size_t size,
    const hsa_amd_memory_access_desc_t* desc, const size_t desc_cnt) {
    return ROCR_DYN(hsa_amd_vmem_set_access)(va, size, desc, desc_cnt);
  }
  static hsa_status_t vmem_get_access(void* va, hsa_access_permission_t* flags,
    const hsa_agent_t agent_handle) {
    return ROCR_DYN(hsa_amd_vmem_get_access)(va, flags, agent_handle);
  }
  static hsa_status_t vmem_export_shareable_handle(int* dmabuf_fd,
    hsa_amd_vmem_alloc_handle_t handle, uint64_t flags) {
    return ROCR_DYN(hsa_amd_vmem_export_shareable_handle)(dmabuf_fd, handle, flags);
  }
  static hsa_status_t vmem_import_shareable_handle(int dmabuf_fd,
    hsa_amd_vmem_alloc_handle_t* handle) {
    return ROCR_DYN(hsa_amd_vmem_import_shareable_handle)(dmabuf_fd, handle);
  }
  static hsa_status_t vmem_retain_alloc_handle(hsa_amd_vmem_alloc_handle_t* allocHandle,
    void* addr) {
    return ROCR_DYN(hsa_amd_vmem_retain_alloc_handle)(allocHandle, addr);
  }
  static hsa_status_t agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold) {
    return ROCR_DYN(hsa_amd_agent_set_async_scratch_limit)(agent, threshold);
  }
  static hsa_status_t vmem_address_reserve_align(void** ptr, size_t size, uint64_t address,
    uint64_t alignment, uint64_t flags) {
    return ROCR_DYN(hsa_amd_vmem_address_reserve_align)(ptr, size, address, alignment, flags);
  }
  static hsa_status_t enable_logging(uint8_t* flags, void* file) {
    return ROCR_DYN(hsa_amd_enable_logging)(flags, file);
  }
  static hsa_status_t memory_get_preferred_copy_engine(hsa_agent_t dst_agent,
    hsa_agent_t src_agent, uint32_t* recommended_ids_mask) {
    return ROCR_DYN(hsa_amd_memory_get_preferred_copy_engine)(
      dst_agent, src_agent, recommended_ids_mask);
  }
  static hsa_status_t ais_file_read(hsa_amd_ais_file_handle_t handle, void* devicePtr,
                                    uint64_t size, int64_t file_offset, uint64_t* size_copied,
                                    int32_t* status) {
    return ROCR_DYN(hsa_amd_ais_file_read)(handle, devicePtr, size, file_offset, size_copied,
                                           status);
  }
  static hsa_status_t ais_file_write(hsa_amd_ais_file_handle_t handle, void* devicePtr,
                                     uint64_t size, int64_t file_offset, uint64_t* size_copied,
                                     int32_t* status) {
    return ROCR_DYN(hsa_amd_ais_file_write)(handle, devicePtr, size, file_offset, size_copied,
                                            status);
  }

  // Image extensions
  static hsa_status_t image_create(hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* image_descriptor,
    const hsa_amd_image_descriptor_t* image_layout,
    const void* image_data, hsa_access_permission_t access_permission,
    hsa_ext_image_t* image) {
    return ROCR_DYN(hsa_amd_image_create)(agent, image_descriptor, image_layout, image_data,
                                          access_permission, image);
  }
  static hsa_status_t image_data_get_info(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor,
    hsa_access_permission_t access_permission, hsa_ext_image_data_info_t* image_data_info) {
    return ROCR_DYN(hsa_ext_image_data_get_info)(agent, image_descriptor, access_permission,
                                                 image_data_info);
  }
  static hsa_status_t image_create(hsa_agent_t agent,
    const hsa_ext_image_descriptor_t* image_descriptor,
    const void* image_data,
    hsa_access_permission_t access_permission,
    hsa_ext_image_t* image) {
    return ROCR_DYN(hsa_ext_image_create)(agent, image_descriptor, image_data,
                                                 access_permission, image);
  }
  static hsa_status_t image_import(hsa_agent_t agent, const void* src_memory,
    size_t src_row_pitch, size_t src_slice_pitch,
    hsa_ext_image_t dst_image,
    const hsa_ext_image_region_t* image_region) {
    return ROCR_DYN(hsa_ext_image_import)(agent, src_memory, src_row_pitch, src_slice_pitch,
                                          dst_image, image_region);
  }
  static hsa_status_t image_export(hsa_agent_t agent, hsa_ext_image_t src_image,
    void* dst_memory, size_t dst_row_pitch,
    size_t dst_slice_pitch,
    const hsa_ext_image_region_t* image_region) {
    return ROCR_DYN(hsa_ext_image_export)(agent, src_image, dst_memory, dst_row_pitch,
                                          dst_slice_pitch, image_region);
  }
  static hsa_status_t image_destroy(hsa_agent_t agent, hsa_ext_image_t image) {
    return ROCR_DYN(hsa_ext_image_destroy)(agent, image);
  }
  static hsa_status_t sampler_create(hsa_agent_t agent,
    const hsa_ext_sampler_descriptor_v2_t* sampler_descriptor, hsa_ext_sampler_t* sampler) {
    return ROCR_DYN(hsa_ext_sampler_create_v2)(agent, sampler_descriptor, sampler);
  }
  static hsa_status_t sampler_destroy(hsa_agent_t agent, hsa_ext_sampler_t sampler) {
    return ROCR_DYN(hsa_ext_sampler_destroy)(agent, sampler);
  }
  static hsa_status_t image_create_with_layout(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor, const void* image_data,
    hsa_access_permission_t access_permission, hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch, size_t image_data_slice_pitch, hsa_ext_image_t* image) {
    return ROCR_DYN(hsa_ext_image_create_with_layout)(
        agent, image_descriptor, image_data, access_permission, image_data_layout,
        image_data_row_pitch, image_data_slice_pitch, image);
  }

 private:
  static RocrEntryPoints cep_;
  static bool is_ready_;
};

}  // namespace roc
}  // namespace amd
