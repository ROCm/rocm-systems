////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_HSA_INTERNAL_H
#define HSA_RUNTIME_CORE_INC_HSA_INTERNAL_H

#include "inc/hsa.h"

namespace rocr {
namespace HSA {

  // Define core namespace interfaces - copy of function declarations in hsa.h
  hsa_status_t hsa_init();
  hsa_status_t hsa_shut_down();
  hsa_status_t
    hsa_system_get_info(hsa_system_info_t attribute, void *value);
  hsa_status_t hsa_extension_get_name(uint16_t extension, const char** name);
  hsa_status_t hsa_system_extension_supported(uint16_t extension, uint16_t version_major,
                                                      uint16_t version_minor, bool* result);
  hsa_status_t hsa_system_major_extension_supported(uint16_t extension,
                                                            uint16_t version_major,
                                                            uint16_t* version_minor, bool* result);
  hsa_status_t
    hsa_system_get_extension_table(uint16_t extension, uint16_t version_major,
    uint16_t version_minor, void *table);
  hsa_status_t hsa_system_get_major_extension_table(uint16_t extension,
                                                            uint16_t version_major,
                                                            size_t table_length, void* table);
  hsa_status_t
    hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void *data),
    void *data);
  hsa_status_t hsa_agent_get_info(hsa_agent_t agent,
    hsa_agent_info_t attribute,
    void *value);
  hsa_status_t hsa_agent_get_exception_policies(hsa_agent_t agent,
    hsa_profile_t profile,
    uint16_t *mask);
  hsa_status_t hsa_cache_get_info(hsa_cache_t cache, hsa_cache_info_t attribute,
                                          void* value);
  hsa_status_t hsa_agent_iterate_caches(
      hsa_agent_t agent, hsa_status_t (*callback)(hsa_cache_t cache, void* data), void* value);
  hsa_status_t
    hsa_agent_extension_supported(uint16_t extension, hsa_agent_t agent,
    uint16_t version_major,
    uint16_t version_minor, bool *result);
  hsa_status_t hsa_agent_major_extension_supported(uint16_t extension, hsa_agent_t agent,
                                                           uint16_t version_major,
                                                           uint16_t* version_minor, bool* result);
  hsa_status_t
    hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t *source,
    void *data),
    void *data, uint32_t private_segment_size,
    uint32_t group_segment_size, hsa_queue_t **queue);
  hsa_status_t
    hsa_soft_queue_create(hsa_region_t region, uint32_t size,
    hsa_queue_type32_t type, uint32_t features,
    hsa_signal_t completion_signal, hsa_queue_t **queue);
  hsa_status_t hsa_queue_destroy(hsa_queue_t *queue);
  hsa_status_t hsa_queue_inactivate(hsa_queue_t *queue);
  uint64_t hsa_queue_load_read_index_scacquire(const hsa_queue_t* queue);
  uint64_t hsa_queue_load_read_index_relaxed(const hsa_queue_t *queue);
  uint64_t hsa_queue_load_write_index_scacquire(const hsa_queue_t* queue);
  uint64_t hsa_queue_load_write_index_relaxed(const hsa_queue_t *queue);
  void hsa_queue_store_write_index_relaxed(const hsa_queue_t *queue,
    uint64_t value);
  void hsa_queue_store_write_index_screlease(const hsa_queue_t* queue, uint64_t value);
  uint64_t hsa_queue_cas_write_index_scacq_screl(const hsa_queue_t* queue,
                                                         uint64_t expected, uint64_t value);
  uint64_t hsa_queue_cas_write_index_scacquire(const hsa_queue_t* queue, uint64_t expected,
                                                       uint64_t value);
  uint64_t hsa_queue_cas_write_index_relaxed(const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);
  uint64_t hsa_queue_cas_write_index_screlease(const hsa_queue_t* queue, uint64_t expected,
                                                       uint64_t value);
  uint64_t hsa_queue_add_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t value);
  uint64_t hsa_queue_add_write_index_scacquire(const hsa_queue_t* queue, uint64_t value);
  uint64_t
    hsa_queue_add_write_index_relaxed(const hsa_queue_t *queue, uint64_t value);
  uint64_t hsa_queue_add_write_index_screlease(const hsa_queue_t* queue, uint64_t value);
  void hsa_queue_store_read_index_relaxed(const hsa_queue_t *queue,
    uint64_t value);
  void hsa_queue_store_read_index_screlease(const hsa_queue_t* queue, uint64_t value);
  hsa_status_t hsa_agent_iterate_regions(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_region_t region, void *data), void *data);
  hsa_status_t hsa_region_get_info(hsa_region_t region,
    hsa_region_info_t attribute,
    void *value);
  hsa_status_t hsa_memory_register(void *address, size_t size);
  hsa_status_t hsa_memory_deregister(void *address, size_t size);
  hsa_status_t
    hsa_memory_allocate(hsa_region_t region, size_t size, void **ptr);
  hsa_status_t hsa_memory_free(void *ptr);
  hsa_status_t hsa_memory_copy(void *dst, const void *src, size_t size);
  hsa_status_t hsa_memory_assign_agent(void *ptr, hsa_agent_t agent,
    hsa_access_permission_t access);
  hsa_status_t
    hsa_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
    const hsa_agent_t *consumers, hsa_signal_t *signal);
  hsa_status_t hsa_signal_destroy(hsa_signal_t signal);
  hsa_signal_value_t hsa_signal_load_relaxed(hsa_signal_t signal);
  hsa_signal_value_t hsa_signal_load_scacquire(hsa_signal_t signal);
  void
    hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_silent_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_silent_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);
  hsa_signal_value_t
    hsa_signal_wait_relaxed(hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint,
    hsa_wait_state_t wait_expectancy_hint);
  hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal,
                                                       hsa_signal_condition_t condition,
                                                       hsa_signal_value_t compare_value,
                                                       uint64_t timeout_hint,
                                                       hsa_wait_state_t wait_expectancy_hint);
  hsa_status_t hsa_signal_group_create(uint32_t num_signals, const hsa_signal_t* signals,
                                               uint32_t num_consumers, const hsa_agent_t* consumers,
                                               hsa_signal_group_t* signal_group);
  hsa_status_t hsa_signal_group_destroy(hsa_signal_group_t signal_group);
  hsa_status_t hsa_signal_group_wait_any_scacquire(hsa_signal_group_t signal_group,
                                                           const hsa_signal_condition_t* conditions,
                                                           const hsa_signal_value_t* compare_values,
                                                           hsa_wait_state_t wait_state_hint,
                                                           hsa_signal_t* signal,
                                                           hsa_signal_value_t* value);
  hsa_status_t hsa_signal_group_wait_any_relaxed(hsa_signal_group_t signal_group,
                                                         const hsa_signal_condition_t* conditions,
                                                         const hsa_signal_value_t* compare_values,
                                                         hsa_wait_state_t wait_state_hint,
                                                         hsa_signal_t* signal,
                                                         hsa_signal_value_t* value);
  void
    hsa_signal_and_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_and_scacquire(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_and_screlease(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_and_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
  void
    hsa_signal_or_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_or_scacquire(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_or_screlease(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_or_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
  void
    hsa_signal_xor_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_xor_scacquire(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_xor_screlease(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_xor_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
  void
    hsa_signal_add_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_add_scacquire(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_add_screlease(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_add_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
  void
    hsa_signal_subtract_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_subtract_scacquire(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_subtract_screlease(hsa_signal_t signal, hsa_signal_value_t value);
  void hsa_signal_subtract_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
  hsa_signal_value_t
    hsa_signal_exchange_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
  hsa_signal_value_t hsa_signal_exchange_scacquire(hsa_signal_t signal,
                                                           hsa_signal_value_t value);
  hsa_signal_value_t hsa_signal_exchange_screlease(hsa_signal_t signal,
                                                           hsa_signal_value_t value);
  hsa_signal_value_t hsa_signal_exchange_scacq_screl(hsa_signal_t signal,
                                                             hsa_signal_value_t value);
  hsa_signal_value_t hsa_signal_cas_relaxed(hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);
  hsa_signal_value_t hsa_signal_cas_scacquire(hsa_signal_t signal,
                                                      hsa_signal_value_t expected,
                                                      hsa_signal_value_t value);
  hsa_signal_value_t hsa_signal_cas_screlease(hsa_signal_t signal,
                                                      hsa_signal_value_t expected,
                                                      hsa_signal_value_t value);
  hsa_signal_value_t hsa_signal_cas_scacq_screl(hsa_signal_t signal,
                                                        hsa_signal_value_t expected,
                                                        hsa_signal_value_t value);

  //===--- Instruction Set Architecture -----------------------------------===//

  hsa_status_t hsa_isa_from_name(
      const char *name,
      hsa_isa_t *isa);
  hsa_status_t hsa_agent_iterate_isas(
      hsa_agent_t agent,
      hsa_status_t (*callback)(hsa_isa_t isa,
                               void *data),
      void *data);
  /* deprecated */ hsa_status_t hsa_isa_get_info(
      hsa_isa_t isa,
      hsa_isa_info_t attribute,
      uint32_t index,
      void *value);
  hsa_status_t hsa_isa_get_info_alt(
      hsa_isa_t isa,
      hsa_isa_info_t attribute,
      void *value);
  hsa_status_t hsa_isa_get_exception_policies(
      hsa_isa_t isa,
      hsa_profile_t profile,
      uint16_t *mask);
  hsa_status_t hsa_isa_get_round_method(
      hsa_isa_t isa,
      hsa_fp_type_t fp_type,
      hsa_flush_mode_t flush_mode,
      hsa_round_method_t *round_method);
  hsa_status_t hsa_wavefront_get_info(
      hsa_wavefront_t wavefront,
      hsa_wavefront_info_t attribute,
      void *value);
  hsa_status_t hsa_isa_iterate_wavefronts(
      hsa_isa_t isa,
      hsa_status_t (*callback)(hsa_wavefront_t wavefront,
                               void *data),
      void *data);
  /* deprecated */ hsa_status_t hsa_isa_compatible(
      hsa_isa_t code_object_isa,
      hsa_isa_t agent_isa,
      bool *result);

  //===--- Code Objects (deprecated) --------------------------------------===//

  /* deprecated */ hsa_status_t hsa_code_object_serialize(
      hsa_code_object_t code_object,
      hsa_status_t (*alloc_callback)(size_t size,
                                     hsa_callback_data_t data,
                                     void **address),
      hsa_callback_data_t callback_data,
      const char *options,
      void **serialized_code_object,
      size_t *serialized_code_object_size);
  /* deprecated */ hsa_status_t hsa_code_object_deserialize(
      void *serialized_code_object,
      size_t serialized_code_object_size,
      const char *options,
      hsa_code_object_t *code_object);
  /* deprecated */ hsa_status_t hsa_code_object_destroy(
      hsa_code_object_t code_object);
  /* deprecated */ hsa_status_t hsa_code_object_get_info(
      hsa_code_object_t code_object,
      hsa_code_object_info_t attribute,
      void *value);
  /* deprecated */ hsa_status_t hsa_code_object_get_symbol(
      hsa_code_object_t code_object,
      const char *symbol_name,
      hsa_code_symbol_t *symbol);
  /* deprecated */ hsa_status_t hsa_code_object_get_symbol_from_name(
      hsa_code_object_t code_object,
      const char *module_name,
      const char *symbol_name,
      hsa_code_symbol_t *symbol);
  /* deprecated */ hsa_status_t hsa_code_symbol_get_info(
      hsa_code_symbol_t code_symbol,
      hsa_code_symbol_info_t attribute,
      void *value);
  /* deprecated */ hsa_status_t hsa_code_object_iterate_symbols(
      hsa_code_object_t code_object,
      hsa_status_t (*callback)(hsa_code_object_t code_object,
                               hsa_code_symbol_t symbol,
                               void *data),
      void *data);

  //===--- Executable -----------------------------------------------------===//

  hsa_status_t hsa_code_object_reader_create_from_file(
      hsa_file_t file,
      hsa_code_object_reader_t *code_object_reader);
  hsa_status_t hsa_code_object_reader_create_from_memory(
      const void *code_object,
      size_t size,
      hsa_code_object_reader_t *code_object_reader);
  hsa_status_t hsa_code_object_reader_destroy(
      hsa_code_object_reader_t code_object_reader);
  /* deprecated */ hsa_status_t hsa_executable_create(
      hsa_profile_t profile,
      hsa_executable_state_t executable_state,
      const char *options,
      hsa_executable_t *executable);
  hsa_status_t hsa_executable_create_alt(
      hsa_profile_t profile,
      hsa_default_float_rounding_mode_t default_float_rounding_mode,
      const char *options,
      hsa_executable_t *executable);
  hsa_status_t hsa_executable_destroy(
      hsa_executable_t executable);
  /* deprecated */ hsa_status_t hsa_executable_load_code_object(
      hsa_executable_t executable,
      hsa_agent_t agent,
      hsa_code_object_t code_object,
      const char *options);
  hsa_status_t hsa_executable_load_program_code_object(
      hsa_executable_t executable,
      hsa_code_object_reader_t code_object_reader,
      const char *options,
      hsa_loaded_code_object_t *loaded_code_object);
  hsa_status_t hsa_executable_load_agent_code_object(
      hsa_executable_t executable,
      hsa_agent_t agent,
      hsa_code_object_reader_t code_object_reader,
      const char *options,
      hsa_loaded_code_object_t *loaded_code_object);
  hsa_status_t hsa_executable_freeze(
      hsa_executable_t executable,
      const char *options);
  hsa_status_t hsa_executable_get_info(
      hsa_executable_t executable,
      hsa_executable_info_t attribute,
      void *value);
  hsa_status_t hsa_executable_global_variable_define(
      hsa_executable_t executable,
      const char *variable_name,
      void *address);
  hsa_status_t hsa_executable_agent_global_variable_define(
      hsa_executable_t executable,
      hsa_agent_t agent,
      const char *variable_name,
      void *address);
  hsa_status_t hsa_executable_readonly_variable_define(
      hsa_executable_t executable,
      hsa_agent_t agent,
      const char *variable_name,
      void *address);
  hsa_status_t hsa_executable_validate(
      hsa_executable_t executable,
      uint32_t *result);
  hsa_status_t hsa_executable_validate_alt(
      hsa_executable_t executable,
      const char *options,
      uint32_t *result);
  /* deprecated */ hsa_status_t hsa_executable_get_symbol(
      hsa_executable_t executable,
      const char *module_name,
      const char *symbol_name,
      hsa_agent_t agent,
      int32_t call_convention,
      hsa_executable_symbol_t *symbol);
  hsa_status_t hsa_executable_get_symbol_by_name(
      hsa_executable_t executable,
      const char *symbol_name,
      const hsa_agent_t *agent,
      hsa_executable_symbol_t *symbol);
  hsa_status_t hsa_executable_symbol_get_info(
      hsa_executable_symbol_t executable_symbol,
      hsa_executable_symbol_info_t attribute,
      void *value);
  /* deprecated */ hsa_status_t hsa_executable_iterate_symbols(
      hsa_executable_t executable,
      hsa_status_t (*callback)(hsa_executable_t executable,
                               hsa_executable_symbol_t symbol,
                               void *data),
      void *data);
  hsa_status_t hsa_executable_iterate_agent_symbols(
      hsa_executable_t executable,
      hsa_agent_t agent,
      hsa_status_t (*callback)(hsa_executable_t exec,
                               hsa_agent_t agent,
                               hsa_executable_symbol_t symbol,
                               void *data),
      void *data);
  hsa_status_t hsa_executable_iterate_program_symbols(
      hsa_executable_t executable,
      hsa_status_t (*callback)(hsa_executable_t exec,
                               hsa_executable_symbol_t symbol,
                               void *data),
      void *data);
  hsa_status_t hsa_get_tile_config(hsa_agent_t agent_handle, void* config);

  //===--- Runtime Notifications ------------------------------------------===//

  hsa_status_t hsa_status_string(
      hsa_status_t status,
      const char **status_string);

}   //  namespace HSA
}   //  namespace rocr

#ifdef BUILDING_HSA_CORE_RUNTIME
//This using declaration is deliberate!
//We want unqualified name resolution to fail when building the runtime.  This is a guard against accidental use of the intercept layer in the runtime.
//using namespace rocr::HSA;
#endif

#endif
