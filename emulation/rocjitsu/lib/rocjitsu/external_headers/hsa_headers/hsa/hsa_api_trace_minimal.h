// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_api_trace_minimal.h
/// @brief Minimal ROCR HSA tools API-table ABI mirror used by rocjitsu hooks.
///
/// @details ROCR passes HSA tools a table whose layout is defined by
/// `hsa_api_trace.h`. The full ROCR trace header pulls in extension headers that
/// are unnecessary for the DBT-only MVP. This file mirrors only the table
/// headers and the core API entries through the code-object load functions that
/// rocjitsu patches. Field order must match ROCR exactly for every mirrored
/// field; callers must validate `CoreApiTable::version.minor_id` before reading
/// or writing function pointers near the tail of the table.

#pragma once

#include "hsa/hsa.h"

#include <cstdint>

/// @brief Version header present at the start of every ROCR API table.
///
/// @details ROCR stores the table size in `minor_id` for the HSA tools ABI. The
/// DBT hook uses that value as an ABI boundary check before dereferencing fields
/// that may not exist in older runtimes.
struct ApiTableVersion {
  uint32_t major_id;
  uint32_t minor_id;
  uint32_t step_id;
  uint32_t reserved;
};

/// @brief Minimal mirror of ROCR's `CoreApiTable` through agent code-object load.
///
/// @details The field order and types intentionally match ROCR
/// `hsa_api_trace.h` through `hsa_executable_load_agent_code_object_fn`.
/// Rocjitsu overwrites the file-reader create function for diagnostics, the
/// memory-reader create/destroy functions for translation state, and the agent
/// code-object loader. Preserving the preceding fields keeps the offsets
/// identical to ROCR without vendoring extension-table dependencies.
struct CoreApiTable {
  ApiTableVersion version;
  decltype(hsa_init) *hsa_init_fn;
  decltype(hsa_shut_down) *hsa_shut_down_fn;
  decltype(hsa_system_get_info) *hsa_system_get_info_fn;
  decltype(hsa_system_extension_supported) *hsa_system_extension_supported_fn;
  decltype(hsa_system_get_extension_table) *hsa_system_get_extension_table_fn;
  decltype(hsa_iterate_agents) *hsa_iterate_agents_fn;
  decltype(hsa_agent_get_info) *hsa_agent_get_info_fn;
  decltype(hsa_queue_create) *hsa_queue_create_fn;
  decltype(hsa_soft_queue_create) *hsa_soft_queue_create_fn;
  decltype(hsa_queue_destroy) *hsa_queue_destroy_fn;
  decltype(hsa_queue_inactivate) *hsa_queue_inactivate_fn;
  decltype(hsa_queue_load_read_index_scacquire) *hsa_queue_load_read_index_scacquire_fn;
  decltype(hsa_queue_load_read_index_relaxed) *hsa_queue_load_read_index_relaxed_fn;
  decltype(hsa_queue_load_write_index_scacquire) *hsa_queue_load_write_index_scacquire_fn;
  decltype(hsa_queue_load_write_index_relaxed) *hsa_queue_load_write_index_relaxed_fn;
  decltype(hsa_queue_store_write_index_relaxed) *hsa_queue_store_write_index_relaxed_fn;
  decltype(hsa_queue_store_write_index_screlease) *hsa_queue_store_write_index_screlease_fn;
  decltype(hsa_queue_cas_write_index_scacq_screl) *hsa_queue_cas_write_index_scacq_screl_fn;
  decltype(hsa_queue_cas_write_index_scacquire) *hsa_queue_cas_write_index_scacquire_fn;
  decltype(hsa_queue_cas_write_index_relaxed) *hsa_queue_cas_write_index_relaxed_fn;
  decltype(hsa_queue_cas_write_index_screlease) *hsa_queue_cas_write_index_screlease_fn;
  decltype(hsa_queue_add_write_index_scacq_screl) *hsa_queue_add_write_index_scacq_screl_fn;
  decltype(hsa_queue_add_write_index_scacquire) *hsa_queue_add_write_index_scacquire_fn;
  decltype(hsa_queue_add_write_index_relaxed) *hsa_queue_add_write_index_relaxed_fn;
  decltype(hsa_queue_add_write_index_screlease) *hsa_queue_add_write_index_screlease_fn;
  decltype(hsa_queue_store_read_index_relaxed) *hsa_queue_store_read_index_relaxed_fn;
  decltype(hsa_queue_store_read_index_screlease) *hsa_queue_store_read_index_screlease_fn;
  decltype(hsa_agent_iterate_regions) *hsa_agent_iterate_regions_fn;
  decltype(hsa_region_get_info) *hsa_region_get_info_fn;
  decltype(hsa_agent_get_exception_policies) *hsa_agent_get_exception_policies_fn;
  decltype(hsa_agent_extension_supported) *hsa_agent_extension_supported_fn;
  decltype(hsa_memory_register) *hsa_memory_register_fn;
  decltype(hsa_memory_deregister) *hsa_memory_deregister_fn;
  decltype(hsa_memory_allocate) *hsa_memory_allocate_fn;
  decltype(hsa_memory_free) *hsa_memory_free_fn;
  decltype(hsa_memory_copy) *hsa_memory_copy_fn;
  decltype(hsa_memory_assign_agent) *hsa_memory_assign_agent_fn;
  decltype(hsa_signal_create) *hsa_signal_create_fn;
  decltype(hsa_signal_destroy) *hsa_signal_destroy_fn;
  decltype(hsa_signal_load_relaxed) *hsa_signal_load_relaxed_fn;
  decltype(hsa_signal_load_scacquire) *hsa_signal_load_scacquire_fn;
  decltype(hsa_signal_store_relaxed) *hsa_signal_store_relaxed_fn;
  decltype(hsa_signal_store_screlease) *hsa_signal_store_screlease_fn;
  decltype(hsa_signal_wait_relaxed) *hsa_signal_wait_relaxed_fn;
  decltype(hsa_signal_wait_scacquire) *hsa_signal_wait_scacquire_fn;
  decltype(hsa_signal_and_relaxed) *hsa_signal_and_relaxed_fn;
  decltype(hsa_signal_and_scacquire) *hsa_signal_and_scacquire_fn;
  decltype(hsa_signal_and_screlease) *hsa_signal_and_screlease_fn;
  decltype(hsa_signal_and_scacq_screl) *hsa_signal_and_scacq_screl_fn;
  decltype(hsa_signal_or_relaxed) *hsa_signal_or_relaxed_fn;
  decltype(hsa_signal_or_scacquire) *hsa_signal_or_scacquire_fn;
  decltype(hsa_signal_or_screlease) *hsa_signal_or_screlease_fn;
  decltype(hsa_signal_or_scacq_screl) *hsa_signal_or_scacq_screl_fn;
  decltype(hsa_signal_xor_relaxed) *hsa_signal_xor_relaxed_fn;
  decltype(hsa_signal_xor_scacquire) *hsa_signal_xor_scacquire_fn;
  decltype(hsa_signal_xor_screlease) *hsa_signal_xor_screlease_fn;
  decltype(hsa_signal_xor_scacq_screl) *hsa_signal_xor_scacq_screl_fn;
  decltype(hsa_signal_exchange_relaxed) *hsa_signal_exchange_relaxed_fn;
  decltype(hsa_signal_exchange_scacquire) *hsa_signal_exchange_scacquire_fn;
  decltype(hsa_signal_exchange_screlease) *hsa_signal_exchange_screlease_fn;
  decltype(hsa_signal_exchange_scacq_screl) *hsa_signal_exchange_scacq_screl_fn;
  decltype(hsa_signal_add_relaxed) *hsa_signal_add_relaxed_fn;
  decltype(hsa_signal_add_scacquire) *hsa_signal_add_scacquire_fn;
  decltype(hsa_signal_add_screlease) *hsa_signal_add_screlease_fn;
  decltype(hsa_signal_add_scacq_screl) *hsa_signal_add_scacq_screl_fn;
  decltype(hsa_signal_subtract_relaxed) *hsa_signal_subtract_relaxed_fn;
  decltype(hsa_signal_subtract_scacquire) *hsa_signal_subtract_scacquire_fn;
  decltype(hsa_signal_subtract_screlease) *hsa_signal_subtract_screlease_fn;
  decltype(hsa_signal_subtract_scacq_screl) *hsa_signal_subtract_scacq_screl_fn;
  decltype(hsa_signal_cas_relaxed) *hsa_signal_cas_relaxed_fn;
  decltype(hsa_signal_cas_scacquire) *hsa_signal_cas_scacquire_fn;
  decltype(hsa_signal_cas_screlease) *hsa_signal_cas_screlease_fn;
  decltype(hsa_signal_cas_scacq_screl) *hsa_signal_cas_scacq_screl_fn;
  decltype(hsa_isa_from_name) *hsa_isa_from_name_fn;
  decltype(hsa_isa_get_info) *hsa_isa_get_info_fn;
  decltype(hsa_isa_compatible) *hsa_isa_compatible_fn;
  decltype(hsa_code_object_serialize) *hsa_code_object_serialize_fn;
  decltype(hsa_code_object_deserialize) *hsa_code_object_deserialize_fn;
  decltype(hsa_code_object_destroy) *hsa_code_object_destroy_fn;
  decltype(hsa_code_object_get_info) *hsa_code_object_get_info_fn;
  decltype(hsa_code_object_get_symbol) *hsa_code_object_get_symbol_fn;
  decltype(hsa_code_symbol_get_info) *hsa_code_symbol_get_info_fn;
  decltype(hsa_code_object_iterate_symbols) *hsa_code_object_iterate_symbols_fn;
  decltype(hsa_executable_create) *hsa_executable_create_fn;
  decltype(hsa_executable_destroy) *hsa_executable_destroy_fn;
  decltype(hsa_executable_load_code_object) *hsa_executable_load_code_object_fn;
  decltype(hsa_executable_freeze) *hsa_executable_freeze_fn;
  decltype(hsa_executable_get_info) *hsa_executable_get_info_fn;
  decltype(hsa_executable_global_variable_define) *hsa_executable_global_variable_define_fn;
  decltype(hsa_executable_agent_global_variable_define)
      *hsa_executable_agent_global_variable_define_fn;
  decltype(hsa_executable_readonly_variable_define) *hsa_executable_readonly_variable_define_fn;
  decltype(hsa_executable_validate) *hsa_executable_validate_fn;
  decltype(hsa_executable_get_symbol) *hsa_executable_get_symbol_fn;
  decltype(hsa_executable_symbol_get_info) *hsa_executable_symbol_get_info_fn;
  decltype(hsa_executable_iterate_symbols) *hsa_executable_iterate_symbols_fn;
  decltype(hsa_status_string) *hsa_status_string_fn;
  decltype(hsa_extension_get_name) *hsa_extension_get_name_fn;
  decltype(hsa_system_major_extension_supported) *hsa_system_major_extension_supported_fn;
  decltype(hsa_system_get_major_extension_table) *hsa_system_get_major_extension_table_fn;
  decltype(hsa_agent_major_extension_supported) *hsa_agent_major_extension_supported_fn;
  decltype(hsa_cache_get_info) *hsa_cache_get_info_fn;
  decltype(hsa_agent_iterate_caches) *hsa_agent_iterate_caches_fn;
  decltype(hsa_signal_silent_store_relaxed) *hsa_signal_silent_store_relaxed_fn;
  decltype(hsa_signal_silent_store_screlease) *hsa_signal_silent_store_screlease_fn;
  decltype(hsa_signal_group_create) *hsa_signal_group_create_fn;
  decltype(hsa_signal_group_destroy) *hsa_signal_group_destroy_fn;
  decltype(hsa_signal_group_wait_any_scacquire) *hsa_signal_group_wait_any_scacquire_fn;
  decltype(hsa_signal_group_wait_any_relaxed) *hsa_signal_group_wait_any_relaxed_fn;
  decltype(hsa_agent_iterate_isas) *hsa_agent_iterate_isas_fn;
  decltype(hsa_isa_get_info_alt) *hsa_isa_get_info_alt_fn;
  decltype(hsa_isa_get_exception_policies) *hsa_isa_get_exception_policies_fn;
  decltype(hsa_isa_get_round_method) *hsa_isa_get_round_method_fn;
  decltype(hsa_wavefront_get_info) *hsa_wavefront_get_info_fn;
  decltype(hsa_isa_iterate_wavefronts) *hsa_isa_iterate_wavefronts_fn;
  decltype(hsa_code_object_get_symbol_from_name) *hsa_code_object_get_symbol_from_name_fn;
  decltype(hsa_code_object_reader_create_from_file) *hsa_code_object_reader_create_from_file_fn;
  decltype(hsa_code_object_reader_create_from_memory) *hsa_code_object_reader_create_from_memory_fn;
  decltype(hsa_code_object_reader_destroy) *hsa_code_object_reader_destroy_fn;
  decltype(hsa_executable_create_alt) *hsa_executable_create_alt_fn;
  decltype(hsa_executable_load_program_code_object) *hsa_executable_load_program_code_object_fn;
  decltype(hsa_executable_load_agent_code_object) *hsa_executable_load_agent_code_object_fn;
};

/// @brief Top-level ROCR API table passed to HSA tools from `OnLoad()`.
///
/// @details The DBT MVP only patches `core_`. Extension tables are intentionally
/// opaque because queue interception, AMD extension APIs, finalizer APIs, image
/// APIs, tools events, and PC sampling are not part of the load-time DBT hook.
struct HsaApiTable {
  ApiTableVersion version;
  CoreApiTable *core_;
  void *amd_ext_;
  void *finalizer_ext_;
  void *image_ext_;
  void *tools_;
  void *pc_sampling_ext_;
};
