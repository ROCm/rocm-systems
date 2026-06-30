// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "ctrl/hsa_init.h"

#include "hsa/hsa.h"
#include "hsa/hsa_api_trace.h"
#include "hsa/hsa_ext_amd.h"

#include "lib/aqlprofile/util/hsa_rsrc_factory.h"

namespace ctrl_test {

void InitHsaTables() {
  static CoreApiTable core{};
  static AmdExtTable amd_ext{};

  core.hsa_init_fn = ::hsa_init;
  core.hsa_shut_down_fn = ::hsa_shut_down;
  core.hsa_status_string_fn = ::hsa_status_string;
  core.hsa_system_get_info_fn = ::hsa_system_get_info;
  core.hsa_system_get_major_extension_table_fn = ::hsa_system_get_major_extension_table;
  core.hsa_iterate_agents_fn = ::hsa_iterate_agents;
  core.hsa_agent_get_info_fn = ::hsa_agent_get_info;
  core.hsa_queue_create_fn = ::hsa_queue_create;
  core.hsa_queue_destroy_fn = ::hsa_queue_destroy;
  core.hsa_queue_load_read_index_relaxed_fn = ::hsa_queue_load_read_index_relaxed;
  core.hsa_queue_load_write_index_relaxed_fn = ::hsa_queue_load_write_index_relaxed;
  core.hsa_queue_store_write_index_relaxed_fn = ::hsa_queue_store_write_index_relaxed;
  core.hsa_signal_create_fn = ::hsa_signal_create;
  core.hsa_signal_destroy_fn = ::hsa_signal_destroy;
  core.hsa_signal_store_relaxed_fn = ::hsa_signal_store_relaxed;
  core.hsa_signal_wait_scacquire_fn = ::hsa_signal_wait_scacquire;
  core.hsa_memory_free_fn = ::hsa_memory_free;
  core.hsa_code_object_reader_create_from_file_fn = ::hsa_code_object_reader_create_from_file;
  core.hsa_code_object_reader_destroy_fn = ::hsa_code_object_reader_destroy;
  core.hsa_executable_create_alt_fn = ::hsa_executable_create_alt;
  core.hsa_executable_load_agent_code_object_fn = ::hsa_executable_load_agent_code_object;
  core.hsa_executable_freeze_fn = ::hsa_executable_freeze;
  core.hsa_executable_get_symbol_fn = ::hsa_executable_get_symbol;

  amd_ext.hsa_amd_memory_pool_get_info_fn = ::hsa_amd_memory_pool_get_info;
  amd_ext.hsa_amd_agent_iterate_memory_pools_fn = ::hsa_amd_agent_iterate_memory_pools;
  amd_ext.hsa_amd_memory_pool_allocate_fn = ::hsa_amd_memory_pool_allocate;
  amd_ext.hsa_amd_agents_allow_access_fn = ::hsa_amd_agents_allow_access;
  amd_ext.hsa_amd_memory_async_copy_fn = ::hsa_amd_memory_async_copy;

  HsaApiTable table{};
  table.core_ = &core;
  table.amd_ext_ = &amd_ext;
  rocprofiler::aqlprofile::hsa_rsrc_factory_init(&table);
}

const hsa_ven_amd_aqlprofile_pfn_t* InTreeAqlProfileApi() {
  // Point each entry at the in-tree (statically linked) implementation from
  // rocprofiler-sdk-aqlprofile. Referencing these symbols also forces the
  // linker to pull the in-tree implementation into the test binary.
  static hsa_ven_amd_aqlprofile_pfn_t api = {};
  api.hsa_ven_amd_aqlprofile_error_string = ::hsa_ven_amd_aqlprofile_error_string;
  api.hsa_ven_amd_aqlprofile_validate_event = ::hsa_ven_amd_aqlprofile_validate_event;
  api.hsa_ven_amd_aqlprofile_start = ::hsa_ven_amd_aqlprofile_start;
  api.hsa_ven_amd_aqlprofile_stop = ::hsa_ven_amd_aqlprofile_stop;
#ifdef AQLPROF_NEW_API
  api.hsa_ven_amd_aqlprofile_read = ::hsa_ven_amd_aqlprofile_read;
#endif
  api.hsa_ven_amd_aqlprofile_legacy_get_pm4 = ::hsa_ven_amd_aqlprofile_legacy_get_pm4;
  api.hsa_ven_amd_aqlprofile_get_info = ::hsa_ven_amd_aqlprofile_get_info;
  api.hsa_ven_amd_aqlprofile_iterate_data = ::hsa_ven_amd_aqlprofile_iterate_data;
  return &api;
}

}  // namespace ctrl_test
