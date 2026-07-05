/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "os/os.hpp"
#include "utils/flags.hpp"
#include "comgrctx.hpp"

namespace amd {
std::once_flag Comgr::initialized;
ComgrEntryPoints Comgr::cep_;
bool Comgr::is_ready_ = false;

bool Comgr::LoadLib() {
#if defined(COMGR_DYN_DLL)
#if defined(_WIN64) && defined(COMGR_DLL_NAME)
  // Use CMake-configured DLL name on 64-bit Windows
  static const char* comgr_lib_name = COMGR_DLL_NAME;
#else
  // Default names for other platforms
  static const char* comgr_lib_name =
      LP64_SWITCH(WINDOWS_SWITCH("amd_comgr32.dll", "libamd_comgr32.so.3"),
                  WINDOWS_SWITCH("amd_comgr.dll", "libamd_comgr.so.3"));
#endif
  cep_.handle = Os::loadLibrary(comgr_lib_name);
  if (nullptr == cep_.handle) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Failed to load COMGR library.");
    return false;
  }
#endif
  GET_COMGR_SYMBOL(amd_comgr_get_version)
  GET_COMGR_SYMBOL(amd_comgr_status_string)
  GET_COMGR_SYMBOL(amd_comgr_get_isa_count)
  GET_COMGR_SYMBOL(amd_comgr_get_isa_name)
  GET_COMGR_SYMBOL(amd_comgr_get_isa_metadata)
  GET_COMGR_SYMBOL(amd_comgr_create_data)
  GET_COMGR_SYMBOL(amd_comgr_release_data)
  GET_COMGR_SYMBOL(amd_comgr_get_data_kind)
  GET_COMGR_SYMBOL(amd_comgr_set_data)
  GET_COMGR_SYMBOL(amd_comgr_set_data_name)
  GET_COMGR_SYMBOL(amd_comgr_get_data)
  GET_COMGR_SYMBOL(amd_comgr_get_data_name)
  GET_COMGR_SYMBOL(amd_comgr_get_data_isa_name)
  GET_COMGR_SYMBOL(amd_comgr_get_data_metadata)
  GET_COMGR_SYMBOL(amd_comgr_destroy_metadata)
  GET_COMGR_SYMBOL(amd_comgr_create_data_set)
  GET_COMGR_SYMBOL(amd_comgr_destroy_data_set)
  GET_COMGR_SYMBOL(amd_comgr_data_set_add)
  GET_COMGR_SYMBOL(amd_comgr_data_set_remove)
  GET_COMGR_SYMBOL(amd_comgr_action_data_count)
  GET_COMGR_SYMBOL(amd_comgr_action_data_get_data)
  GET_COMGR_SYMBOL(amd_comgr_create_action_info)
  GET_COMGR_SYMBOL(amd_comgr_destroy_action_info)
  GET_COMGR_SYMBOL(amd_comgr_action_info_set_isa_name)
  GET_COMGR_SYMBOL(amd_comgr_action_info_get_isa_name)
  GET_COMGR_SYMBOL(amd_comgr_action_info_set_language)
  GET_COMGR_SYMBOL(amd_comgr_action_info_get_language)
  GET_COMGR_SYMBOL(amd_comgr_action_info_set_option_list)
  GET_COMGR_SYMBOL(amd_comgr_action_info_get_option_list_count)
  GET_COMGR_SYMBOL(amd_comgr_action_info_get_option_list_item)
  GET_COMGR_SYMBOL(amd_comgr_action_info_set_working_directory_path)
  GET_COMGR_SYMBOL(amd_comgr_action_info_get_working_directory_path)
  GET_COMGR_SYMBOL(amd_comgr_action_info_set_logging)
  GET_COMGR_SYMBOL(amd_comgr_action_info_get_logging)
  GET_COMGR_SYMBOL(amd_comgr_do_action)
  GET_COMGR_SYMBOL(amd_comgr_get_metadata_kind)
  GET_COMGR_SYMBOL(amd_comgr_get_metadata_string)
  GET_COMGR_SYMBOL(amd_comgr_get_metadata_map_size)
  GET_COMGR_SYMBOL(amd_comgr_iterate_map_metadata)
  GET_COMGR_SYMBOL(amd_comgr_metadata_lookup)
  GET_COMGR_SYMBOL(amd_comgr_get_metadata_list_size)
  GET_COMGR_SYMBOL(amd_comgr_index_list_metadata)
  GET_COMGR_SYMBOL(amd_comgr_iterate_symbols)
  GET_COMGR_SYMBOL(amd_comgr_symbol_lookup)
  GET_COMGR_SYMBOL(amd_comgr_symbol_get_info)
  GET_COMGR_SYMBOL(amd_comgr_demangle_symbol_name)
  GET_COMGR_SYMBOL(amd_comgr_populate_mangled_names)
  GET_COMGR_SYMBOL(amd_comgr_get_mangled_name)
  GET_COMGR_SYMBOL(amd_comgr_populate_name_expression_map)
  GET_COMGR_SYMBOL(amd_comgr_map_name_expression_to_symbol_name)
  GET_COMGR_SYMBOL(amd_comgr_action_info_set_device_lib_linking)
  GET_COMGR_SYMBOL(amd_comgr_lookup_code_object)
  GET_COMGR_OPTIONAL_SYMBOL(amd_comgr_action_info_set_bundle_entry_ids)
  is_ready_ = true;
  size_t major = 0, minor = 0;
  get_version(&major, &minor);
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "Loaded COMGR library version %zu.%zu.", major, minor);
  return true;
}

}  // namespace amd
