/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#if defined(USE_COMGR_LIBRARY)
#include "os/os.hpp"
#include "utils/flags.hpp"
#include "comgrctx.hpp"

namespace amd {
std::once_flag Comgr::initialized;
ComgrEntryPoints Comgr::cep_;
bool Comgr::is_ready_ = false;

bool Comgr::LoadLib(bool is_versioned) {
#if defined(COMGR_DYN_DLL)

  if (is_versioned) {
#if defined(HIP_MAJOR_VERSION) && defined(HIP_MAJOR_VERSION)
    std::string comgr_versioned_name, major_version, minor_version;
    const std::string kComgrPrefix = "amd_comgr";

    if (HIP_MAJOR_VERSION > 9) {
      major_version = std::to_string(HIP_MAJOR_VERSION);
    } else {
      major_version = "0" + std::to_string(HIP_MAJOR_VERSION);
    }

    if (HIP_MINOR_VERSION > 9) {
      minor_version = std::to_string(HIP_MINOR_VERSION);
    } else {
      minor_version = "0" + std::to_string(HIP_MINOR_VERSION);
    }

    comgr_versioned_name = kComgrPrefix + major_version + minor_version + std::string(".dll");

    static const char* comgr_lib_name =
        LP64_SWITCH(WINDOWS_SWITCH("amd_comgr32.dll", "libamd_comgr32.so.3"),
                    WINDOWS_SWITCH(comgr_versioned_name.c_str(), "libamd_comgr.so.3"));
    cep_.handle = Os::loadLibrary(comgr_lib_name);
#endif
  } else {
    std::string comgr_major_dll =
        "amd_comgr_" + std::to_string(AMD_COMGR_INTERFACE_VERSION_MAJOR) + ".dll";
    static const char* comgr_lib_name =
        LP64_SWITCH(WINDOWS_SWITCH("amd_comgr32.dll", "libamd_comgr32.so.3"),
                    WINDOWS_SWITCH(comgr_major_dll.c_str(), "libamd_comgr.so.3"));
    cep_.handle = Os::loadLibrary(comgr_lib_name);
  }
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
#endif
