/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <dlfcn.h>
#include <string.h>

#include <fstream>
#include <map>

#include "common/rdc_fields_supported.h"
#include "rdc/rdc.h"
#include "rdc/rdc_private.h"
#include "rdc_lib/RdcHandler.h"
#include "rdc_lib/RdcLibraryLoader.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

static amd::rdc::RdcLibraryLoader rdc_lib_loader;

rdc_status_t rdc_init(uint64_t) { return RDC_ST_OK; }

rdc_status_t rdc_shutdown() { return rdc_lib_loader.unload(); }

rdc_status_t rdc_connect(const char* ipAddress, rdc_handle_t* p_rdc_handle, const char* root_ca,
                         const char* client_cert, const char* client_key) {
  amd::rdc::RdcHandler* (*func_make_handler)(const char*, const char*, const char*, const char*) =
      nullptr;

  if (!ipAddress || !p_rdc_handle) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  rdc_status_t status = rdc_lib_loader.load("librdc_client.so", &func_make_handler);
  if (status != RDC_ST_OK) {
    *p_rdc_handle = nullptr;
    return status;
  }

  *p_rdc_handle =
      static_cast<rdc_handle_t>(func_make_handler(ipAddress, root_ca, client_cert, client_key));
  return RDC_ST_OK;
}

rdc_status_t rdc_disconnect(rdc_handle_t p_rdc_handle) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }
  delete static_cast<amd::rdc::RdcHandler*>(p_rdc_handle);
  p_rdc_handle = nullptr;
  return RDC_ST_OK;
}

rdc_status_t rdc_start_embedded(rdc_operation_mode_t op_mode, rdc_handle_t* p_rdc_handle) {
  amd::rdc::RdcHandler* (*func_make_handler)(rdc_operation_mode_t) = nullptr;
  if (!p_rdc_handle) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  rdc_status_t status = rdc_lib_loader.load("librdc.so", &func_make_handler);
  if (status != RDC_ST_OK) {
    *p_rdc_handle = nullptr;
    return status;
  }

  *p_rdc_handle = static_cast<rdc_handle_t>(func_make_handler(op_mode));

  return RDC_ST_OK;
}

rdc_status_t rdc_stop_embedded(rdc_handle_t p_rdc_handle) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }
  delete static_cast<amd::rdc::RdcHandler*>(p_rdc_handle);
  p_rdc_handle = nullptr;
  return RDC_ST_OK;
}

rdc_status_t rdc_field_update_all(rdc_handle_t p_rdc_handle, uint32_t wait_for_update) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_field_update_all(wait_for_update);
}

rdc_status_t rdc_job_get_stats(rdc_handle_t p_rdc_handle, const char job_id[64],
                               rdc_job_info_t* p_job_info) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_job_get_stats(job_id, p_job_info);
}

rdc_status_t rdc_job_start_stats(rdc_handle_t p_rdc_handle, rdc_gpu_group_t groupId,
                                 const char job_id[64], uint64_t update_freq) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_job_start_stats(groupId, job_id, update_freq);
}

rdc_status_t rdc_job_remove(rdc_handle_t p_rdc_handle, const char job_id[64]) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_job_remove(job_id);
}

rdc_status_t rdc_job_remove_all(rdc_handle_t p_rdc_handle) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_job_remove_all();
}

rdc_status_t rdc_job_stop_stats(rdc_handle_t p_rdc_handle, const char job_id[64]) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_job_stop_stats(job_id);
}

rdc_status_t rdc_group_gpu_create(rdc_handle_t p_rdc_handle, rdc_group_type_t type,
                                  const char* group_name, rdc_gpu_group_t* p_rdc_group_id) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_group_gpu_create(type, group_name, p_rdc_group_id);
}

rdc_status_t rdc_group_gpu_add(rdc_handle_t p_rdc_handle, rdc_gpu_group_t groupId,
                               uint32_t gpuIndex) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_group_gpu_add(groupId, gpuIndex);
}

// TODO: rewrite get_all to allow different types
rdc_status_t rdc_device_get_all(rdc_handle_t p_rdc_handle,
                                uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count) {
  if (!p_rdc_handle || !count) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_device_get_all(gpu_index_list, count);
}

rdc_status_t rdc_device_get_attributes(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                       rdc_device_attributes_t* p_rdc_attr) {
  if (!p_rdc_handle || !p_rdc_attr) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_device_get_attributes(gpu_index, p_rdc_attr);
}

rdc_status_t rdc_device_get_component_version(rdc_handle_t p_rdc_handle, rdc_component_t component,
                                              rdc_component_version_t* p_rdc_compv) {
  if (!p_rdc_handle || !p_rdc_compv) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_device_get_component_version(component, p_rdc_compv);
}

rdc_status_t rdc_group_field_create(rdc_handle_t p_rdc_handle, uint32_t num_field_ids,
                                    rdc_field_t* field_ids, const char* field_group_name,
                                    rdc_field_grp_t* rdc_field_group_id) {
  if (!p_rdc_handle || !field_ids || !field_group_name || !rdc_field_group_id) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_group_field_create(num_field_ids, field_ids, field_group_name, rdc_field_group_id);
}

rdc_status_t rdc_group_field_get_info(rdc_handle_t p_rdc_handle, rdc_field_grp_t rdc_field_group_id,
                                      rdc_field_group_info_t* field_group_info) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_group_field_get_info(rdc_field_group_id, field_group_info);
}

rdc_status_t rdc_group_gpu_get_info(rdc_handle_t p_rdc_handle, rdc_gpu_group_t p_rdc_group_id,
                                    rdc_group_info_t* p_rdc_group_info) {
  if (!p_rdc_handle || !p_rdc_group_info) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_group_gpu_get_info(p_rdc_group_id, p_rdc_group_info);
}

rdc_status_t rdc_group_get_all_ids(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id_list[],
                                   uint32_t* count) {
  if (!p_rdc_handle || !count) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_group_get_all_ids(group_id_list, count);
}

rdc_status_t rdc_group_field_get_all_ids(rdc_handle_t p_rdc_handle,
                                         rdc_field_grp_t field_group_id_list[], uint32_t* count) {
  if (!p_rdc_handle || !count) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_group_field_get_all_ids(field_group_id_list, count);
}

rdc_status_t rdc_field_watch(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                             rdc_field_grp_t field_group_id, uint64_t update_freq,
                             double max_keep_age, uint32_t max_keep_samples) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_field_watch(group_id, field_group_id, update_freq, max_keep_age, max_keep_samples);
}

rdc_status_t rdc_field_get_latest_value(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                        rdc_field_t field, rdc_field_value* value) {
  if (!p_rdc_handle || !value) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_field_get_latest_value(gpu_index, field, value);
}

rdc_status_t rdc_field_get_value_since(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                       rdc_field_t field, uint64_t since_time_stamp,
                                       uint64_t* next_since_time_stamp, rdc_field_value* value) {
  if (!p_rdc_handle || !next_since_time_stamp || !value) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_field_get_value_since(gpu_index, field, since_time_stamp, next_since_time_stamp, value);
}

rdc_status_t rdc_field_unwatch(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                               rdc_field_grp_t field_group_id) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_field_unwatch(group_id, field_group_id);
}

rdc_status_t rdc_group_gpu_destroy(rdc_handle_t p_rdc_handle, rdc_gpu_group_t p_rdc_group_id) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_group_gpu_destroy(p_rdc_group_id);
}

rdc_status_t rdc_group_field_destroy(rdc_handle_t p_rdc_handle,
                                     rdc_field_grp_t rdc_field_group_id) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_group_field_destroy(rdc_field_group_id);
}

rdc_status_t rdc_diagnostic_run(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                                rdc_diag_level_t level, const char* config, size_t config_size,
                                rdc_diag_response_t* response, rdc_diag_callback_t* callback) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_diagnostic_run(group_id, level, config, config_size, response, callback);
}

rdc_status_t rdc_test_case_run(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                               rdc_diag_test_cases_t test_case, const char* config,
                               size_t config_size, rdc_diag_test_result_t* result,
                               rdc_diag_callback_t* callback) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_test_case_run(group_id, test_case, config, config_size, result, callback);
}

rdc_status_t get_mixed_component_version(rdc_handle_t p_rdc_handle, mixed_component_t component,
                                         mixed_component_version_t* p_mixed_compv) {
  if (!p_rdc_handle || !p_mixed_compv) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->get_mixed_component_version(component, p_mixed_compv);
}

const char* rdc_status_string(rdc_status_t result) {
  switch (result) {
    case RDC_ST_OK:
      return "Success";
    case RDC_ST_NOT_SUPPORTED:
      return "Not supported";
    case RDC_ST_FAIL_LOAD_MODULE:
      return "Fail to load module";
    case RDC_ST_INVALID_HANDLER:
      return "Invalid handler";
    case RDC_ST_NOT_FOUND:
      return "Cannot find the value";
    case RDC_ST_BAD_PARAMETER:
      return "Invalid parameters";
    case RDC_ST_SMI_ERROR:
      return "SMI error";
    case RDC_ST_MAX_LIMIT:
      return "The max limit reached";
    case RDC_ST_CONFLICT:
      return "Conflict with current state";
    case RDC_ST_ALREADY_EXIST:
      return "The value already exists";
    case RDC_ST_CLIENT_ERROR:
      return "RDC Client error";
    case RDC_ST_INSUFF_RESOURCES:
      return "Not enough resources to complete operation";
    case RDC_ST_FILE_ERROR:
      return "Failed to access a file";
    case RDC_ST_NO_DATA:
      return "Data was requested, but none was found";
    case RDC_ST_PERM_ERROR:
      return "Insufficient permission to complete operation";
    case RDC_ST_CORRUPTED_EEPROM:
      return "EEPROM is corrupted";
    case RDC_ST_UNKNOWN_ERROR:
      return "Unknown error";
    default:
      return "Unknown";
  }
}

const char* rdc_diagnostic_result_string(rdc_diag_result_t result) {
  switch (result) {
    case RDC_DIAG_RESULT_PASS:
      return "Pass";
    case RDC_DIAG_RESULT_SKIP:
      return "Skip";
    case RDC_DIAG_RESULT_WARN:
      return "Warn";
    case RDC_DIAG_RESULT_FAIL:
      return "Fail";
    default:
      return "Unknown";
  }
}

rdc_status_t rdc_config_set(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            rdc_config_setting_t setting) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_config_set(group_id, setting);
}

rdc_status_t rdc_config_get(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            rdc_config_setting_list_t* settings) {
  if (!p_rdc_handle || settings == nullptr) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_config_get(group_id, settings);
}

rdc_status_t rdc_config_clear(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_config_clear(group_id);
}

const char* field_id_string(rdc_field_t field_id) {
  amd::rdc::fld_id2name_map_t& field_id_to_descript = amd::rdc::get_field_id_description_from_id();
  return field_id_to_descript.find(field_id)->second.label.c_str();
}

rdc_field_t get_field_id_from_name(const char* name) {
  rdc_field_t value;
  if (amd::rdc::get_field_id_from_name(name, &value)) {
    return value;
  }
  return RDC_FI_INVALID;
}

rdc_status_t rdc_health_set(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            unsigned int components) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_health_set(group_id, components);
}

rdc_status_t rdc_health_get(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            unsigned int* components) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_health_get(group_id, components);
}

rdc_status_t rdc_health_check(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                              rdc_health_response_t* response) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_health_check(group_id, response);
}

rdc_status_t rdc_health_clear(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_health_clear(group_id);
}

char* strncpy_with_null(char* dest, const char* src, size_t n) {
  if (n == 0) {
    return dest;
  }
  strncpy(dest, src, n - 1);
  dest[n - 1] = '\0';
  return dest;
}

rdc_status_t rdc_policy_set(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            rdc_policy_t policy) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_policy_set(group_id, policy);
}

rdc_status_t rdc_policy_get(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id, uint32_t* count,
                            rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_policy_get(group_id, count, policies);
}

rdc_status_t rdc_policy_delete(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                               rdc_policy_condition_type_t condition_type) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }

  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_policy_delete(group_id, condition_type);
}

rdc_status_t rdc_policy_register(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                                 rdc_policy_register_callback callback) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }
  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_policy_register(group_id, callback);
}
rdc_status_t rdc_policy_unregister(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }
  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_policy_unregister(group_id);
}
rdc_status_t rdc_device_topology_get(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                     rdc_device_topology_t* results) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }
  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_device_topology_get(gpu_index, results);
}
rdc_status_t rdc_link_status_get(rdc_handle_t p_rdc_handle, rdc_link_status_t* results) {
  if (!p_rdc_handle) {
    return RDC_ST_INVALID_HANDLER;
  }
  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->rdc_link_status_get(results);
}

rdc_status_t rdc_get_num_partition(rdc_handle_t p_rdc_handle, uint32_t index,
                                   uint16_t* num_partition) {
  if (!p_rdc_handle || !num_partition) {
    return RDC_ST_INVALID_HANDLER;
  }
  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_get_num_partition(index, num_partition);
}

rdc_status_t rdc_instance_profile_get(rdc_handle_t p_rdc_handle, uint32_t entity_index,
                                      rdc_instance_resource_type_t resource_type,
                                      rdc_resource_profile_t* profile) {
  if (!p_rdc_handle || !profile) {
    return RDC_ST_INVALID_HANDLER;
  }
  return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)
      ->rdc_instance_profile_get(entity_index, resource_type, profile);
}

const char* get_rocm_path(const char* search_string) {
  // set default rocm path in case lookup fails
  static std::string rocm_path("/opt/rocm");
  const char* rocm_path_env = getenv("ROCM_PATH");
  if (rocm_path_env != nullptr) {
    rocm_path = rocm_path_env;
  }

  std::ifstream file("/proc/self/maps");

  if (!file.is_open()) {
    RDC_LOG(RDC_DEBUG, "CANT OPEN FILE");
    return rocm_path.c_str();
  }

  std::string line;
  while (getline(file, line)) {
    size_t index_end = line.find(search_string);
    size_t index_start = index_end;
    if (index_end == std::string::npos) {
      // no library on this line
      continue;
    }
    // walk index backwards until it reaches a space
    while ((index_start > 0) && (line[index_start - 1] != ' ')) {
      index_start--;
    }
    // extract library path, drop library name
    rocm_path = line.substr(index_start, index_end - index_start);
    // appending "../" should result in "/opt/rocm/lib/.." or similar
    rocm_path += "..";
    RDC_LOG(RDC_DEBUG, "FOUND SOMETHING!");
    return rocm_path.c_str();
  }

  return rocm_path.c_str();
}
