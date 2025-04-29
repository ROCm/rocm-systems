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
#include "rdc_lib/impl/RdcEmbeddedHandler.h"

#include <string.h>

#include "amd_smi/amdsmi.h"
#include "common/rdc_fields_supported.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/impl/RdcCacheManagerImpl.h"
#include "rdc_lib/impl/RdcConfigSettingsImpl.h"
#include "rdc_lib/impl/RdcGroupSettingsImpl.h"
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include "rdc_lib/impl/RdcMetricsUpdaterImpl.h"
#include "rdc_lib/impl/RdcModuleMgrImpl.h"
#include "rdc_lib/impl/RdcNotificationImpl.h"
#include "rdc_lib/impl/RdcPartitionImpl.h"
#include "rdc_lib/impl/RdcPolicyImpl.h"
#include "rdc_lib/impl/RdcTopologyLinkImpl.h"
#include "rdc_lib/impl/RdcWatchTableImpl.h"
#include "rdc_lib/rdc_common.h"

namespace {
// call the smi_init when load library
// and smi_shutdown when unload the library.
class smi_initializer {
  smi_initializer() {
    // Make sure smi will not be initialized multiple times
    amdsmi_shut_down();
    amdsmi_status_t ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      throw amd::rdc::RdcException(RDC_ST_FAIL_LOAD_MODULE, "SMI initialize fail");
    }
  }
  ~smi_initializer() { amdsmi_shut_down(); }

 public:
  static smi_initializer& getInstance() {
    static smi_initializer instance;
    return instance;
  }
};

static smi_initializer& in = smi_initializer::getInstance();
}  // namespace

amd::rdc::RdcHandler* make_handler(rdc_operation_mode_t op_mode) {
  return new amd::rdc::RdcEmbeddedHandler(op_mode);
}

namespace amd {
namespace rdc {

// TODO(bill_liu): make it configurable
const uint32_t METIC_UPDATE_FREQUENCY = 1000;  // 1000 microseconds by default

RdcEmbeddedHandler::RdcEmbeddedHandler(rdc_operation_mode_t mode)
    : partition_(new RdcPartitionImpl()),
      group_settings_(new RdcGroupSettingsImpl(partition_)),
      cache_mgr_(new RdcCacheManagerImpl()),
      metric_fetcher_(new RdcMetricFetcherImpl()),
      rdc_module_mgr_(new RdcModuleMgrImpl(metric_fetcher_)),
      rdc_notif_(new RdcNotificationImpl()),
      watch_table_(new RdcWatchTableImpl(group_settings_, cache_mgr_, metric_fetcher_,
                                         rdc_module_mgr_, rdc_notif_)),
      metrics_updater_(new RdcMetricsUpdaterImpl(watch_table_, METIC_UPDATE_FREQUENCY)),
      policy_(new RdcPolicyImpl(group_settings_, metric_fetcher_)),
      topologylink_(new RdcTopologyLinkImpl(group_settings_, metric_fetcher_)),
      config_handler_(new RdcConfigSettingsImpl(group_settings_)) {
  if (mode == RDC_OPERATION_MODE_AUTO) {
    RDC_LOG(RDC_DEBUG, "Run RDC with RDC_OPERATION_MODE_AUTO");
    metrics_updater_->start();
  }
}

RdcEmbeddedHandler::~RdcEmbeddedHandler() { metrics_updater_->stop(); }

// JOB API
rdc_status_t RdcEmbeddedHandler::rdc_job_start_stats(rdc_gpu_group_t groupId, const char job_id[64],
                                                     uint64_t update_freq) {
  rdc_gpu_gauges_t gpu_gauges;
  rdc_status_t status = get_gpu_gauges(&gpu_gauges);
  if (status != RDC_ST_OK) return status;

  return watch_table_->rdc_job_start_stats(groupId, job_id, update_freq, gpu_gauges);
}

rdc_status_t RdcEmbeddedHandler::get_gpu_gauges(rdc_gpu_gauges_t* gpu_gauges) {
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  uint32_t count = 0;

  if (gpu_gauges == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  rdc_status_t status = rdc_device_get_all(gpu_index_list, &count);
  if (status != RDC_ST_OK) {
    return status;
  }

  // Fetch total memory and current ecc errors
  for (uint32_t i = 0; i < count; i++) {
    rdc_field_value value;
    status = metric_fetcher_->fetch_smi_field(gpu_index_list[i], RDC_FI_GPU_MEMORY_TOTAL, &value);
    if (status != RDC_ST_OK) {
      RDC_LOG(RDC_ERROR, "Fail to get total memory of GPU " << gpu_index_list[i]);
      return status;
    }
    gpu_gauges->insert(
        {{gpu_index_list[i], RDC_FI_GPU_MEMORY_TOTAL}, static_cast<uint64_t>(value.value.l_int)});

    status = metric_fetcher_->fetch_smi_field(gpu_index_list[i], RDC_FI_ECC_CORRECT_TOTAL, &value);
    if (status == RDC_ST_OK) {
      gpu_gauges->insert({{gpu_index_list[i], RDC_FI_ECC_CORRECT_TOTAL},
                          static_cast<uint64_t>(value.value.l_int)});
    }
    status =
        metric_fetcher_->fetch_smi_field(gpu_index_list[i], RDC_FI_ECC_UNCORRECT_TOTAL, &value);
    if (status == RDC_ST_OK) {
      gpu_gauges->insert({{gpu_index_list[i], RDC_FI_ECC_UNCORRECT_TOTAL},
                          static_cast<uint64_t>(value.value.l_int)});
    }
  }
  return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_job_get_stats(const char job_id[64],
                                                   rdc_job_info_t* p_job_info) {
  if (p_job_info == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  rdc_gpu_gauges_t gpu_gauges;
  rdc_status_t status = get_gpu_gauges(&gpu_gauges);
  if (status != RDC_ST_OK) return status;

  return cache_mgr_->rdc_job_get_stats(job_id, gpu_gauges, p_job_info);
}

rdc_status_t RdcEmbeddedHandler::rdc_job_stop_stats(const char job_id[64]) {
  rdc_gpu_gauges_t gpu_gauges;
  rdc_status_t status = get_gpu_gauges(&gpu_gauges);
  if (status != RDC_ST_OK) return status;

  return watch_table_->rdc_job_stop_stats(job_id, gpu_gauges);
}

rdc_status_t RdcEmbeddedHandler::rdc_job_remove(const char job_id[64]) {
  return watch_table_->rdc_job_remove(job_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_job_remove_all() { return watch_table_->rdc_job_remove_all(); }

// Discovery API
rdc_status_t RdcEmbeddedHandler::rdc_device_get_all(uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES],
                                                    uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }
  rdc_field_value device_count;
  rdc_status_t status = metric_fetcher_->fetch_smi_field(0, RDC_FI_GPU_COUNT, &device_count);
  if (status != RDC_ST_OK) {
    return status;
  }

  // Assign the index to the index list
  *count = device_count.value.l_int;
  for (uint32_t i = 0; i < *count; i++) {
    gpu_index_list[i] = i;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_device_get_attributes(uint32_t gpu_index,
                                                           rdc_device_attributes_t* p_rdc_attr) {
  if (!p_rdc_attr) {
    return RDC_ST_BAD_PARAMETER;
  }
  rdc_field_value device_name;
  rdc_status_t status = metric_fetcher_->fetch_smi_field(gpu_index, RDC_FI_DEV_NAME, &device_name);
  strncpy_with_null(p_rdc_attr->device_name, device_name.value.str, RDC_MAX_STR_LENGTH);
  return status;
}

rdc_status_t RdcEmbeddedHandler::rdc_device_get_component_version(
    rdc_component_t component, rdc_component_version_t* p_rdc_compv) {
  if (!p_rdc_compv) {
    return RDC_ST_BAD_PARAMETER;
  }

  if (component == RDC_AMDSMI_COMPONENT) {
    amdsmi_status_t ret;
    amdsmi_version_t ver = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};

    ret = amdsmi_get_lib_version(&ver);

    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "Failed to obtain the version of the server's amd-smi library. reason: "
                             << (ret == AMDSMI_STATUS_INVAL ? "Invalid parameters" : "unknown"));
      return RDC_ST_SMI_ERROR;
    }

    strncpy_with_null(p_rdc_compv->version, ver.build, RDC_MAX_VERSION_STR_LENGTH);
    return RDC_ST_OK;
  } else {
    return RDC_ST_BAD_PARAMETER;
  }
}

// Group API
rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_create(rdc_group_type_t type, const char* group_name,
                                                      rdc_gpu_group_t* p_rdc_group_id) {
  if (!group_name || !p_rdc_group_id) {
    return RDC_ST_BAD_PARAMETER;
  }

  rdc_status_t status = group_settings_->rdc_group_gpu_create(group_name, p_rdc_group_id);

  if (status != RDC_ST_OK || type == RDC_GROUP_EMPTY) {
    return status;
  }

  // Add All GPUs to the group
  uint32_t count = 0;
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  status = rdc_device_get_all(gpu_index_list, &count);
  if (status != RDC_ST_OK) {
    return status;
  }
  for (uint32_t i = 0; i < count; i++) {
    status = rdc_group_gpu_add(*p_rdc_group_id, gpu_index_list[i]);
  }

  return status;
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_add(rdc_gpu_group_t group_id, uint32_t gpu_index) {
  uint32_t count = 0;
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  rdc_status_t status = rdc_device_get_all(gpu_index_list, &count);
  if (status != RDC_ST_OK) {
    return status;
  }

  rdc_entity_info_t info = rdc_get_info_from_entity_index(gpu_index);

  uint32_t physical_gpu = info.device_index;

  bool is_gpu_exist = false;
  for (uint32_t i = 0; i < count; i++) {
    if (gpu_index_list[i] == physical_gpu) {
      is_gpu_exist = true;
      break;
    }
  }

  if (!is_gpu_exist) {
    RDC_LOG(RDC_INFO, "Fail to add GPU index " << gpu_index << " to group " << group_id
                                               << " as the GPU index is invalid.");
    return RDC_ST_NOT_FOUND;
  }

  return group_settings_->rdc_group_gpu_add(group_id, gpu_index);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_field_create(uint32_t num_field_ids,
                                                        rdc_field_t* field_ids,
                                                        const char* field_group_name,
                                                        rdc_field_grp_t* rdc_field_group_id) {
  if (!field_group_name || !rdc_field_group_id || !field_ids) {
    return RDC_ST_BAD_PARAMETER;
  }

  // Check the field is valid or not
  if (num_field_ids <= RDC_MAX_FIELD_IDS_PER_FIELD_GROUP) {
    for (uint32_t i = 0; i < num_field_ids; i++) {
      if (!is_field_valid(field_ids[i])) {
        RDC_LOG(RDC_INFO, "Fail to create field group with unknown field id " << field_ids[i]);
        return RDC_ST_NOT_SUPPORTED;
      }
    }
  } else {
    return RDC_ST_MAX_LIMIT;
  }

  return group_settings_->rdc_group_field_create(num_field_ids, field_ids, field_group_name,
                                                 rdc_field_group_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_field_get_info(
    rdc_field_grp_t rdc_field_group_id, rdc_field_group_info_t* field_group_info) {
  if (!field_group_info) {
    return RDC_ST_BAD_PARAMETER;
  }
  return group_settings_->rdc_group_field_get_info(rdc_field_group_id, field_group_info);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_get_info(rdc_gpu_group_t p_rdc_group_id,
                                                        rdc_group_info_t* p_rdc_group_info) {
  if (!p_rdc_group_info) {
    return RDC_ST_BAD_PARAMETER;
  }

  return group_settings_->rdc_group_gpu_get_info(p_rdc_group_id, p_rdc_group_info);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_get_all_ids(rdc_gpu_group_t group_id_list[],
                                                       uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }
  return group_settings_->rdc_group_get_all_ids(group_id_list, count);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_field_get_all_ids(rdc_field_grp_t field_group_id_list[],
                                                             uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }
  return group_settings_->rdc_group_field_get_all_ids(field_group_id_list, count);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_destroy(rdc_gpu_group_t p_rdc_group_id) {
  return group_settings_->rdc_group_gpu_destroy(p_rdc_group_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_field_destroy(rdc_field_grp_t rdc_field_group_id) {
  return group_settings_->rdc_group_field_destroy(rdc_field_group_id);
}

// Field API
rdc_status_t RdcEmbeddedHandler::rdc_field_watch(rdc_gpu_group_t group_id,
                                                 rdc_field_grp_t field_group_id,
                                                 uint64_t update_freq, double max_keep_age,
                                                 uint32_t max_keep_samples) {
  return watch_table_->rdc_field_watch(group_id, field_group_id, update_freq, max_keep_age,
                                       max_keep_samples);
}

rdc_status_t RdcEmbeddedHandler::rdc_field_get_latest_value(uint32_t gpu_index, rdc_field_t field,
                                                            rdc_field_value* value) {
  if (!value) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (!is_field_valid(field)) {
    RDC_LOG(RDC_INFO, "Fail to get latest value with unknown field id " << field);
    return RDC_ST_NOT_SUPPORTED;
  }
  return cache_mgr_->rdc_field_get_latest_value(gpu_index, field, value);
}

rdc_status_t RdcEmbeddedHandler::rdc_field_get_value_since(uint32_t gpu_index, rdc_field_t field,
                                                           uint64_t since_time_stamp,
                                                           uint64_t* next_since_time_stamp,
                                                           rdc_field_value* value) {
  if (!next_since_time_stamp || !value) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (!is_field_valid(field)) {
    RDC_LOG(RDC_INFO, "Fail to get value since with unknown field id " << field);
    return RDC_ST_NOT_SUPPORTED;
  }
  return cache_mgr_->rdc_field_get_value_since(gpu_index, field, since_time_stamp,
                                               next_since_time_stamp, value);
}

rdc_status_t RdcEmbeddedHandler::rdc_field_unwatch(rdc_gpu_group_t group_id,
                                                   rdc_field_grp_t field_group_id) {
  return watch_table_->rdc_field_unwatch(group_id, field_group_id);
}

// Diagnostic API
rdc_status_t RdcEmbeddedHandler::rdc_diagnostic_run(rdc_gpu_group_t group_id,
                                                    rdc_diag_level_t level, const char* config,
                                                    size_t config_size,
                                                    rdc_diag_response_t* response,
                                                    rdc_diag_callback_t* callback) {
  if (!response) {
    return RDC_ST_BAD_PARAMETER;
  }

  // Get GPU group information
  rdc_group_info_t rdc_group_info;
  rdc_status_t status = rdc_group_gpu_get_info(group_id, &rdc_group_info);
  if (status != RDC_ST_OK) return status;

  auto diag = rdc_module_mgr_->get_diagnostic_module();
  return diag->rdc_diagnostic_run(rdc_group_info, level, config, config_size, response, callback);
}

rdc_status_t RdcEmbeddedHandler::rdc_test_case_run(rdc_gpu_group_t group_id,
                                                   rdc_diag_test_cases_t test_case,
                                                   const char* config, size_t config_size,
                                                   rdc_diag_test_result_t* result,
                                                   rdc_diag_callback_t* callback) {
  if (!result) {
    return RDC_ST_BAD_PARAMETER;
  }
  // Get GPU group information
  rdc_group_info_t rdc_group_info;
  rdc_status_t status = rdc_group_gpu_get_info(group_id, &rdc_group_info);
  if (status != RDC_ST_OK) return status;

  auto diag = rdc_module_mgr_->get_diagnostic_module();
  return diag->rdc_test_case_run(test_case, rdc_group_info.entity_ids, rdc_group_info.count, config,
                                 config_size, result, callback);
}

// Control API
rdc_status_t RdcEmbeddedHandler::rdc_field_update_all(uint32_t wait_for_update) {
  if (wait_for_update == 1) {
    return watch_table_->rdc_field_update_all();
  }

  //  Async update the field and return immediately.
  updater_ = std::async(std::launch::async, [this]() { watch_table_->rdc_field_update_all(); });

  return RDC_ST_OK;
}

// It is just a client interface under the GRPC framework and is not used as an RDC API.
// Just write an empty function to solve compilation errors
rdc_status_t RdcEmbeddedHandler::get_mixed_component_version(
    mixed_component_t component, mixed_component_version_t* p_mixed_compv) {
  (void)(component);
  (void)(p_mixed_compv);
  return RDC_ST_OK;
}

// Policy API
rdc_status_t RdcEmbeddedHandler::rdc_policy_set(rdc_gpu_group_t group_id, rdc_policy_t policy) {
  return policy_->rdc_policy_set(group_id, policy);
}

rdc_status_t RdcEmbeddedHandler::rdc_policy_get(rdc_gpu_group_t group_id, uint32_t* count,
                                                rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) {
  if (count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  return policy_->rdc_policy_get(group_id, count, policies);
}

rdc_status_t RdcEmbeddedHandler::rdc_policy_delete(rdc_gpu_group_t group_id,
                                                   rdc_policy_condition_type_t condition_type) {
  return policy_->rdc_policy_delete(group_id, condition_type);
}

rdc_status_t RdcEmbeddedHandler::rdc_policy_register(rdc_gpu_group_t group_id,
                                                     rdc_policy_register_callback callback) {
  return policy_->rdc_policy_register(group_id, callback);
}

rdc_status_t RdcEmbeddedHandler::rdc_policy_unregister(rdc_gpu_group_t group_id) {
  return policy_->rdc_policy_unregister(group_id);
}

// Health API
rdc_status_t RdcEmbeddedHandler::rdc_health_set(rdc_gpu_group_t group_id, unsigned int components) {
  if (0 == components) {
    return RDC_ST_BAD_PARAMETER;
  }

  return watch_table_->rdc_health_set(group_id, components);
}

rdc_status_t RdcEmbeddedHandler::rdc_health_get(rdc_gpu_group_t group_id,
                                                unsigned int* components) {
  if (components == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  return watch_table_->rdc_health_get(group_id, components);
}

rdc_status_t RdcEmbeddedHandler::rdc_health_check(rdc_gpu_group_t group_id,
                                                  rdc_health_response_t* response) {
  if (response == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  return watch_table_->rdc_health_check(group_id, response);
}

rdc_status_t RdcEmbeddedHandler::rdc_health_clear(rdc_gpu_group_t group_id) {
  return watch_table_->rdc_health_clear(group_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_device_topology_get(uint32_t gpu_index,
                                                         rdc_device_topology_t* results) {
  return topologylink_->rdc_device_topology_get(gpu_index, results);
}

rdc_status_t RdcEmbeddedHandler::rdc_link_status_get(rdc_link_status_t* results) {
  return topologylink_->rdc_link_status_get(results);
}

// Set one configure
rdc_status_t RdcEmbeddedHandler::rdc_config_set(rdc_gpu_group_t group_id,
                                                rdc_config_setting_t setting) {
  return config_handler_->rdc_config_set(group_id, setting);
}

// Get the setting
rdc_status_t RdcEmbeddedHandler::rdc_config_get(rdc_gpu_group_t group_id,
                                                rdc_config_setting_list_t* settings) {
  return config_handler_->rdc_config_get(group_id, settings);
}

// Clear the setting
rdc_status_t RdcEmbeddedHandler::rdc_config_clear(rdc_gpu_group_t group_id) {
  return config_handler_->rdc_config_clear(group_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_get_num_partition(uint32_t index, uint16_t* num_partition) {
  return partition_->rdc_get_num_partition_impl(index, num_partition);
}

rdc_status_t RdcEmbeddedHandler::rdc_instance_profile_get(
    uint32_t entity_index, rdc_instance_resource_type_t resource_type,
    rdc_resource_profile_t* profile) {
  return partition_->rdc_instance_profile_get_impl(entity_index, resource_type, profile);
}
}  // namespace rdc
}  // namespace amd
