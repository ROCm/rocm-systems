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

#include "rdc_lib/impl/RdcWatchTableImpl.h"

#include <sys/time.h>

#include <algorithm>
#include <ctime>
#include <map>
#include <sstream>
#include <unordered_map>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcWatchTableImpl::RdcWatchTableImpl(const RdcGroupSettingsPtr& group_settings,
                                     const RdcCacheManagerPtr& cache_mgr,
                                     const RdcMetricFetcherPtr& metric_fetcher,
                                     const RdcModuleMgrPtr& module_mgr,
                                     const RdcNotificationPtr& notif)
    : group_settings_(group_settings),
      cache_mgr_(cache_mgr),
      metric_fetcher_(metric_fetcher),
      rdc_module_mgr_(module_mgr),
      notifications_(notif),
      last_cleanup_time_(0) {}

rdc_status_t RdcWatchTableImpl::rdc_job_start_stats(rdc_gpu_group_t group_id, const char job_id[64],
                                                    uint64_t update_freq,
                                                    const rdc_gpu_gauges_t& gpu_gauges) {
  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    if (job_watch_table_.find(job_id) != job_watch_table_.end()) {
      return RDC_ST_ALREADY_EXIST;
    }
  } while (0);

  std::vector<RdcFieldKey> fields_in_watch;
  rdc_status_t result = get_fields_from_group(group_id, JOB_FIELD_ID, fields_in_watch);
  if (result != RDC_ST_OK) {
    return result;
  }
  if (fields_in_watch.size() == 0) {
    RDC_LOG(RDC_ERROR, "Fail to start job " << job_id << ". The group " << group_id
                                            << " must contain at least one GPU.");
    return RDC_ST_NOT_FOUND;
  }

  // Add process start/stop for all jobs
  rdc_group_info_t ginfo;
  result = group_settings_->rdc_group_gpu_get_info(group_id, &ginfo);
  if (result != RDC_ST_OK) {
    return result;
  }

  for (uint32_t ix = 0; ix < ginfo.count; ++ix) {
    uint32_t gpu = ginfo.entity_ids[ix];
    fields_in_watch.emplace_back(gpu, RDC_EVNT_NOTIF_PROCESS_START);
    fields_in_watch.emplace_back(gpu, RDC_EVNT_NOTIF_PROCESS_END);
  }

  JobWatchTableEntry jentry{group_id, fields_in_watch};
  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    job_watch_table_.insert({job_id, jentry});
  } while (0);

  rdc_field_group_info_t finfo;

  result = group_settings_->rdc_group_field_get_info(JOB_FIELD_ID, &finfo);
  if (result != RDC_ST_OK) {
    return result;
  }

  result = cache_mgr_->rdc_job_start_stats(job_id, ginfo, finfo, gpu_gauges);
  if (result != RDC_ST_OK) {
    return result;
  }

  // At last, when every thing sets up, starts to watch the fields.
  result = rdc_field_watch(group_id, JOB_FIELD_ID, update_freq, 0, 0);
  return result;
}

rdc_status_t RdcWatchTableImpl::rdc_job_stop_stats(const char job_id[64],
                                                   const rdc_gpu_gauges_t& gpu_gauge) {
  uint32_t job_group_id;
  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    auto job = job_watch_table_.find(job_id);
    if (job == job_watch_table_.end()) {
      return RDC_ST_NOT_FOUND;
    }
    job_group_id = job->second.group_id;
  } while (0);

  rdc_status_t result = rdc_field_unwatch(job_group_id, JOB_FIELD_ID);
  if (result != RDC_ST_OK) {
    return result;
  }

  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    job_watch_table_.erase(job_id);
  } while (0);

  result = cache_mgr_->rdc_job_stop_stats(job_id, gpu_gauge);

  return result;
}

rdc_status_t RdcWatchTableImpl::rdc_job_remove(const char job_id[64]) {
  rdc_gpu_gauges_t gpu_gauge;
  rdc_job_stop_stats(job_id, gpu_gauge);
  return cache_mgr_->rdc_job_remove(job_id);
}

rdc_status_t RdcWatchTableImpl::rdc_job_remove_all() {
  // Get all the job ids;
  std::vector<std::string> v;
  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    for (auto ite = job_watch_table_.begin(); ite != job_watch_table_.end(); ite++) {
      v.push_back(ite->first);
    }
  } while (0);

  // Stop them
  for (auto job = v.begin(); job != v.end(); job++) {
    rdc_gpu_gauges_t gpu_gauge;
    rdc_job_stop_stats(const_cast<char*>(job->c_str()), gpu_gauge);
  }

  return cache_mgr_->rdc_job_remove_all();
}

rdc_status_t RdcWatchTableImpl::get_fields_from_group(rdc_gpu_group_t group_id,
                                                      rdc_field_grp_t field_group_id,
                                                      std::vector<RdcFieldKey>& fields) {
  rdc_field_group_info_t finfo;
  rdc_group_info_t ginfo;
  rdc_status_t result = group_settings_->rdc_group_gpu_get_info(group_id, &ginfo);
  if (result != RDC_ST_OK) {
    return result;
  }

  result = group_settings_->rdc_group_field_get_info(field_group_id, &finfo);
  if (result != RDC_ST_OK) {
    return result;
  }

  for (uint32_t i = 0; i < ginfo.count; i++) {    // GPUs
    for (uint32_t j = 0; j < finfo.count; j++) {  // Fields
      fields.push_back(RdcFieldKey({ginfo.entity_ids[i], finfo.field_ids[j]}));
    }
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::rdc_field_watch(rdc_gpu_group_t group_id,
                                                rdc_field_grp_t field_group_id,
                                                uint64_t update_freq, double max_keep_age,
                                                uint32_t max_keep_samples) {
  std::lock_guard<std::mutex> guard(watch_mutex_);
  RdcFieldGroupKey gkey({group_id, field_group_id});
  auto table_iter = watch_table_.find(gkey);

  // Already in the watch table
  if (table_iter != watch_table_.end()) {
    if (table_iter->second.is_watching) {
      return RDC_ST_CONFLICT;
    } else {  // delete to overwrite
      watch_table_.erase(table_iter);
    }
  }

  // The field settings for this watch
  FieldSettings f;
  f.update_freq = update_freq;
  f.max_keep_age = max_keep_age;
  f.max_keep_samples = max_keep_samples;
  f.last_update_time = 0;
  f.is_watching = true;

  // Get individual fields for the watch
  std::vector<RdcFieldKey> fields_in_watch;
  rdc_status_t result = get_fields_from_group(group_id, field_group_id, fields_in_watch);
  if (result != RDC_ST_OK) {
    return result;
  }

  // Check for rocprof fields in partitions
  rdc_group_info_t ginfo;
  result = group_settings_->rdc_group_gpu_get_info(group_id, &ginfo);
  if (result != RDC_ST_OK) {
    return result;
  }
  bool groupHasPartition = false;
  for (unsigned int i = 0; i < ginfo.count; i++) {
    uint32_t entityId = ginfo.entity_ids[i];
    rdc_entity_info_t info = rdc_get_info_from_entity_index(entityId);
    if (info.entity_role == RDC_DEVICE_ROLE_PARTITION_INSTANCE) {
      groupHasPartition = true;
      break;
    }
  }

  rdc_field_group_info_t field_info;
  result = group_settings_->rdc_group_field_get_info(field_group_id, &field_info);
  if (result != RDC_ST_OK) {
    return result;
  }
  bool groupHasRocprof = false;
  if (result == RDC_ST_OK) {
    for (unsigned int i = 0; i < field_info.count; i++) {
      rdc_field_t fid = field_info.field_ids[i];
      if (fid >= 800 && fid < 900) {  // Rocprof fields in the 800's
        groupHasRocprof = true;
        break;
      }
    }
  }

  if (groupHasPartition && groupHasRocprof) {
    return RDC_ST_NOT_SUPPORTED;
  }

  // See if any of the fields are notification fields, and
  // set them up, if so.
  result = notifications_->set_listen_events(fields_in_watch);
  if (result != RDC_ST_OK) {
    RDC_LOG(RDC_DEBUG, "Error in configuring for event notification. Return " << result);
  }
  // Skip not supported fields
  uint32_t unsupported_fields = 0;
  auto rdc_telemetry = rdc_module_mgr_->get_telemetry_module();
  if (rdc_telemetry) {
    uint32_t field_ids[MAX_NUM_FIELDS];
    uint32_t field_count;
    result = rdc_telemetry->rdc_telemetry_fields_query(field_ids, &field_count);
    if (result == RDC_ST_OK) {
      RDC_LOG(RDC_DEBUG, "The system support " << field_count << " fields");
      for (auto it = fields_in_watch.begin(); it != fields_in_watch.end();) {
        bool not_supported = true;
        for (uint32_t fi = 0; fi < field_count; fi++) {
          if (field_ids[fi] == it->second) {
            not_supported = false;
            break;
          }
        }
        if (not_supported) {
          if (!notifications_->is_notification_event(it->second)) {
            unsupported_fields++;
          }
          it = fields_in_watch.erase(it);
        } else {
          it++;
        }
      }  // end for
    }    // end if
  }
  if (unsupported_fields > 0) {
    RDC_LOG(RDC_DEBUG, "Skip watch " << unsupported_fields << " fields as they are not supported.");
  }

  // Update the fields_to_watch_
  auto f_in_watch_iter = fields_in_watch.begin();

  for (; f_in_watch_iter != fields_in_watch.end(); f_in_watch_iter++) {
    auto ite = fields_to_watch_.find(*f_in_watch_iter);
    if (ite == fields_to_watch_.end()) {  // A new field
      fields_to_watch_.insert({*f_in_watch_iter, f});
    } else {  // Merge the settings
      auto& f_in_table = ite->second;
      f_in_table.max_keep_age = std::max(f_in_table.max_keep_age, max_keep_age);
      f_in_table.max_keep_samples = std::max(f_in_table.max_keep_samples, max_keep_samples);
      if (f_in_table.is_watching) {  // Already watching
        f_in_table.update_freq = std::min(f_in_table.update_freq, update_freq);
      } else {  // Not watching before
        f_in_table.is_watching = true;
        f_in_table.update_freq = update_freq;
      }
    }
  }

  // Add to the watch table
  watch_table_.insert({gkey, f});

  // Notify the telemetry_module to watch those fields
  if (rdc_telemetry) {
    std::vector<rdc_gpu_field_t> fields;
    auto fields_to_watch_iter = fields_to_watch_.begin();
    for (; fields_to_watch_iter != fields_to_watch_.end(); fields_to_watch_iter++) {
      if (fields_to_watch_iter->second.is_watching) {
        fields.push_back({fields_to_watch_iter->first.first, fields_to_watch_iter->first.second});
      }
    }
    rdc_telemetry->rdc_telemetry_fields_watch(&fields[0], fields.size());
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::update_field_in_table_when_unwatch(const RdcFieldGroupKey& entry) {
  // Get individual fields for this unwatch
  std::vector<RdcFieldKey> fields;
  rdc_status_t result = get_fields_from_group(entry.first, entry.second, fields);
  if (result != RDC_ST_OK) {
    return result;
  }

  // Unwatch will only impact the update_freq, but not the max_keep_age
  // and max_keep_samples. Walk through  watch_table_ to get new update
  // frequency for all fields and store it in update_frequencies
  std::map<RdcFieldKey, uint64_t> update_frequencies;
  auto w_iter = watch_table_.begin();
  for (; w_iter != watch_table_.end(); w_iter++) {
    // Skip the table is not in watching status
    if (w_iter->second.is_watching == false) {
      continue;
    }

    // Get all fields in current table
    std::vector<RdcFieldKey> watch_fields;
    result = get_fields_from_group(w_iter->first.first, w_iter->first.second, watch_fields);
    if (result != RDC_ST_OK) {
      return result;
    }

    // Get the update_freq
    auto fields_in_table_iter = watch_fields.begin();
    for (; fields_in_table_iter != watch_fields.end(); fields_in_table_iter++) {
      auto f_in_freq_iter = update_frequencies.find(*fields_in_table_iter);
      if (f_in_freq_iter == update_frequencies.end()) {
        update_frequencies.insert({*fields_in_table_iter, w_iter->second.update_freq});
      } else {
        f_in_freq_iter->second = std::min(f_in_freq_iter->second, w_iter->second.update_freq);
      }
    }
  }

  // Update the fields that impacted by this unwatch
  auto fite = fields.begin();
  std::vector<rdc_gpu_field_t> unwatch_fields;
  for (; fite != fields.end(); fite++) {
    // Turn off any notification fields
    if (notifications_->is_notification_event(fite->second)) {
      notifications_->stop_listening(fite->first);
      continue;
    }

    auto f_in_table = fields_to_watch_.find((*fite));
    if (f_in_table == fields_to_watch_.end()) {  // Not in fields_to_watch_
      unwatch_fields.push_back({fite->first, fite->second});
      continue;
    }

    auto freq_iter = update_frequencies.find(*fite);
    if (freq_iter == update_frequencies.end()) {
      f_in_table->second.is_watching = false;
      unwatch_fields.push_back({fite->first, fite->second});
    } else {
      f_in_table->second.update_freq = freq_iter->second;
    }
  }

  // Notify the telemetry_module to unwatch those fields
  auto rdc_telemetry = rdc_module_mgr_->get_telemetry_module();
  if (rdc_telemetry) {
    rdc_telemetry->rdc_telemetry_fields_unwatch(&unwatch_fields[0], unwatch_fields.size());
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::rdc_field_unwatch(rdc_gpu_group_t group_id,
                                                  rdc_field_grp_t field_group_id) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

  std::lock_guard<std::mutex> guard(watch_mutex_);
  // Set is_watching = false
  auto ite = watch_table_.find(RdcFieldGroupKey({group_id, field_group_id}));
  if (ite == watch_table_.end()) {
    return RDC_ST_NOT_FOUND;
  }
  ite->second.is_watching = false;
  ite->second.last_update_time = now;

  // Update the fields_to_watch_
  return update_field_in_table_when_unwatch(ite->first);
}

rdc_status_t RdcWatchTableImpl::create_health_field_group(unsigned int components,
                                                          rdc_field_grp_t* field_group_id) {
  // set filed ids
  std::vector<rdc_field_t> field_ids{};
  if (components & RDC_HEALTH_WATCH_PCIE) {
    field_ids.push_back(RDC_HEALTH_PCIE_REPLAY_COUNT);
  }

  if (components & RDC_HEALTH_WATCH_XGMI) {
    field_ids.push_back(RDC_HEALTH_XGMI_ERROR);
  }

  if (components & RDC_HEALTH_WATCH_MEM) {
    field_ids.push_back(RDC_FI_ECC_UNCORRECT_TOTAL);
    field_ids.push_back(RDC_HEALTH_RETIRED_PAGE_NUM);
    field_ids.push_back(RDC_HEALTH_PENDING_PAGE_NUM);
    field_ids.push_back(RDC_HEALTH_RETIRED_PAGE_LIMIT);
  }

  if (components & RDC_HEALTH_WATCH_EEPROM) {
    field_ids.push_back(RDC_HEALTH_EEPROM_CONFIG_VALID);
  }

  if (components & RDC_HEALTH_WATCH_THERMAL) {
    field_ids.push_back(RDC_HEALTH_THERMAL_THROTTLE_TIME);
  }

  if (components & RDC_HEALTH_WATCH_POWER) {
    field_ids.push_back(RDC_HEALTH_POWER_THROTTLE_TIME);
  }

  if (0 == field_ids.size()) {
    RDC_LOG(RDC_ERROR, "Fail to health set. The components must contain at least one watch.");
    return RDC_ST_BAD_PARAMETER;
  }

  const std::string field_group_name("health-field-group");
  return group_settings_->rdc_group_field_create(field_ids.size(), field_ids.data(),
                                                 field_group_name.c_str(), field_group_id);
}

rdc_status_t RdcWatchTableImpl::rdc_health_set(rdc_gpu_group_t group_id, unsigned int components) {
  // remove old health for same group_id
  rdc_health_clear(group_id);

  // create a field group base on the components
  rdc_field_grp_t field_group_id;
  rdc_status_t result = create_health_field_group(components, &field_group_id);
  if (result != RDC_ST_OK) {
    return result;
  }

  // get field key
  std::vector<RdcFieldKey> fields_in_watch;
  result = get_fields_from_group(group_id, field_group_id, fields_in_watch);
  if (result != RDC_ST_OK) {
    return result;
  }

  // add to the health watch table
  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    HealthWatchTableEntry hentry{components, field_group_id, fields_in_watch};
    health_watch_table_.insert({group_id, hentry});
  } while (0);

  for (auto fields = fields_in_watch.begin(); fields != fields_in_watch.end(); fields++) {
    // get initial values
    rdc_field_value value;
    result = metric_fetcher_->fetch_smi_field(fields->first, fields->second, &value);
    if (result != RDC_ST_OK) break;

    // set initial values to cache
    result = cache_mgr_->rdc_health_set(group_id, fields->first, value);
    if (result != RDC_ST_OK) break;
  }

  // Start to watch the fields and update fields per 1 second.
  result = rdc_field_watch(group_id, field_group_id, 1000000, 0, 0);
  return result;
}

rdc_status_t RdcWatchTableImpl::rdc_health_get(rdc_gpu_group_t group_id, unsigned int* components) {
  if (nullptr == components) return RDC_ST_BAD_PARAMETER;

  std::lock_guard<std::mutex> guard(watch_mutex_);
  auto table_iter = health_watch_table_.find(group_id);

  // already in the health watch table
  if (table_iter != health_watch_table_.end())
    *components = table_iter->second.components;
  else
    *components = 0;

  return RDC_ST_OK;
}

bool RdcWatchTableImpl::add_health_incident(uint32_t gpu_index, rdc_health_system_t component,
                                            rdc_health_result_t health, uint32_t err_code,
                                            std::string err_msg, rdc_health_incidents_t* incident,
                                            rdc_health_response_t* response) {
  bool result = false;

  incident->gpu_index = gpu_index;
  incident->component = component;
  incident->health = health;
  incident->error.code = err_code;
  strncpy_with_null(incident->error.msg, err_msg.c_str(), MAX_HEALTH_MSG_LENGTH);

  if (incident->health > response->overall_health) response->overall_health = incident->health;
  response->incidents_count++;
  if (response->incidents_count >= HEALTH_MAX_ERROR_ITEMS) {
    RDC_LOG(RDC_INFO, "Health incidents are full!");
    result = true;
  }

  return (result);
}

rdc_status_t RdcWatchTableImpl::get_start_end_values(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                                     rdc_field_t field, uint64_t start_timestamp,
                                                     rdc_field_value* start_value,
                                                     rdc_field_value* end_value) {
  if ((nullptr == start_value) && (nullptr == end_value)) return RDC_ST_BAD_PARAMETER;

  rdc_status_t result = RDC_ST_OK;
  if (nullptr != start_value) {
    // get the values of the field at the start_timestamp/end_timestampe
    result = cache_mgr_->rdc_health_get_values(group_id, gpu_index, field, start_timestamp, 0,
                                               start_value, nullptr);
    if (result != RDC_ST_OK) {
      RDC_LOG(RDC_ERROR, "Error get gpu: " << gpu_index << " field: " << field
                                           << " history data. Return: " << result);
      return result;
    }
  }

  // get end values
  result = metric_fetcher_->fetch_smi_field(gpu_index, field, end_value);
  if (result != RDC_ST_OK)
    RDC_LOG(RDC_ERROR, "Error get gpu: " << gpu_index << " field: " << field
                                         << " current data. Return: " << result);

  return result;
}

rdc_status_t RdcWatchTableImpl::pcie_check(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                           rdc_health_response_t* response) {
  // get field start/end values
  rdc_field_value start = {}, end = {};
  uint64_t start_timestamp = static_cast<uint64_t>(time(nullptr) - 60) * 1000;
  // get the history data last 1 minute
  rdc_status_t result = get_start_end_values(group_id, gpu_index, RDC_HEALTH_PCIE_REPLAY_COUNT,
                                             start_timestamp, &start, &end);
  if (result != RDC_ST_OK) return result;

  uint64_t pcie_replay_count = end.value.l_int - start.value.l_int;
  if (pcie_replay_count > PCIE_MAX_REPLAYS_PERMIN) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    std::string err_msg = "Detected ";
    err_msg += std::to_string(pcie_replay_count);
    err_msg += " PCIe replays per minute exceeding the max limit ";
    err_msg += std::to_string(PCIE_MAX_REPLAYS_PERMIN);
    err_msg += ".";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_PCIE, RDC_HEALTH_RESULT_WARN,
                            RDC_FR_PCI_REPLAY_RATE, err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::xgmi_check(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                           rdc_health_response_t* response) {
  // get field start/end values
  rdc_field_value end = {};
  rdc_status_t result =
      get_start_end_values(group_id, gpu_index, RDC_HEALTH_XGMI_ERROR, 0, nullptr, &end);
  if (result != RDC_ST_OK) return result;

  amdsmi_xgmi_status_t status = static_cast<amdsmi_xgmi_status_t>(end.value.l_int);
  if (AMDSMI_XGMI_STATUS_NO_ERRORS != status) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    uint32_t err_code;
    std::string err_msg = "Detected ";
    if (AMDSMI_XGMI_STATUS_ERROR == status) {
      err_msg += " a single XGMI error";
      err_code = RDC_FR_XGMI_SINGLE_ERROR;
    } else {
      err_msg += " multiple XGMI errors";
      err_code = RDC_FR_XGMI_MULTIPLE_ERROR;
    }
    err_msg += ".";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_XGMI, RDC_HEALTH_RESULT_FAIL, err_code,
                            err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::memory_check(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                             rdc_health_response_t* response) {
  // get field start/end values
  rdc_field_value start = {}, end = {};
  rdc_status_t result =
      get_start_end_values(group_id, gpu_index, RDC_FI_ECC_UNCORRECT_TOTAL, 0, nullptr, &end);
  if (result != RDC_ST_OK) return result;

  uint64_t ecc_uncorrectable_count = 0;
  ecc_uncorrectable_count = end.value.l_int;
  if (ecc_uncorrectable_count > 0) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    std::string err_msg = "Detected ";
    err_msg += std::to_string(ecc_uncorrectable_count);
    err_msg += " uncorrectable ECC error(s) since last GPU reset.";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_MEM, RDC_HEALTH_RESULT_FAIL,
                            RDC_FR_ECC_UNCORRECTABLE_DETECTED, err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;
  }

  result = get_start_end_values(group_id, gpu_index, RDC_HEALTH_PENDING_PAGE_NUM, 0, nullptr, &end);
  if (result != RDC_ST_OK) return result;

  uint64_t num_pages = end.value.l_int;
  if (num_pages > 0) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    std::string err_msg = "Detected ";
    err_msg += std::to_string(num_pages);
    err_msg += " pending retired page(s).";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_MEM, RDC_HEALTH_RESULT_WARN,
                            RDC_FR_PENDING_PAGE_RETIREMENTS, err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;
  }

  // get retired page number
  result = get_start_end_values(group_id, gpu_index, RDC_HEALTH_RETIRED_PAGE_NUM, 0, nullptr, &end);
  if (result != RDC_ST_OK) return result;
  uint64_t retired_page = end.value.l_int;

  // get retired page threshold
  result =
      get_start_end_values(group_id, gpu_index, RDC_HEALTH_RETIRED_PAGE_LIMIT, 0, nullptr, &end);
  if (result != RDC_ST_OK) return result;
  uint32_t retired_page_threshold = end.value.l_int;

  if (retired_page > retired_page_threshold) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    std::string err_msg = "Detected ";
    err_msg += std::to_string(retired_page);
    err_msg += " retired pages exceeding the max limit: ";
    err_msg += std::to_string(retired_page_threshold);
    err_msg += ".";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_MEM, RDC_HEALTH_RESULT_FAIL,
                            RDC_FR_RETIRED_PAGES_LIMIT, err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;

    return RDC_ST_OK;
  }

  if (retired_page > 0) {
    uint64_t start_timestamp = static_cast<uint64_t>(time(nullptr) - 604800) * 1000;
    // get retired page number last 1 week
    result = get_start_end_values(group_id, gpu_index, RDC_HEALTH_RETIRED_PAGE_NUM, start_timestamp,
                                  &start, &end);
    if (result != RDC_ST_OK) return result;

    retired_page = end.value.l_int - start.value.l_int;
    if (retired_page > 1) {
      rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

      std::string err_msg = "Detected ";
      err_msg += std::to_string(retired_page);
      err_msg += " retired pages more than one in the last week.";

      // add incident
      if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_MEM, RDC_HEALTH_RESULT_FAIL,
                              RDC_FR_RETIRED_PAGES_UNCORRECTABLE_LIMIT, err_msg, incident,
                              response))
        return RDC_ST_MAX_LIMIT;
    }
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::eeprom_check(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                             rdc_health_response_t* response) {
  rdc_field_value end = {};
  rdc_status_t result =
      get_start_end_values(group_id, gpu_index, RDC_FI_ECC_UNCORRECT_TOTAL, 0, nullptr, &end);
  if (result != RDC_ST_OK && result != RDC_ST_CORRUPTED_EEPROM) return result;

  if (result == RDC_ST_CORRUPTED_EEPROM) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    std::string err_msg = "Detected a corrupt EEPROM since last GPU reset.";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_EEPROM, RDC_HEALTH_RESULT_WARN,
                            RDC_FR_CORRUPT_EEPROM, err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::thermal_check(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                              rdc_health_response_t* response) {
  // get field start/end values
  rdc_field_value start = {}, end = {};
  uint64_t start_timestamp = static_cast<uint64_t>(time(nullptr) - 60) * 1000;
  // get the history data last 1 minute
  rdc_status_t result = get_start_end_values(group_id, gpu_index, RDC_HEALTH_THERMAL_THROTTLE_TIME,
                                             start_timestamp, &start, &end);
  if (result != RDC_ST_OK) return result;

  uint64_t acc_socket_thrm = end.value.l_int - start.value.l_int;
  if (0 < acc_socket_thrm) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    std::string err_msg = "Detected ";
    err_msg += std::to_string(acc_socket_thrm);
    err_msg += " clock throttling due to thermal violation in the last minute.";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_THERMAL, RDC_HEALTH_RESULT_WARN,
                            RDC_FR_CLOCKS_THROTTLE_THERMAL, err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::power_check(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                            rdc_health_response_t* response) {
  // get field start/end values
  rdc_field_value start = {}, end = {};
  uint64_t start_timestamp = static_cast<uint64_t>(time(nullptr) - 60) * 1000;
  // get the history data last 1 minute
  rdc_status_t result = get_start_end_values(group_id, gpu_index, RDC_HEALTH_POWER_THROTTLE_TIME,
                                             start_timestamp, &start, &end);
  if (result != RDC_ST_OK) return result;

  uint64_t acc_ppt_pwr = end.value.l_int - start.value.l_int;
  if (0 < acc_ppt_pwr) {
    rdc_health_incidents_t* incident = &response->incidents[response->incidents_count];

    std::string err_msg = "Detected ";
    err_msg += std::to_string(acc_ppt_pwr);
    err_msg += " Detected clock throttling due to power violation in the last minute.";

    // add incident
    if (add_health_incident(gpu_index, RDC_HEALTH_WATCH_POWER, RDC_HEALTH_RESULT_WARN,
                            RDC_FR_CLOCKS_THROTTLE_POWER, err_msg, incident, response))
      return RDC_ST_MAX_LIMIT;
  }

  return RDC_ST_OK;
}
rdc_status_t RdcWatchTableImpl::rdc_health_check(rdc_gpu_group_t group_id,
                                                 rdc_health_response_t* response) {
  if (nullptr == response) return RDC_ST_BAD_PARAMETER;

  unsigned int components = 0;
  std::vector<RdcFieldKey> fields_in_watch;
  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    auto health = health_watch_table_.find(group_id);
    if (health == health_watch_table_.end()) return RDC_ST_NOT_FOUND;
    components = health->second.components;
    fields_in_watch = health->second.fields;
  } while (0);

  rdc_group_info_t ginfo;
  rdc_status_t result = group_settings_->rdc_group_gpu_get_info(group_id, &ginfo);
  if (result != RDC_ST_OK) return result;

  for (auto fields = fields_in_watch.begin(); fields != fields_in_watch.end(); fields++) {
    // get current values
    rdc_field_value value;
    result = metric_fetcher_->fetch_smi_field(fields->first, fields->second, &value);
    if (result != RDC_ST_OK) break;

    // set current values to cache
    result = cache_mgr_->rdc_update_health_stats(group_id, fields->first, value);
    if (result != RDC_ST_OK) break;
  }

  // init response
  response->overall_health = RDC_HEALTH_RESULT_PASS;
  response->incidents_count = 0;

  for (uint32_t gindex = 0; gindex < ginfo.count; gindex++) {
    // PCIe
    if (components & RDC_HEALTH_WATCH_PCIE) {
      result = pcie_check(group_id, ginfo.entity_ids[gindex], response);
      if (result == RDC_ST_MAX_LIMIT) return result;
    }

    // XGMI
    if (components & RDC_HEALTH_WATCH_XGMI) {
      result = xgmi_check(group_id, ginfo.entity_ids[gindex], response);
      if (result == RDC_ST_MAX_LIMIT) return result;
    }

    // Memory
    if (components & RDC_HEALTH_WATCH_MEM) {
      result = memory_check(group_id, ginfo.entity_ids[gindex], response);
      if (result == RDC_ST_MAX_LIMIT) return result;
    }

    // EEPROM
    if (components & RDC_HEALTH_WATCH_EEPROM) {
      result = eeprom_check(group_id, ginfo.entity_ids[gindex], response);
      if (result == RDC_ST_MAX_LIMIT) return result;
    }

    // Thermal
    if (components & RDC_HEALTH_WATCH_THERMAL) {
      result = thermal_check(group_id, ginfo.entity_ids[gindex], response);
      if (result == RDC_ST_MAX_LIMIT) return result;
    }

    // Power
    if (components & RDC_HEALTH_WATCH_POWER) {
      result = power_check(group_id, ginfo.entity_ids[gindex], response);
      if (result == RDC_ST_MAX_LIMIT) return result;
    }
  }  // end of for gindex

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::rdc_health_clear(rdc_gpu_group_t group_id) {
  rdc_field_grp_t field_group_id;

  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    auto health = health_watch_table_.find(group_id);
    if (health == health_watch_table_.end()) {
      return RDC_ST_NOT_FOUND;
    }
    field_group_id = health->second.field_group_id;
  } while (0);

  // at first, unwatch the old fields.
  rdc_status_t result = rdc_field_unwatch(group_id, field_group_id);
  if (result != RDC_ST_OK) {
    return result;
  }

  // destroy the old field group
  group_settings_->rdc_group_field_destroy(field_group_id);

  do {  //< lock guard for thread safe
    std::lock_guard<std::mutex> guard(watch_mutex_);
    health_watch_table_.erase(group_id);
  } while (0);

  result = cache_mgr_->rdc_health_clear(group_id);

  return RDC_ST_OK;
}

bool RdcWatchTableImpl::is_job_watch_field(uint32_t gpu_index, rdc_field_t field_id,
                                           std::string& job_id) const {
  RdcFieldKey key{gpu_index, field_id};

  for (auto ite = job_watch_table_.begin(); ite != job_watch_table_.end(); ite++) {
    auto& fields = ite->second.fields;
    if (std::find(fields.begin(), fields.end(), key) != fields.end()) {
      job_id = ite->first;
      return true;
    }
  }

  return false;
}

bool RdcWatchTableImpl::is_health_watch_field(uint32_t gpu_index, rdc_field_t field_id,
                                              rdc_gpu_group_t& group_id) const {
  RdcFieldKey key{gpu_index, field_id};

  for (auto ite = health_watch_table_.begin(); ite != health_watch_table_.end(); ite++) {
    auto& fields = ite->second.fields;
    if (std::find(fields.begin(), fields.end(), key) != fields.end()) {
      group_id = ite->first;
      return true;
    }
  }

  return false;
}

rdc_status_t RdcWatchTableImpl::handle_fields(rdc_gpu_field_value_t* values, uint32_t num_values,
                                              void* user_data) {
  if (values == nullptr || user_data == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  RdcWatchTableImpl* watchTable = static_cast<RdcWatchTableImpl*>(user_data);

  for (uint32_t i = 0; i < num_values; i++) {
    auto gpu_index = values[i].gpu_index;
    auto field_id = values[i].field_value.field_id;

    // Always Update the timestamp
    auto ite = watchTable->fields_to_watch_.find({gpu_index, field_id});
    if (ite != watchTable->fields_to_watch_.end()) {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
      ite->second.last_update_time = now;
    }

    // Only cache valid results
    if (values[i].field_value.status != RDC_ST_OK) {
      continue;
    }

    // Update the cache
    watchTable->cache_mgr_->rdc_update_cache(gpu_index, values[i].field_value);

    // Update the job stats cache
    std::string job_id;
    if (watchTable->is_job_watch_field(gpu_index, field_id, job_id)) {
      watchTable->cache_mgr_->rdc_update_job_stats(gpu_index, job_id, values[i].field_value);
    }

    // Update the health stats cache
    rdc_gpu_group_t group_id;
    if (watchTable->is_health_watch_field(gpu_index, field_id, group_id)) {
      watchTable->cache_mgr_->rdc_update_health_stats(group_id, gpu_index, values[i].field_value);
    }
  }
  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::rdc_field_update_all() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

  // Collect all fields need to be updated for bulk fetch
  std::vector<rdc_gpu_field_t> fields;
  std::lock_guard<std::mutex> guard(watch_mutex_);
  auto fite = fields_to_watch_.begin();
  for (; fite != fields_to_watch_.end(); fite++) {
    // Is this field need to be updated?
    uint64_t track_freq = fite->second.update_freq / 1000;
    if (!fite->second.is_watching || fite->second.last_update_time + track_freq > now) {
      continue;
    }
    fields.push_back({fite->first.first, fite->first.second});
  }

  if (fields.size() != 0) {
    auto rdc_telemetry = rdc_module_mgr_->get_telemetry_module();
    if (rdc_telemetry) {
      rdc_telemetry->rdc_telemetry_fields_value_get(&fields[0], fields.size(),
                                                    RdcWatchTableImpl::handle_fields, this);
    } else {
      RDC_LOG(RDC_ERROR, "RdcWatchTableImpl: Fail to get the telemetry module");
    }
  }

  // Clean up is expensive, only do it once per second
  if (now - last_cleanup_time_ > 1000) {
    clean_up();
    last_cleanup_time_ = now;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::rdc_notif_update_cache(rdc_evnt_notification_t* events,
                                                       uint32_t num_events) {
  if (events == nullptr || num_events == 0) {
    return RDC_ST_BAD_PARAMETER;
  }
  std::lock_guard<std::mutex> guard(watch_mutex_);

  for (uint32_t i = 0; i < num_events; i++) {
    auto gpu_index = events[i].gpu_id;
    auto field_id = events[i].field.field_id;

    // Always Update the timestamp
    auto ite = fields_to_watch_.find({gpu_index, field_id});
    if (ite != fields_to_watch_.end()) {
      ite->second.last_update_time = events[i].field.ts;
    }

    // Only cache valid results
    if (events[i].field.status != RDC_ST_OK) {
      continue;
    }

    // Update the cache
    cache_mgr_->rdc_update_cache(gpu_index, events[i].field);

    // Update the job stats cache
    std::string job_id;
    if (is_job_watch_field(gpu_index, field_id, job_id)) {
      cache_mgr_->rdc_update_job_stats(gpu_index, job_id, events[i].field);
    }

    // Update the health stats cache
    rdc_gpu_group_t group_id;
    if (is_health_watch_field(gpu_index, field_id, group_id)) {
      cache_mgr_->rdc_update_health_stats(group_id, gpu_index, events[i].field);
    }
  }
  return RDC_ST_OK;
}

rdc_status_t RdcWatchTableImpl::rdc_field_listen_notif(uint32_t timeout_ms) {
  rdc_status_t ret;
  rdc_evnt_notification_t events[kMaxRSMIEvents];
  uint32_t num_events = kMaxRSMIEvents;

  ret = notifications_->listen(events, &num_events, timeout_ms);

  // Update cache
  if (ret == RDC_ST_OK && num_events) {
    ret = rdc_notif_update_cache(events, num_events);
  }
  return ret;
}

void RdcWatchTableImpl::clean_up() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

  // Clean the cache and the fields_to_watch_ table
  auto fite = fields_to_watch_.begin();
  while (fite != fields_to_watch_.end()) {
    cache_mgr_->evict_cache(fite->first.first, fite->first.second, fite->second.max_keep_samples,
                            fite->second.max_keep_age);
    if (!fite->second.is_watching &&
        fite->second.last_update_time + fite->second.max_keep_age * 1000 < now) {
      fite = fields_to_watch_.erase(fite);
    } else {
      ++fite;
    }
  }

  // Clean the watch table
  auto wite = watch_table_.begin();
  while (wite != watch_table_.end()) {
    if (!wite->second.is_watching &&
        wite->second.last_update_time + wite->second.max_keep_age * 1000 < now) {
      wite = watch_table_.erase(wite);
    } else {
      ++wite;
    }
  }

  // Debug log every 30 seconds
  if (now / 1000 % 30 == 0) {
    debug_status();
  }
}

void RdcWatchTableImpl::debug_status() {
  RDC_LOG(RDC_DEBUG, "fields_to_watch_:" << fields_to_watch_.size()
                                         << " watch_table_:" << watch_table_.size()
                                         << " job_watch_table_:" << job_watch_table_.size()
                                         << " health_watch_table_:" << health_watch_table_.size()
                                         << " cache stats:" << cache_mgr_->get_cache_stats());

  if (watch_table_.size() > 0) {
    RDC_LOG(RDC_DEBUG, "watch table details:");
  }
  for (auto wite = watch_table_.begin(); wite != watch_table_.end(); wite++) {
    RDC_LOG(RDC_DEBUG, wite->first.first << "," << wite->first.second
                                         << ": age:" << wite->second.max_keep_age
                                         << ", samples:" << wite->second.max_keep_samples
                                         << ", is_watching:" << wite->second.is_watching
                                         << ", last_update_time:" << wite->second.last_update_time
                                         << ", update_freq:" << wite->second.update_freq);
  }

  if (job_watch_table_.size() > 0) {
    RDC_LOG(RDC_DEBUG, "job watch table details: ");
  }
  for (auto jite = job_watch_table_.begin(); jite != job_watch_table_.end(); jite++) {
    std::stringstream strstream;
    for (const auto& p : jite->second.fields) {
      strstream << "<" << p.first << "," << p.second << "> ";
    }
    RDC_LOG(RDC_DEBUG,
            jite->first << ": " << jite->second.group_id << " fields : " << strstream.str());
  }

  if (health_watch_table_.size() > 0) {
    RDC_LOG(RDC_DEBUG, "health watch table details: ");
  }
  for (auto hite = health_watch_table_.begin(); hite != health_watch_table_.end(); hite++) {
    std::stringstream strstream;
    for (const auto& p : hite->second.fields) {
      strstream << "<" << p.first << "," << p.second << "> ";
    }
    RDC_LOG(RDC_DEBUG, "group id : " << hite->first << " components : " << hite->second.components
                                     << " fields : " << strstream.str());
  }

  if (fields_to_watch_.size() > 0) {
    RDC_LOG(RDC_DEBUG, "fields to watch details:");
  }
  for (auto fite = fields_to_watch_.begin(); fite != fields_to_watch_.end(); fite++) {
    RDC_LOG(RDC_DEBUG, fite->first.first << "," << fite->first.second
                                         << ": age:" << fite->second.max_keep_age
                                         << ", samples:" << fite->second.max_keep_samples
                                         << ", is_watching:" << fite->second.is_watching
                                         << ", last_update_time:" << fite->second.last_update_time
                                         << ", update_freq:" << fite->second.update_freq);
  }
}

}  // namespace rdc
}  // namespace amd
