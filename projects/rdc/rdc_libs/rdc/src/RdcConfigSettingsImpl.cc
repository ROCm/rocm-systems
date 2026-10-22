/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "rdc_lib/impl/RdcConfigSettingsImpl.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <limits>
#include <type_traits>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

namespace {

constexpr uint64_t kMicrowattsPerWatt = 1'000'000;
constexpr uint64_t kHzPerMHz = 1'000'000;

// The clock that a clock limit setting limits.
amdsmi_clk_type_t clock_of(rdc_config_type_t type) {
  return type == RDC_CFG_MEMORY_CLOCK_LIMIT ? AMDSMI_CLK_TYPE_MEM : AMDSMI_CLK_TYPE_GFX;
}

// Gives one GPU a setting: the power cap, or the upper limit of a clock.
amdsmi_status_t apply_setting(amdsmi_processor_handle processor_handle,
                              const rdc_config_setting_t& setting) {
  if (setting.type == RDC_CFG_POWER_LIMIT) {
    return amdsmi_set_power_cap(processor_handle, 0, setting.target_value * kMicrowattsPerWatt);
  }
  return amdsmi_set_gpu_clk_limit(processor_handle, clock_of(setting.type), CLK_LIMIT_MAX,
                                  setting.target_value);
}

// Puts one GPU's setting of the given type back to its default: the default power
// cap, or the automatic performance level for a clock whose upper limit is not its
// top frequency level.
amdsmi_status_t reset_setting(amdsmi_processor_handle processor_handle, rdc_config_type_t type) {
  if (type == RDC_CFG_POWER_LIMIT) {
    amdsmi_power_cap_info_t cap_info = {};
    amdsmi_status_t status = amdsmi_get_power_cap_info(processor_handle, 0, &cap_info);
    if (status != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl: Failed to get power cap : " << status);
      return status;
    }
    if (cap_info.power_cap == cap_info.default_power_cap) {
      return AMDSMI_STATUS_SUCCESS;
    }
    status = amdsmi_set_power_cap(processor_handle, 0, cap_info.default_power_cap);
    if (status != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl: Failed to reset power cap : " << status);
    }
    return status;
  }

  const amdsmi_clk_type_t clk = clock_of(type);
  amdsmi_frequencies_t freqs = {};
  amdsmi_status_t status = amdsmi_get_clk_freq(processor_handle, clk, &freqs);
  if (status != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl: Failed to get clock " << clk << " freq: " << status);
    return status;
  }
  amdsmi_clk_info_t info = {};
  status = amdsmi_get_clock_info(processor_handle, clk, &info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl: Failed to get clock " << clk << " info: " << status);
    return status;
  }
  // A table with no level, or with more than it holds, has no top level to
  // compare with: reset anyway.
  if (freqs.num_supported > 0 && freqs.num_supported <= AMDSMI_MAX_NUM_FREQUENCIES &&
      info.max_clk * kHzPerMHz == freqs.frequency[freqs.num_supported - 1]) {
    return AMDSMI_STATUS_SUCCESS;
  }
  status = amdsmi_set_gpu_perf_level(processor_handle, AMDSMI_DEV_PERF_LEVEL_AUTO);
  if (status != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR,
            "RdcConfigSettingsImpl: Failed to reset clock " << clk << " limit : " << status);
  }
  return status;
}

// Gives one GPU a cached setting again if the GPU no longer has it. The power cap
// is compared in microwatts, a clock's upper limit in MHz.
void reapply_setting(amdsmi_processor_handle processor_handle,
                     const rdc_config_setting_t& setting) {
  uint64_t current = 0;
  uint64_t target = setting.target_value;
  amdsmi_status_t status;
  if (setting.type == RDC_CFG_POWER_LIMIT) {
    amdsmi_power_cap_info_t cap_info = {};
    status = amdsmi_get_power_cap_info(processor_handle, 0, &cap_info);
    current = cap_info.power_cap;
    target *= kMicrowattsPerWatt;
  } else {
    amdsmi_clk_info_t info = {};
    status = amdsmi_get_clock_info(processor_handle, clock_of(setting.type), &info);
    current = info.max_clk;
  }
  if (status != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl::monitorSettings(): Failed to read setting "
                           << setting.type << " : " << status);
    return;
  }
  if (current == target) {
    return;
  }
  RDC_LOG(RDC_INFO, "RdcConfigSettingsImpl::monitorSettings(): Mismatched setting "
                        << setting.type << ", resetting");
  status = apply_setting(processor_handle, setting);
  if (status != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl::monitorSettings(): Failed to set setting "
                           << setting.type << " : " << status);
  }
}

}  // namespace

RdcConfigSettingsImpl::RdcConfigSettingsImpl(const RdcGroupSettingsPtr& group_settings)
    : group_settings_(group_settings), is_running_(false) {}

// Settings that were never cleared leave the monitoring thread running.
RdcConfigSettingsImpl::~RdcConfigSettingsImpl() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_running_ = false;
  }
  cv_.notify_all();
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }
}

// Monitoring thread of gpu settings
void RdcConfigSettingsImpl::monitorSettings() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (is_running_) {
    // No setting to check: wait for one, or for the destructor, without waking up.
    if (cached_group_settings_.empty()) {
      cv_.wait(lock);
      continue;
    }
    if (cv_.wait_for(lock, monitor_period_, [this] { return !is_running_; })) {
      break;
    }
    for (const auto& group_pair : cached_group_settings_) {
      rdc_group_info_t rdc_group_info;
      if (get_group_info(group_pair.first, &rdc_group_info) != RDC_ST_OK) {
        // Error log handled in get_group_info
        continue;
      }
      for (uint32_t i = 0; i < rdc_group_info.count; ++i) {
        amdsmi_processor_handle processor_handle;
        amdsmi_status_t status =
            get_processor_handle_from_id(rdc_group_info.entity_ids[i], &processor_handle);
        if (status != AMDSMI_STATUS_SUCCESS) {
          RDC_LOG(RDC_ERROR,
                  "RdcConfigSettingsImpl::monitorSettings(): get_processor_handle_from_id failed: "
                      << status);
          continue;
        }
        // Each setting on its own: one that cannot be read or set does not keep the
        // others from being checked.
        for (const auto& setting_pair : group_pair.second) {
          reapply_setting(processor_handle, setting_pair.second);
        }
      }
    }
  }
  RDC_LOG(RDC_INFO, "RdcConfigSettingsImpl Monitoring Thread Stopped");
}

// The group's GPUs. A group table that reports success without filling the info in
// gives an empty group, and a count above what the info holds is an error.
rdc_status_t RdcConfigSettingsImpl::get_group_info(rdc_gpu_group_t group_id,
                                                   rdc_group_info_t* rdc_group_info) {
  *rdc_group_info = {};
  rdc_status_t status = group_settings_->rdc_group_gpu_get_info(group_id, rdc_group_info);
  if (status != RDC_ST_OK) {
    RDC_LOG(RDC_ERROR,
            "RdcConfigSettingsImpl::get_group_info(): rdc_group_gpu_get_info failed : " << status);
    return status;
  }
  if (rdc_group_info->count > RDC_GROUP_MAX_ENTITIES) {
    RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl::get_group_info(): group "
                           << group_id << " reports " << rdc_group_info->count
                           << " GPUs, more than " << RDC_GROUP_MAX_ENTITIES);
    return RDC_ST_UNKNOWN_ERROR;
  }
  return RDC_ST_OK;
}

RdcConfigSettingsImpl::RdcConfigSettingsImpl(const RdcGroupSettingsPtr& group_settings,
                                             std::chrono::milliseconds monitor_period)
    : group_settings_(group_settings), is_running_(false), monitor_period_(monitor_period) {}

// Set configuration setting
rdc_status_t RdcConfigSettingsImpl::rdc_config_set(rdc_gpu_group_t group_id,
                                                   rdc_config_setting_t setting) {
  // A client can send any number as the type. Compare it as an integer: loading a
  // value outside rdc_config_type_t through the enum is undefined.
  std::underlying_type_t<rdc_config_type_t> type;
  std::memcpy(&type, &setting.type, sizeof(type));
  if (type != RDC_CFG_GFX_CLOCK_LIMIT && type != RDC_CFG_MEMORY_CLOCK_LIMIT &&
      type != RDC_CFG_POWER_LIMIT) {
    RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl::rdc_config_set(): unknown setting type " << type);
    return RDC_ST_BAD_PARAMETER;
  }
  // The power cap is set in microwatts, which must fit in 64 bits.
  if (setting.type == RDC_CFG_POWER_LIMIT &&
      setting.target_value > std::numeric_limits<uint64_t>::max() / kMicrowattsPerWatt) {
    RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl::rdc_config_set(): power limit too large: "
                           << setting.target_value);
    return RDC_ST_BAD_PARAMETER;
  }

  // One change at a time, and none while the monitoring thread checks the GPUs.
  std::lock_guard<std::mutex> lock(mutex_);

  // Get the group info for gpu_index list
  rdc_group_info_t rdc_group_info;
  if (get_group_info(group_id, &rdc_group_info) != RDC_ST_OK) {
    return RDC_ST_UNKNOWN_ERROR;
  }
  // Nothing to apply the setting to, so do not cache it as if it succeeded.
  if (rdc_group_info.count == 0) {
    RDC_LOG(RDC_ERROR,
            "RdcConfigSettingsImpl::rdc_config_set(): group " << group_id << " has no GPU");
    return RDC_ST_BAD_PARAMETER;
  }

  // The GPUs this call changed, to put back if a later one fails.
  std::vector<amdsmi_processor_handle> changed;
  amdsmi_status_t amd_ret = AMDSMI_STATUS_SUCCESS;
  for (uint32_t i = 0; i < rdc_group_info.count; ++i) {
    amdsmi_processor_handle processor_handle;
    amd_ret = get_processor_handle_from_id(rdc_group_info.entity_ids[i], &processor_handle);
    if (amd_ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(
          RDC_ERROR,
          "RdcConfigSettingsImpl::rdc_config_set(): Failed to get processor handle : " << amd_ret);
      break;
    }
    amd_ret = apply_setting(processor_handle, setting);
    if (amd_ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl::rdc_config_set(): Failed to set setting "
                             << type << " : " << amd_ret);
      break;
    }
    changed.push_back(processor_handle);
  }

  if (amd_ret != AMDSMI_STATUS_SUCCESS) {
    // Put the GPUs that took the setting back: to the group's setting of this type
    // if it has one, else to the default.
    const rdc_config_setting_t* previous = nullptr;
    auto group_iter = cached_group_settings_.find(group_id);
    if (group_iter != cached_group_settings_.end()) {
      auto setting_iter = group_iter->second.find(setting.type);
      if (setting_iter != group_iter->second.end()) {
        previous = &setting_iter->second;
      }
    }
    for (auto it = changed.rbegin(); it != changed.rend(); ++it) {
      const amdsmi_status_t undo =
          previous != nullptr ? apply_setting(*it, *previous) : reset_setting(*it, setting.type);
      if (undo != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR,
                "RdcConfigSettingsImpl::rdc_config_set(): Failed to put a GPU back : " << undo);
      }
    }
    return RDC_ST_UNKNOWN_ERROR;
  }

  cached_group_settings_[group_id][setting.type] = setting;
  if (!is_running_) {
    is_running_ = true;
    monitor_thread_ = std::thread(&RdcConfigSettingsImpl::monitorSettings, this);
    RDC_LOG(RDC_INFO, "RdcConfigSettingsImpl Monitoring Thread Started");
  }
  // An idle monitoring thread waits for a setting.
  cv_.notify_all();
  return RDC_ST_OK;
}

// Display user configured settings
rdc_status_t RdcConfigSettingsImpl::rdc_config_get(rdc_gpu_group_t group_id,
                                                   rdc_config_setting_list_t* settings) {
  // Ensure group_id exists in the cache
  std::lock_guard<std::mutex> lock(mutex_);
  auto group_iter = cached_group_settings_.find(group_id);
  if (group_iter == cached_group_settings_.end()) {
    RDC_LOG(RDC_ERROR, "rdc_config_get: group_id not found in cache: " << RDC_ST_NOT_FOUND);
    return RDC_ST_NOT_FOUND;
  }

  // Iterate through cached settings for this group
  int i = 0;
  for (const auto& setting_pair : group_iter->second) {
    if (i >= RDC_MAX_CONFIG_SETTINGS) {
      RDC_LOG(RDC_ERROR,
              "RdcConfigSettingsImpl::rdc_config_get: more settings than RDC_MAX_CONFIG_SETTINGS: "
                  << RDC_ST_MAX_LIMIT);
      return RDC_ST_MAX_LIMIT;
    }

    settings->settings[i].type = setting_pair.first;
    settings->settings[i].target_value = setting_pair.second.target_value;
    ++i;
  }

  settings->total_settings = i;
  return RDC_ST_OK;
}

// Clear cache of user configured settings
rdc_status_t RdcConfigSettingsImpl::rdc_config_clear(rdc_gpu_group_t group_id) {
  // Check if group_id has any cached settings
  std::lock_guard<std::mutex> lock(mutex_);
  auto group_iter = cached_group_settings_.find(group_id);
  if (group_iter == cached_group_settings_.end()) {
    // No cached settings for this group, nothing to clear
    return RDC_ST_OK;
  }

  rdc_group_info_t rdc_group_info;
  if (get_group_info(group_id, &rdc_group_info) != RDC_ST_OK) {
    return RDC_ST_UNKNOWN_ERROR;
  }
  // Reset each cached setting on each GPU of the group: one that fails does not keep
  // the others from being reset.
  amdsmi_status_t amd_ret = AMDSMI_STATUS_SUCCESS;
  for (uint32_t i = 0; i < rdc_group_info.count; ++i) {
    amdsmi_processor_handle processor_handle;
    amdsmi_status_t status =
        get_processor_handle_from_id(rdc_group_info.entity_ids[i], &processor_handle);
    if (status != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(
          RDC_ERROR,
          "RdcConfigSettingsImpl::rdc_config_clear(): Failed to get processor handle : " << status);
      amd_ret = status;
      continue;
    }
    for (const auto& setting_pair : group_iter->second) {
      status = reset_setting(processor_handle, setting_pair.first);
      if (status != AMDSMI_STATUS_SUCCESS) {
        amd_ret = status;
      }
    }
  }

  // The monitoring thread stops checking the group and waits for the next setting.
  cached_group_settings_.erase(group_iter);
  return (amd_ret == AMDSMI_STATUS_SUCCESS) ? RDC_ST_OK : RDC_ST_UNKNOWN_ERROR;
}

}  // namespace rdc
}  // namespace amd
