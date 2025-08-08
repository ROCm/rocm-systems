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
#include <ctime>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcConfigSettingsImpl::RdcConfigSettingsImpl(const RdcGroupSettingsPtr& group_settings)
    : group_settings_(group_settings), is_running_(false) {}

// Monitoring thread of gpu settings
void RdcConfigSettingsImpl::monitorSettings() {
  rdc_gpu_group_t group_id;
  amdsmi_processor_handle processor_handle;
  amdsmi_status_t status;
  rdc_status_t rdc_status;
  rdc_group_info_t rdc_group_info = {};
  amdsmi_power_cap_info_t cap_info = {};
  amdsmi_clk_info_t info = {};
  uint64_t cached_value;

  while (true) {
    {  // Scope block for mutex
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::minutes(1),
                   [this] { return !is_running_ || cached_group_settings_.empty(); });

      if (!is_running_ || cached_group_settings_.empty()) {
        break;  // Stop if the thread is requested to stop or settings are empty
      }

      for (const auto& group_pair : cached_group_settings_) {
        group_id = group_pair.first;
        const auto& cached_settings = group_pair.second;
        rdc_status = get_group_info(group_id, &rdc_group_info);
        if (rdc_status != RDC_ST_OK) {
          // Error log handled in get_group_info
          continue;
        }
        for (unsigned int i = 0; i < rdc_group_info.count; ++i) {
          status = get_processor_handle_from_id(rdc_group_info.entity_ids[i], &processor_handle);
          if (status != AMDSMI_STATUS_SUCCESS) {
            RDC_LOG(RDC_ERROR,
                    "RdcConfigSettingsImpl::monitorSettings(): get_processor_handle_from_id faied: "
                        << status);
            continue;
          }

          // Power cap
          status = amdsmi_get_power_cap_info(processor_handle, 0, &cap_info);
          if (status != AMDSMI_STATUS_SUCCESS) {
            RDC_LOG(RDC_ERROR,
                    "RdcConfigSettingsImpl::monitorSettings(); amdsmi_get_power_cap_info failed: "
                        << status);
            continue;
          }

          auto power_cap_it = cached_settings.find(RDC_CFG_POWER_LIMIT);
          if (power_cap_it != cached_settings.end()) {
            cached_value = power_cap_it->second.target_value;
            if (microwattsToWatts(cap_info.power_cap) != cached_value) {
              RDC_LOG(
                  RDC_INFO,
                  "RdcConfigSettingsImpl::monitorSettings(); Mismatched Power values, resetting");
              status = amdsmi_set_power_cap(processor_handle, 0, wattsToMicrowatts(cached_value));
              if (status != AMDSMI_STATUS_SUCCESS) {
                RDC_LOG(
                    RDC_ERROR,
                    "RdcConfigSettingsImpl::monitorSettings(); amdsmi_set_power_cap_info failed: "
                        << status);
                continue;
              }
            }
          }

          // Mem clock
          status = amdsmi_get_clock_info(processor_handle, AMDSMI_CLK_TYPE_MEM, &info);
          if (status != AMDSMI_STATUS_SUCCESS) {
            RDC_LOG(
                RDC_ERROR,
                "RdcConfigSettingsImpl::monitorSettings(); amdsmi_get_clock_info for mem failed: "
                    << status);
            continue;
          }

          auto mem_clk_it = cached_settings.find(RDC_CFG_MEMORY_CLOCK_LIMIT);
          if (mem_clk_it != cached_settings.end()) {
            cached_value = mem_clk_it->second.target_value;
            if (info.max_clk != cached_value) {
              status = amdsmi_set_gpu_clk_limit(processor_handle, AMDSMI_CLK_TYPE_MEM,
                                                CLK_LIMIT_MAX, cached_value);
              if (status != AMDSMI_STATUS_SUCCESS) {
                RDC_LOG(RDC_ERROR,
                        "RdcConfigSettingsImpl::monitorSettings(); amdsmi_set_gpu_clk_limit failed "
                        "for mem clk: "
                            << status);
                continue;
              }
            }
          }

          // GFX clock
          status = amdsmi_get_clock_info(processor_handle, AMDSMI_CLK_TYPE_GFX, &info);
          if (status != AMDSMI_STATUS_SUCCESS) {
            RDC_LOG(
                RDC_ERROR,
                "RdcConfigSettingsImpl::monitorSettings(); amdsmi_get_clock_info for gfx failed: "
                    << status);
            continue;
          }

          auto gfx_clk_it = cached_settings.find(RDC_CFG_GFX_CLOCK_LIMIT);
          if (gfx_clk_it != cached_settings.end()) {
            cached_value = gfx_clk_it->second.target_value;
            if (info.max_clk != cached_value) {
              status = amdsmi_set_gpu_clk_limit(processor_handle, AMDSMI_CLK_TYPE_GFX,
                                                CLK_LIMIT_MAX, cached_value);
              if (status != AMDSMI_STATUS_SUCCESS) {
                RDC_LOG(RDC_ERROR,
                        "RdcConfigSettingsImpl::monitorSettings(); amdsmi_set_gpu_clk_limit failed "
                        "for gfx clk: "
                            << status);
                continue;
              }
            }
          }
        }
      }
    }
  }
  RDC_LOG(RDC_INFO, "RdcConfigSettingsImpl Monitoring Thread Stopped");
}

uint64_t RdcConfigSettingsImpl::wattsToMicrowatts(uint64_t watts) const {
  return watts * 1'000'000;
}

uint64_t RdcConfigSettingsImpl::microwattsToWatts(int microwatts) const {
  return microwatts / 1'000'000;
}

uint64_t RdcConfigSettingsImpl::mHzToHz(uint64_t mhz) const { return mhz * 1000000ULL; }

rdc_status_t RdcConfigSettingsImpl::get_group_info(rdc_gpu_group_t group_id,
                                                   rdc_group_info_t* rdc_group_info) {
  rdc_status_t status = group_settings_->rdc_group_gpu_get_info(group_id, rdc_group_info);
  if (status != RDC_ST_OK) {
    RDC_LOG(RDC_ERROR,
            "RdcConfigSettingsImpl::rdc_config_set(): rdc_group_gpu_get_info failed : " << status);
  }
  return status;
}

// Set configuration setting
rdc_status_t RdcConfigSettingsImpl::rdc_config_set(rdc_gpu_group_t group_id,
                                                   rdc_config_setting_t setting) {
  amdsmi_processor_handle processor_handle;
  amdsmi_status_t amd_ret;

  // Get the group info for gpu_index list
  rdc_group_info_t rdc_group_info;
  if (get_group_info(group_id, &rdc_group_info) != RDC_ST_OK) {
    return RDC_ST_UNKNOWN_ERROR;
  }

  for (unsigned int i = 0; i < rdc_group_info.count; ++i) {
    amd_ret = get_processor_handle_from_id(rdc_group_info.entity_ids[i], &processor_handle);
    if (amd_ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(
          RDC_ERROR,
          "RdcConfigSettingsImpl::rdc_config_set(): Failed to get processor handle : " << amd_ret);
      break;
    }

    if (setting.type == RDC_CFG_POWER_LIMIT) {
      amd_ret = amdsmi_set_power_cap(processor_handle, 0, wattsToMicrowatts(setting.target_value));
      if (amd_ret != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR,
                "RdcConfigSettingsImpl::rdc_config_set: amdsmi_set_power_cap failed : " << amd_ret);
        break;
      }
    } else if (setting.type == RDC_CFG_MEMORY_CLOCK_LIMIT) {
      amd_ret = amdsmi_set_gpu_clk_limit(processor_handle, AMDSMI_CLK_TYPE_MEM, CLK_LIMIT_MAX,
                                         setting.target_value);
      if (amd_ret != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(
            RDC_ERROR,
            "RdcConfigSettingsImpl::rdc_config_set: amdsmi_set_gpu_clk_limit failed : " << amd_ret);
        break;
      }
    } else if (setting.type == RDC_CFG_GFX_CLOCK_LIMIT) {
      amd_ret = amdsmi_set_gpu_clk_limit(processor_handle, AMDSMI_CLK_TYPE_GFX, CLK_LIMIT_MAX,
                                         setting.target_value);
      if (amd_ret != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(
            RDC_ERROR,
            "RdcConfigSettingsImpl::rdc_config_set: amdsmi_set_gpu_clk_limit failed : " << amd_ret);
        break;
      }
    }
  }

  if (amd_ret == AMDSMI_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> lock(mutex_);
    cached_group_settings_[group_id][setting.type] = setting;
    if (!is_running_) {
      is_running_ = true;
      monitor_thread_ = std::thread(&RdcConfigSettingsImpl::monitorSettings, this);
      RDC_LOG(RDC_INFO, "RdcConfigSettingsImpl Monitoring Thread Started");
    }
    return RDC_ST_OK;
  } else {
    return RDC_ST_UNKNOWN_ERROR;
  }
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
  amdsmi_status_t amd_ret = AMDSMI_STATUS_SUCCESS;
  amdsmi_processor_handle processor_handle;

  // Check if group_id has any cached settings
  std::unique_lock<std::mutex> lock(mutex_);
  auto group_iter = cached_group_settings_.find(group_id);
  if (group_iter == cached_group_settings_.end()) {
    // No cached settings for this group, nothing to clear
    return RDC_ST_OK;
  }

  rdc_group_info_t rdc_group_info;
  if (get_group_info(group_id, &rdc_group_info) != RDC_ST_OK) {
    return RDC_ST_UNKNOWN_ERROR;
  }
  // Iterate over each GPU in the group and clear only the cached settings
  for (unsigned int i = 0; i < rdc_group_info.count; ++i) {
    amd_ret = get_processor_handle_from_id(rdc_group_info.entity_ids[i], &processor_handle);
    if (amd_ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR,
              "RdcConfigSettingsImpl::rdc_config_clear(): Failed to get processor handle : "
                  << amd_ret);
      break;
    }

    // Reset power cap if it was set
    if (group_iter->second.find(RDC_CFG_POWER_LIMIT) != group_iter->second.end()) {
      amdsmi_power_cap_info_t cap_info = {};
      amd_ret = amdsmi_get_power_cap_info(processor_handle, 0, &cap_info);
      if (amd_ret == AMDSMI_STATUS_SUCCESS && cap_info.power_cap != cap_info.default_power_cap) {
        amd_ret = amdsmi_set_power_cap(processor_handle, 0, cap_info.default_power_cap);
        if (amd_ret != AMDSMI_STATUS_SUCCESS) {
          RDC_LOG(RDC_ERROR, "RdcConfigSettingsImpl::rdc_config_clear: Failed to reset power cap : "
                                 << amd_ret);
          break;
        }
      }
    }

    // Reset GFX clock limit if it was set
    if (group_iter->second.find(RDC_CFG_GFX_CLOCK_LIMIT) != group_iter->second.end()) {
      amdsmi_frequencies_t freqs = {};
      amdsmi_clk_info_t info = {};

      amd_ret = amdsmi_get_clk_freq(processor_handle, AMDSMI_CLK_TYPE_GFX, &freqs);
      if (amd_ret != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR,
                "RdcConfigSettingsImpl::rdc_config_clear: Failed to get GFX freq: " << amd_ret);
        break;
      }

      amd_ret = amdsmi_get_clock_info(processor_handle, AMDSMI_CLK_TYPE_GFX, &info);
      if (amd_ret != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR,
                "RdcConfigSettingsImpl::rdc_config_clear: Failed to get GFX info: " << amd_ret);
        break;
      }

      uint64_t curr = mHzToHz(info.max_clk);
      uint64_t maxf = freqs.frequency[freqs.num_supported - 1];
      if (curr != maxf) {
        amd_ret = amdsmi_set_gpu_perf_level(processor_handle, AMDSMI_DEV_PERF_LEVEL_AUTO);
        if (amd_ret != AMDSMI_STATUS_SUCCESS) {
          RDC_LOG(RDC_ERROR,
                  "RdcConfigSettingsImpl::rdc_config_clear: Failed to reset GFX clock limit : "
                      << amd_ret);
          break;
        }
      }
    }

    // Reset memory clock limit if it was set
    if (group_iter->second.find(RDC_CFG_MEMORY_CLOCK_LIMIT) != group_iter->second.end()) {
      amdsmi_frequencies_t freqs = {};
      amdsmi_clk_info_t info = {};

      amd_ret = amdsmi_get_clk_freq(processor_handle, AMDSMI_CLK_TYPE_MEM, &freqs);
      if (amd_ret != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR,
                "RdcConfigSettingsImpl::rdc_config_clear: Failed to get MEM freq: " << amd_ret);
        break;
      }

      amd_ret = amdsmi_get_clock_info(processor_handle, AMDSMI_CLK_TYPE_MEM, &info);
      if (amd_ret != AMDSMI_STATUS_SUCCESS) {
        RDC_LOG(RDC_ERROR,
                "RdcConfigSettingsImpl::rdc_config_clear: Failed to get MEM info: " << amd_ret);
        break;
      }

      uint64_t curr = mHzToHz(info.max_clk);
      uint64_t maxf = freqs.frequency[freqs.num_supported - 1];
      if (curr != maxf) {
        amd_ret = amdsmi_set_gpu_perf_level(processor_handle, AMDSMI_DEV_PERF_LEVEL_AUTO);
        if (amd_ret != AMDSMI_STATUS_SUCCESS) {
          RDC_LOG(RDC_ERROR,
                  "RdcConfigSettingsImpl::rdc_config_clear: Failed to reset memory clock limit:"
                      << amd_ret);
          break;
        }
      }
    }
  }

  cached_group_settings_.erase(group_id);

  if (cached_group_settings_.empty()) {
    is_running_ = false;
    cv_.notify_all();
    lock.unlock();

    if (monitor_thread_.joinable()) {
      monitor_thread_.join();  // Wait for the thread to finish
    }

    RDC_LOG(RDC_INFO, "RdcConfigSettingsImpl Monitoring Thread Stopped");
  }

  return (amd_ret == AMDSMI_STATUS_SUCCESS) ? RDC_ST_OK : RDC_ST_UNKNOWN_ERROR;
}

}  // namespace rdc
}  // namespace amd
