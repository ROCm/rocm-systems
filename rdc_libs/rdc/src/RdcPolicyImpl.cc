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

#include "rdc_lib/impl/RdcPolicyImpl.h"

#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <ctime>
#include <map>
#include <sstream>
#include <unordered_map>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcPolicyImpl::RdcPolicyImpl(const RdcGroupSettingsPtr& group_settings,
                             const RdcMetricFetcherPtr& metric_fetcher)
    : group_settings_(group_settings), metric_fetcher_(metric_fetcher), start_(true) {
  thread_ = std::thread([this]() {
    while (start_) {
      rdc_policy_check_condition();
      usleep(500);  // 500ms
    }
  });
}

RdcPolicyImpl::~RdcPolicyImpl() {
  start_ = false;
  thread_.join();
}

rdc_status_t RdcPolicyImpl::rdc_policy_set(rdc_gpu_group_t group_id, rdc_policy_t policy) {
  rdc_status_t status = RDC_ST_NOT_SUPPORTED;
  std::lock_guard<std::mutex> guard(policy_mutex_);

  // parameters check
  if (policy.condition.type >= RDC_POLICY_COND_MAX) {
    status = RDC_ST_BAD_PARAMETER;
    return status;
  }

  if (policy.action > RDC_POLICY_ACTION_GPU_RESET) {
    status = RDC_ST_BAD_PARAMETER;
    return status;
  }

  // check if support RDC_POLICY_COND_MAX_PAGE_RETRIED
  if (RDC_POLICY_COND_MAX_PAGE_RETRIED == policy.condition.type) {
    uint32_t gpu_index;
    rdc_group_info_t group_info;
    rdc_field_value value;

    status = group_settings_->rdc_group_gpu_get_info(group_id, &group_info);
    for (unsigned int i = 0; i < group_info.count; i++) {
      gpu_index = group_info.entity_ids[i];

      status = metric_fetcher_->fetch_smi_field(gpu_index, RDC_FI_GPU_PAGE_RETRIED, &value);
      if (status == RDC_ST_SMI_ERROR) return RDC_ST_NOT_SUPPORTED;
    }
  }

  auto it = settings_.find(group_id);
  if (it != settings_.end()) {
    std::vector<rdc_policy_t>& policies = it->second;

    bool exist = false;
    for (auto& itpolicy : policies) {
      // if exist, overwrite the value and action
      if (itpolicy.condition.type == policy.condition.type) {
        itpolicy.condition.value = policy.condition.value;
        itpolicy.action = policy.action;
        exist = true;
      }
    }
    if (!exist) {
      policies.push_back(policy);
    }

    status = RDC_ST_OK;
  } else {
    std::vector<rdc_policy_t> policies;
    policies.push_back(policy);

    settings_.insert(std::make_pair(group_id, policies));
    status = RDC_ST_OK;
  }

  return status;
}

rdc_status_t RdcPolicyImpl::rdc_policy_get(rdc_gpu_group_t group_id, uint32_t* count,
                                           rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) {
  rdc_status_t status = RDC_ST_NOT_SUPPORTED;

  std::lock_guard<std::mutex> guard(policy_mutex_);

  auto it = settings_.find(group_id);
  if (it != settings_.end()) {
    std::vector<rdc_policy_t>& policies_ref = it->second;
    uint32_t i = 0;
    for (auto itpolicy : policies_ref) {
      // if exist
      policies[i].condition.type = itpolicy.condition.type;
      policies[i].condition.value = itpolicy.condition.value;
      policies[i].action = itpolicy.action;
      ++i;
    }
    *count = i;

    status = RDC_ST_OK;
  } else {
    status = RDC_ST_NOT_FOUND;
  }

  return status;
}

rdc_status_t RdcPolicyImpl::rdc_policy_delete(rdc_gpu_group_t group_id,
                                              rdc_policy_condition_type_t condition_type) {
  rdc_status_t status = RDC_ST_NOT_FOUND;

  std::lock_guard<std::mutex> guard(policy_mutex_);

  auto it = settings_.find(group_id);
  if (it != settings_.end()) {
    std::vector<rdc_policy_t>& policies_ref = it->second;

    auto itpolicy = policies_ref.begin();
    while (itpolicy != policies_ref.end()) {
      if (itpolicy->condition.type == condition_type) {
        status = RDC_ST_OK;
        itpolicy = policies_ref.erase(itpolicy);
      } else {
        ++itpolicy;
      }
    }
  } else {
    status = RDC_ST_NOT_FOUND;
  }

  return status;
}

rdc_status_t RdcPolicyImpl::rdc_policy_register(rdc_gpu_group_t group_id,
                                                rdc_policy_register_callback callback)

{
  rdc_status_t status = RDC_ST_NOT_FOUND;

  std::lock_guard<std::mutex> guard(policy_mutex_);

  auto run = callbacks_.find(group_id);
  if (run != callbacks_.end()) {
    run->second = callback;
    status = RDC_ST_OK;
  } else {
    callbacks_.insert(std::make_pair(group_id, callback));
    status = RDC_ST_OK;
  }
  return status;
}

rdc_status_t RdcPolicyImpl::rdc_policy_unregister(rdc_gpu_group_t group_id) {
  rdc_status_t status = RDC_ST_OK;

  std::lock_guard<std::mutex> guard(policy_mutex_);

  callbacks_.erase(group_id);
  return status;
}

void RdcPolicyImpl::rdc_policy_check_condition() {
  // go through the settings
  for (auto it : settings_) {
    rdc_gpu_group_t group_id = it.first;
    std::vector<rdc_policy_t> policies = it.second;
    rdc_policy_register_callback callback = rdc_policy_get_callback(group_id);

    for (auto policy : policies) {
      rdc_status_t status;
      rdc_field_value value;
      rdc_group_info_t group_info;
      uint32_t gpu_index;

      status = group_settings_->rdc_group_gpu_get_info(group_id, &group_info);
      for (unsigned int i = 0; i < group_info.count; i++) {
        rdc_field_t map[RDC_POLICY_COND_MAX] = {RDC_FI_GPU_PAGE_RETRIED, RDC_FI_GPU_TEMP,
                                                RDC_FI_POWER_USAGE};

        gpu_index = group_info.entity_ids[i];
        status = metric_fetcher_->fetch_smi_field(gpu_index, map[policy.condition.type], &value);
        if (status == RDC_ST_OK) {
          if (value.value.l_int > policy.condition.value) {
            // callback if needed
            if (callback) {
              rdc_policy_callback_response_t response = {1, policy.condition, group_id,
                                                         value.value.l_int};
              callback(&response);
            }

            if (RDC_POLICY_ACTION_GPU_RESET == policy.action) {
              rdc_policy_gpu_reset(gpu_index);
            }
          }
        }
      }
    }
  }
}

rdc_policy_register_callback RdcPolicyImpl::rdc_policy_get_callback(rdc_gpu_group_t group_id) {
  rdc_policy_register_callback cb = nullptr;
  auto it = callbacks_.find(group_id);
  if (it != callbacks_.end()) {
    cb = it->second;
  }
  return cb;
}

void RdcPolicyImpl::rdc_policy_gpu_reset(uint32_t gpu_index) {
  amdsmi_processor_handle processor_handle = {};

  amdsmi_status_t err = get_processor_handle_from_id(gpu_index, &processor_handle);
  if (err == AMDSMI_STATUS_SUCCESS) {
    amdsmi_reset_gpu(processor_handle);
  }
}

}  // namespace rdc
}  // namespace amd
