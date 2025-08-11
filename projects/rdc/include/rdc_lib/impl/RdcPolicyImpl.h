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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCPOLICYIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCPOLICYIMPL_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>
#include <future>

#include "amd_smi/amdsmi.h"
#include "rdc_lib/RdcPolicy.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcGroupSettings.h"

namespace amd {
namespace rdc {

class RdcPolicyImpl : public RdcPolicy {
 public:
  RdcPolicyImpl(const RdcGroupSettingsPtr& group_settings, const RdcMetricFetcherPtr& metric_fetcher);
  ~RdcPolicyImpl();

  rdc_status_t rdc_policy_set(rdc_gpu_group_t group_id, rdc_policy_t policy) override;

  rdc_status_t rdc_policy_get(rdc_gpu_group_t group_id, uint32_t* count,
                              rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) override;

  rdc_status_t rdc_policy_delete(rdc_gpu_group_t group_id,
                                 rdc_policy_condition_type_t condition_type) override;

  rdc_status_t rdc_policy_register(rdc_gpu_group_t group_id,rdc_policy_register_callback callback) override;

  rdc_status_t rdc_policy_unregister(rdc_gpu_group_t group_id) override;

 private:
  RdcGroupSettingsPtr group_settings_;
  RdcMetricFetcherPtr metric_fetcher_;
  std::mutex policy_mutex_;
  std::thread thread_;
  bool start_;

  std::map<rdc_gpu_group_t, std::vector<rdc_policy_t> > settings_;
  std::map<rdc_gpu_group_t, rdc_policy_register_callback> callbacks_;

  void rdc_policy_check_condition();
  void rdc_policy_gpu_reset(uint32_t gpu_index);
  rdc_policy_register_callback rdc_policy_get_callback(rdc_gpu_group_t group_id);
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCPOLICYIMPL_H_
