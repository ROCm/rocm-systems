/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

// RdcPolicyImpl calls the policy callbacks without holding its policy lock.
// rdcd relies on that: its PolicyCallback() takes rdcd's own policy lock, under
// which rdcd calls rdc_policy_register(). A callback under RDC's lock would
// let the two locks deadlock, which ThreadSanitizer reports as a lock-order
// inversion once both orders have run.
//
// RdcPolicyImpl.cc is compiled into this binary; the group settings, metric
// fetcher and amdsmi calls it uses are the fakes below.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/impl/RdcPolicyImpl.h"

namespace {

constexpr rdc_gpu_group_t kGroup = 1;

// Group kGroup holds GPU 0; no other group exists.
class OneGroup : public amd::rdc::RdcGroupSettings {
 public:
  rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t group_id, rdc_group_info_t* info) override {
    if (group_id != kGroup) {
      return RDC_ST_GROUP_NOT_FOUND;
    }
    std::memset(info, 0, sizeof(*info));
    info->count = 1;
    info->entity_ids[0] = 0;
    return RDC_ST_OK;
  }
  rdc_status_t rdc_group_gpu_create(const char*, rdc_gpu_group_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_gpu_destroy(rdc_gpu_group_t) override { return RDC_ST_NOT_SUPPORTED; }
  rdc_status_t rdc_group_gpu_add(rdc_gpu_group_t, uint32_t) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_get_all_ids(rdc_gpu_group_t[], uint32_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_field_create(uint32_t, rdc_field_t*, const char*,
                                      rdc_field_grp_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_field_add_field(rdc_field_grp_t, rdc_field_t) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_field_destroy(rdc_field_grp_t) override { return RDC_ST_NOT_SUPPORTED; }
  rdc_status_t rdc_group_field_get_info(rdc_field_grp_t, rdc_field_group_info_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_field_get_all_ids(rdc_field_grp_t[], uint32_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
};

// Every GPU reads 100 for every field, above the policy's threshold of 1.
class HotGpus : public amd::rdc::RdcMetricFetcher {
 public:
  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t fetch_smi_field(uint32_t, rdc_field_t field_id, rdc_field_value* value) override {
    value->field_id = field_id;
    value->status = AMDSMI_STATUS_SUCCESS;
    value->type = INTEGER;
    value->value.l_int = 100;
    return RDC_ST_OK;
  }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
};

// Stands for rdcd's policy lock.
std::mutex g_rdcd_lock;
std::atomic<int> g_callbacks{0};

int callback(rdc_policy_callback_response_t*) {
  std::lock_guard<std::mutex> lock(g_rdcd_lock);  // as rdcd's PolicyCallback() does
  g_callbacks++;
  return 0;
}

}  // namespace

// Fakes for the calls RdcPolicyImpl.cc makes to reset a GPU; no policy here resets one.
namespace amd {
namespace rdc {
amdsmi_status_t get_processor_handle_from_id(uint32_t, amdsmi_processor_handle*) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}
}  // namespace rdc
}  // namespace amd
amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle) { return AMDSMI_STATUS_NOT_SUPPORTED; }

TEST(PolicyRace, CallbacksRunWithoutThePolicyLock) {
  amd::rdc::RdcPolicyImpl policy(std::make_shared<OneGroup>(), std::make_shared<HotGpus>());
  rdc_policy_t hot = {};
  hot.condition = {RDC_POLICY_COND_THERMAL, 1};
  hot.action = RDC_POLICY_ACTION_NONE;
  ASSERT_EQ(policy.rdc_policy_set(kGroup, hot), RDC_ST_OK);

  // Register under rdcd's lock, as rdcd does, then again right after each
  // callback, between two checks (500 ms apart): the two lock orders never
  // overlap, so a wrong order shows up as a report, not a hang.
  {
    std::lock_guard<std::mutex> lock(g_rdcd_lock);
    ASSERT_EQ(policy.rdc_policy_register(kGroup, callback), RDC_ST_OK);
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (int seen = 0; seen < 3 && std::chrono::steady_clock::now() < deadline;) {
    if (g_callbacks == seen) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    seen = g_callbacks;
    std::lock_guard<std::mutex> lock(g_rdcd_lock);
    ASSERT_EQ(policy.rdc_policy_register(kGroup, callback), RDC_ST_OK);
  }
  EXPECT_GE(g_callbacks, 3) << "the policy thread did not call back within 30 s";
  EXPECT_EQ(policy.rdc_policy_register(kGroup + 1, callback), RDC_ST_GROUP_NOT_FOUND);
  EXPECT_EQ(policy.rdc_policy_unregister(kGroup), RDC_ST_OK);
}
