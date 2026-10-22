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

// Configuration settings of several GPU groups set, read and cleared from
// several threads at once, while the monitoring thread re-applies them every
// millisecond. A clear that removed the last setting used to join the
// monitoring thread outside the lock while a set started a new one in the same
// std::thread (a data race, and std::terminate when the old thread was still
// joinable).
//
// RdcConfigSettingsImpl.cc and RdcLogger.cc are compiled into this binary; the
// GPU groups, get_processor_handle_from_id() and the amd-smi calls are the fakes
// below.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/impl/RdcConfigSettingsImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

constexpr uint32_t kThreads = 4;
constexpr int kRounds = 200;
constexpr uint64_t kDefaultCapUw = 750ULL * 1000 * 1000;

// The fake GPUs' power caps; the threads and the monitoring thread call the
// fakes at once.
std::mutex g_mutex;
uint64_t g_cap_uw[kThreads] = {kDefaultCapUw, kDefaultCapUw, kDefaultCapUw, kDefaultCapUw};

bool gpu_of(amdsmi_processor_handle handle, uint32_t* gpu) {
  const auto value = reinterpret_cast<uintptr_t>(handle);
  if (value < 0x1000 || value >= 0x1000 + kThreads) {
    return false;
  }
  *gpu = static_cast<uint32_t>(value - 0x1000);
  return true;
}

// Group g + 1 holds GPU g.
class OneGpuGroups : public amd::rdc::RdcGroupSettings {
 public:
  rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t group, rdc_group_info_t* info) override {
    if (group == 0 || group > kThreads) {
      return RDC_ST_NOT_FOUND;
    }
    std::memset(info, 0, sizeof(*info));
    info->count = 1;
    info->entity_ids[0] = group - 1;
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

}  // namespace

// Fakes for what RdcConfigSettingsImpl.cc calls. Another race test that defines
// get_processor_handle_from_id() too must make its copy weak or drop it: these
// tests need a handle for every GPU, and fail if they get none.
namespace amd {
namespace rdc {
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle) {
  if (gpu_id >= kThreads || processor_handle == nullptr) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  *processor_handle =
      reinterpret_cast<amdsmi_processor_handle>(static_cast<uintptr_t>(0x1000 + gpu_id));
  return AMDSMI_STATUS_SUCCESS;
}
}  // namespace rdc
}  // namespace amd

amdsmi_status_t amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle, uint32_t,
                                          amdsmi_power_cap_info_t* info) {
  uint32_t gpu = 0;
  if (!gpu_of(processor_handle, &gpu) || info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  std::memset(info, 0, sizeof(*info));
  info->power_cap = g_cap_uw[gpu];
  info->default_power_cap = kDefaultCapUw;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_power_cap(amdsmi_processor_handle processor_handle, uint32_t,
                                     uint64_t cap) {
  uint32_t gpu = 0;
  if (!gpu_of(processor_handle, &gpu)) {
    return AMDSMI_STATUS_INVAL;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  g_cap_uw[gpu] = cap;
  return AMDSMI_STATUS_SUCCESS;
}

// The tests set power limits only; the clock calls are never reached.
amdsmi_status_t amdsmi_get_clock_info(amdsmi_processor_handle, amdsmi_clk_type_t,
                                      amdsmi_clk_info_t*) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle, amdsmi_clk_type_t,
                                         amdsmi_clk_limit_type_t, uint64_t) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle, amdsmi_clk_type_t,
                                    amdsmi_frequencies_t*) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t amdsmi_set_gpu_perf_level(amdsmi_processor_handle, amdsmi_dev_perf_level_t) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

// Each thread sets, reads and clears its own group's power limit, so the last
// setting goes away and comes back again and again.
TEST(ConfigRace, SetsAndClearsOfSeveralGroupsAtOnce) {
  std::atomic<int> failures{0};
  {
    amd::rdc::RdcConfigSettingsImpl config(std::make_shared<OneGpuGroups>(),
                                           std::chrono::milliseconds(1));
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < kThreads; ++t) {
      threads.emplace_back([&config, &failures, t] {
        const rdc_gpu_group_t group = t + 1;
        const rdc_config_setting_t power = {RDC_CFG_POWER_LIMIT, 300 + t};
        for (int round = 0; round < kRounds; ++round) {
          if (config.rdc_config_set(group, power) != RDC_ST_OK) {
            failures++;
          }
          rdc_config_setting_list_t list = {};
          if (config.rdc_config_get(group, &list) != RDC_ST_OK || list.total_settings != 1) {
            failures++;
          }
          if (config.rdc_config_clear(group) != RDC_ST_OK) {
            failures++;
          }
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  EXPECT_EQ(failures.load(), 0);
}
