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

// rdc_config_set(), rdc_config_get() and rdc_config_clear(), and the thread
// that re-applies the settings, on four fake GPUs.
//
// librdc cannot be loaded without a GPU (it initializes amd-smi when loaded), so
// this binary builds RdcConfigSettingsImpl.cc itself. The GPU groups, the
// amd-smi calls that file makes and get_processor_handle_from_id() are the
// fakes below; librdc_bootstrap provides the logger. The monitoring thread
// re-applies the settings every 10 ms in the Monitor tests and never in the
// others.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/impl/RdcConfigSettingsImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

constexpr uint32_t kGpus = 4;
constexpr uint64_t kMicro = 1000 * 1000;
constexpr uint64_t kDefaultCapUw = 750 * kMicro;
// Each clock's top level, where the automatic performance level puts its limit.
constexpr uint64_t kTopGfxMhz = 2100;
constexpr uint64_t kTopMemMhz = 1300;

constexpr uint32_t kGfx = RDC_CFG_GFX_CLOCK_LIMIT;
constexpr uint32_t kMem = RDC_CFG_MEMORY_CLOCK_LIMIT;
constexpr uint32_t kPower = RDC_CFG_POWER_LIMIT;

// The GPU groups of the fake group table.
constexpr rdc_gpu_group_t kAll = 1;     // GPUs 0 to 3
constexpr rdc_gpu_group_t kFirst = 2;   // GPU 0
constexpr rdc_gpu_group_t kSecond = 3;  // GPU 1
constexpr rdc_gpu_group_t kNone = 4;    // no GPU
constexpr rdc_gpu_group_t kGroups[] = {kAll, kFirst, kSecond, kNone};

// One amd-smi call: the clock it names (-1 if none) and, for a setter, the
// value it was asked to write.
struct Call {
  std::string api;
  uint32_t gpu;
  int clk;
  uint64_t value;
};

// A fake GPU: its power cap and the upper limit of each clock.
struct Gpu {
  uint64_t cap_uw = kDefaultCapUw;
  uint64_t gfx_max_mhz = kTopGfxMhz;
  uint64_t mem_max_mhz = kTopMemMhz;
};

// The fakes' state, which the monitoring thread uses too.
std::mutex g_mutex;
Gpu g_gpus[kGpus];
std::vector<Call> g_calls;
std::vector<uint32_t> g_lookups;                       // the IDs get_processor_handle_from_id() got
std::set<std::pair<std::string, uint32_t>> g_failing;  // calls that fail: API, GPU
uint32_t g_num_freq = 3;                               // levels in every frequency table

void reset_fakes() {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (auto& gpu : g_gpus) {
    gpu = Gpu();
  }
  g_calls.clear();
  g_lookups.clear();
  g_failing.clear();
  g_num_freq = 3;
}

void fail(const std::string& api, uint32_t gpu) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_failing.insert({api, gpu});
}

void set_num_freq(uint32_t levels) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_num_freq = levels;
}

Gpu gpu(uint32_t index) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_gpus[index];
}

void set_gpu(uint32_t index, const Gpu& value) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_gpus[index] = value;
}

size_t calls_so_far() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_calls.size();
}

// The calls of api (every API if empty), from the from-th call on.
std::vector<Call> calls(const std::string& api = "", size_t from = 0) {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::vector<Call> out;
  for (size_t i = from; i < g_calls.size(); ++i) {
    if (api.empty() || g_calls[i].api == api) {
      out.push_back(g_calls[i]);
    }
  }
  return out;
}

std::vector<uint32_t> lookups() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_lookups;
}

// The GPU index of each call, in order.
std::vector<uint32_t> gpus_of(const std::vector<Call>& list) {
  std::vector<uint32_t> out;
  for (const auto& call : list) {
    out.push_back(call.gpu);
  }
  return out;
}

amdsmi_processor_handle handle_of(uint32_t gpu) {
  return reinterpret_cast<amdsmi_processor_handle>(static_cast<uintptr_t>(0x1000 + gpu));
}

bool gpu_of(amdsmi_processor_handle handle, uint32_t* gpu) {
  const auto value = reinterpret_cast<uintptr_t>(handle);
  if (value < 0x1000 || value >= 0x1000 + kGpus) {
    return false;
  }
  *gpu = static_cast<uint32_t>(value - 0x1000);
  return true;
}

uint64_t* max_mhz(Gpu* gpu, amdsmi_clk_type_t clk) {
  if (clk == AMDSMI_CLK_TYPE_GFX) {
    return &gpu->gfx_max_mhz;
  }
  if (clk == AMDSMI_CLK_TYPE_MEM) {
    return &gpu->mem_max_mhz;
  }
  return nullptr;
}

// Records a call on a fake GPU (g_mutex held); returns the status the call
// fails with, or AMDSMI_STATUS_SUCCESS.
amdsmi_status_t enter(const char* api, amdsmi_processor_handle handle, int clk, uint64_t value,
                      uint32_t* gpu) {
  if (!gpu_of(handle, gpu)) {
    return AMDSMI_STATUS_INVAL;
  }
  g_calls.push_back({api, *gpu, clk, value});
  return g_failing.count({api, *gpu}) != 0 ? AMDSMI_STATUS_UNKNOWN_ERROR : AMDSMI_STATUS_SUCCESS;
}

// How the fake group table answers rdc_group_gpu_get_info() for a group.
enum class Report {
  kFilled,        // as RDC's group table does
  kUnfilled,      // RDC_ST_OK without writing the output
  kOverCapacity,  // a count one above RDC_GROUP_MAX_ENTITIES
};

class FakeGroups : public amd::rdc::RdcGroupSettings {
 public:
  void report(rdc_gpu_group_t group, Report how) {
    std::lock_guard<std::mutex> lock(mutex_);
    how_[group] = how;
  }

  int info_calls() {
    std::lock_guard<std::mutex> lock(mutex_);
    return info_calls_;
  }

  rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t group, rdc_group_info_t* info) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++info_calls_;
    const auto it = gpus_.find(group);
    if (it == gpus_.end()) {
      return RDC_ST_NOT_FOUND;
    }
    switch (how_[group]) {
      case Report::kUnfilled:
        return RDC_ST_OK;
      case Report::kOverCapacity:
        info->count = RDC_GROUP_MAX_ENTITIES + 1;
        for (uint32_t i = 0; i < RDC_GROUP_MAX_ENTITIES; ++i) {
          info->entity_ids[i] = i % kGpus;
        }
        return RDC_ST_OK;
      case Report::kFilled:
        break;
    }
    std::snprintf(info->group_name, sizeof(info->group_name), "group%u", group);
    info->count = static_cast<uint32_t>(it->second.size());
    for (uint32_t i = 0; i < info->count; ++i) {
      info->entity_ids[i] = it->second[i];
    }
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

 private:
  std::mutex mutex_;
  const std::map<rdc_gpu_group_t, std::vector<uint32_t>> gpus_ = {
      {kAll, {0, 1, 2, 3}}, {kFirst, {0}}, {kSecond, {1}}, {kNone, {}}};
  std::map<rdc_gpu_group_t, Report> how_;
  int info_calls_ = 0;
};

// A setting with any type value, as a client can send one: the value is copied
// into the enum's bytes, not converted.
rdc_config_setting_t setting(uint32_t type, uint64_t value) {
  rdc_config_setting_t out;
  std::memset(&out, 0, sizeof(out));
  static_assert(sizeof(out.type) == sizeof(type), "rdc_config_type_t has 32 bits");
  std::memcpy(&out.type, &type, sizeof(type));
  out.target_value = value;
  return out;
}

// Larger than the frames of a config call; well within the default 8 MB stack.
constexpr size_t kDirt = 1 << 20;

// Fills the stack below the caller's frame with byte, where the frames of the
// caller's next call go. Not instrumented by AddressSanitizer, which would put a
// redzone between the array and the top of the frame, and so leave the top of the
// next frame clean.
__attribute__((noinline, no_sanitize_address)) void dirty_stack(unsigned char byte) {
  unsigned char dirt[kDirt];
  memset(dirt, byte, sizeof(dirt));
  asm volatile("" : : "r"(dirt) : "memory");
}

// Whether an uninitialized local, in a frame instrumented like the config
// calls, shows the byte of dirty_stack(): if not, this build places frames
// differently, and a check after dirty_stack() proves nothing. It reads
// uninitialized memory on purpose, which MemorySanitizer would report.
__attribute__((noinline)) bool shows_dirt(unsigned char byte) {
  unsigned char local[4096];
  asm volatile("" : : "r"(local) : "memory");
  return local[0] == byte && local[sizeof(local) - 1] == byte;
}

double time_factor() {
  const char* factor = std::getenv("RDC_TEST_TIME_FACTOR");
  return factor != nullptr && std::atof(factor) > 0 ? std::atof(factor) : 1.0;
}

class Config : public ::testing::Test {
 protected:
  Config() : Config(std::chrono::hours(1)) {}
  explicit Config(std::chrono::milliseconds period) : period_(period) {}

  void SetUp() override {
    reset_fakes();
    groups_ = std::make_shared<FakeGroups>();
    config_ = std::make_unique<amd::rdc::RdcConfigSettingsImpl>(groups_, period_);
  }

  // Clears what the test set, so that no setting outlives the test.
  void TearDown() override {
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_failing.clear();
      g_num_freq = 3;
    }
    for (auto group : kGroups) {
      groups_->report(group, Report::kFilled);
      config_->rdc_config_clear(group);
    }
    config_.reset();
  }

  rdc_status_t set(rdc_gpu_group_t group, uint32_t type, uint64_t value) {
    return config_->rdc_config_set(group, setting(type, value));
  }

  rdc_status_t clear(rdc_gpu_group_t group) { return config_->rdc_config_clear(group); }

  // rdc_config_get()'s status and the settings it reports, as {type: value}.
  std::pair<rdc_status_t, std::map<uint32_t, uint64_t>> get(rdc_gpu_group_t group) {
    rdc_config_setting_list_t list;
    std::memset(&list, 0, sizeof(list));
    const rdc_status_t status = config_->rdc_config_get(group, &list);
    std::map<uint32_t, uint64_t> out;
    for (uint32_t i = 0;
         status == RDC_ST_OK && i < list.total_settings && i < RDC_MAX_CONFIG_SETTINGS; ++i) {
      uint32_t type = 0;
      std::memcpy(&type, &list.settings[i].type, sizeof(type));
      out[type] = list.settings[i].target_value;
    }
    return {status, out};
  }

  // Waits until the monitoring thread has gone over `groups` cached groups
  // `passes` more times (it looks each one up once per pass).
  bool wait_for_passes(int passes, int groups) {
    const int target = groups_->info_calls() + passes * groups;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(10 * time_factor());
    while (groups_->info_calls() < target) {
      if (std::chrono::steady_clock::now() > deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }

  std::chrono::milliseconds period_;
  std::shared_ptr<FakeGroups> groups_;
  std::unique_ptr<amd::rdc::RdcConfigSettingsImpl> config_;
};

// The monitoring thread re-applies the settings every 10 ms.
class ConfigMonitor : public Config {
 protected:
  ConfigMonitor() : Config(std::chrono::milliseconds(10)) {}
};

}  // namespace

// Fakes for what RdcConfigSettingsImpl.cc calls.
namespace amd {
namespace rdc {
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_lookups.push_back(gpu_id);
  if (gpu_id >= kGpus || processor_handle == nullptr) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  *processor_handle = handle_of(gpu_id);
  return AMDSMI_STATUS_SUCCESS;
}
}  // namespace rdc
}  // namespace amd

amdsmi_status_t amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle,
                                          uint32_t sensor_ind, amdsmi_power_cap_info_t* info) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint32_t gpu = 0;
  if (auto status = enter(__func__, processor_handle, -1, 0, &gpu)) {
    return status;
  }
  if (info == nullptr || sensor_ind != 0) {
    return AMDSMI_STATUS_INVAL;
  }
  std::memset(info, 0, sizeof(*info));
  info->power_cap = g_gpus[gpu].cap_uw;
  info->default_power_cap = kDefaultCapUw;
  info->max_power_cap = std::numeric_limits<uint64_t>::max();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_power_cap(amdsmi_processor_handle processor_handle, uint32_t sensor_ind,
                                     uint64_t cap) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint32_t gpu = 0;
  if (auto status = enter(__func__, processor_handle, -1, cap, &gpu)) {
    return status;
  }
  if (sensor_ind != 0) {
    return AMDSMI_STATUS_INVAL;
  }
  g_gpus[gpu].cap_uw = cap;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_clock_info(amdsmi_processor_handle processor_handle,
                                      amdsmi_clk_type_t clk_type, amdsmi_clk_info_t* info) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint32_t gpu = 0;
  if (auto status = enter(__func__, processor_handle, clk_type, 0, &gpu)) {
    return status;
  }
  const uint64_t* max = max_mhz(&g_gpus[gpu], clk_type);
  if (info == nullptr || max == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  std::memset(info, 0, sizeof(*info));
  info->min_clk = 100;
  info->max_clk = static_cast<uint32_t>(*max);
  info->clk = info->max_clk;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle,
                                         amdsmi_clk_type_t clk_type,
                                         amdsmi_clk_limit_type_t limit_type, uint64_t clk_value) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint32_t gpu = 0;
  if (auto status = enter(__func__, processor_handle, clk_type, clk_value, &gpu)) {
    return status;
  }
  uint64_t* max = max_mhz(&g_gpus[gpu], clk_type);
  if (max == nullptr || limit_type != CLK_LIMIT_MAX) {
    return AMDSMI_STATUS_INVAL;
  }
  *max = clk_value;
  return AMDSMI_STATUS_SUCCESS;
}

// g_num_freq levels, the last one the clock's top. An empty table, or one with
// more levels than the array holds, breaks amd-smi's contract on purpose.
amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle,
                                    amdsmi_clk_type_t clk_type, amdsmi_frequencies_t* f) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint32_t gpu = 0;
  if (auto status = enter(__func__, processor_handle, clk_type, 0, &gpu)) {
    return status;
  }
  if (f == nullptr || max_mhz(&g_gpus[gpu], clk_type) == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  std::memset(f, 0, sizeof(*f));
  f->num_supported = g_num_freq;
  const uint32_t filled = std::min<uint32_t>(g_num_freq, AMDSMI_MAX_NUM_FREQUENCIES);
  for (uint32_t i = 0; i < filled; ++i) {
    f->frequency[i] = (100 + i) * kMicro;
  }
  if (filled > 0) {
    f->frequency[filled - 1] = (clk_type == AMDSMI_CLK_TYPE_MEM ? kTopMemMhz : kTopGfxMhz) * kMicro;
  }
  return AMDSMI_STATUS_SUCCESS;
}

// The automatic level puts both clock limits back at the top.
amdsmi_status_t amdsmi_set_gpu_perf_level(amdsmi_processor_handle processor_handle,
                                          amdsmi_dev_perf_level_t perf_lvl) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint32_t gpu = 0;
  if (auto status = enter(__func__, processor_handle, -1, perf_lvl, &gpu)) {
    return status;
  }
  if (perf_lvl == AMDSMI_DEV_PERF_LEVEL_AUTO) {
    g_gpus[gpu].gfx_max_mhz = kTopGfxMhz;
    g_gpus[gpu].mem_max_mhz = kTopMemMhz;
  }
  return AMDSMI_STATUS_SUCCESS;
}

// A setting reaches every GPU of the group, rdc_config_get() reports it, and
// rdc_config_clear() puts every GPU back to its default.
TEST_F(Config, SetReachesEveryGpuAndClearPutsThemBack) {
  ASSERT_EQ(set(kAll, kPower, 400), RDC_ST_OK);
  ASSERT_EQ(set(kAll, kGfx, 1500), RDC_ST_OK);
  for (uint32_t i = 0; i < kGpus; ++i) {
    EXPECT_EQ(gpu(i).cap_uw, 400 * kMicro) << "GPU " << i;
    EXPECT_EQ(gpu(i).gfx_max_mhz, 1500u) << "GPU " << i;
  }
  const auto got = get(kAll);
  EXPECT_EQ(got.first, RDC_ST_OK);
  EXPECT_EQ(got.second, (std::map<uint32_t, uint64_t>{{kGfx, 1500}, {kPower, 400}}));

  EXPECT_EQ(clear(kAll), RDC_ST_OK);
  for (uint32_t i = 0; i < kGpus; ++i) {
    EXPECT_EQ(gpu(i).cap_uw, kDefaultCapUw) << "GPU " << i;
    EXPECT_EQ(gpu(i).gfx_max_mhz, kTopGfxMhz) << "GPU " << i;
  }
  EXPECT_EQ(get(kAll).first, RDC_ST_NOT_FOUND);
}

// A type other than the three is a bad parameter: nothing reaches a GPU or
// stays cached. It used to be cached and reported as set.
TEST_F(Config, SetRejectsAnUnknownSettingType) {
  for (uint32_t type : {3u, 99u, 0x80000000u}) {
    EXPECT_EQ(set(kAll, type, 300), RDC_ST_BAD_PARAMETER) << "type " << type;
  }
  EXPECT_TRUE(calls().empty());
  EXPECT_EQ(get(kAll).first, RDC_ST_NOT_FOUND);
}

// An empty group is a bad parameter, whatever the stack held: the status used
// to be read uninitialized, so the setting was cached and reported as set, or
// failed, by chance.
TEST_F(Config, SetRejectsAnEmptyGroupWhateverTheStackHeld) {
  for (unsigned char byte : {0xA5, 0x5A, 0x00}) {
    dirty_stack(byte);
    ASSERT_TRUE(shows_dirt(byte)) << "this build leaves no dirt in a new frame";
    dirty_stack(byte);
    EXPECT_EQ(set(kNone, kPower, 400), RDC_ST_BAD_PARAMETER) << "stack byte " << int{byte};
  }
  EXPECT_TRUE(calls().empty());
  EXPECT_EQ(get(kNone).first, RDC_ST_NOT_FOUND);
}

// A group table that reports success without writing the group's GPUs gives an
// empty group, not the GPU IDs the stack held.
TEST_F(Config, SetReadsNoGpuFromGroupInfoLeftUnfilled) {
  groups_->report(kFirst, Report::kUnfilled);
  for (unsigned char byte : {0xA5, 0x5A}) {
    dirty_stack(byte);
    ASSERT_TRUE(shows_dirt(byte)) << "this build leaves no dirt in a new frame";
    dirty_stack(byte);
    EXPECT_EQ(set(kFirst, kPower, 400), RDC_ST_BAD_PARAMETER) << "stack byte " << int{byte};
  }
  EXPECT_TRUE(lookups().empty());
  EXPECT_TRUE(calls().empty());
}

TEST_F(Config, ClearReadsNoGpuFromGroupInfoLeftUnfilled) {
  ASSERT_EQ(set(kFirst, kPower, 400), RDC_ST_OK);
  const size_t looked_up = lookups().size();
  const size_t called = calls_so_far();
  groups_->report(kFirst, Report::kUnfilled);
  dirty_stack(0xA5);
  ASSERT_TRUE(shows_dirt(0xA5)) << "this build leaves no dirt in a new frame";
  dirty_stack(0xA5);
  EXPECT_EQ(clear(kFirst), RDC_ST_OK);
  EXPECT_EQ(lookups().size(), looked_up);
  EXPECT_TRUE(calls("", called).empty());
  EXPECT_EQ(get(kFirst).first, RDC_ST_NOT_FOUND);
}

// A count above what rdc_group_info_t holds fails the call before any GPU is
// looked up; it used to read entity IDs past the array.
TEST_F(Config, GroupInfoAboveCapacityIsAnError) {
  groups_->report(kAll, Report::kOverCapacity);
  EXPECT_EQ(set(kAll, kPower, 400), RDC_ST_UNKNOWN_ERROR);
  EXPECT_TRUE(lookups().empty());
  EXPECT_TRUE(calls().empty());

  groups_->report(kAll, Report::kFilled);
  ASSERT_EQ(set(kAll, kPower, 400), RDC_ST_OK);
  const size_t looked_up = lookups().size();
  groups_->report(kAll, Report::kOverCapacity);
  EXPECT_EQ(clear(kAll), RDC_ST_UNKNOWN_ERROR);
  EXPECT_EQ(lookups().size(), looked_up);
}

// A power limit whose microwatts do not fit in 64 bits is a bad parameter; it
// used to wrap around and set another limit (18446744074210 W set 500.448384 W).
TEST_F(Config, SetRejectsAPowerLimitTooLargeForMicrowatts) {
  const uint64_t largest = std::numeric_limits<uint64_t>::max() / kMicro;
  EXPECT_EQ(set(kFirst, kPower, largest + 1), RDC_ST_BAD_PARAMETER);
  EXPECT_EQ(set(kFirst, kPower, 18446744074210ULL), RDC_ST_BAD_PARAMETER);
  EXPECT_TRUE(calls("amdsmi_set_power_cap").empty());
  EXPECT_EQ(get(kFirst).first, RDC_ST_NOT_FOUND);

  // The largest one that fits reaches the GPU exactly.
  ASSERT_EQ(set(kFirst, kPower, largest), RDC_ST_OK);
  EXPECT_EQ(gpu(0).cap_uw, largest * kMicro);
}

// Clearing a clock limit when amd-smi reports an empty frequency table (or more
// levels than the table holds) puts the clock back to the automatic level. It
// used to read the level before the first one, far outside the table.
class ClockReset : public Config,
                   public ::testing::WithParamInterface<std::pair<uint32_t, uint32_t>> {};

TEST_P(ClockReset, ResetsToTheAutomaticLevel) {
  const uint32_t type = GetParam().first;
  const uint32_t levels = GetParam().second;
  ASSERT_EQ(set(kFirst, type, 1000), RDC_ST_OK);
  set_num_freq(levels);
  const size_t called = calls_so_far();
  EXPECT_EQ(clear(kFirst), RDC_ST_OK);
  EXPECT_EQ(gpus_of(calls("amdsmi_set_gpu_perf_level", called)), std::vector<uint32_t>{0});
  EXPECT_EQ(type == kGfx ? gpu(0).gfx_max_mhz : gpu(0).mem_max_mhz,
            type == kGfx ? kTopGfxMhz : kTopMemMhz);
}

INSTANTIATE_TEST_SUITE_P(Config, ClockReset,
                         ::testing::Values(std::make_pair(kGfx, 0u), std::make_pair(kMem, 0u),
                                           std::make_pair(kGfx, AMDSMI_MAX_NUM_FREQUENCIES + 1u),
                                           std::make_pair(kMem, AMDSMI_MAX_NUM_FREQUENCIES + 1u)),
                         [](const ::testing::TestParamInfo<std::pair<uint32_t, uint32_t>>& info) {
                           return std::string(info.param.first == kGfx ? "Gfx" : "Mem") +
                                  (info.param.second == 0 ? "EmptyTable" : "OversizedTable");
                         });

// A set that a GPU rejects puts the GPUs it already changed back to their
// default, leaves the GPUs after it alone and caches nothing. It used to leave
// the changed GPUs at the new value, which nothing cleared.
TEST_F(Config, ARejectedSetPutsTheChangedGpusBackToTheirDefault) {
  fail("amdsmi_set_power_cap", 2);
  EXPECT_EQ(set(kAll, kPower, 400), RDC_ST_UNKNOWN_ERROR);
  for (uint32_t i = 0; i < kGpus; ++i) {
    EXPECT_EQ(gpu(i).cap_uw, kDefaultCapUw) << "GPU " << i;
  }
  for (const auto& call : calls("amdsmi_set_power_cap")) {
    EXPECT_NE(call.gpu, 3u) << "GPU 3 comes after the GPU that failed";
  }
  EXPECT_EQ(get(kAll).first, RDC_ST_NOT_FOUND);
}

// ... or back to the group's previous setting of that type, which stays.
TEST_F(Config, ARejectedSetPutsTheChangedGpusBackToTheGroupsSetting) {
  ASSERT_EQ(set(kAll, kPower, 400), RDC_ST_OK);
  fail("amdsmi_set_power_cap", 2);
  EXPECT_EQ(set(kAll, kPower, 500), RDC_ST_UNKNOWN_ERROR);
  for (uint32_t i = 0; i < kGpus; ++i) {
    EXPECT_EQ(gpu(i).cap_uw, 400 * kMicro) << "GPU " << i;
  }
  const auto got = get(kAll);
  EXPECT_EQ(got.first, RDC_ST_OK);
  EXPECT_EQ(got.second, (std::map<uint32_t, uint64_t>{{kPower, 400}}));
}

TEST_F(Config, ARejectedClockLimitPutsTheChangedGpusBackToTheAutomaticLevel) {
  fail("amdsmi_set_gpu_clk_limit", 1);
  EXPECT_EQ(set(kAll, kGfx, 1500), RDC_ST_UNKNOWN_ERROR);
  for (uint32_t i = 0; i < kGpus; ++i) {
    EXPECT_EQ(gpu(i).gfx_max_mhz, kTopGfxMhz) << "GPU " << i;
  }
  EXPECT_EQ(gpus_of(calls("amdsmi_set_gpu_perf_level")), std::vector<uint32_t>{0});
  EXPECT_EQ(get(kAll).first, RDC_ST_NOT_FOUND);
}

// A GPU that cannot be reset does not stop the others, or its own other
// settings, from being reset; the call reports the failure, and the group's
// settings are gone. It used to stop at the first failure. Each of GPU 1's two
// settings fails in turn, so that the one the clear visits first fails once.
TEST_F(Config, ClearResetsEveryGpuAndSettingWhenOneFails) {
  for (const std::string api : {"amdsmi_set_power_cap", "amdsmi_set_gpu_perf_level"}) {
    reset_fakes();
    ASSERT_EQ(set(kAll, kPower, 400), RDC_ST_OK);
    ASSERT_EQ(set(kAll, kGfx, 1500), RDC_ST_OK);
    fail(api, 1);
    EXPECT_EQ(clear(kAll), RDC_ST_UNKNOWN_ERROR) << api;
    const bool power_failed = api == "amdsmi_set_power_cap";
    for (uint32_t i = 0; i < kGpus; ++i) {
      EXPECT_EQ(gpu(i).cap_uw, i == 1 && power_failed ? 400 * kMicro : kDefaultCapUw)
          << api << ", GPU " << i;
      EXPECT_EQ(gpu(i).gfx_max_mhz, i == 1 && !power_failed ? 1500 : kTopGfxMhz)
          << api << ", GPU " << i;
    }
    EXPECT_EQ(get(kAll).first, RDC_ST_NOT_FOUND) << api;
  }
}

// The monitoring thread puts a power cap that changed back, and leaves one that
// did not alone, at any size: it used to compare whole watts converted through
// an int, which re-applied every cap above 2147 W on every pass and missed a
// change of less than a watt.
TEST_F(ConfigMonitor, LeavesAnUnchangedPowerCapAbove2147WattsAlone) {
  ASSERT_EQ(set(kFirst, kPower, 3000), RDC_ST_OK);
  const size_t called = calls_so_far();
  ASSERT_TRUE(wait_for_passes(3, 1)) << "the monitoring thread made no pass";
  EXPECT_TRUE(calls("amdsmi_set_power_cap", called).empty());
}

TEST_F(ConfigMonitor, PutsBackAPowerCapChangedByLessThanAWatt) {
  ASSERT_EQ(set(kFirst, kPower, 300), RDC_ST_OK);
  Gpu changed = gpu(0);
  changed.cap_uw = 300 * kMicro + kMicro / 2;
  set_gpu(0, changed);
  ASSERT_TRUE(wait_for_passes(3, 1)) << "the monitoring thread made no pass";
  EXPECT_EQ(gpu(0).cap_uw, 300 * kMicro);
}

// Each pass looks a group's GPUs up afresh: a group whose lookup succeeds
// without filling its GPUs in gets no GPU, not the previous group's. It used to
// apply its settings to the GPUs of the group before it.
TEST_F(ConfigMonitor, AppliesNoGroupsSettingToAnotherGroupsGpus) {
  ASSERT_EQ(set(kFirst, kPower, 300), RDC_ST_OK);
  ASSERT_EQ(set(kSecond, kPower, 500), RDC_ST_OK);
  groups_->report(kSecond, Report::kUnfilled);
  const size_t called = calls_so_far();
  ASSERT_TRUE(wait_for_passes(3, 2)) << "the monitoring thread made no pass";
  for (const auto& call : calls("amdsmi_set_power_cap", called)) {
    EXPECT_FALSE(call.gpu == 0 && call.value == 500 * kMicro)
        << "GPU 0 got the second group's limit";
  }
}

// The monitoring thread reads only what is set: with a clock limit only, no
// power cap and no other clock.
TEST_F(ConfigMonitor, ReadsOnlyTheSettingsOfTheGroup) {
  ASSERT_EQ(set(kFirst, kGfx, 1500), RDC_ST_OK);
  const size_t called = calls_so_far();
  ASSERT_TRUE(wait_for_passes(3, 1)) << "the monitoring thread made no pass";
  EXPECT_TRUE(calls("amdsmi_get_power_cap_info", called).empty());
  for (const auto& call : calls("amdsmi_get_clock_info", called)) {
    EXPECT_EQ(call.clk, AMDSMI_CLK_TYPE_GFX);
  }
}

// A power cap that cannot be read does not keep a clock limit from being put
// back. The thread used to skip the GPU's other settings after a failed read.
TEST_F(ConfigMonitor, PutsBackAClockLimitWhenThePowerCapCannotBeRead) {
  ASSERT_EQ(set(kFirst, kPower, 400), RDC_ST_OK);
  ASSERT_EQ(set(kFirst, kGfx, 1500), RDC_ST_OK);
  fail("amdsmi_get_power_cap_info", 0);
  Gpu changed = gpu(0);
  changed.gfx_max_mhz = kTopGfxMhz;
  set_gpu(0, changed);
  ASSERT_TRUE(wait_for_passes(3, 1)) << "the monitoring thread made no pass";
  EXPECT_EQ(gpu(0).gfx_max_mhz, 1500u);
}
