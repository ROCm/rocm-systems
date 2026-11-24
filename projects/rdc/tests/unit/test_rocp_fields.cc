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

// The profiler fields RdcRocpBase derives from its counters: the FLOPS percents, which divide
// the ops per SIMD per cycle by the GPU's peak, and OCC_ELAPSED, which spreads the occupancy per
// active CU over the elapsed cycles. Both read close to 0 on a busy GPU when RdcRocpBase divided
// the FLOPS percents by the sample window and OCC_ELAPSED by the active cycles.
//
// The binary compiles RdcRocpBase.cc only. The counter sampler and the HSA, rocprofiler-sdk and
// AMD SMI calls it makes are replaced by the fakes below, so it needs neither the profiler nor a
// GPU. librdc_bootstrap provides the logger and the entity codec.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_modules/rdc_rocp/RdcRocpBase.h"
#include "rdc_modules/rdc_rocp/RdcRocpCounterSampler.h"

namespace {

// The GPUs of the fake, by RDC GPU index, which is also their profiler agent index. The counter
// values are what every sample of that GPU reads; a counter that is absent is not sampled.
struct FakeGpu {
  const char* name;
  uint32_t simd_per_cu;
  std::map<std::string, double> counters;
};
std::vector<FakeGpu> g_gpus;

// The counters and window of the last sample.
std::vector<std::string> g_sampled;
uint64_t g_duration_us = 0;

uint64_t kfd_id(size_t gpu) { return 0x1000 + gpu; }

// The GPU an agent or a processor handle stands for: its index plus 1.
size_t gpu_of(uint64_t handle) { return static_cast<size_t>(handle - 1); }

}  // namespace

extern "C" {

hsa_status_t hsa_init() { return HSA_STATUS_SUCCESS; }

hsa_status_t hsa_shut_down() { return HSA_STATUS_SUCCESS; }

hsa_status_t hsa_status_string(hsa_status_t, const char** status_string) {
  *status_string = "fake";
  return HSA_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_is_initialized(int* status) {
  *status = 1;
  return ROCPROFILER_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_kfd_info(amdsmi_processor_handle processor_handle,
                                        amdsmi_kfd_info_t* info) {
  *info = {};
  info->kfd_id = kfd_id(gpu_of(reinterpret_cast<uintptr_t>(processor_handle)));
  return AMDSMI_STATUS_SUCCESS;
}

}  // extern "C"

namespace amd {
namespace rdc {

std::shared_ptr<const GpuHandleTable> get_flat_gpu_table() {
  auto table = std::make_shared<GpuHandleTable>();
  for (size_t gpu = 0; gpu < g_gpus.size(); ++gpu) {
    auto* handle = reinterpret_cast<amdsmi_processor_handle>(static_cast<uintptr_t>(gpu + 1));
    table->push_back({handle, static_cast<uint32_t>(gpu), 0, true});
  }
  return table;
}

CounterSampler::CounterSampler(rocprofiler_agent_id_t agent) : agent_(agent) {}

CounterSampler::~CounterSampler() = default;

std::vector<std::shared_ptr<CounterSampler>>& CounterSampler::get_samplers() {
  static std::vector<std::shared_ptr<CounterSampler>> samplers;
  samplers.clear();
  for (size_t gpu = 0; gpu < g_gpus.size(); ++gpu) {
    samplers.push_back(std::make_shared<CounterSampler>(rocprofiler_agent_id_t{gpu + 1}));
  }
  return samplers;
}

std::vector<rocprofiler_agent_v0_t> CounterSampler::get_available_agents() {
  std::vector<rocprofiler_agent_v0_t> agents;
  for (size_t gpu = 0; gpu < g_gpus.size(); ++gpu) {
    rocprofiler_agent_v0_t agent = {};
    agent.size = sizeof(agent);
    agent.id.handle = gpu + 1;
    agent.type = ROCPROFILER_AGENT_TYPE_GPU;
    agent.name = g_gpus[gpu].name;
    agent.simd_per_cu = g_gpus[gpu].simd_per_cu;
    agent.gpu_id = kfd_id(gpu);
    agents.push_back(agent);
  }
  return agents;
}

std::unordered_map<std::string, rocprofiler_counter_id_t> CounterSampler::get_supported_counters(
    rocprofiler_agent_id_t agent) {
  std::unordered_map<std::string, rocprofiler_counter_id_t> supported;
  for (const auto& [name, value] : g_gpus[gpu_of(agent.handle)].counters) {
    supported[name] = rocprofiler_counter_id_t{supported.size() + 1};
  }
  return supported;
}

void CounterSampler::sample_counter_values(const std::vector<std::string>& counters,
                                           std::vector<rocprofiler_record_counter_t>& out,
                                           uint64_t duration) {
  g_sampled = counters;
  g_duration_us = duration;
  out.clear();
  for (const auto& name : counters) {
    const auto& values = g_gpus[gpu_of(agent_.handle)].counters;
    const auto it = values.find(name);
    if (it != values.end()) {
      rocprofiler_record_counter_t record = {};
      record.counter_value = it->second;
      out.push_back(record);
    }
  }
}

void CounterSampler::sample_counters_with_packing(const std::vector<std::string>& counters,
                                                  std::map<std::string, double>& out_values,
                                                  uint64_t duration) {
  g_sampled = counters;
  g_duration_us = duration;
  for (const auto& name : counters) {
    const auto& values = g_gpus[gpu_of(agent_.handle)].counters;
    const auto it = values.find(name);
    if (it != values.end()) {
      out_values[name] = it->second;
    }
  }
}

}  // namespace rdc
}  // namespace amd

namespace {

using amd::rdc::RdcRocpBase;

constexpr uint32_t kMi300 = 0;
constexpr uint32_t kMi200 = 1;

// One field of one GPU: the lookup's status, the field's status and its value.
struct Read {
  rdc_status_t status = RDC_ST_UNKNOWN_ERROR;
  rdc_status_t field_status = RDC_ST_UNKNOWN_ERROR;
  double value = NAN;
};

class RocpFieldsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_gpus = {{"gfx942", 4, {}}, {"gfx90a", 4, {}}};
    g_sampled.clear();
    g_duration_us = 0;
  }

  // Reads fields of one GPU in one batch, the way the telemetry module does.
  std::vector<Read> bulk(uint32_t gpu, const std::vector<rdc_field_t>& fields) {
    std::vector<rdc_gpu_field_t> gpu_fields;
    for (const auto field : fields) {
      gpu_fields.push_back({gpu, field});
    }
    std::vector<rdc_field_value_data> values;
    std::vector<rdc_field_type_t> types;
    std::vector<rdc_status_t> statuses;
    const rdc_status_t status = rocp_.rocp_lookup_bulk(gpu_fields, values, types, statuses);
    std::vector<Read> reads(fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
      reads[i].status = status;
      if (i < statuses.size()) reads[i].field_status = statuses[i];
      if (i < values.size()) reads[i].value = values[i].dbl;
    }
    return reads;
  }

  Read bulk(uint32_t gpu, rdc_field_t field) { return bulk(gpu, std::vector{field})[0]; }

  // Reads one field with rocp_lookup().
  Read single(uint32_t gpu, rdc_field_t field) {
    rdc_field_value_data value = {};
    rdc_field_type_t type = INTEGER;
    Read read;
    read.status = rocp_.rocp_lookup({gpu, field}, &value, &type);
    read.field_status = read.status;
    read.value = value.dbl;
    return read;
  }

  bool sampled(const std::string& counter) const {
    return std::find(g_sampled.begin(), g_sampled.end(), counter) != g_sampled.end();
  }

  RdcRocpBase rocp_;
};

// RDC_OPS_*_PER_SIMDCYCLE is ops per SIMD per cycle. An MI300 series GPU peaks at 2048 fp16 and
// 256 fp32 or fp64 ops per CU per cycle, so with 4 SIMDs per CU at 512 and 64 per SIMD.
TEST_F(RocpFieldsTest, FlopsPercentsAreTheRatioToThePeak) {
  g_gpus[kMi300].counters = {{"RDC_OPS_16_PER_SIMDCYCLE", 384.0},
                             {"RDC_OPS_32_PER_SIMDCYCLE", 48.0},
                             {"RDC_OPS_64_PER_SIMDCYCLE", 16.0}};

  const auto reads =
      bulk(kMi300, {RDC_FI_PROF_EVAL_FLOPS_16_PERCENT, RDC_FI_PROF_EVAL_FLOPS_32_PERCENT,
                    RDC_FI_PROF_EVAL_FLOPS_64_PERCENT});

  for (const auto& read : reads) {
    EXPECT_EQ(read.status, RDC_ST_OK);
    EXPECT_EQ(read.field_status, RDC_ST_OK);
  }
  EXPECT_DOUBLE_EQ(reads[0].value, 0.75);
  EXPECT_DOUBLE_EQ(reads[1].value, 0.75);
  EXPECT_DOUBLE_EQ(reads[2].value, 0.25);
}

// SIMD_UTILIZATION makes the batch sample for 500 ms instead of 100 ms. The FLOPS percents are
// per cycle, so the window must not change them; the rate fields are per millisecond of it.
TEST_F(RocpFieldsTest, FlopsPercentsDoNotDependOnTheSampleWindow) {
  g_gpus[kMi300].counters = {
      {"RDC_OPS_32_PER_SIMDCYCLE", 48.0}, {"TOTAL_32_OPS", 1.0e9}, {"SIMD_UTILIZATION", 0.5}};

  const auto reads = bulk(kMi300, {RDC_FI_PROF_EVAL_FLOPS_32_PERCENT, RDC_FI_PROF_EVAL_FLOPS_32});
  EXPECT_EQ(g_duration_us, 100000u);
  const auto with_simd = bulk(kMi300, {RDC_FI_PROF_EVAL_FLOPS_32_PERCENT, RDC_FI_PROF_EVAL_FLOPS_32,
                                       RDC_FI_PROF_SIMD_UTILIZATION});
  EXPECT_EQ(g_duration_us, 500000u);

  EXPECT_DOUBLE_EQ(reads[0].value, 0.75);
  EXPECT_DOUBLE_EQ(with_simd[0].value, 0.75);
  EXPECT_DOUBLE_EQ(reads[1].value, 1.0e9 / 100.0);
  EXPECT_DOUBLE_EQ(with_simd[1].value, 1.0e9 / 500.0);
}

// An MI200 series GPU peaks at 1024 fp16 ops per CU per cycle.
TEST_F(RocpFieldsTest, Fp16PercentUsesTheMi200Peak) {
  g_gpus[kMi200].counters = {{"RDC_OPS_16_PER_SIMDCYCLE", 192.0}};

  const Read read = bulk(kMi200, RDC_FI_PROF_EVAL_FLOPS_16_PERCENT);

  EXPECT_EQ(read.field_status, RDC_ST_OK);
  EXPECT_DOUBLE_EQ(read.value, 0.75);
}

// The GPU was active half of the elapsed cycles, with 40 waves per active CU while it was.
TEST_F(RocpFieldsTest, OccElapsedSpreadsTheOccupancyOverTheElapsedCycles) {
  g_gpus[kMi300].counters = {
      {"MeanOccupancyPerActiveCU", 40.0}, {"GRBM_GUI_ACTIVE", 3.0e8}, {"GRBM_COUNT", 6.0e8}};

  const Read read = bulk(kMi300, RDC_FI_PROF_OCC_ELAPSED);

  EXPECT_TRUE(sampled("GRBM_COUNT"));
  EXPECT_EQ(read.status, RDC_ST_OK);
  EXPECT_EQ(read.field_status, RDC_ST_OK);
  EXPECT_DOUBLE_EQ(read.value, 20.0);
}

TEST_F(RocpFieldsTest, OccElapsedOfAnIdleGpuIsZero) {
  g_gpus[kMi300].counters = {
      {"MeanOccupancyPerActiveCU", 0.0}, {"GRBM_GUI_ACTIVE", 0.0}, {"GRBM_COUNT", 6.0e8}};

  const Read read = bulk(kMi300, RDC_FI_PROF_OCC_ELAPSED);

  EXPECT_EQ(read.field_status, RDC_ST_OK);
  EXPECT_DOUBLE_EQ(read.value, 0.0);
}

TEST_F(RocpFieldsTest, OccElapsedFailsWithZeroElapsedCycles) {
  g_gpus[kMi300].counters = {
      {"MeanOccupancyPerActiveCU", 40.0}, {"GRBM_GUI_ACTIVE", 3.0e8}, {"GRBM_COUNT", 0.0}};

  const Read read = bulk(kMi300, RDC_FI_PROF_OCC_ELAPSED);

  EXPECT_EQ(read.status, RDC_ST_OK);
  EXPECT_EQ(read.field_status, RDC_ST_BAD_PARAMETER);
}

TEST_F(RocpFieldsTest, OccElapsedFailsWithoutAnInput) {
  for (const std::string missing : {"MeanOccupancyPerActiveCU", "GRBM_COUNT"}) {
    g_gpus[kMi300].counters = {
        {"MeanOccupancyPerActiveCU", 40.0}, {"GRBM_GUI_ACTIVE", 3.0e8}, {"GRBM_COUNT", 6.0e8}};
    g_gpus[kMi300].counters.erase(missing);

    const Read read = bulk(kMi300, RDC_FI_PROF_OCC_ELAPSED);

    EXPECT_EQ(read.status, RDC_ST_OK) << missing;
    EXPECT_EQ(read.field_status, RDC_ST_BAD_PARAMETER) << missing;
  }
}

// The fields OCC_ELAPSED is derived from can be in the batch too, before or after it: each input
// is sampled once, and every field reads its own value. (ACTIVE_CYCLES and OCC_ELAPSED both list
// GRBM_GUI_ACTIVE; rocprofiler-sdk ignores the repeated counter.)
TEST_F(RocpFieldsTest, OccElapsedWithItsInputsInTheBatch) {
  g_gpus[kMi300].counters = {
      {"MeanOccupancyPerActiveCU", 40.0}, {"GRBM_GUI_ACTIVE", 3.0e8}, {"GRBM_COUNT", 6.0e8}};
  const std::map<rdc_field_t, double> expected = {{RDC_FI_PROF_OCC_ELAPSED, 20.0},
                                                  {RDC_FI_PROF_ELAPSED_CYCLES, 6.0e8},
                                                  {RDC_FI_PROF_ACTIVE_CYCLES, 3.0e8},
                                                  {RDC_FI_PROF_OCC_PER_ACTIVE_CU, 40.0}};
  std::vector<rdc_field_t> fields;
  for (const auto& [field, value] : expected) {
    fields.push_back(field);
  }

  for (const auto& order : {fields, std::vector<rdc_field_t>(fields.rbegin(), fields.rend())}) {
    const auto reads = bulk(kMi300, order);

    for (size_t i = 0; i < order.size(); ++i) {
      EXPECT_EQ(reads[i].field_status, RDC_ST_OK) << order[i];
      EXPECT_DOUBLE_EQ(reads[i].value, expected.at(order[i])) << order[i];
    }
    EXPECT_EQ(std::count(g_sampled.begin(), g_sampled.end(), "GRBM_COUNT"), 1);
    EXPECT_EQ(std::count(g_sampled.begin(), g_sampled.end(), "MeanOccupancyPerActiveCU"), 1);
  }
}

// rocp_lookup() reads each counter in its own sample, and must derive the same values. It looks
// the GPU up before it starts the profiler, so a bulk read starts the profiler first.
TEST_F(RocpFieldsTest, SingleLookupDerivesTheSameValues) {
  g_gpus[kMi300].counters = {{"RDC_OPS_32_PER_SIMDCYCLE", 48.0},
                             {"MeanOccupancyPerActiveCU", 40.0},
                             {"GRBM_GUI_ACTIVE", 3.0e8},
                             {"GRBM_COUNT", 6.0e8}};
  ASSERT_EQ(bulk(kMi300, RDC_FI_PROF_ACTIVE_CYCLES).field_status, RDC_ST_OK);

  const Read percent = single(kMi300, RDC_FI_PROF_EVAL_FLOPS_32_PERCENT);
  const Read occupancy = single(kMi300, RDC_FI_PROF_OCC_ELAPSED);

  EXPECT_EQ(percent.status, RDC_ST_OK);
  EXPECT_DOUBLE_EQ(percent.value, 0.75);
  EXPECT_EQ(occupancy.status, RDC_ST_OK);
  EXPECT_DOUBLE_EQ(occupancy.value, 20.0);
}

}  // namespace
