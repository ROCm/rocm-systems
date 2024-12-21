/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocp/RdcRocpBase.h"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// #include "hsa.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"
#include "rdc_modules/rdc_rocp/RdcRocpCounterSampler.h"

namespace amd {
namespace rdc {

double RdcRocpBase::run_profiler(uint32_t gpu_index, rdc_field_t field) {
  thread_local std::vector<rocprofiler_record_counter_t> records;

  // initialize hsa. hsa_init() will also load the profiler libs under the hood
  hsa_status_t status = HSA_STATUS_SUCCESS;

  auto counter_sampler = CounterSampler::get_samplers()[gpu_index];
  if (!counter_sampler) {
    RDC_LOG(RDC_ERROR, "Error: Counter sampler not found for GPU index " << gpu_index << std::endl);
    throw std::runtime_error("Counter sampler not found");
  }

  auto field_it = field_to_metric.find(field);
  if (field_it == field_to_metric.end()) {
    RDC_LOG(RDC_ERROR,
            "Error: Field " << field << " not found in field_to_metric map." << std::endl);
    throw std::out_of_range("Field not found in field_to_metric map");
  }
  const std::string& metric_id = field_it->second;

  try {
    counter_sampler->sample_counter_values({metric_id}, records, collection_duration_us_k);
  } catch (const std::exception& e) {
    RDC_LOG(RDC_ERROR, "Error while sampling counter values: " << e.what() << std::endl);
    throw;
  }

  // Aggregate counter values. Rocprof v1/v2 summed values across dimensions.
  double value = 0.0;
  for (auto& record : records) {
    value += record.counter_value;  // Summing up values from all dimensions.
  }

  return value;
}

const char* RdcRocpBase::get_field_id_from_name(rdc_field_t field) {
  auto it = field_to_metric.find(field);
  if (it == field_to_metric.end()) {
    RDC_LOG(RDC_ERROR,
            "Error: Field ID " << field << " not found in field_to_metric map." << std::endl);
    throw std::out_of_range("Field ID not found in field_to_metric map");
  }

  return field_to_metric.at(field);
}

const std::vector<rdc_field_t> RdcRocpBase::get_field_ids() {
  std::vector<rdc_field_t> field_ids;
  for (auto& [k, v] : field_to_metric) {
    field_ids.push_back(k);
  }
  return field_ids;
}

RdcRocpBase::RdcRocpBase() {
  // all fields
  static const std::map<rdc_field_t, const char*> temp_field_map_k = {
      {RDC_FI_PROF_OCCUPANCY_PERCENT, "OccupancyPercent"},
      {RDC_FI_PROF_ACTIVE_CYCLES, "GRBM_GUI_ACTIVE"},
      {RDC_FI_PROF_ACTIVE_WAVES, "SQ_WAVES"},
      {RDC_FI_PROF_ELAPSED_CYCLES, "GRBM_COUNT"},
      {RDC_FI_PROF_TENSOR_ACTIVE_PERCENT,
       "MfmaUtil"},  // same as TENSOR_ACTIVE but available for more GPUs
      {RDC_FI_PROF_GPU_UTIL_PERCENT, "GPU_UTIL"},
      // metrics below are divided by time passed
      {RDC_FI_PROF_EVAL_MEM_R_BW, "FETCH_SIZE"},
      {RDC_FI_PROF_EVAL_MEM_W_BW, "WRITE_SIZE"},
      {RDC_FI_PROF_EVAL_FLOPS_16, "TOTAL_16_OPS"},
      {RDC_FI_PROF_EVAL_FLOPS_32, "TOTAL_32_OPS"},
      {RDC_FI_PROF_EVAL_FLOPS_64, "TOTAL_64_OPS"},
      {RDC_FI_PROF_VALU_PIPE_ISSUE_UTIL, "ValuPipeIssueUtil"},
      {RDC_FI_PROF_SM_ACTIVE, "VALUBusy"},
  };

  hsa_status_t status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    const char* errstr = nullptr;
    hsa_status_string(status, &errstr);
    throw std::runtime_error("hsa error code: " + std::to_string(status) + " " + errstr);
  }

  // check rocprofiler
  if (int rocp_status = 0;
      rocprofiler_is_initialized(&rocp_status) == ROCPROFILER_STATUS_SUCCESS && rocp_status != 1) {
    throw std::runtime_error("Rocprofiler is not initialized. status: " +
                             std::to_string(rocp_status));
  }

  std::vector<std::string> all_fields;
  std::vector<std::string> checked_fields;

  // populate list of agents
  agents = CounterSampler::get_available_agents();
  RDC_LOG(RDC_DEBUG, "Agent count: " << agents.size());
  samplers = CounterSampler::get_samplers();

  // populate fields
  for (auto& [k, v] : temp_field_map_k) {
    all_fields.push_back(v);
  }

  // find intersection of supported and requested fields
  for (uint32_t gpu_index = 0; gpu_index < agents.size(); gpu_index++) {
    auto& cs = *samplers[gpu_index];
    RDC_LOG(RDC_DEBUG,
            "gpu_index[" << gpu_index << "] = node_id[" << agents[gpu_index].node_id << "]");
    for (auto& [str, id] : cs.get_supported_counters(cs.get_agent())) {
      checked_fields.emplace_back(str);
    }

    for (auto& [k, v] : temp_field_map_k) {
      auto found = std::find(checked_fields.begin(), checked_fields.end(), v);
      if (found != checked_fields.end()) {
        field_to_metric.insert({k, v});
      }
    }
  }

  RDC_LOG(RDC_DEBUG, "Rocprofiler supports " << field_to_metric.size() << " fields");
}

RdcRocpBase::~RdcRocpBase() {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  status = hsa_shut_down();
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_shut_down();
  assert(status == HSA_STATUS_ERROR_NOT_INITIALIZED);
}

rdc_status_t RdcRocpBase::rocp_lookup(rdc_gpu_field_t gpu_field, double* value) {
  const auto& gpu_index = gpu_field.gpu_index;
  const auto& field = gpu_field.field_id;

  if (value == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  const auto start_time = std::chrono::high_resolution_clock::now();
  *value = run_profiler(gpu_index, field);
  const auto stop_time = std::chrono::high_resolution_clock::now();
  // extra processing required
  if (eval_fields.find(field) != eval_fields.end()) {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    *value = *value / elapsed;
  }
  // GPU_UTIL metric is available on more GPUs than ENGINE_ACTIVE.
  // ENGINE_ACTIVE = GPU_UTIL/100, so do the math ourselves
  if (field == RDC_FI_PROF_GPU_UTIL_PERCENT) {
    *value = *value / 100.0F;
  }
  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
