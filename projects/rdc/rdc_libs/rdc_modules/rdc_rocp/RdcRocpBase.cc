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

#include <rocprofiler/rocprofiler.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <utility>

// #include "hsa.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

static hsa_status_t get_agent_handle_cb(hsa_agent_t agent, void* agent_arr) {
  hsa_device_type_t type;
  hsa_agent_arr_t* agent_arr_ = (hsa_agent_arr_t*)agent_arr;

  hsa_status_t hsa_errno = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    return hsa_errno;
  }

  if (type == HSA_DEVICE_TYPE_GPU) {
    if (agent_arr_->count >= agent_arr_->capacity) {
      agent_arr_->capacity *= 2;
      agent_arr_->agents =
          (hsa_agent_t*)realloc(agent_arr_->agents, agent_arr_->capacity * sizeof(hsa_agent_t));
      // realloc might set agents to nullptr upon failure
      assert(agent_arr_->agents != nullptr);
    }
    agent_arr_->agents[agent_arr_->count] = agent;
    ++agent_arr_->count;
  }

  return HSA_STATUS_SUCCESS;
}

void RdcRocpBase::read_features(rocprofiler_t* context, const unsigned feature_count) {
  hsa_status_t hsa_errno = rocprofiler_read(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  hsa_errno = rocprofiler_get_data(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  hsa_errno = rocprofiler_get_metrics(context);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  for (auto i = 0; i < feature_count; i++) {
    switch (features[0][i].data.kind) {
      case ROCPROFILER_DATA_KIND_DOUBLE:
        metrics[features[0][i].name] = features[0][i].data.result_double;
        break;
      default:
        RDC_LOG(RDC_ERROR, "ERROR: Unexpected feature kind: " << features[0][i].data.kind);
    }
  }
}

static int get_agents(hsa_agent_arr_t* agent_arr) {
  int errcode = 0;
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;

  agent_arr->capacity = 1;
  agent_arr->count = 0;
  agent_arr->agents = (hsa_agent_t*)calloc(agent_arr->capacity, sizeof(hsa_agent_t));
  assert(agent_arr->agents);

  hsa_errno = hsa_iterate_agents(get_agent_handle_cb, agent_arr);
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    errcode = -1;

    agent_arr->capacity = 0;
    agent_arr->count = 0;
    free(agent_arr->agents);
  }

  return errcode;
}

bool createHsaQueue(hsa_queue_t** queue, hsa_agent_t gpu_agent) {
  // create a single-producer queue
  // TODO: check if API args are correct, especially UINT32_MAX
  hsa_status_t status;
  status = hsa_queue_create(gpu_agent, 64, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX,
                            UINT32_MAX, queue);
  if (status != HSA_STATUS_SUCCESS) fprintf(stdout, "Queue creation failed");

  // TODO: warning: is it really required!! ??
  status = hsa_amd_queue_set_priority(*queue, HSA_AMD_QUEUE_PRIORITY_HIGH);
  if (status != HSA_STATUS_SUCCESS) fprintf(stdout, "HSA Queue Priority Set Failed");

  return (status == HSA_STATUS_SUCCESS);
}

int RdcRocpBase::run_profiler(const char* feature_name) {
  const char* events[features_count] = {feature_name};

  // initialize hsa. hsa_init() will also load the profiler libs under the hood
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;

  for (int i = 0; i < dev_count; ++i) {
    for (int j = 0; j < features_count; ++j) {
      features[i][j].kind = (rocprofiler_feature_kind_t)ROCPROFILER_FEATURE_KIND_METRIC;
      features[i][j].name = events[j];
    }
  }

  rocprofiler_t* contexts[dev_count] = {0};
  for (int i = 0; i < dev_count; ++i) {
    rocprofiler_properties_t properties = {
        queues[i],
        64,
        NULL,
        NULL,
    };
    int mode = (ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_SINGLEGROUP);
    hsa_errno = rocprofiler_open(agent_arr.agents[i], features[i], features_count, &contexts[i],
                                 mode, &properties);
    const char* error_string;
    rocprofiler_error_string(&error_string);
    if (error_string != NULL) {
      fprintf(stdout, "%s", error_string);
      fflush(stdout);
    }
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < dev_count; ++i) {
    hsa_errno = rocprofiler_start(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  // this is the duration for which the counter increments from zero.
  usleep(10000);

  for (int i = 0; i < dev_count; ++i) {
    hsa_errno = rocprofiler_stop(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < dev_count; ++i) {
    // printf("Iteration %d\n", loopcount++);
    // fprintf(stdout, "------ Collecting Device[%d] -------\n", i);
    read_features(contexts[i], features_count);
  }

  usleep(100);

  for (int i = 0; i < dev_count; ++i) {
    hsa_errno = rocprofiler_close(contexts[i]);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  return 0;
}

const char* RdcRocpBase::get_field_id_from_name(rdc_field_t field) {
  return counter_map_k.at(field);
}

const std::vector<rdc_field_t> RdcRocpBase::get_field_ids() {
  std::vector<rdc_field_t> field_ids;
  for (auto& [k, v] : counter_map_k) {
    field_ids.push_back(k);
  }
  return field_ids;
}

RdcRocpBase::RdcRocpBase() {
  counter_map_k = {
      {RDC_FI_PROF_CU_UTILIZATION, "CU_UTILIZATION"},
      {RDC_FI_PROF_CU_OCCUPANCY, "CU_OCCUPANCY"},
      {RDC_FI_PROF_FLOPS_16, "FLOPS_16"},
      {RDC_FI_PROF_FLOPS_32, "FLOPS_32"},
      {RDC_FI_PROF_FLOPS_64, "FLOPS_64"},
      {RDC_FI_PROF_ACTIVE_CYCLES, "ACTIVE_CYCLES"},
      {RDC_FI_PROF_ACTIVE_WAVES, "ACTIVE_WAVES"},
      {RDC_FI_PROF_ELAPSED_CYCLES, "ELAPSED_CYCLES"},
  };

  // populate monitored fields
  std::cout << "Size of counter_map_k: " << counter_map_k.size() << "\n";

  for (auto& k : counter_map_k) {
    printf("metric %d = %s\n", k.first, k.second);
  }
  for (auto& [k, v] : counter_map_k) {
    const char* str = v;
    metrics.emplace(std::make_pair(str, 0.0));
  }
  assert(metrics.size() == counter_map_k.size());

  printf("Metric size %d\n", (int)metrics.size());
  for (auto& metric : metrics) {
    printf("Metric: %s\n", metric.first);
  }

  hsa_status_t err = hsa_init();
  if (err != HSA_STATUS_SUCCESS) {
    const char* errstr = nullptr;
    hsa_status_string(err, &errstr);
    throw std::runtime_error("hsa error code: " + std::to_string(err) + " " + errstr);
  }

  // populate list of agents
  int errcode = get_agents(&agent_arr);
  if (errcode != 0) {
    return;
  }
  printf("number of devices: %u\n", agent_arr.count);
  printf("devices being profiled: %u\n", dev_count);

  for (int i = 0; i < dev_count; ++i) {
    int j = 0;
    for (auto& metric : metrics) {
      features[i][j].kind = (rocprofiler_feature_kind_t)ROCPROFILER_FEATURE_KIND_METRIC;
      features[i][j].name = metric.first;
      printf("Metric[%d]: %s\n", j, features[i][j].name);
      j++;
    }
  }

  for (int i = 0; i < dev_count; ++i) {
    if (!createHsaQueue(&queues[i], agent_arr.agents[i])) {
      fprintf(stdout, "can't create queues[%d]\n", i);
    }
  }
}

RdcRocpBase::~RdcRocpBase() {
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;
  for (int i = 0; i < dev_count; ++i) {
    hsa_errno = rocprofiler_stop(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < dev_count; ++i) {
    hsa_errno = rocprofiler_close(contexts[i]);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  hsa_errno = hsa_shut_down();
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  hsa_errno = hsa_shut_down();
  assert(hsa_errno == HSA_STATUS_ERROR_NOT_INITIALIZED);
}

rdc_status_t RdcRocpBase::rocp_lookup(pair_gpu_field_t gpu_field, double* value) {
  if (value == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (status != HSA_STATUS_SUCCESS) {
    return Rocp2RdcError(status);
  }
  switch (gpu_field.second) {
    default:
      run_profiler(counter_map_k.at(gpu_field.second));
      *value = metrics[counter_map_k.at(gpu_field.second)];
      break;
  }
  return Rocp2RdcError(status);
}

rdc_status_t RdcRocpBase::Rocp2RdcError(hsa_status_t rocm_status) {
  switch (rocm_status) {
    case HSA_STATUS_SUCCESS:
      return RDC_ST_OK;
    default:
      return RDC_ST_UNKNOWN_ERROR;
  }
}

}  // namespace rdc
}  // namespace amd
