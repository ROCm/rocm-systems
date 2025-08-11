/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <iostream>
#include <vector>
#include <sstream>
#include <unistd.h>  // for usleep and sleep
#include <cstdint>

#include "agent_props.h"
#include "common/base_rocr.h"

#include <algorithm>

#include "suites/functional/agent_props.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const uint32_t kNumBufferElements = 256;

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}

static const char kSubTestSeparator[] = "  **************************";

static void PrintAgentPropsSubtestHeader(const char *header) {
  std::cout << "  *** " << header << " ***" << std::endl;
}

AgentPropTest::AgentPropTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("  *** Query RocR Agent Properties ***");
  set_description("  *** Checks properties of Agent's on a system ***");
}

AgentPropTest::~AgentPropTest(void) {
}
typedef struct {
  bool valid;
  hsa_ven_amd_pcs_method_kind_t method;
  hsa_ven_amd_pcs_units_t units;
  size_t interval;
  size_t min_interval;
  size_t max_interval;
  size_t latency;
  size_t buffer_size;
  hsa_ven_amd_pcs_t handle;
} pc_sampling_session_t;
typedef struct {
  size_t buffer_size;
  uint8_t* buffer;
  size_t bytes_received;
} pc_sampling_copy_data_t;
hsa_status_t pc_sampling_info_callback(const hsa_ven_amd_pcs_configuration_t* configuration,
                                       void* _hosttrap_session) {
  pc_sampling_session_t* session = (pc_sampling_session_t*)_hosttrap_session;
  std::cout << "  PC sampling config:" << std::endl;
  std::cout << "    method:" << configuration->method << std::endl;
  std::cout << "    units:" << configuration->units << std::endl;
  std::cout << "    min_interval:" << configuration->min_interval << std::endl;
  std::cout << "    max_interval:" << configuration->max_interval << std::endl;
  std::cout << std::endl;
  if (configuration->method == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1 ||
      configuration->method == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1) {
    session->valid = true;
    session->method = configuration->method;
    session->units = configuration->units;
    session->min_interval = configuration->min_interval;
    session->max_interval = configuration->max_interval;
  }
  return HSA_STATUS_SUCCESS;
}
void pc_sampling_data_ready_callback(void* client_callback_data, size_t data_size,
                                     size_t lost_sample_count,
                                     hsa_ven_amd_pcs_data_copy_callback_t data_copy_callback,
                                     void* hsa_callback_data) {
  pc_sampling_copy_data_t* results = (pc_sampling_copy_data_t*)client_callback_data;
  std::cout << "  PC sampling data ready:" << std::endl;
  std::cout << "    size:" << data_size << std::endl;
  std::cout << "    lost_sample_count:" << lost_sample_count << std::endl;
  results->bytes_received += data_size;
  if (data_copy_callback(hsa_callback_data, /* size_to_copy */ data_size,
                         /* buffer */ results->buffer) != HSA_STATUS_SUCCESS) {
    std::cout << "Failed to data_copy_callback" << std::endl;
    return;
  }
}
void AgentPropTest::PcSampling() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Test PC Sampling (HostTrap and Stochastic)");
  }
  size_t buffer_sizes[] = {510 * sizeof(perf_sample_hosttrap_v1_t)};
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (!gpus.size()) {
    if (verbosity() > 0) {
      std::cout << "  *** Execution skipped - no GPUs found "
                << " ***" << std::endl;
    }
    return;
  }
  std::string selected_method =
      "stochastic";  // You could set this based on a command-line input parser.
  std::vector<hsa_ven_amd_pcs_method_kind_t> sampling_methods;
  if (selected_method == "hosttrap") {
    sampling_methods.push_back(HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1);
  } else if (selected_method == "stochastic") {
    sampling_methods.push_back(HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1);
  } else {
    sampling_methods.push_back(HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1);
    sampling_methods.push_back(HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1);
  }
  for (hsa_ven_amd_pcs_method_kind_t method : sampling_methods) {
    pc_sampling_session_t session = {};
    err = hsa_ven_amd_pcs_iterate_configuration(gpus[0], &pc_sampling_info_callback, &session);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    std::cout << std::endl;
    if (!session.valid || session.method != method) {
      if (verbosity() > 0) {
        std::cout << "  *** Execution skipped - GPU does not support this sampling method "
                  << " ***" << std::endl;
      }
      continue;
    }
    hsa_amd_memory_pool_t gpu_pool;
    const size_t gpu_buffer_size = 128 * 1024 * 1024;
    err = hsa_amd_agent_iterate_memory_pools(gpus[0], rocrtst::GetGlobalMemoryPool, &gpu_pool);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    uint32_t* gpuMemToFill;
    err = hsa_amd_memory_pool_allocate(gpu_pool, gpu_buffer_size, 0,
                                       reinterpret_cast<void**>(&gpuMemToFill));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_agent_t allowed_agents[2] = {cpus[0], gpus[0]};
    err = hsa_amd_agents_allow_access(2, allowed_agents, NULL, gpuMemToFill);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    session.method = HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1;
    session.units = HSA_VEN_AMD_PCS_INTERVAL_UNITS_MICRO_SECONDS;
    session.interval = 1;
    session.latency = 1000;
    session.buffer_size = 64 * 1024;

    for (unsigned i = 0; i < sizeof(buffer_sizes) / sizeof(buffer_sizes[0]); i++) {
      std::cout << "Testing with buffer size:" << buffer_sizes[i] << std::endl;
      session.buffer_size = buffer_sizes[i];
      pc_sampling_copy_data_t results = {};
      results.buffer_size = session.buffer_size;
      results.bytes_received = 0;
      results.buffer = static_cast<uint8_t*>(malloc(results.buffer_size));
      if (!results.buffer) {
        std::cout << "Failed to allocate local buffer" << std::endl;
        return;
      }
      std::cout << "    Calling hsa_ven_amd_pcs_create" << std::endl;
      err = hsa_ven_amd_pcs_create(gpus[0], session.method, session.units, session.interval,
                                   session.latency, session.buffer_size,
                                   &pc_sampling_data_ready_callback, &results, &session.handle);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      std::cout << "    Calling hsa_ven_amd_pcs_start" << std::endl;
      err = hsa_ven_amd_pcs_start(session.handle);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      uint64_t count = 0;
      do {
        hsa_amd_memory_fill(gpuMemToFill, 0x55, gpu_buffer_size / 4);
        sched_yield();
        if (!(count++ % 1000)) {
          hsa_ven_amd_pcs_flush(session.handle);
        }
        usleep((rand() % 8000));
      } while (results.bytes_received < (session.buffer_size * 50));
      std::cout << "    Calling hsa_ven_amd_pcs_stop" << std::endl;
      err = hsa_ven_amd_pcs_stop(session.handle);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      sleep(3);
      std::cout << "    Calling hsa_ven_amd_pcs_destroy" << std::endl;
      err = hsa_ven_amd_pcs_destroy(session.handle);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      hsa_memory_free(gpuMemToFill);
      free(results.buffer);
    }
    if (verbosity() > 0) {
      std::cout << "  *** Execution completed - subtest Passed "
                << " ***" << std::endl;
    }
  }
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void AgentPropTest::SetUp(void) {
  TestBase::SetUp();
  std::cout << "  *** Initialize ROCr Runtime and " 
            << "acquire handles of agents" << " ***" << std::endl;
}

void AgentPropTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void AgentPropTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void AgentPropTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  std::cout << std::endl;
  for (uint32_t idx = 0 ; idx < this->propList_.size(); ++idx) {
    std::cout << this->propList_[idx] << std::endl;
  }
  return;
}

void AgentPropTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

// Extend this method to query for agent properties that are
// currently not tested
void AgentPropTest::QueryAgentProp(hsa_agent_t agent,
                                   hsa_agent_info_t prop) {
  hsa_status_t err;
  hsa_device_type_t agType;
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agType);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::stringstream ss;
  ss << "  Agent " << "(";
  switch (agType) {
    case HSA_DEVICE_TYPE_CPU:
      ss << "CPU) : ";
      break;
    case HSA_DEVICE_TYPE_GPU:
      ss << "GPU) : ";
      break;
    case HSA_DEVICE_TYPE_DSP:
      ss << "DSP) : ";
      break;
    case HSA_DEVICE_TYPE_AIE:
      ss << "AIE) : ";
      break;
  }

  // Print the agent property
  uint32_t key = uint32_t(prop);
  switch (key) {
  // Retrieves UUID property value of the agent
  case HSA_AMD_AGENT_INFO_UUID: {
    char uuid[32];
    err = hsa_agent_get_info(agent, prop, (void*)&uuid[0]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    ss << uuid;
    propList_.push_back(ss.str());
    break;
  }
  default:
      FAIL();
  }
}

void AgentPropTest::QueryAgentUUID() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Query GPU and CPU Agent's UUID");
  }

  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (uint32_t idx = 0 ; idx < cpus.size(); ++idx) {
    QueryAgentProp(cpus[idx], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID);
  }

  for (uint32_t idx = 0 ; idx < gpus.size(); ++idx) {
    QueryAgentProp(gpus[idx], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID);
  }

  if (verbosity() > 0) {
    std::cout << "  *** Execution completed - subtest Passed " << " ***" << std::endl;
  }
}

#undef RET_IF_HSA_ERR
