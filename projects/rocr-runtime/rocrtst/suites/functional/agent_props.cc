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

#include <algorithm>
#include <iostream>
#include <vector>

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
  case HSA_AMD_AGENT_INFO_CLOCK_COUNTERS: {
    std::stringstream str_s;

    hsa_amd_clock_counters_t counters = {0};

    err = hsa_agent_get_info(agent, prop, &counters);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    str_s << "\n\n  Clock Counters";
    str_s << "\n  Clock Frequency: "  << counters.system_clock_frequency << "\n";
    str_s << "  GPU Clock counter: "  << counters.gpu_clock_counter << "\n";
    str_s << "  System system_clock_counter: "  << counters.system_clock_counter << "\n";
    str_s << "  CPU Clock counter: "  << counters.cpu_clock_counter << "\n";
    propList_.push_back(str_s.str());

    ASSERT_NE(0, counters.system_clock_frequency);
    ASSERT_NE(0, counters.gpu_clock_counter);
    ASSERT_NE(0, counters.system_clock_counter);
    ASSERT_NE(0, counters.cpu_clock_counter);

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

void AgentPropTest::QueryAgentClockCounters() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Query Agent's Clock Counters");
  }

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (uint32_t idx = 0 ; idx < gpus.size(); ++idx) {
    QueryAgentProp(gpus[idx], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CLOCK_COUNTERS);
  }

  if (verbosity() > 0) {
    std::cout << "  *** Execution completed - subtest Passed " << " ***" << std::endl;
  }
}

#undef RET_IF_HSA_ERR
