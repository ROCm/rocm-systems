/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <sys/sysinfo.h>

#include "suites/functional/time_stamp.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

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


TimeStamp::TimeStamp(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("RocR verify clock counter");
  set_description("This series of tests captures various sets of  timestamps from KFD,"
    " to get snapshots of CPU vs GPU tickets/clock counters.");
}

TimeStamp::~TimeStamp(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void TimeStamp::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  return;
}

void TimeStamp::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void TimeStamp::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TimeStamp::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void TimeStamp::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const char kSubTestSeparator[] = "  **************************";

static void PrinTimeStampSubtestHeader(const char *header) {
  std::cout << "  *** TimeStamp Subtest: " << header << " ***" << std::endl;
}

void TimeStamp::TimeStampTest (void) {
  const int NUM_COUNTERS = 50;
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;
  char ag_name[64];
  hsa_device_type_t ag_type;
  PrinTimeStampSubtestHeader("TimeStampTest: verifing clock counter IOCTLs");

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::cout<<"Total number of agent_pools: "<<agent_pools.size()<<std::endl;
  for (auto a : agent_pools) {
      err = hsa_agent_get_info(a->agent, HSA_AGENT_INFO_NAME, ag_name);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      err = hsa_agent_get_info(a->agent, HSA_AGENT_INFO_DEVICE, &ag_type);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      if (verbosity() > 0){
          std::cout << std::endl<<"  Agent: " << ag_name;
      }
      switch (ag_type) {
          case HSA_DEVICE_TYPE_GPU:{
              hsa_amd_clock_counters_t counter[NUM_COUNTERS];
              if (verbosity() > 0){
                std::cout << "(GPU)"<<std::endl;
              }
              for(int i = 0; i < NUM_COUNTERS ;i++){
                ASSERT_EQ(hsa_agent_get_info(a->agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_CLOCK_COUNTERS,
                        &counter[i]), HSA_STATUS_SUCCESS);
                if(i > 0){
                    ASSERT_GE(counter[i].gpu_clock_counter, counter[i-1].gpu_clock_counter);
                    ASSERT_GE(counter[i].cpu_clock_counter, counter[i-1].cpu_clock_counter);
                    ASSERT_GE(counter[i].system_clock_counter, counter[i-1].system_clock_counter);
                 }
                if (verbosity() > 0){
                    std::cout<<" gpu_clock_counter:    "<<counter[i].gpu_clock_counter <<std::endl;
                    std::cout<<" cpu_clock_counter:    "<<counter[i].cpu_clock_counter <<std::endl;
                    std::cout<<" system_clock_counter: "<<counter[i].system_clock_counter <<std::endl;
                    std::cout<<std::endl;
                 }
              }
            }
            break;
          default:
            std::cout<<std::endl;
            break;
        } //switch ends.
    } //for loop iterating over agent_pools ends
}


#undef RET_IF_HSA_ERR
