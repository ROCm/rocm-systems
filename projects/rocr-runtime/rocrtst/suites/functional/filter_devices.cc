/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates. 
* 
* SPDX-License-Identifier: MIT
*/
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

#include "suites/functional/filter_devices.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/os.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const char kSubTestSeparator[] = "  **************************";

static void PrintFilterSubtestHeader(const char *header) {
  std::cout << "  *** Filter Devices Subtest: " << header << " ***" << std::endl;
}

FilterDevicesTest::FilterDevicesTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("RocR Filter Devices Tests");
  set_description("This series of tests check ROCR_VISIBLE_DEVICES "
                  "environment variable functionality for filtering GPU devices.");
  initial_gpu_count_ = 0;
}

FilterDevicesTest::~FilterDevicesTest(void) {}

void FilterDevicesTest::SetUp(void) {
  TestBase::SetUp();
}

void FilterDevicesTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void FilterDevicesTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void FilterDevicesTest::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  return;
}

void FilterDevicesTest::Close() {
  TestBase::Close();
}

void FilterDevicesTest::TestRocrVisibleDevicesFiltering() {
  PrintFilterSubtestHeader("ROCR_VISIBLE_DEVICES Comprehensive Test");

  const char* rvd_env = std::getenv("ROCR_VISIBLE_DEVICES");
  if (rvd_env != nullptr) {
    if (verbosity() > 0) {
      std::cout << "ROCR_VISIBLE_DEVICES is already set to: " 
                                    << rvd_env << ", Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &available_gpus_));
  initial_gpu_count_ = available_gpus_.size();

  if (initial_gpu_count_ < 2) {
    if (verbosity() > 0) {
      std::cout << "Test requires at least 2 GPUs. Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  std::vector<std::string> gpu_uuids;
  for (const auto& gpu : available_gpus_) {
    char uuid[64] = {0};
    ASSERT_SUCCESS(hsa_agent_get_info(gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID), uuid));
    gpu_uuids.push_back(std::string(uuid));
  }

  // ROCR_VISIBLE_DEVICES=0 (single GPU)
  if (verbosity() > 0) {
    std::cout << "Testing with single device index." << std::endl;
  }
  rocrtst::SetEnv("ROCR_VISIBLE_DEVICES", "0");
  ASSERT_SUCCESS(hsa_shut_down());
  ASSERT_SUCCESS(hsa_init());
  
  std::vector<hsa_agent_t> filtered_gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &filtered_gpus));
  ASSERT_EQ(filtered_gpus.size(), 1) << "Expected 1 GPU, found " << filtered_gpus.size();

  // ROCR_VISIBLE_DEVICES > number of physical devices
  if (verbosity() > 0) {
    std::cout << "Testing with invalid device index." << std::endl;
  }
  std::string invalid_index = std::to_string(initial_gpu_count_ + 5);
  rocrtst::SetEnv("ROCR_VISIBLE_DEVICES", invalid_index.c_str());
  ASSERT_SUCCESS(hsa_shut_down());
  ASSERT_SUCCESS(hsa_init());
  
  filtered_gpus.clear();
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &filtered_gpus));
  ASSERT_EQ(filtered_gpus.size(), 0) << "Expected 0 GPUs for invalid index, found " << filtered_gpus.size();

  // ROCR_VISIBLE_DEVICES=1,0 (swapped indices)
  if (verbosity() > 0) {
    std::cout << "Testing with swapped device indices." << std::endl;
  }
  rocrtst::SetEnv("ROCR_VISIBLE_DEVICES", "1,0");
  ASSERT_SUCCESS(hsa_shut_down());
  ASSERT_SUCCESS(hsa_init());
  
  filtered_gpus.clear();
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &filtered_gpus));
  ASSERT_EQ(filtered_gpus.size(), 2) << "Expected 2 GPUs, found " << filtered_gpus.size();
  
  if (gpu_uuids.size() >= 2) {
    char new_uuid_0[64] = {0};
    char new_uuid_1[64] = {0};
    
    ASSERT_SUCCESS(hsa_agent_get_info(filtered_gpus[0], static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID), new_uuid_0));
    ASSERT_SUCCESS(hsa_agent_get_info(filtered_gpus[1], static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID), new_uuid_1));
    
    ASSERT_EQ(std::string(new_uuid_0), gpu_uuids[1]) << "GPU 0 UUID should be swapped";
    ASSERT_EQ(std::string(new_uuid_1), gpu_uuids[0]) << "GPU 1 UUID should be swapped";
  }

  // ROCR_VISIBLE_DEVICES=GPU-XX (UUID selection)
  if (verbosity() > 0) {
    std::cout << "Testing with UUID filters." << std::endl;
  }
  if (!gpu_uuids.empty()) {
    std::string uuid_filter = gpu_uuids[0];
    rocrtst::SetEnv("ROCR_VISIBLE_DEVICES", uuid_filter.c_str());
    ASSERT_SUCCESS(hsa_shut_down());
    ASSERT_SUCCESS(hsa_init());
    
    filtered_gpus.clear();
    ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &filtered_gpus));
    ASSERT_EQ(filtered_gpus.size(), 1) << "Expected 1 GPU for UUID filter, found " << filtered_gpus.size();
    
    char selected_uuid[64] = {0};
    ASSERT_SUCCESS(hsa_agent_get_info(filtered_gpus[0], static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID), selected_uuid));
    ASSERT_EQ(std::string(selected_uuid), gpu_uuids[0]) << "Selected GPU UUID should match";
  }

  // Clean up
  unsetenv("ROCR_VISIBLE_DEVICES");
  ASSERT_SUCCESS(hsa_shut_down());
  ASSERT_SUCCESS(hsa_init());

  if (verbosity() > 0) {
    std::cout << "\nAll subtests passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}