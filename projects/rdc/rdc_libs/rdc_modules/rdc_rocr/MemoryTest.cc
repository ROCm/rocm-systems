/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocr/MemoryTest.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace amd {
namespace rdc {

static const uint32_t kNumBufferElements = 256;

MemoryTest::MemoryTest(uint32_t gpu_index) : TestBase(gpu_index) {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("Max Single Allocation Memory Test");
  set_description(
      "This series of tests check memory allocation limits, extent"
      " of GPU access to system memory and other memory related "
      "functionality.");
}

MemoryTest::~MemoryTest(void) {}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
hsa_status_t MemoryTest::SetUp(void) {
  hsa_status_t err = HSA_STATUS_SUCCESS;

  TestBase::SetUp();

  err = SetDefaultAgents(this);
  if (err != HSA_STATUS_SUCCESS) return err;

  err = SetPoolsTypical(this);
  return err;
}

void MemoryTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void MemoryTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

hsa_status_t MemoryTest::TestAllocate(hsa_amd_memory_pool_t pool, size_t sz) {
  void* ptr;
  hsa_status_t err;

  err = hsa_amd_memory_pool_allocate(pool, sz, 0, &ptr);

  if (err == HSA_STATUS_SUCCESS) {
    err = hsa_memory_free(ptr);
  }

  return err;
}

static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char* header) {
  RDC_LOG(RDC_DEBUG, "  *** Memory Subtest: " << header << " ***");
}

// Test Fixtures
hsa_status_t MemoryTest::MaxSingleAllocationTest(hsa_agent_t ag, hsa_amd_memory_pool_t pool) {
  hsa_status_t err = HSA_STATUS_SUCCESS;

  pool_info_t pool_i;
  char ag_name[64];
  hsa_device_type_t ag_type;

  err = hsa_agent_get_info(ag, HSA_AGENT_INFO_NAME, ag_name);
  if (err != HSA_STATUS_SUCCESS) return err;

  err = hsa_agent_get_info(ag, HSA_AGENT_INFO_DEVICE, &ag_type);
  if (err != HSA_STATUS_SUCCESS) return err;

  uint32_t node = 0;
  err = hsa_agent_get_info(ag, HSA_AGENT_INFO_NODE, &node);
  if (err != HSA_STATUS_SUCCESS) return err;

  if (verbosity() > 0) {
    std::string device_type;
    switch (ag_type) {
      case HSA_DEVICE_TYPE_CPU:
        device_type = "CPU";
        break;
      case HSA_DEVICE_TYPE_GPU:
        device_type = "GPU";
        break;
      case HSA_DEVICE_TYPE_DSP:
        device_type = "DSP";
        break;
    }
    RDC_LOG(RDC_DEBUG, "  Agent: " << ag_name << " Node " << node << " (" << device_type << ")");
  }

  err = AcquirePoolInfo(pool, &pool_i);
  if (err != HSA_STATUS_SUCCESS) return err;

  if (verbosity() > 0) {
    DumpMemoryPoolInfo(&pool_i, 2);
  }

  if (!pool_i.alloc_allowed || pool_i.alloc_granule == 0 || pool_i.alloc_alignment == 0) {
    if (verbosity() > 0) {
      RDC_LOG(RDC_DEBUG, "  Test not applicable. Skipping.");
    }
    return err;
  }
  // Do everything in "granule" units
  auto gran_sz = pool_i.alloc_granule;
  auto pool_sz = pool_i.aggregate_alloc_max / gran_sz;

  // Neg. test: Try to allocate more than the pool size
  err = TestAllocate(pool, pool_sz * gran_sz + gran_sz);
  if (err != HSA_STATUS_ERROR_INVALID_ALLOCATION) return err;

  auto max_alloc_size = pool_sz / 2;
  uint64_t upper_bound = pool_sz;
  uint64_t lower_bound = 0;

  while (true) {
    err = TestAllocate(pool, max_alloc_size * gran_sz);

    if (err != HSA_STATUS_SUCCESS || err != HSA_STATUS_ERROR_OUT_OF_RESOURCES) return err;

    if (err == HSA_STATUS_SUCCESS) {
      lower_bound = max_alloc_size;
      max_alloc_size += (upper_bound - lower_bound) / 2;
    } else if (err == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      upper_bound = max_alloc_size;
      max_alloc_size -= (upper_bound - lower_bound) / 2;
    }

    if ((upper_bound - lower_bound) < 2) {
      break;
    }

    if (upper_bound <= lower_bound) {
      RDC_LOG(RDC_ERROR, "Wrong upper bound and lower bound");
      return err;
    }
  }

  if (verbosity() > 0) {
    RDC_LOG(RDC_DEBUG, "  Biggest single allocation size for this pool is "
                           << (max_alloc_size * gran_sz) / 1024 << "KB.");
    RDC_LOG(RDC_DEBUG, "  This is " << static_cast<float>(max_alloc_size) / pool_sz * 100
                                    << "% of the total.");
  }

  if (ag_type == HSA_DEVICE_TYPE_GPU) {
    if ((float)max_alloc_size / pool_sz < (float)15 / 16) {
      RDC_LOG(RDC_ERROR, "the allocate size is wrong");
      throw_if_error(HSA_STATUS_ERROR, "The allocate size is wrong");
    }
    // EXPECT_GE((float)max_alloc_size/pool_sz, (float)15/16);
  }
  if (verbosity() > 0) {
    std::cout << kSubTestSeparator << std::endl;
  }

  return err;
}

hsa_status_t MemoryTest::MaxSingleAllocationTest(void) {
  hsa_status_t err = HSA_STATUS_SUCCESS;
  std::vector<std::shared_ptr<agent_pools_t>> agent_pools;

  PrintMemorySubtestHeader("Maximum Single Allocation in Memory Pools");

  err = GetAgentPools(&agent_pools);
  throw_if_error(err, "GetAgentPools pool fail.");

  hsa_agent_t current_gpu;
  err = get_agent_by_gpu_index(gpu_index_, &current_gpu);
  throw_if_error(err, "Get agent by GPU index fail.");

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    if (a->agent.handle != current_gpu.handle) continue;
    for (auto p : a->pools) {
      pool_idx++;
      RDC_LOG(RDC_DEBUG, "  Pool " << pool_idx << ":");
      err = MaxSingleAllocationTest(a->agent, p);
      throw_if_error(err, "MaxSingleAllocationTest .");
      per_gpu_info_ += title();
      per_gpu_info_ += " Pool ";
      per_gpu_info_ += std::to_string(pool_idx);
      per_gpu_info_ += " test pass. ";
    }
  }
  gpu_info_ += title();
  gpu_info_ += " for GPU ";
  gpu_info_ += std::to_string(gpu_index_);
  gpu_info_ += " Pass. ";

  return err;
}

}  // namespace rdc
}  // namespace amd
