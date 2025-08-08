/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
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


#include <fcntl.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>

#include "suites/functional/memory_alignment.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "common/concurrent_utils.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"


static const uint32_t kNumThreads = 4096;

typedef struct control_block {
    hsa_amd_memory_pool_t* pool;
} cb_t;

// Callback function which will call upon when need
// to allocate memory from the pool in the thread.
static void CallbackVerifyPoolAlignmendFunc(void *data) {
  hsa_status_t err;
  cb_t *cb = reinterpret_cast<cb_t*>(data);

  rocrtst::pool_info_t info;
  memset(&info, 0, sizeof(rocrtst::pool_info_t));
  err = rocrtst::AcquirePoolInfo(*(cb->pool), &info);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  if (info.alloc_allowed) {
    // Get the allocated alignment size
    size_t alignment_size = info.alloc_alignment;
    EXPECT_TRUE(alignment_size);
    // Verifies the alignment attribute is a power of 2
    if (info.size != 0) {
      EXPECT_TRUE((alignment_size&&(!(alignment_size&(alignment_size-1)))));
    }
  }
  return;
}


MemoryAlignmentTest::MemoryAlignmentTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  set_title("RocR Memory Alignment Test");
  set_description(" This test verifies that each memory pool of the agent that"
  " has HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED alloc memory, It is "
  " aligned as specified by the HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT"
  " and has the alignment attribute is a power of 2.");
}

MemoryAlignmentTest::~MemoryAlignmentTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void MemoryAlignmentTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  return;
}

void MemoryAlignmentTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryAlignmentTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryAlignmentTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryAlignmentTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}




static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char *header) {
  std::cout << "  *** Memory Functional Subtest: " << header << " ***" << std::endl;
}

static void PrintAgentNameAndType(hsa_agent_t agent) {
  hsa_status_t err;

  char ag_name[64];
  hsa_device_type_t ag_type;

  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, ag_name);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::cout << "  Agent: " << ag_name << " (";
  switch (ag_type) {
    case HSA_DEVICE_TYPE_CPU:
      std::cout << "CPU)";
      break;
    case HSA_DEVICE_TYPE_GPU:
      std::cout << "GPU)";
      break;
    case HSA_DEVICE_TYPE_DSP:
      std::cout << "DSP)";
      break;
    case HSA_DEVICE_TYPE_AIE:
      std::cout << "AIE)";
      break;
    }
  std::cout << std::endl;
  return;
}



void MemoryAlignmentTest::MemoryPoolAlignment(hsa_agent_t agent,
                                                hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (verbosity() > 0) {
    PrintAgentNameAndType(agent);
  }

  if (pool_i.alloc_allowed) {
    // Get the allocated alignment size
    size_t alignment_size = pool_i.alloc_alignment;
    EXPECT_TRUE(alignment_size);
    // Verifies the alignment attribute is a power of 2
    if (pool_i.size != 0) {
      EXPECT_TRUE((alignment_size&&(!(alignment_size&(alignment_size-1)))));
    }

    // verifies that alignment attribute is a power of 2 in different threads
    rocrtst::test_group* tg_concurrent = rocrtst::TestGroupCreate(kNumThreads);
    // The control blocks are used to pass data to the threads
    uint32_t kk;
    cb_t cb[kNumThreads];
    for (kk = 0; kk < kNumThreads; kk++) {
      cb[kk].pool = &pool;
      rocrtst::TestGroupAdd(tg_concurrent, &CallbackVerifyPoolAlignmendFunc, &cb[kk], 1);
    }

    // Create threads for each test
    rocrtst::TestGroupThreadCreate(tg_concurrent);

    // Start to run tests
    rocrtst::TestGroupStart(tg_concurrent);

    // Wait all tests finish
    rocrtst::TestGroupWait(tg_concurrent);

    // Exit all tests
    rocrtst::TestGroupExit(tg_concurrent);

    // Destroy thread group and cleanup resources
    rocrtst::TestGroupDestroy(tg_concurrent);
  }
  return;
}


void MemoryAlignmentTest::MemoryPoolAlignment(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("MemoryPoolAlignment in Basic func & Stress Test");
  }

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      if (verbosity() > 0) {
        std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      }
      MemoryPoolAlignment(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

