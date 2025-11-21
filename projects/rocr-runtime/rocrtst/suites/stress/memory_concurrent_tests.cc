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
#include <string>

#include "suites/stress/memory_concurrent_tests.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "common/concurrent_utils.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"


static const uint32_t kNumThreads = 1024;
static const uint32_t kMaxAllocSize = 1024 * 1024;




typedef struct control_block {
    hsa_amd_memory_pool_t* pool;
    size_t alloc_size;
    void* alloc_pointer;
} cb_t;


// Callback function which will call upon when need
// to allocate memory from the pool in the thread.
static void CallbackHSAMemoryAllocateFunc(void *data) {
  hsa_status_t err;
  cb_t *cb = static_cast<cb_t*>(data);

  err = hsa_amd_memory_pool_allocate(*(cb->pool),
                               cb->alloc_size, 0,
                               reinterpret_cast<void**>(&(cb->alloc_pointer)));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  return;
}

// Callback function which will call upon when need
// to Free memory from the pool in the thread.
static void CallbackHSAMemoryFreeFunc(void *data) {
  hsa_status_t err;
  cb_t *cb = static_cast<cb_t*>(data);

  err = hsa_memory_free(cb->alloc_pointer);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  return;
}

typedef struct thread_data_get_pool_info_s {
    // The current pool
    hsa_amd_memory_pool_t pool;
    // The pool info retrieved from main thread
    rocrtst::pool_info_t* info;
    // Consistency check result
    int consistency;
} thread_data_get_pool_info_t;

// Callback function which will call upon when need
// to Fetch different info for the pool in the thread.
static void CallbackGetPoolInfo(void* data) {
  hsa_status_t err;

  thread_data_get_pool_info_t* thread_data =
              static_cast<thread_data_get_pool_info_t*>(data);

  rocrtst::pool_info_t info;
  memset(&info, 0, sizeof(rocrtst::pool_info_t));
  err = rocrtst::AcquirePoolInfo(thread_data->pool, &info);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (*(thread_data->info) == info) {
    // The pool info is consistent with the one got from the main thread
    thread_data->consistency = 1;
  } else {
    thread_data->consistency = 0;
  }
}

MemoryConcurrentTest::MemoryConcurrentTest(bool launch_Concurrent_Allocate_,
                      bool launch_Concurrent_Free_ ,
                      bool launch_Concurrent_PoolGetInfo_) :TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  std::string name;
  std::string desc;

  name = "RocR Memory Concurrent";
  desc = "These series of tests are Stress tests which contains different subtests ";

  if (launch_Concurrent_Allocate_) {
    name += " Allocate";
    desc += " This test Verify that memory can be concurrently allocated from pool"
            " and thread safety while allocating memory from different threads"
            " on ROCR agents";
  } else if (launch_Concurrent_Free_) {
    name += " Free";
    desc += " This test thet memory Verify can be concurrently freed from pool"
            " and thread safety while memory free from different threads"
            " on ROCR agents";
  } else if (launch_Concurrent_PoolGetInfo_) {
    name += " PoolGetInfo";
    desc += " This test Verify that memory pool info can be concurrently "
            " get from different threads on ROCR agents";
  }
  set_title(name);
  set_description(desc);
}

MemoryConcurrentTest::~MemoryConcurrentTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void MemoryConcurrentTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  return;
}

void MemoryConcurrentTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryConcurrentTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryConcurrentTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryConcurrentTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}




static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char *header) {
  std::cout << "  *** Memory Stress Subtest: " << header << " ***" << std::endl;
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

// This test verify check  memory can be
// concurrently allocated from pool on ROCR agents
void MemoryConcurrentTest::MemoryConcurrentAllocate(hsa_agent_t agent,
                                               hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (verbosity() > 0) {
    PrintAgentNameAndType(agent);
  }

  // Determine if allocation is allowed in this memory pool
  bool alloc = false;
  err = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);

  if (alloc) {
    size_t alloc_size;
    size_t total_vram_size;
    hsa_device_type_t ag_type;

    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                      &total_vram_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // If VRAM size <= 512MB, it should be APU whose VRAM is carved from system memory
    // and much smaller than dGPU. Change the threshold accordingly.
    if (total_vram_size <= 536870912 && ag_type == HSA_DEVICE_TYPE_GPU) {
      // Make sure do not allocate more than 1/4 of the available vram size
      err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                                &total_vram_size);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      alloc_size = (total_vram_size*1/4 <= kMaxAllocSize*kNumThreads) ? total_vram_size*1/(4*kNumThreads): kMaxAllocSize;
    } else {
      // Make sure do not allocate more than 3/4 of the vram size
      alloc_size = (total_vram_size*3/4 <= kMaxAllocSize*kNumThreads) ? total_vram_size*3/(4*kNumThreads): kMaxAllocSize;
    }

    // Page align the alloc_size
    alloc_size = alloc_size - (alloc_size & ((1 << 12) - 1));

    // Create a test group
    rocrtst::test_group* tg_concurrent = rocrtst::TestGroupCreate(kNumThreads);

    // The control blocks are used to pass data to the threads
    uint32_t kk;
    cb_t cb[kNumThreads];
    for (kk = 0; kk < kNumThreads; kk++) {
      cb[kk].pool = &pool;
      cb[kk].alloc_size = alloc_size;
      rocrtst::TestGroupAdd(tg_concurrent, &CallbackHSAMemoryAllocateFunc, &cb[kk], 1);
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

    // Check for overlapping addresses
    char *addr1, *addr2;
    for (kk = 0; kk < kNumThreads; ++kk) {
      addr1 = reinterpret_cast<char *>(cb[kk].alloc_pointer);
      addr2 = addr1+alloc_size;
      ASSERT_NE(reinterpret_cast<void *>(addr1), nullptr);
      uint32_t ll;
      for (ll = kk+1; ll < kNumThreads; ++ll) {
        if (addr1 < reinterpret_cast<char *>(cb[ll].alloc_pointer)) {
          ASSERT_LE(addr2, reinterpret_cast<char *>(cb[ll].alloc_pointer));
        }
        if (addr2 > reinterpret_cast<char *>(cb[ll].alloc_pointer)+alloc_size) {
          ASSERT_GE(addr1, reinterpret_cast<char *>(cb[ll].alloc_pointer)+alloc_size);
        }
      }
    }

    for (uint32_t ii = 0; ii < kNumThreads; ii++) {
      err = hsa_memory_free(cb[ii].alloc_pointer);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    }
  }
  return;
}




// This test verify check  memory can be
// concurrently allocated from pool on ROCR agents
void MemoryConcurrentTest::MemoryConcurrentFree(hsa_agent_t agent,
                                                hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (verbosity() > 0) {
    PrintAgentNameAndType(agent);
  }

  // Determine if allocation is allowed in this pool
  bool alloc = false;
  err = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  if (alloc) {
    // Get the maximum allocation size
    size_t alloc_size;
    size_t total_vram_size;
    hsa_device_type_t ag_type;

    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                      &total_vram_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // If VRAM size <= 512MB, it should be APU whose VRAM is carved from system memory
    // and much smaller than dGPU. Change the threshold accordingly.
    if (total_vram_size <= 536870912 && ag_type == HSA_DEVICE_TYPE_GPU) {
      // Make sure do not allocate more than 1/4 of the available vram size
      err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                                &total_vram_size);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      alloc_size = (total_vram_size*1/4 <= kMaxAllocSize*kNumThreads) ? total_vram_size*1/(4*kNumThreads): kMaxAllocSize;
    } else {
      // Make sure do not allocate more than 3/4 of the vram size
      alloc_size = (total_vram_size*3/4 <= kMaxAllocSize*kNumThreads) ? total_vram_size*3/(4*kNumThreads): kMaxAllocSize;
    }

    // Page align the alloc_size
    alloc_size = alloc_size - (alloc_size & ((1 << 12) - 1));

    // Create a test group
    rocrtst::test_group* tg_concurrent = rocrtst::TestGroupCreate(kNumThreads);

    // The control blocks are used to pass data to the threads
    uint32_t kk;
    cb_t cb[kNumThreads];
    for (kk = 0; kk < kNumThreads; kk++) {
      cb[kk].pool = &pool;
      cb[kk].alloc_size = alloc_size;
      err = hsa_amd_memory_pool_allocate(*(cb[kk].pool), cb[kk].alloc_size, 0, &(cb[kk].alloc_pointer));
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      rocrtst::TestGroupAdd(tg_concurrent, &CallbackHSAMemoryFreeFunc, &cb[kk], 1);
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


// This test verify if each Agent pool's attribute information
// is consistent across multiple thread.
void MemoryConcurrentTest::MemoryConcurrentPoolGetInfo(hsa_agent_t agent,
                                                hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (verbosity() > 0) {
    PrintAgentNameAndType(agent);
  }


  uint32_t kk;
  thread_data_get_pool_info_t thread_data[kNumThreads];

  // Create a test group
  rocrtst::test_group* tg_concurrent = rocrtst::TestGroupCreate(kNumThreads);

  for (kk = 0; kk < kNumThreads; kk++) {
    thread_data[kk].pool = pool;
    thread_data[kk].info = &pool_i;
    thread_data[kk].consistency = 0;
    rocrtst::TestGroupAdd(tg_concurrent, &CallbackGetPoolInfo, thread_data + kk, 1);
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

  // Verify pool info is consistent among all threads
  for (kk = 0; kk < kNumThreads; kk++) {
    ASSERT_EQ(thread_data[kk].consistency, 1);
  }
  return;
}



void MemoryConcurrentTest::MemoryConcurrentAllocate(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("MemoryConcurrentAllocate in Stress Test");
  }
  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      if (verbosity() > 0) {
        std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      }
      MemoryConcurrentAllocate(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void MemoryConcurrentTest::MemoryConcurrentFree(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("MemoryConcurrentFree in Stress Test");
  }

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      if (verbosity() > 0) {
        std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      }
      MemoryConcurrentFree(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void MemoryConcurrentTest::MemoryConcurrentPoolGetInfo(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("MemoryConcurrentPoolGetInfo in Stress Test");
  }
  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      if (verbosity() > 0) {
        std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      }
      MemoryConcurrentPoolGetInfo(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}
