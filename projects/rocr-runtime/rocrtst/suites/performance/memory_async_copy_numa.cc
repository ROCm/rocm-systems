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

#include <hwloc.h>
#include <hwloc/linux-libnuma.h>
#include <numa.h>

#include <vector>
#include <algorithm>

#include "common/base_rocr.h"
#include "suites/test_common/test_base.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/performance/memory_async_copy_numa.h"
#include "common/base_rocr_utils.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"

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

MemoryAsyncCopyNUMA::MemoryAsyncCopyNUMA(void) : MemoryAsyncCopy() {
  set_title("Asynchronous Memory Copy Bandwidth Using NUMA aware allocation");
  set_description("This test measures bandwidth to/from Host from/to GPU "
      "using hsa_amd_memory_async_copy() to copy buffers of various length "
      "from memory pool to another. Host memory is allocated using NUMA "
      "aware allocators. Bandwidth performance using NUMA should, at worst, "
      "be as good as using the standard hsa allocator.");

  do_p2p_ = false;
}

MemoryAsyncCopyNUMA::~MemoryAsyncCopyNUMA(void) {
}

void MemoryAsyncCopyNUMA::Run(void) {
  int ret;
  TestBase::Run();

  hwloc_bitmap_t cpu_bind_set = nullptr;
  char *a;

  // Bind CPU
  cpu_bind_set = hwloc_bitmap_alloc();

  hwloc_cpuset_from_nodeset(topology_, cpu_bind_set, cpu_hwl_numa_nodeset_);

  ASSERT_FALSE((bool)hwloc_bitmap_iszero(cpu_bind_set));

  if (hwloc_bitmap_isfull(cpu_bind_set)) {
    std::cout <<
     "All cpus associated with NUMA node. No hwloc cpu binding will be done."
                                                                 << std::endl;
  } else {
    hwloc_bitmap_t cpu_bind_set_chk = nullptr;
    cpu_bind_set_chk = hwloc_bitmap_alloc();

    hwloc_bitmap_singlify(cpu_bind_set);
    ret = hwloc_set_cpubind(topology_, cpu_bind_set, HWLOC_CPUBIND_PROCESS);
    ASSERT_TRUE(ret == 0 &&
          "hwloc: cpubind not supported or cannot be enforced. Check errno.");

    hwloc_get_cpubind(topology_, cpu_bind_set_chk, 0);

    if (verbosity() >= VERBOSE_STANDARD) {
      hwloc_bitmap_asprintf(&a, cpu_bind_set);
      printf("write hwloc cpubind mask: %s\n", a);
      hwloc_bitmap_asprintf(&a, cpu_bind_set_chk);
      printf("read hwloc cpubind mask: %s\n", a);
    }
    ASSERT_TRUE(hwloc_bitmap_isequal(cpu_bind_set, cpu_bind_set_chk) &&
                                              "Unexpected hwloc cpubind set");
    hwloc_bitmap_free(cpu_bind_set_chk);

    // Bind Memory
    ret = hwloc_set_membind_nodeset(topology_, cpu_hwl_numa_nodeset_,
                                     HWLOC_MEMBIND_BIND, 0);
    ASSERT_TRUE(ret == 0 &&
          "hwloc: membind not supported or cannot be enforced. Check errno.");
  }
  for (Transaction t : tran_) {
    RunBenchmarkWithVerification(&t);
  }

  hwloc_bitmap_free(cpu_bind_set);
}

void MemoryAsyncCopyNUMA::RunBenchmarkWithVerification(Transaction *t) {
  hsa_status_t err;
  void* ptr_src;
  void* ptr_dst;

  size_t size = t->max_size * 1024;

  hsa_amd_memory_pool_t src_pool =  pool_info_[t->src]->pool_;
  hsa_agent_t dst_agent = pool_info_[t->dst]->owner_agent_info()->agent();
  hsa_amd_memory_pool_t dst_pool = pool_info_[t->dst]->pool_;

  hsa_agent_t src_agent = pool_info_[t->src]->owner_agent_info()->agent();

  PrintTransactionType(t);

  // Allocate resources...
  void *locked_mem;

  // We are relying a previous call to hwloc_set_membind_nodeset() to set
  // policy
  void *local_alloc = hwloc_alloc(topology_, size);
  ASSERT_TRUE(local_alloc != nullptr && "hwloc_alloc_membind() failed");
  hsa_agent_t gpu_agent = ((t->type == H2D || t->type == H2DRemote) ?
                                                       dst_agent : src_agent);

  // 1. We should specify the gpu agent here as the cpu already has
  // access to the system memory.
  // 2. The host can only use the pointer assigned from the system mem.
  // alloc. call (e.g., "local_alloc" below). The gpu agent can only use the
  // pointer returned by the lock call (e.g., "locked_mem" below). This is
  // a current (as of August 2017) limitation of KFD.
  err = hsa_amd_memory_lock(local_alloc, size, &gpu_agent, 1, &locked_mem);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (t->type == D2H || t->type == D2HRemote) {
    err = hsa_amd_memory_pool_allocate(src_pool, size, 0, &ptr_src);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    ptr_dst = locked_mem;
  } else if (t->type == H2D || t->type == H2DRemote) {
    err = hsa_amd_memory_pool_allocate(dst_pool, size, 0, &ptr_dst);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    ptr_src = locked_mem;
  } else {
    ASSERT_EQ(t->type, P2P);
    std::cout << "Skipping P2P for NUMA test" << std::endl;
    return;
  }
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  void* host_ptr_src = NULL;
  void* host_ptr_dst = NULL;
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0,
                                     reinterpret_cast<void**>(&host_ptr_src));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0,
                                     reinterpret_cast<void**>(&host_ptr_dst));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_signal_t s;
  err = hsa_signal_create(1, 0, NULL, &s);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Deallocate resources...
  MAKE_SCOPE_GUARD([&]() {
    // NOTE that the host memory pointer (local_alloc) must be used below
    err = hsa_amd_memory_unlock(local_alloc);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    if (t->type == D2H) {
      err = hsa_amd_memory_pool_free(ptr_src);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    } else {
      err = hsa_amd_memory_pool_free(ptr_dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    }
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    // numa_free(local_alloc, size);
    hwloc_free(topology_, local_alloc, size);
    err = hsa_amd_memory_pool_free(host_ptr_src);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_amd_memory_pool_free(host_ptr_dst);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_signal_destroy(s);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  });

  hsa_agent_t *cpy_ag = nullptr;
  // **** First copy from the system buffer source to the test source pool
  // Acquire the appropriate access; prefer GPU agent over CPU where there
  // is a choice. We don't need to do this is the test source happens to
  // be the host pool

  err = hsa_amd_memory_fill(host_ptr_src, 1, size/sizeof(uint32_t));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_fill(host_ptr_dst, 0, size/sizeof(uint32_t));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (t->type == D2H) {
    cpy_ag = AcquireAsyncCopyAccess(ptr_src, src_pool, &src_agent,
                                        host_ptr_src, sys_pool_, &cpu_agent_);
    if (cpy_ag == nullptr) {
      std::cout << "Agents " << t->src << " and " << t->dst <<
                              "cannot access each other's pool." << std::endl;
      std::cout << "Skipping..." << std::endl;
      return;
    }
    ASSERT_NE(cpy_ag, nullptr);

    err = hsa_amd_memory_async_copy(ptr_src, *cpy_ag, host_ptr_src, *cpy_ag,
                                                            size, 0, NULL, s);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1,
                                         uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
    {}

    memset(local_alloc, 0, size);
  } else {  // H2D
    cpy_ag = AcquireAsyncCopyAccess(ptr_dst, dst_pool, &dst_agent,
                                        host_ptr_dst, sys_pool_, &cpu_agent_);
    if (cpy_ag == nullptr) {
      std::cout << "Agents " << t->src << " and " << t->dst <<
                              "cannot access each other's pool." << std::endl;
      std::cout << "Skipping..." << std::endl;
      return;
    }
    ASSERT_NE(cpy_ag, nullptr);

    err = hsa_amd_memory_async_copy(ptr_src, *cpy_ag, host_ptr_src, *cpy_ag,
                                                            size, 0, NULL, s);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1,
                                         uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
    {}

    memset(local_alloc, 1, size);
  }

  int iterations = RealIterationNum();

  // **** Next, copy from the test source pool to the test destination pool
  // Prefer a gpu agent to a cpu agent

  ASSERT_NE(cpy_ag, nullptr);

  cpy_ag = AcquireAsyncCopyAccess(ptr_dst, dst_pool, &dst_agent,
              ptr_src, src_pool, &src_agent);
  if (cpy_ag == nullptr) {
    std::cout << "Agents " << t->src << " and " << t->dst <<
                            "cannot access each other's pool." << std::endl;
    std::cout << "Skipping..." << std::endl;
    return;
  }
  ASSERT_NE(cpy_ag, nullptr);

  for (int i = 0; i < kNumGranularity; i++) {
    if (Size[i] > size) {
      break;
    }

    std::vector<double> time;

    for (int it = 0; it < iterations; it++) {
      if (verbosity() >= VERBOSE_PROGRESS) {
        std::cout << ".";
        std::cout.flush();
      }

      hsa_signal_store_relaxed(t->signal, 1);

      rocrtst::PerfTimer copy_timer;
      int index = copy_timer.CreateTimer();

      copy_timer.StartTimer(index);
      err = hsa_amd_memory_async_copy(ptr_dst, *cpy_ag, ptr_src, *cpy_ag,
                                                 Size[i], 0, NULL, t->signal);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      while (hsa_signal_wait_scacquire(t->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                         uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
      {}

      copy_timer.StopTimer(index);

      hsa_signal_store_relaxed(s, 1);

      err = AcquireAccess(dst_agent, sys_pool_, host_ptr_dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      if (t->type == D2H) {
        memcpy(host_ptr_dst, local_alloc, size);
      } else {
        err = hsa_amd_memory_async_copy(host_ptr_dst, dst_agent, ptr_dst,
                                                 dst_agent, size, 0, NULL, s);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err);

        while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1,
                                       uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
          {}
      }

      if (memcmp(host_ptr_src, host_ptr_dst, Size[i])) {
        verified_ = false;
      }
      // Push the result back to vector time
      time.push_back(copy_timer.ReadTimer(index));
    }

    if (verbosity() >= VERBOSE_PROGRESS) {
      std::cout << std::endl;
    }

    // Get Min copy time
    t->min_time->push_back(*std::min_element(time.begin(), time.end()));
    // Get mean copy time and store to the array
    t->benchmark_copy_time->push_back(GetMeanTime(&time));
  }
}

#undef RET_IF_HSA_ERR
