/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates. 
 * 
 * SPDX-License-Identifier: MIT
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
#include "suites/performance/memory_async_copy_on_engine.h"
#include "common/base_rocr_utils.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"

#define GPU_MEMORY_THRESHOLD 536870912
#define RET_IF_HSA_ERR(err)                                                                        \
  {                                                                                                \
    if ((err) != HSA_STATUS_SUCCESS) {                                                             \
      const char* msg = 0;                                                                         \
      hsa_status_string(err, &msg);                                                                \
      EXPECT_EQ(HSA_STATUS_SUCCESS, err) << msg;                                                   \
      return (err);                                                                                \
    }                                                                                              \
  }

MemoryAsyncCopyOnEngine::MemoryAsyncCopyOnEngine(void) :
    MemoryAsyncCopy() {
  set_title("Asynchronous Memory Copy On Engine Bandwidth");
  set_description("This test measures bandwidth to/from Host from/to GPU "
      "and Peer to Peer using hsa_amd_memory_async_copy_on_engine() to copy "
      "buffers of various length from memory pool to another.");
}


void MemoryAsyncCopyOnEngine::RunBenchmarkWithVerification(Transaction *t) {
  hsa_status_t err;
  void* ptr_src;
  void* ptr_dst;
  size_t src_alloc_size;
  size_t dst_alloc_size;
  size_t max_alloc_size;
  size_t size;
  hsa_device_type_t ag_type;


  size_t max_trans_size = t->max_size * 1024;

  hsa_amd_memory_pool_t src_pool =  pool_info_[t->src]->pool_;
  hsa_agent_t dst_agent = pool_info_[t->dst]->owner_agent_info()->agent();
  hsa_amd_memory_pool_t dst_pool = pool_info_[t->dst]->pool_;
  hsa_agent_t src_agent = pool_info_[t->src]->owner_agent_info()->agent();

  PrintTransactionType(t);

  err = hsa_amd_memory_pool_get_info(src_pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                      &src_alloc_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_agent_get_info(src_agent, HSA_AGENT_INFO_DEVICE, &ag_type);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  if (src_alloc_size <= GPU_MEMORY_THRESHOLD && ag_type == HSA_DEVICE_TYPE_GPU) {
    err = hsa_agent_get_info(src_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                              &src_alloc_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  err = hsa_amd_memory_pool_get_info(dst_pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                      &dst_alloc_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_agent_get_info(dst_agent, HSA_AGENT_INFO_DEVICE, &ag_type);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  if (dst_alloc_size <= GPU_MEMORY_THRESHOLD && ag_type == HSA_DEVICE_TYPE_GPU) {
    err = hsa_agent_get_info(dst_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                              &dst_alloc_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  max_alloc_size = (src_alloc_size < dst_alloc_size) ? src_alloc_size: dst_alloc_size;

  if (dst_alloc_size <= GPU_MEMORY_THRESHOLD && ag_type == HSA_DEVICE_TYPE_GPU)
    size = (max_alloc_size/3 <= max_trans_size) ? max_alloc_size/3: max_trans_size;
  else
    size = (max_alloc_size/2 <= max_trans_size) ? max_alloc_size/2: max_trans_size;

  err = hsa_amd_memory_pool_allocate(src_pool, size, 0,
                      &ptr_src);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_pool_allocate(dst_pool, size, 0,
                      &ptr_dst);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);


  // rocrtst::CommonCleanUp data
  void* host_ptr_src = NULL;
  void* host_ptr_dst = NULL;
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0,
                                     reinterpret_cast<void**>(&host_ptr_src));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0,
                                     reinterpret_cast<void**>(&host_ptr_dst));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_fill(host_ptr_src, 1, size/sizeof(uint32_t));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_fill(host_ptr_dst, 0, size/sizeof(uint32_t));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_signal_t s;
  err = hsa_signal_create(1, 0, NULL, &s);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);


  // Deallocate resources...
  MAKE_SCOPE_GUARD([&]() {
    err = hsa_amd_memory_pool_free(ptr_src);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_amd_memory_pool_free(ptr_dst);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_amd_memory_pool_free(host_ptr_src);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_amd_memory_pool_free(host_ptr_dst);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_signal_destroy(s);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  });

  // **** First copy from the system buffer source to the test source pool
  // Acquire the appropriate access; prefer GPU agent over CPU where there
  // is a choice.
  hsa_agent_t *cpy_ag = nullptr;
  cpy_ag = AcquireAsyncCopyAccess(ptr_src, src_pool, &src_agent, host_ptr_src,
                                                     sys_pool_, &cpu_agent_);
  if (cpy_ag == nullptr) {
    std::cout << "Agents " << t->src << " and " << t->dst <<
                              "cannot access each other's pool." << std::endl;
    std::cout << "Skipping..." << std::endl;
    return;
  }

  err = hsa_amd_memory_async_copy(ptr_src, *cpy_ag, host_ptr_src, *cpy_ag,
                                                            size, 0, NULL, s);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                   HSA_WAIT_STATE_ACTIVE))
  {}

  int iterations = RealIterationNum();

  // **** Next, copy from the test source pool to the test destination pool
  // Prefer a gpu agent to a cpu agent

  cpy_ag = AcquireAsyncCopyAccess(ptr_dst, dst_pool, &dst_agent, ptr_src,
                                                        src_pool, &src_agent);
  if (cpy_ag == nullptr) {
    std::cout << "Owner agents for pools" << t->src << " and " <<
                   t->dst << " cannot access each other's pool." << std::endl;
    std::cout << "Skipping..." << std::endl;
    return;
  }

  for (int i = 0; i < kNumGranularity; i++) {
    if (Size[i] > size) {
      printf("Skip test with block size %s\n", Str[i]);
      break;
    }
    printf("Start test with block size %s\n",Str[i]);

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
      uint32_t preferred_mask = 0;
      uint32_t engine_ids_mask = 0;

      err = hsa_amd_memory_get_preferred_copy_engine(dst_agent, src_agent, &preferred_mask);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      err = hsa_amd_memory_copy_engine_status(dst_agent, src_agent, &engine_ids_mask);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      preferred_mask = preferred_mask ? (preferred_mask & engine_ids_mask) : engine_ids_mask;
      engine_ids_mask = preferred_mask ? preferred_mask : engine_ids_mask;

      if (engine_ids_mask == 0) {
          std::cout << "WARNING: No available copy engine detected. Exiting test." << std::endl;
          return;
      } 
      hsa_amd_sdma_engine_id_t engine_id = 
          static_cast<hsa_amd_sdma_engine_id_t>(1 << (ffs(engine_ids_mask) - 1));
      
      err = hsa_amd_memory_async_copy_on_engine(ptr_dst, dst_agent, ptr_src, src_agent,
                                              Size[i], 0, NULL, t->signal,
                                              engine_id, false);
      
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      while (hsa_signal_wait_scacquire(t->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                         uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
      {}

      copy_timer.StopTimer(index);

      hsa_signal_store_relaxed(s, 1);

      err = AcquireAccess(dst_agent, sys_pool_,
                    host_ptr_dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);


      err = hsa_amd_memory_async_copy(host_ptr_dst, cpu_agent_, ptr_dst,
                                                 dst_agent, Size[i], 0, NULL, s);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1,
                                       uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
      {}

      err = AcquireAccess(cpu_agent_, sys_pool_, host_ptr_dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

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
