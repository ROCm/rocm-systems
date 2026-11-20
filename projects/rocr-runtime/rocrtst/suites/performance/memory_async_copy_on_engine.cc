/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates. 
 * 
 * SPDX-License-Identifier: MIT
 */

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
  void* ptr_src = nullptr;
  void* ptr_dst = nullptr;
  size_t size = 0;

  hsa_amd_memory_pool_t src_pool =  pool_info_[t->src]->pool_;
  hsa_agent_t dst_agent = pool_info_[t->dst]->owner_agent_info()->agent();
  hsa_amd_memory_pool_t dst_pool = pool_info_[t->dst]->pool_;
  hsa_agent_t src_agent = pool_info_[t->src]->owner_agent_info()->agent();

  PrintTransactionType(t);

  err = BenchmarkAllocLimit(t, &size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (size == 0) {
    std::cout << "Skipping: host pool cannot hold this copy." << std::endl;
    return;
  }

  const bool src_is_cpu =
      pool_info_[t->src]->owner_agent_info()->device_type() == HSA_DEVICE_TYPE_CPU;
  const bool dst_is_cpu =
      pool_info_[t->dst]->owner_agent_info()->device_type() == HSA_DEVICE_TYPE_CPU;

  void* allocated = nullptr;
  err = hsa_amd_memory_pool_allocate(src_pool, size, 0, &allocated);
  if (src_is_cpu && HostPoolShort(err)) {
    std::cout << "Skipping: host pool cannot allocate " << size
              << " bytes." << std::endl;
    return;
  }
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  ptr_src = allocated;

  allocated = nullptr;
  err = hsa_amd_memory_pool_allocate(dst_pool, size, 0, &allocated);
  if (dst_is_cpu && HostPoolShort(err)) {
    std::cout << "Skipping: host pool cannot allocate " << size
              << " bytes." << std::endl;
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_free(ptr_src));
    return;
  }
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  ptr_dst = allocated;

  // rocrtst::CommonCleanUp data
  void* host_ptr_src = nullptr;
  void* host_ptr_dst = nullptr;
  allocated = nullptr;
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0, &allocated);
  if (HostPoolShort(err)) {
    std::cout << "Skipping: host pool cannot allocate " << size
              << " bytes." << std::endl;
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_free(ptr_src));
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_free(ptr_dst));
    return;
  }
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  host_ptr_src = allocated;

  allocated = nullptr;
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0, &allocated);
  if (HostPoolShort(err)) {
    std::cout << "Skipping: host pool cannot allocate " << size
              << " bytes." << std::endl;
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_free(ptr_src));
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_free(ptr_dst));
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_free(host_ptr_src));
    return;
  }
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  host_ptr_dst = allocated;

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

  err = hsa_amd_memory_async_copy(ptr_src, src_agent, host_ptr_src, cpu_agent_,
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

  for (int i = 0; i < Granularities.size(); i++) {
    if (Granularities[i].Size > size) {
      printf("Skip test with block size %s\n", Granularities[i].Str);
      break;
    }
    printf("Start test with block size %s\n", Granularities[i].Str);

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
                                              Granularities[i].Size, 0, NULL, t->signal,
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
                                                 dst_agent, Granularities[i].Size, 0, NULL, s);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1,
                                       uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
      {}

      err = AcquireAccess(cpu_agent_, sys_pool_, host_ptr_dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      if (memcmp(host_ptr_src, host_ptr_dst, Granularities[i].Size)) {
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
