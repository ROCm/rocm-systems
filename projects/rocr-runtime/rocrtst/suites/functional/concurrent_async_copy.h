/*
•	Copyright © Advanced Micro Devices, Inc., or its affiliates.
•	
•	SPDX-License-Identifier: MIT
*/


#ifndef ROCRTST_SUITES_FUNCTIONAL_CONCURRENT_ASYNC_COPY_H_
#define ROCRTST_SUITES_FUNCTIONAL_CONCURRENT_ASYNC_COPY_H_

#include <pthread.h>
#include <vector>
#include <atomic>
#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/test_common/test_base.h"

// Shared data structure for all threads
struct SdmaRaceThreadData {
  hsa_agent_t gpu_agent;
  hsa_amd_memory_pool_t gpu_pool;
  hsa_amd_memory_pool_t sys_pool;
  std::atomic<int> error_count;

  SdmaRaceThreadData() : error_count(0) {}
};

// Per-thread data structure
struct ThreadContext {
  SdmaRaceThreadData* shared;
  int thread_id;
  void* src_buffer;
  void* dst_buffer;
  hsa_signal_t signal;
};

/**
 * @brief Test class for concurrent async copy operations
 *
 * This test reproduces the TSAN race condition on sdma_blit_used_mask_
 * by spawning multiple threads that concurrently perform async memory
 * copies, all targeting the same GPU agent.
 *
 * Expected TSAN report (before fix):
 *   Read/Write race at amd_gpu_agent.cpp:2292 in GetBlitObject()
 */
class ConcurrentAsyncCopy : public TestBase {
 public:
  ConcurrentAsyncCopy(void);
  virtual ~ConcurrentAsyncCopy(void);

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  void TestSdmaDoorbellRace(void);
  void TestSdmaEngineExhaustion(void);
  void TestSignalCompletionRace(void);
  void TestVaryingCopySizeStress(void);
  void TestConcurrentCopyAndKernelDispatch(void);

 private:
  // Reusable thread management members - cleaned up in Close()
  std::vector<ThreadContext> contexts_;
  std::vector<pthread_t> sdma_threads_;
  std::vector<pthread_t> kernel_threads_;
  hsa_queue_t* queue_;
  bool cleanup_required_;

  void CleanupResources();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_CONCURRENT_ASYNC_COPY_H_
