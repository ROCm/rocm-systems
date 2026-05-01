/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/concurrent_async_copy.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa_ext_amd.h"

#include <vector>
#include <atomic>
#include <thread>

// Use MANY threads with SMALL copies to maximize doorbell/wptr contention
static const size_t kCopySize_1 = 64;        // Tiny 64-byte copies for max submission rate
static const int kNumThreads = 100;        // Very high thread count
static const int kCopiesPerThread = 1000;  // Many iterations per thread

static const size_t kMaxCopySize = 256 * 1024; // 256 KB max buffer size
// Varying copy sizes to trigger different SDMA engine selection
static const size_t kCopySizes[] = {
  4 * 1024,     // 4 KB
  8 * 1024,     // 8 KB
  16 * 1024,    // 16 KB
  32 * 1024,    // 32 KB
  64 * 1024,    // 64 KB
  128 * 1024,   // 128 KB
  256 * 1024    // 256 KB
};
static const int kNumCopySizes = sizeof(kCopySizes) / sizeof(kCopySizes[0]);

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


static void PrintMemorySubtestHeader(const char *header) {
  std::cout << "  *** Memory Subtest: " << header << " ***" << std::endl;
}

hsa_status_t copy_frm_system_to_GPU_memory(ThreadContext* ctx, size_t copy_size = kCopySize_1){
    hsa_status_t status;

    //reset signal
    hsa_signal_store_screlease(ctx->signal, 1);

    // Perform async copy from system to GPU memory
    status = hsa_amd_memory_async_copy(
        ctx->dst_buffer,
        ctx->shared->gpu_agent,
        ctx->src_buffer,
        ctx->shared->gpu_agent,
        copy_size,
        0,
        nullptr,
        ctx->signal);

    if (status != HSA_STATUS_SUCCESS) {
      const char* msg = nullptr;
      hsa_status_string(status, &msg);
      std::cout << "[Thread " << ctx->thread_id << "] async_copy FAILED: " << msg << std::endl;
      return HSA_STATUS_ERROR;
    }

    hsa_signal_value_t result = hsa_signal_wait_scacquire(
        ctx->signal, HSA_SIGNAL_CONDITION_LT, 1,
        UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);

    return HSA_STATUS_SUCCESS;
}

hsa_status_t copy_frm_GPU_to_system_memory(ThreadContext* ctx ,size_t copy_size = kCopySize_1){

    hsa_status_t status;
    //reset signal
    hsa_signal_store_screlease(ctx->signal, 1);

    status = hsa_amd_memory_async_copy(
        ctx->src_buffer,        // System memory
        ctx->shared->gpu_agent,
        ctx->dst_buffer,        // GPU memory
        ctx->shared->gpu_agent,
        copy_size,              // Same varying size
        0,
        nullptr,
        ctx->signal);

    if (status != HSA_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    hsa_signal_wait_scacquire(ctx->signal, HSA_SIGNAL_CONDITION_LT, 1,
                               UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    return HSA_STATUS_SUCCESS;
}

// Thread worker function that performs async copies
static void* async_copy_worker(void* arg) {
  ThreadContext* ctx = reinterpret_cast<ThreadContext*>(arg);
  hsa_status_t status;

  // Perform multiple async copies to maximize race window
  for (int i = 0; i < kCopiesPerThread; i++) {
    // Vary copy size to trigger different SDMA engine selection
    size_t copy_size = kCopySizes[(ctx->thread_id * kCopiesPerThread + i) % kNumCopySizes];

    if( HSA_STATUS_SUCCESS != copy_frm_system_to_GPU_memory(ctx, copy_size)){
      ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    // Copy back from GPU to system (triggers GetBlitObject again)
    if(HSA_STATUS_SUCCESS != copy_frm_GPU_to_system_memory(ctx, copy_size)){
      ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
  }
  return nullptr;
}



// Thread worker function - performs many small async copies rapidly
static void* sdma_race_worker(void* arg) {
  ThreadContext* ctx = reinterpret_cast<ThreadContext*>(arg);

  // Hammer the SDMA submission path with tiny copies
  for (int i = 0; i < kCopiesPerThread; i++) {
    // Small async copy from system to GPU memory
    // This will trigger SubmitCommand() -> ReleaseWriteAddress() ->
    // UpdateWriteAndDoorbellRegister() where the race occurs
    if( HSA_STATUS_SUCCESS != copy_frm_system_to_GPU_memory(ctx)){
      ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
  }
  return nullptr;
}
ConcurrentAsyncCopy::ConcurrentAsyncCopy(void) : TestBase() {
  set_num_iteration(1);
  set_title("SDMA read write TSAN Test");
  set_description("Reproduces Mutilpe race condition on sdmaBlit"
                  "having multiple threads concurrently call GetBlitObject()"
                  "/BlitSdma::UpdateWriteAndDoorbellRegister()");
}

ConcurrentAsyncCopy::~ConcurrentAsyncCopy(void) {}

void ConcurrentAsyncCopy::SetUp(void) {
  //ASSERT_SUCCESS(hsa_init());
  TestBase::SetUp();
  //return;
}

void ConcurrentAsyncCopy::Run(void) {
    if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void ConcurrentAsyncCopy::TestConcurrentAsyncCopy(){
  PrintMemorySubtestHeader("Reproduces race condition on sdma_blit_used_mask_ by"
                  "having multiple threads concurrently call GetBlitObject()");
  hsa_status_t status;

  // Set default agents (CPU and GPU)
  status = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Find typical memory pools
  status = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Get GPU agent and pools from base class
  hsa_agent_t gpu_agent = *gpu_device1();
  hsa_amd_memory_pool_t sys_pool = cpu_pool();
  hsa_amd_memory_pool_t gpu_pool = device_pool();

  // Set up shared data
  SdmaRaceThreadData shared_data;
  shared_data.gpu_agent = gpu_agent;
  shared_data.gpu_pool = gpu_pool;
  shared_data.sys_pool = sys_pool;

  // Create thread contexts
  std::vector<ThreadContext> contexts(kNumThreads);
  std::vector<pthread_t> threads(kNumThreads);

  // Allocate buffers and signals for each thread
  for (int i = 0; i < kNumThreads; i++) {
    contexts[i].shared = &shared_data;
    contexts[i].thread_id = i;

    // Allocate system memory (source) - use max size to accommodate all copy sizes
    status = hsa_amd_memory_pool_allocate(sys_pool, kMaxCopySize, 0,
                                           &contexts[i].src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Make system memory accessible to GPU
    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          contexts[i].src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Initialize source buffer
    memset(contexts[i].src_buffer, 0xAB + i, kMaxCopySize);

    // Allocate GPU memory (destination) - use max size
    status = hsa_amd_memory_pool_allocate(gpu_pool, kMaxCopySize, 0,
                                           &contexts[i].dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Make GPU memory accessible from CPU for verification
    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          contexts[i].dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Create signal for async copy
    status = hsa_signal_create(1, 0, nullptr, &contexts[i].signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }

  // Create threads
  for (int i = 0; i < kNumThreads; i++) {
    int rc = pthread_create(&threads[i], nullptr, async_copy_worker,
                             &contexts[i]);
    ASSERT_EQ(0, rc);
  }

  // Start all threads simultaneously to maximize race condition
  std::cout << "Starting " << kNumThreads << " threads for concurrent async copy...\n";

  // Wait for all threads to complete
  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(threads[i], nullptr);
  }

  std::cout << "All threads completed.\n";
  ASSERT_EQ(0, shared_data.error_count.load());

  // Cleanup
  for (int i = 0; i < kNumThreads; i++) {
    hsa_signal_destroy(contexts[i].signal);
    hsa_amd_memory_pool_free(contexts[i].src_buffer);
    hsa_amd_memory_pool_free(contexts[i].dst_buffer);
  }
}


void ConcurrentAsyncCopy::TestSdmaDoorbellRace() {
  PrintMemorySubtestHeader("Reproduces race condition in BlitSdma::UpdateWriteAndDoorbellRegister()"
                  "at line 1300 (*queue_wptr_ write) by having many threads"
                  "concurrently submit small SDMA commands");
  hsa_status_t status;

  // Set default agents (CPU and GPU)
  status = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  // Find typical memory pools
  status = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Get GPU agent and pools
  hsa_agent_t gpu_agent = *gpu_device1();
  hsa_amd_memory_pool_t sys_pool = cpu_pool();
  hsa_amd_memory_pool_t gpu_pool = device_pool();
  // Set up shared data
  SdmaRaceThreadData shared_data;
  shared_data.gpu_agent = gpu_agent;
  shared_data.gpu_pool = gpu_pool;
  shared_data.sys_pool = sys_pool;

  // Create thread contexts
  std::vector<ThreadContext> contexts(kNumThreads);
  std::vector<pthread_t> threads(kNumThreads);
  // Allocate buffers and signals for each thread
  for (int i = 0; i < kNumThreads; i++) {
    contexts[i].shared = &shared_data;
    contexts[i].thread_id = i;

    // Allocate small system memory buffer
    status = hsa_amd_memory_pool_allocate(sys_pool, kCopySize_1, 0,
                                           &contexts[i].src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Make system memory accessible to GPU
    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          contexts[i].src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Initialize source buffer
    memset(contexts[i].src_buffer, 0xAB + i, kCopySize_1);

    // Allocate small GPU memory buffer
    status = hsa_amd_memory_pool_allocate(gpu_pool, kCopySize_1, 0,
                                           &contexts[i].dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Make GPU memory accessible from CPU
    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          contexts[i].dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Create signal for async copy
    status = hsa_signal_create(1, 0, nullptr, &contexts[i].signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }
  // Create threads - they start working immediately (no sync barrier)
  std::cout << "Creating " << kNumThreads << " threads with " << kCopiesPerThread << " copies each..." << std::endl;
  std::cout << "Total SDMA submissions: " << (kNumThreads * kCopiesPerThread) << std::endl;

  for (int i = 0; i < kNumThreads; i++) {
    int rc = pthread_create(&threads[i], nullptr, sdma_race_worker,
                             &contexts[i]);
    ASSERT_EQ(0, rc);
  }

  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(threads[i], nullptr);
  }

  std::cout << "All threads completed!" << std::endl;
  ASSERT_EQ(0, shared_data.error_count.load());

  // Cleanup
  for (int i = 0; i < kNumThreads; i++) {
    hsa_signal_destroy(contexts[i].signal);
    hsa_amd_memory_pool_free(contexts[i].src_buffer);
    hsa_amd_memory_pool_free(contexts[i].dst_buffer);
  }
}

void ConcurrentAsyncCopy::Close() {
  //ASSERT_SUCCESS(hsa_shut_down())
    TestBase::Close();
}

void ConcurrentAsyncCopy::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void ConcurrentAsyncCopy::DisplayResults(void) const {
  TestBase::DisplayResults();
}


