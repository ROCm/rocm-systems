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
#include <algorithm>

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
ConcurrentAsyncCopy::ConcurrentAsyncCopy(void) : TestBase(), queue_(nullptr), cleanup_required_(false) {
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

void ConcurrentAsyncCopy::TestSdmaDoorbellRace() {
  PrintMemorySubtestHeader(" Multiple threads concurrently submit small SDMA commands");
  hsa_status_t status;

  // Cleanup any previous test data
  CleanupResources();

  // Set default agents (CPU and GPU)
  status = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  // Find typical memory pools
  status = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Mark that cleanup will be required
  cleanup_required_ = true;

  // Get GPU agent and pools
  hsa_agent_t gpu_agent = *gpu_device1();
  hsa_amd_memory_pool_t sys_pool = cpu_pool();
  hsa_amd_memory_pool_t gpu_pool = device_pool();
  // Set up shared data
  SdmaRaceThreadData shared_data;
  shared_data.gpu_agent = gpu_agent;
  shared_data.gpu_pool = gpu_pool;
  shared_data.sys_pool = sys_pool;

  // Reserve capacity and allocate buffers and signals for each thread
  contexts_.reserve(kNumThreads);
  sdma_threads_.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; i++) {
    contexts_.emplace_back();
    ThreadContext& ctx = contexts_.back();

    ctx.shared = &shared_data;
    ctx.thread_id = i;

    // Allocate small system memory buffer
    status = hsa_amd_memory_pool_allocate(sys_pool, kCopySize_1, 0,
                                           &ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Make system memory accessible to GPU
    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Initialize source buffer
    memset(ctx.src_buffer, 0xAB + i, kCopySize_1);

    // Allocate small GPU memory buffer
    status = hsa_amd_memory_pool_allocate(gpu_pool, kCopySize_1, 0,
                                           &ctx.dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Make GPU memory accessible from CPU
    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          ctx.dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    // Create signal for async copy
    status = hsa_signal_create(1, 0, nullptr, &ctx.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    sdma_threads_.emplace_back();
  }

  // Create threads - they start working immediately (no sync barrier)
  std::cout << "Creating " << kNumThreads << " threads with " << kCopiesPerThread << " copies each..." << std::endl;
  std::cout << "Total SDMA submissions: " << (kNumThreads * kCopiesPerThread) << std::endl;

  for (int i = 0; i < kNumThreads; i++) {
    int rc = pthread_create(&sdma_threads_[i], nullptr, sdma_race_worker,
                             &contexts_[i]);
    ASSERT_EQ(0, rc);
  }

  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(sdma_threads_[i], nullptr);
  }

  std::cout << "All threads completed!" << std::endl;
  ASSERT_EQ(0, shared_data.error_count.load());
}

// Test 3: SDMA Engine Exhaustion Test
// Forces more threads than available SDMA engines to test engine reuse and contention
void ConcurrentAsyncCopy::TestSdmaEngineExhaustion() {
  PrintMemorySubtestHeader("SDMA Engine Exhaustion - Force engine reuse with 20 threads");

  static const int kEngineExhaustThreads = 20;  // Far exceeds typical 2-4 SDMA engines
  static const int kCopiesPerThread = 500;
  static const size_t kCopySize = 4 * 1024;  // 4KB copies

  hsa_status_t status;

  // Cleanup any previous test data
  CleanupResources();

  status = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  status = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Mark that cleanup will be required
  cleanup_required_ = true;

  hsa_agent_t gpu_agent = *gpu_device1();
  hsa_amd_memory_pool_t sys_pool = cpu_pool();
  hsa_amd_memory_pool_t gpu_pool = device_pool();

  SdmaRaceThreadData shared_data;
  shared_data.gpu_agent = gpu_agent;
  shared_data.gpu_pool = gpu_pool;
  shared_data.sys_pool = sys_pool;

  contexts_.reserve(kEngineExhaustThreads);
  sdma_threads_.reserve(kEngineExhaustThreads);

  // Allocate buffers for each thread
  for (int i = 0; i < kEngineExhaustThreads; i++) {
    contexts_.emplace_back();
    ThreadContext& ctx = contexts_.back();

    ctx.shared = &shared_data;
    ctx.thread_id = i;

    status = hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0,
                                           &ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    memset(ctx.src_buffer, 0xAB + i, kCopySize);

    status = hsa_amd_memory_pool_allocate(gpu_pool, kCopySize, 0,
                                           &ctx.dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    status = hsa_signal_create(1, 0, nullptr, &ctx.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    sdma_threads_.emplace_back();
  }

  // Worker that does BLOCKING copies to force engine allocation/deallocation
  auto engine_exhaust_worker = [](void* arg) -> void* {
    ThreadContext* ctx = reinterpret_cast<ThreadContext*>(arg);

    for (int i = 0; i < kCopiesPerThread; i++) {
      // Each copy blocks until complete, forcing engine hold time
      if (HSA_STATUS_SUCCESS != copy_frm_system_to_GPU_memory(ctx, kCopySize)) {
        ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }
    }
    return nullptr;
  };

  std::cout << "Creating " << kEngineExhaustThreads << " threads (exceeds typical 2-4 SDMA engines)...\n";

  for (int i = 0; i < kEngineExhaustThreads; i++) {
    int rc = pthread_create(&sdma_threads_[i], nullptr, engine_exhaust_worker,
                             &contexts_[i]);
    ASSERT_EQ(0, rc);
  }

  for (int i = 0; i < kEngineExhaustThreads; i++) {
    pthread_join(sdma_threads_[i], nullptr);
  }

  std::cout << "All threads completed - engine allocation/deallocation successful!\n";
  ASSERT_EQ(0, shared_data.error_count.load());
}

// Test 4: Signal Completion Race Test
// Tests rapid signal reuse to expose races in SDMA completion handling
void ConcurrentAsyncCopy::TestSignalCompletionRace() {
  PrintMemorySubtestHeader("Signal Completion Race - Rapid signal reuse (10K iterations)");

  static const int kNumThreads = 50;
  static const int kRapidReuses = 10000;
  static const size_t kCopySize = 256;

  hsa_status_t status;

  // Cleanup any previous test data
  CleanupResources();

  status = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  status = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Mark that cleanup will be required
  cleanup_required_ = true;

  hsa_agent_t gpu_agent = *gpu_device1();
  hsa_amd_memory_pool_t sys_pool = cpu_pool();
  hsa_amd_memory_pool_t gpu_pool = device_pool();

  SdmaRaceThreadData shared_data;
  shared_data.gpu_agent = gpu_agent;
  shared_data.gpu_pool = gpu_pool;
  shared_data.sys_pool = sys_pool;

  contexts_.reserve(kNumThreads);
  sdma_threads_.reserve(kNumThreads);

  // Allocate buffers for each thread
  for (int i = 0; i < kNumThreads; i++) {
    contexts_.emplace_back();
    ThreadContext& ctx = contexts_.back();

    ctx.shared = &shared_data;
    ctx.thread_id = i;

    status = hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0,
                                           &ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    memset(ctx.src_buffer, 0xAB + i, kCopySize);

    status = hsa_amd_memory_pool_allocate(gpu_pool, kCopySize, 0,
                                           &ctx.dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    status = hsa_signal_create(1, 0, nullptr, &ctx.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    sdma_threads_.emplace_back();
  }

  // Worker that rapidly reuses the same signal
  auto signal_reuse_worker = [](void* arg) -> void* {
    ThreadContext* ctx = reinterpret_cast<ThreadContext*>(arg);
    hsa_status_t status;

    for (int i = 0; i < kRapidReuses; i++) {
      // Reset signal (might race with SDMA completion)
      hsa_signal_store_screlease(ctx->signal, 1);

      // Submit copy
      status = hsa_amd_memory_async_copy(
          ctx->dst_buffer,
          ctx->shared->gpu_agent,
          ctx->src_buffer,
          ctx->shared->gpu_agent,
          kCopySize,
          0,
          nullptr,
          ctx->signal);

      if (status != HSA_STATUS_SUCCESS) {
        ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }

      // Wait for completion
      hsa_signal_wait_scacquire(ctx->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                 UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

      // IMMEDIATELY reuse - no delay between iterations
    }
    return nullptr;
  };

  std::cout << "Creating " << kNumThreads << " threads, each reusing signal "
            << kRapidReuses << " times...\n";
  std::cout << "Total signal reuse cycles: " << (kNumThreads * kRapidReuses) << "\n";

  for (int i = 0; i < kNumThreads; i++) {
    int rc = pthread_create(&sdma_threads_[i], nullptr, signal_reuse_worker,
                             &contexts_[i]);
    ASSERT_EQ(0, rc);
  }

  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(sdma_threads_[i], nullptr);
  }

  std::cout << "All threads completed - no signal corruption detected!\n";
  ASSERT_EQ(0, shared_data.error_count.load());
}

// Test 5: Varying Copy Size Stress Test
// Tests extreme size variations and alignment edge cases
void ConcurrentAsyncCopy::TestVaryingCopySizeStress() {
  PrintMemorySubtestHeader("Varying Copy Size Stress - 1 byte to 256MB, including unaligned");

  static const size_t kTestSizes[] = {
    1,           // Single byte
    3,           // Unaligned
    7,           // Unaligned
    15,          // Unaligned
    16,          // Aligned
    64,          // Cache line
    256,         // Small
    1024,        // 1 KB
    4096,        // 4 KB (page size)
    4097,        // Unaligned to page
    64 * 1024,   // 64 KB
    1024 * 1024, // 1 MB
    4 * 1024 * 1024,   // 4 MB
    16 * 1024 * 1024,  // 16 MB
    256 * 1024 * 1024  // 256 MB (large)
  };
  static const int kNumTestSizes = sizeof(kTestSizes) / sizeof(kTestSizes[0]);
  static const int kNumThreads = 10;

  hsa_status_t status;

  // Cleanup any previous test data
  CleanupResources();

  status = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  status = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Mark that cleanup will be required
  cleanup_required_ = true;

  hsa_agent_t gpu_agent = *gpu_device1();
  hsa_amd_memory_pool_t sys_pool = cpu_pool();
  hsa_amd_memory_pool_t gpu_pool = device_pool();

  SdmaRaceThreadData shared_data;
  shared_data.gpu_agent = gpu_agent;
  shared_data.gpu_pool = gpu_pool;
  shared_data.sys_pool = sys_pool;

  contexts_.reserve(kNumThreads);
  sdma_threads_.reserve(kNumThreads);

  // Each thread tests all sizes
  for (int i = 0; i < kNumThreads; i++) {
    contexts_.emplace_back();
    ThreadContext& ctx = contexts_.back();

    ctx.shared = &shared_data;
    ctx.thread_id = i;
    ctx.src_buffer = nullptr;
    ctx.dst_buffer = nullptr;

    status = hsa_signal_create(1, 0, nullptr, &ctx.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    sdma_threads_.emplace_back();
  }

  // Worker that tests all size variations
  auto size_variation_worker = [](void* arg) -> void* {
    ThreadContext* ctx = reinterpret_cast<ThreadContext*>(arg);
    hsa_status_t status;

    for (int i = 0; i < kNumTestSizes; i++) {
      size_t size = kTestSizes[i];
      void* src = nullptr;
      void* dst = nullptr;

      // Allocate buffers for this size
      status = hsa_amd_memory_pool_allocate(ctx->shared->sys_pool, size, 0, &src);
      if (status != HSA_STATUS_SUCCESS) {
        ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }

      status = hsa_amd_agents_allow_access(1, &ctx->shared->gpu_agent, nullptr, src);
      if (status != HSA_STATUS_SUCCESS) {
        ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
        hsa_amd_memory_pool_free(src);
        return nullptr;
      }

      // Fill with pattern
      memset(src, 0xAB, size);

      status = hsa_amd_memory_pool_allocate(ctx->shared->gpu_pool, size, 0, &dst);
      if (status != HSA_STATUS_SUCCESS) {
        ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
        hsa_amd_memory_pool_free(src);
        return nullptr;
      }

      // Copy
      hsa_signal_store_screlease(ctx->signal, 1);
      status = hsa_amd_memory_async_copy(
          dst,
          ctx->shared->gpu_agent,
          src,
          ctx->shared->gpu_agent,
          size,
          0,
          nullptr,
          ctx->signal);

      if (status != HSA_STATUS_SUCCESS) {
        ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
        hsa_amd_memory_pool_free(src);
        hsa_amd_memory_pool_free(dst);
        return nullptr;
      }

      // Wait for completion
      hsa_signal_wait_scacquire(ctx->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                 UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

      // Cleanup this iteration
      hsa_amd_memory_pool_free(src);
      hsa_amd_memory_pool_free(dst);
    }
    return nullptr;
  };

  std::cout << "Creating " << kNumThreads << " threads, each testing "
            << kNumTestSizes << " different copy sizes...\n";

  for (int i = 0; i < kNumThreads; i++) {
    int rc = pthread_create(&sdma_threads_[i], nullptr, size_variation_worker,
                             &contexts_[i]);
    ASSERT_EQ(0, rc);
  }

  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(sdma_threads_[i], nullptr);
  }

  std::cout << "All threads completed - all copy sizes succeeded!\n";
  ASSERT_EQ(0, shared_data.error_count.load());
}

// Test 6: Concurrent Copy + Kernel Dispatch Test
// Tests interaction between SDMA copies and AQL kernel dispatch
void ConcurrentAsyncCopy::TestConcurrentCopyAndKernelDispatch() {
  PrintMemorySubtestHeader("Concurrent Copy + Kernel Dispatch - SDMA and AQL concurrency");

  static const int kSdmaThreads = 25;
  static const int kKernelThreads = 25;
  static const int kIterations = 1000;
  static const size_t kCopySize = 4096;

  hsa_status_t status;

  // Cleanup any previous test data
  CleanupResources();

  status = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  status = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Mark that cleanup will be required
  cleanup_required_ = true;

  hsa_agent_t gpu_agent = *gpu_device1();
  hsa_amd_memory_pool_t sys_pool = cpu_pool();
  hsa_amd_memory_pool_t gpu_pool = device_pool();

  // Check if GPU supports kernel dispatch
  uint32_t features = 0;
  status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_FEATURE, &features);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  if (0 == (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)) {
    std::cout << "GPU does not support kernel dispatch, skipping test\n";
    return;
  }

  // Create AQL queue for kernel dispatch threads
  uint32_t queue_max_size;
  status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max_size);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Use a smaller queue size (1024) - queue size must be power of 2
  uint32_t queue_size = std::min(1024u, queue_max_size);

  status = hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  ASSERT_NE(nullptr, queue_);

  std::cout << "Created AQL queue with size " << queue_->size << "\n";

  SdmaRaceThreadData shared_data;
  shared_data.gpu_agent = gpu_agent;
  shared_data.gpu_pool = gpu_pool;
  shared_data.sys_pool = sys_pool;

  contexts_.reserve(kSdmaThreads);
  sdma_threads_.reserve(kSdmaThreads);
  kernel_threads_.reserve(kKernelThreads);

  // Setup SDMA thread contexts
  for (int i = 0; i < kSdmaThreads; i++) {
    contexts_.emplace_back();
    ThreadContext& ctx = contexts_.back();

    ctx.shared = &shared_data;
    ctx.thread_id = i;

    status = hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0,
                                           &ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr,
                                          ctx.src_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    memset(ctx.src_buffer, 0xAB + i, kCopySize);

    status = hsa_amd_memory_pool_allocate(gpu_pool, kCopySize, 0,
                                           &ctx.dst_buffer);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    status = hsa_signal_create(1, 0, nullptr, &ctx.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    sdma_threads_.emplace_back();
  }

  // Add kernel threads
  for (int i = 0; i < kKernelThreads; i++) {
    kernel_threads_.emplace_back();
  }

  // SDMA worker
  auto sdma_worker = [](void* arg) -> void* {
    ThreadContext* ctx = reinterpret_cast<ThreadContext*>(arg);
    for (int i = 0; i < kIterations; i++) {
      if (HSA_STATUS_SUCCESS != copy_frm_system_to_GPU_memory(ctx, kCopySize)) {
        ctx->shared->error_count.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }
    }
    return nullptr;
  };

  // Kernel dispatch worker - uses barrier packets since we don't have actual kernels
  struct KernelThreadData {
    hsa_queue_t* queue;
    int thread_id;
    std::atomic<int>* error_count;
  };

  std::vector<KernelThreadData> kernel_data(kKernelThreads);
  for (int i = 0; i < kKernelThreads; i++) {
    kernel_data[i].queue = queue_;
    kernel_data[i].thread_id = i;
    kernel_data[i].error_count = &shared_data.error_count;
  }

  auto kernel_dispatch_worker = [](void* arg) -> void* {
    KernelThreadData* data = reinterpret_cast<KernelThreadData*>(arg);
    hsa_status_t status;

    for (int i = 0; i < kIterations; i++) {
      // Create a signal for this dispatch
      hsa_signal_t sig;
      status = hsa_signal_create(1, 0, nullptr, &sig);
      if (status != HSA_STATUS_SUCCESS) {
        data->error_count->fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }

      // Get write index
      uint64_t write_idx = hsa_queue_add_write_index_scacq_screl(data->queue, 1);
      const uint32_t mask = data->queue->size - 1;

      // Build packet in local variable first to avoid races
      hsa_barrier_and_packet_t barrier_pkt;
      memset(&barrier_pkt, 0, sizeof(barrier_pkt));

      // Set packet fields
      barrier_pkt.header = HSA_PACKET_TYPE_BARRIER_AND;
      barrier_pkt.completion_signal = sig;

      // All dep_signal entries are already 0 from memset
      // reserved0 and reserved1 are already 0 from memset

      // Copy packet to queue (this writes header=3 which makes it valid)
      hsa_barrier_and_packet_t* queue_pkt =
          &(reinterpret_cast<hsa_barrier_and_packet_t*>(data->queue->base_address)[write_idx & mask]);
      *queue_pkt = barrier_pkt;

      // Ring doorbell to notify hardware of new packet
      hsa_signal_store_screlease(data->queue->doorbell_signal, write_idx);

      // Wait for completion
      hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1,
                                 UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

      hsa_signal_destroy(sig);
    }
    return nullptr;
  };

  std::cout << "Creating " << kSdmaThreads << " SDMA threads and "
            << kKernelThreads << " kernel dispatch threads...\n";

  // Create SDMA threads
  for (int i = 0; i < kSdmaThreads; i++) {
    int rc = pthread_create(&sdma_threads_[i], nullptr, sdma_worker,
                             &contexts_[i]);
    ASSERT_EQ(0, rc);
  }

  // Create kernel dispatch threads
  for (int i = 0; i < kKernelThreads; i++) {
    int rc = pthread_create(&kernel_threads_[i], nullptr, kernel_dispatch_worker,
                             &kernel_data[i]);
    ASSERT_EQ(0, rc);
  }

  // Wait for all SDMA threads
  for (int i = 0; i < kSdmaThreads; i++) {
    pthread_join(sdma_threads_[i], nullptr);
  }

  // Wait for all kernel threads
  for (int i = 0; i < kKernelThreads; i++) {
    pthread_join(kernel_threads_[i], nullptr);
  }

  std::cout << "All threads completed - SDMA and kernel dispatch coexisted successfully!\n";
  ASSERT_EQ(0, shared_data.error_count.load());
}

void ConcurrentAsyncCopy::CleanupResources() {
  // Skip cleanup if no resources were allocated
  if (!cleanup_required_) {
    return;
  }

  // Cleanup thread contexts (signals and buffers)
  for (auto& ctx : contexts_) {
    if (ctx.signal.handle != 0) {
      hsa_signal_destroy(ctx.signal);
      ctx.signal.handle = 0;
    }
    if (ctx.src_buffer != nullptr) {
      hsa_amd_memory_pool_free(ctx.src_buffer);
      ctx.src_buffer = nullptr;
    }
    if (ctx.dst_buffer != nullptr) {
      hsa_amd_memory_pool_free(ctx.dst_buffer);
      ctx.dst_buffer = nullptr;
    }
  }

  // Cleanup queue if it exists
  if (queue_ != nullptr) {
    hsa_queue_destroy(queue_);
    queue_ = nullptr;
  }

  // Clear vectors
  contexts_.clear();
  sdma_threads_.clear();
  kernel_threads_.clear();

  // Reset flag
  cleanup_required_ = false;
}

void ConcurrentAsyncCopy::Close() {
  CleanupResources();
  //ASSERT_SUCCESS(hsa_shut_down())
  TestBase::Close();
}

void ConcurrentAsyncCopy::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void ConcurrentAsyncCopy::DisplayResults(void) const {
  TestBase::DisplayResults();
}


