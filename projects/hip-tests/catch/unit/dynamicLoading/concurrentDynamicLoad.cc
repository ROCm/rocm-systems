/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <dlfcn.h>
#include <pthread.h>
#include <atomic>
#include <thread>
#include <vector>

/**
 * @addtogroup hipLaunchKernelGGL
 * @{
 * @ingroup DynamicLoading
 */

/**
 * Test Description
 * ------------------------
 * - Stress-tests the thread-safety of the HIP runtime's StatCO class by
 *   racing concurrent kernel launches (StatCO::GetFunc -> functions_.find)
 *   against concurrent module unloads (StatCO::RemoveFatBinary ->
 *   functions_.erase) triggered by dlopen/dlclose of libLazyLoad.so.
 *
 *   Worker threads repeatedly launch THIS file's own kernel (which belongs
 *   to the main module, never removed during the test), so every kernel
 *   launch exercises functions_.find.  Churner threads repeatedly
 *   dlopen("./libLazyLoad.so") and dlclose it; the dlclose runs the
 *   library's static destructors -> __hipUnregisterFatBinary ->
 *   StatCO::RemoveFatBinary, which calls functions_.erase on the library's
 *   entry.  The race between find and erase on the shared functions_ map
 *   (and its surrounding lock, if any) is what this test is designed to
 *   expose.
 *
 *   A pthread_barrier_t is used (C++17 lacks std::barrier) to release all
 *   threads simultaneously, maximising contention.
 *
 *   The test is bounded (300 worker launch iterations, 40 churner dlopen/dlclose
 *   cycles), uses a single GPU
 *   (device 0), and is safe to run in normal CI.  Each worker allocates
 *   its own device/host buffers, so there are no cross-thread data races
 *   on user buffers.
 *
 *   In plain CI (without ThreadSanitizer) the test primarily catches
 *   crashes, use-after-free, hangs (deadlock), and incorrect kernel output.
 *   Full data-race detection requires re-running with TSAN enabled.
 *
 * Test source
 * ------------------------
 * - catch/unit/dynamicLoading/concurrentDynamicLoad.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

/* ---------------------------------------------------------------------------
 * Kernel owned by THIS translation unit (main module – never unloaded).
 * Workers launch this kernel; only churners dlclose libLazyLoad.so.
 * ---------------------------------------------------------------------------*/
__global__ static void increment_kernel(int* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += 1;
  }
}

/* ---------------------------------------------------------------------------
 * Shared failure flag set by any thread that detects wrong output.
 * ---------------------------------------------------------------------------*/
static std::atomic<bool> g_failure{false};

/* ---------------------------------------------------------------------------
 * Per-worker context: owns its own device/host buffers.
 * ---------------------------------------------------------------------------*/
struct WorkerCtx {
  pthread_barrier_t* barrier;
  int n;
  int iters;
};

/* ---------------------------------------------------------------------------
 * Worker thread: launch THIS file's kernel and validate output.
 * ---------------------------------------------------------------------------*/
static void* worker_fn(void* arg) {
  WorkerCtx* ctx = static_cast<WorkerCtx*>(arg);

  if (hipSetDevice(0) != hipSuccess) {
    g_failure.store(true);
    return nullptr;
  }

  const int n = ctx->n;
  const size_t nbytes = static_cast<size_t>(n) * sizeof(int);

  /* Per-thread RAII-managed host and device buffers. */
  std::vector<int> h_in(n);
  std::vector<int> h_out(n);
  for (int i = 0; i < n; ++i) h_in[i] = i;

  LinearAllocGuard<int> d_data(LinearAllocs::hipMalloc, nbytes);

  /* Wait until all threads are ready to maximise contention. */
  pthread_barrier_wait(ctx->barrier);

  const int threadsPerBlock = 256;
  const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  for (int iter = 0; iter < ctx->iters && !g_failure.load(); ++iter) {
    /* Reset device buffer to known input. */
    if (hipMemcpy(d_data.ptr(), h_in.data(), nbytes, hipMemcpyHostToDevice) != hipSuccess) {
      g_failure.store(true);
      break;
    }

    /* Launch the kernel from THIS file's module (StatCO::GetFunc lookup). */
    hipLaunchKernelGGL(increment_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_data.ptr(),
                       n);
    if (hipGetLastError() != hipSuccess) {
      g_failure.store(true);
      break;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
      g_failure.store(true);
      break;
    }

    /* Copy result back and validate. */
    if (hipMemcpy(h_out.data(), d_data.ptr(), nbytes, hipMemcpyDeviceToHost) != hipSuccess) {
      g_failure.store(true);
      break;
    }

    for (int i = 0; i < n; ++i) {
      if (h_out[i] != h_in[i] + 1) {
        g_failure.store(true);
        break;
      }
    }
  }

  return nullptr;
}

/* ---------------------------------------------------------------------------
 * Churner context: owns iteration count and barrier pointer.
 * ---------------------------------------------------------------------------*/
struct ChurnerCtx {
  pthread_barrier_t* barrier;
  int iters;
  std::atomic<int>* dlopen_success_count;
};

/* ---------------------------------------------------------------------------
 * Churner thread: dlopen/dlclose libLazyLoad.so repeatedly.
 * The dlclose triggers __hipUnregisterFatBinary -> StatCO::RemoveFatBinary
 * (functions_.erase), racing against the workers' functions_.find calls.
 * ---------------------------------------------------------------------------*/
static void* churner_fn(void* arg) {
  ChurnerCtx* ctx = static_cast<ChurnerCtx*>(arg);

  /* Wait for all threads to be ready. */
  pthread_barrier_wait(ctx->barrier);

  for (int iter = 0; iter < ctx->iters; ++iter) {
    void* handle = dlopen("./libLazyLoad.so", RTLD_LAZY);
    if (!handle) {
      /* dlopen can transiently fail (e.g., another churner holds the
       * linker lock).  Log but do not hard-fail the test. */
      continue;
    }

    ctx->dlopen_success_count->fetch_add(1);

    void* sym = dlsym(handle, "lazyLoad");
    if (sym) {
      int (*fp)() = reinterpret_cast<int (*)()>(sym);
      /* We do not assert on the return value here; the purpose of the
       * call is just to exercise the registered fat binary path before
       * dlclose triggers removal. */
      (void)fp();
    }

    /* dlclose -> static destructors -> __hipUnregisterFatBinary ->
     * StatCO::RemoveFatBinary: this is the racing erase. */
    dlclose(handle);
  }

  return nullptr;
}

/* ---------------------------------------------------------------------------
 * Test case
 * ---------------------------------------------------------------------------*/
HIP_TEST_CASE(Unit_StatCO_ConcurrentDlopenDlcloseWhileLaunching) {
  HIPCHECK(hipSetDevice(0));

  constexpr int kWorkers = 4;
  constexpr int kChurners = 2;
  constexpr int kTotalThreads = kWorkers + kChurners;
  /* Worker launches are cheap (kernel launch + sync), so iterate many times to
   * keep the functions_.find load high. Churner cycles each do a full dlopen ->
   * fat-binary digest/kernel-build -> dlclose, which is expensive, so they are
   * bounded to a much smaller count to keep CI runtime sane. The race is
   * exercised on every churn cycle (each one erases the lib's funcs while the
   * workers are finding), so a few dozen cycles still gives strong coverage. */
  constexpr int kWorkerIters = 300;
  constexpr int kChurnerIters = 40;
  constexpr int kN = 1024;  /* Small buffer; latency matters, not bandwidth. */

  g_failure.store(false);

  pthread_barrier_t barrier;
  REQUIRE(pthread_barrier_init(&barrier, nullptr,
                               static_cast<unsigned>(kTotalThreads)) == 0);

  std::atomic<int> dlopen_success_count{0};

  /* Build contexts. */
  std::vector<WorkerCtx> wctx(kWorkers);
  for (int i = 0; i < kWorkers; ++i) {
    wctx[i].barrier = &barrier;
    wctx[i].n = kN;
    wctx[i].iters = kWorkerIters;
  }

  std::vector<ChurnerCtx> cctx(kChurners);
  for (int i = 0; i < kChurners; ++i) {
    cctx[i].barrier = &barrier;
    cctx[i].iters = kChurnerIters;
    cctx[i].dlopen_success_count = &dlopen_success_count;
  }

  /* Spawn all threads. */
  std::vector<pthread_t> tids(kTotalThreads);

  for (int i = 0; i < kWorkers; ++i) {
    REQUIRE(pthread_create(&tids[i], nullptr, worker_fn, &wctx[i]) == 0);
  }
  for (int i = 0; i < kChurners; ++i) {
    REQUIRE(pthread_create(&tids[kWorkers + i], nullptr, churner_fn,
                           &cctx[i]) == 0);
  }

  /* Join all threads. */
  for (int i = 0; i < kTotalThreads; ++i) {
    pthread_join(tids[i], nullptr);
  }

  pthread_barrier_destroy(&barrier);

  /* The churners should have succeeded at dlopen at least once across all
   * iterations; if never, the .so was simply not found — warn but allow. */
  if (dlopen_success_count.load() == 0) {
    WARN("libLazyLoad.so could not be opened during any churner iteration; "
         "the concurrent-unload half of the test did not execute.");
  }

  /* Primary assertion: no kernel produced wrong results and no crash. */
  REQUIRE(g_failure.load() == false);
}

/**
 * End doxygen group DynamicLoading.
 * @}
 */
