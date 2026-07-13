/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Child-process executable for Unit_StatCO_ManagedVarConcurrentFirstTouch.
 *
 * WHY A CHILD PROCESS:
 *   StatCO::InitManagedVarDevicePtr sets a per-device atomic flag that is
 *   PROCESS-GLOBAL state in the PlatformState singleton.  It is triggered by
 *   the very first call to ihipModuleLaunchKernel in the process.  When tests
 *   run inside the main Catch2 binary, earlier test cases have already launched
 *   kernels, so the flag is already set long before this test runs and the
 *   concurrent first-touch race window is permanently closed.  By running in a
 *   fresh child process (spawned via hip::SpawnProc), the barrier-synchronised
 *   threads below perform the FIRST kernel launch of the entire process,
 *   maximising the chance of overlapping first-touch paths colliding on
 *   InitManagedVarDevicePtr.
 *
 * WHAT RACE IS TARGETED:
 *   Tests static code object race conditions. Specifically managed variables
 *
 * IMPORTANT CONSTRAINT:
 *   No HIP kernel must be launched before the barrier-synchronised burst.
 *   Any premature launch would call ihipModuleLaunchKernel and set the flag,
 *   closing the race window before the threads are released.
 */

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <vector>

// --------------------------------------------------------------------------
// Managed global variable under test.  The HIP runtime must call
// InitManagedVarDevicePtr the first time a kernel that references this
// symbol is launched on each device.
// --------------------------------------------------------------------------
__managed__ int g_managedVal = 42;

// --------------------------------------------------------------------------
// Kernel: write a sentinel into the managed variable so we can verify
// the managed pointer was correctly initialised.
// --------------------------------------------------------------------------
__global__ void setManagedKernel(int value) {
  // Only thread 0 writes to avoid races between GPU threads.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_managedVal = value;
  }
}

// --------------------------------------------------------------------------
// Error-checking macro (plain main style, matching hipGetFuncBySymbol_exe.cc).
// --------------------------------------------------------------------------
#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      printf("HIP error '%s' (%d) at %s:%d\n",                                \
             hipGetErrorString(_err), _err, __FILE__, __LINE__);               \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// --------------------------------------------------------------------------
// Number of threads released simultaneously at the barrier.
// --------------------------------------------------------------------------
static constexpr int N_THREADS = 8;

// --------------------------------------------------------------------------
// Per-thread context passed via the pthread API.
// --------------------------------------------------------------------------
struct ThreadCtx {
  pthread_barrier_t* barrier;    // shared barrier across all N_THREADS
  int deviceId;                  // which device each thread targets
  int expectedVal;               // value the kernel will write
  std::atomic<bool>* anyError;   // set true on any failure in any thread
};

// --------------------------------------------------------------------------
// Worker: wait at the barrier, then perform the first kernel launch on
// the assigned device.  All N_THREADS threads release simultaneously to
// maximise overlap inside InitManagedVarDevicePtr.
// --------------------------------------------------------------------------
static void* workerThread(void* arg) {
  ThreadCtx* ctx = reinterpret_cast<ThreadCtx*>(arg);

  // Select the device for this thread.  hipSetDevice must be called per-thread
  // because HIP device context is per-thread on the host.
  if (hipSetDevice(ctx->deviceId) != hipSuccess) {
    ctx->anyError->store(true, std::memory_order_relaxed);
    return nullptr;
  }

  // ---- CRITICAL: no HIP kernel may be launched before this point ----
  // All threads synchronise here so that their subsequent kernel launches
  // arrive at ihipModuleLaunchKernel (and InitManagedVarDevicePtr) as
  // simultaneously as pthreads scheduling allows.
  pthread_barrier_wait(ctx->barrier);
  // -------------------------------------------------------------------

  // First kernel launch of the entire process on this device.
  // This call must trigger InitManagedVarDevicePtr inside the HIP runtime.
  hipLaunchKernelGGL(setManagedKernel, dim3(1), dim3(1), 0, nullptr,
                     ctx->expectedVal);

  if (hipGetLastError() != hipSuccess) {
    ctx->anyError->store(true, std::memory_order_relaxed);
    return nullptr;
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    ctx->anyError->store(true, std::memory_order_relaxed);
    return nullptr;
  }

  return nullptr;
}

// --------------------------------------------------------------------------
// Run the concurrent first-touch burst on one device.
// Returns 0 on success, -1 on any failure.
// --------------------------------------------------------------------------
static int runBurstOnDevice(int deviceId) {
  if (hipSetDevice(deviceId) != hipSuccess) {
    printf("hipSetDevice(%d) failed\n", deviceId);
    return -1;
  }

  // Reset the managed variable to a known initial state before the burst so
  // that we can verify the kernel actually wrote to it afterwards.
  // NOTE: this is a plain host write — no kernel launch, no flag set.
  g_managedVal = 0;
  // Ensure the managed memory is accessible on the host side before threads
  // start (the variable was declared __managed__ so this is valid on the host).
  HIP_CHECK(hipDeviceSynchronize());

  const int sentinel = 99;

  pthread_barrier_t barrier;
  if (pthread_barrier_init(&barrier, nullptr, N_THREADS) != 0) {
    printf("pthread_barrier_init failed\n");
    return -1;
  }

  std::atomic<bool> anyError{false};

  std::vector<pthread_t> threads(N_THREADS);
  std::vector<ThreadCtx> ctxs(N_THREADS);

  for (int i = 0; i < N_THREADS; ++i) {
    ctxs[i].barrier    = &barrier;
    ctxs[i].deviceId   = deviceId;
    ctxs[i].expectedVal = sentinel;
    ctxs[i].anyError   = &anyError;
  }

  // Spawn all worker threads before any of them reaches the barrier.
  for (int i = 0; i < N_THREADS; ++i) {
    if (pthread_create(&threads[i], nullptr, workerThread, &ctxs[i]) != 0) {
      printf("pthread_create failed for thread %d\n", i);
      anyError.store(true, std::memory_order_relaxed);
      // Join threads already created.
      for (int j = 0; j < i; ++j) pthread_join(threads[j], nullptr);
      pthread_barrier_destroy(&barrier);
      return -1;
    }
  }

  for (int i = 0; i < N_THREADS; ++i) {
    pthread_join(threads[i], nullptr);
  }
  pthread_barrier_destroy(&barrier);

  if (anyError.load(std::memory_order_relaxed)) {
    printf("Device %d: one or more threads reported an error\n", deviceId);
    return -1;
  }

  // After all threads have run the kernel and synchronised, the managed
  // variable must hold the sentinel written by the last kernel to execute.
  // Because multiple GPU threads all wrote the same value (sentinel), any
  // ordering is acceptable — we just verify the value is correct.
  HIP_CHECK(hipSetDevice(deviceId));
  HIP_CHECK(hipDeviceSynchronize());

  if (g_managedVal != sentinel) {
    printf("Device %d: g_managedVal = %d, expected %d\n",
           deviceId, g_managedVal, sentinel);
    return -1;
  }

  printf("Device %d: OK (g_managedVal = %d)\n", deviceId, g_managedVal);
  return 0;
}

// --------------------------------------------------------------------------
// Entry point — plain main(), same style as hipGetFuncBySymbol_exe.cc.
// --------------------------------------------------------------------------
int main() {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
    printf("No HIP devices found\n");
    return -1;
  }

  for (int dev = 0; dev < deviceCount; ++dev) {
    if (runBurstOnDevice(dev) != 0) {
      return -1;
    }
  }

  return 0;
}
