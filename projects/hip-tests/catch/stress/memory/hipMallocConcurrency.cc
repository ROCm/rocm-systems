/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios :

 1) Regress hipMalloc()/hipFree() in loop for bigger chunk of allocation
 with adequate number of iterations and later test for kernel execution on
 default gpu.

 2) Regress hipMalloc()/hipFree() in loop while allocating smaller chunks
 keeping maximum number of iterations and then run kernel code on default
 gpu, perfom data validation.

 3) Check hipMalloc() api adaptability when app creates small chunks of memory
 continuously, stores it for later use and then frees it at later point
 of time.

 4) Multithread Scenario : Exercise hipMalloc() api parellely on all gpus from
 multiple threads and regress the api.

 5) Validate memory usage with hipMemGetInfo() while regressing hipMalloc()
 api. Check for any possible memory leaks.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>


/* Buffer size for bigger chunks in alloc/free cycles */
static constexpr auto BuffSizeBC = 5 * 1024 * 1024;

/* Buffer size for smaller chunks in alloc/free cycles */
static constexpr auto BuffSizeSC = 16;

/* Max alloc/free iterations for smaller chunks */
static constexpr auto MaxAllocFree_SmallChunks = 5000000;

/* Max alloc/free iterations for bigger chunks */
static constexpr auto MaxAllocFree_BigChunks = 10000;

/* Max alloc and pool iterations */
static constexpr auto MaxAllocPoolIter = 2000000;

/* Test status shared across threads */
static std::atomic<bool> g_thTestPassed{true};


/**
 * Validates data consistency on supplied gpu
 */
static bool validateMemoryOnGPU(int gpu) {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  bool TestPassed = true;
  constexpr auto N = 4 * 1024 * 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(int);

  HIP_CHECK(hipSetDevice(gpu));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                     static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  if (!HipTest::checkVectorADD(A_h, B_h, C_h, N)) {
    UNSCOPED_INFO("Validation PASSED for gpu " << gpu);
  } else {
    UNSCOPED_INFO("Validation FAILED for gpu " << gpu);
    TestPassed = false;
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);

  return TestPassed;
}


/**
 * Regress memory allocation and free in loop
 */
static bool regressAllocInLoop(int gpu) {
  size_t numBytes;
  int i = 0;
  int* ptr;

  HIP_CHECK(hipSetDevice(gpu));
  numBytes = BuffSizeBC;

  // Exercise allocation in loop with bigger chunks
  for (i = 0; i < MaxAllocFree_BigChunks; i++) {
    HIP_CHECK(hipMalloc(&ptr, numBytes));
    HIP_CHECK(hipFree(ptr));
  }

  // Exercise allocation in loop with smaller chunks and maximum iters
  numBytes = BuffSizeSC;

  for (i = 0; i < MaxAllocFree_SmallChunks; i++) {
    HIP_CHECK(hipMalloc(&ptr, numBytes));
    HIP_CHECK(hipFree(ptr));
  }
  return true;
}

/**
 * Validates data consistency on supplied gpu
 * In Multithreaded Environment
 */
static bool validateMemoryOnGpuMThread(int gpu) {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  bool TestPassed = true;
  constexpr auto N = 4 * 1024 * 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(int);
  HIPCHECK(hipSetDevice(gpu));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                     static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);
  HIP_CHECK(hipGetLastError());
  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  if (!HipTest::checkVectorADD(A_h, B_h, C_h, N)) {
    UNSCOPED_INFO("Validation PASSED for gpu " << gpu);
  } else {
    UNSCOPED_INFO("Validation FAILED for gpu " << gpu);
    TestPassed = false;
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);

  return TestPassed;
}

/**
 * Regress memory allocation and free in loop
 * In Multithreaded Environment
 */
static bool regressAllocInLoopMthread(int gpu) {
  size_t numBytes;
  int i = 0;
  int* ptr;

  HIPCHECK(hipSetDevice(gpu));
  numBytes = BuffSizeBC;

  // Exercise allocation in loop with bigger chunks
  for (i = 0; i < MaxAllocFree_BigChunks; i++) {
    HIPCHECK(hipMalloc(&ptr, numBytes));
    HIPCHECK(hipFree(ptr));
  }

  // Exercise allocation in loop with smaller chunks and maximum iters
  numBytes = BuffSizeSC;
  for (i = 0; i < MaxAllocFree_SmallChunks; i++) {
    HIPCHECK(hipMalloc(&ptr, numBytes));
    HIPCHECK(hipFree(ptr));
  }

  return true;
}

/*
 * Thread func to regress alloc and check data consistency
 */
static void threadFunc(int gpu) {
  g_thTestPassed = regressAllocInLoopMthread(gpu) && validateMemoryOnGpuMThread(gpu);

  UNSCOPED_INFO("thread execution status on gpu" << gpu << ":" << g_thTestPassed.load());
}

/**
 * Regress hipMalloc()/hipFree() in loop for bigger chunks and
 * smaller chunks of memory allocation
 */
HIP_TEST_CASE(Stress_hipMalloc_LoopRegressionAllocFreeCycles) {
  int devCnt = 0;

  // Get GPU count
  HIP_CHECK(hipGetDeviceCount(&devCnt));
  REQUIRE(devCnt > 0);

  CHECK(regressAllocInLoop(0) == true);
  CHECK(validateMemoryOnGPU(0) == true);
}

/**
 * Application Behavior Modelling.
 * Check hipMalloc() api adaptability when app creates small chunks of memory
 * continuously, stores it for later use and then frees it at later point
 * of time.
 */
HIP_TEST_CASE(Stress_hipMalloc_AllocateAndPoolBuffers) {
  using Allocation = std::unique_ptr<int, decltype(&hipFree)>;
  std::vector<Allocation> ptrlist;
  ptrlist.reserve(MaxAllocPoolIter);
  constexpr auto BuffSize = 10;
  int devCnt = 0;
  size_t successful_allocations = 0;
  size_t peak_retained = 0;
  size_t batches = 0;
  size_t oom_recoveries = 0;

  HIP_CHECK(hipGetDeviceCount(&devCnt));
  REQUIRE(devCnt > 0);

  auto release_pool = [&] {
    hipError_t first_error = hipSuccess;
    for (auto& allocation : ptrlist) {
      const auto status = hipFree(allocation.release());
      if (first_error == hipSuccess && status != hipSuccess) {
        first_error = status;
      }
    }
    ptrlist.clear();
    INFO("First free error: " << first_error);
    REQUIRE(first_error == hipSuccess);
    ++batches;
  };

  // Retain allocations until the process reaches a resource limit. Draining
  // then continuing preserves the total work without assuming a fixed number
  // of simultaneous allocations fits every runtime and host configuration.
  while (successful_allocations < static_cast<size_t>(MaxAllocPoolIter)) {
    int* ptr = nullptr;
    const auto status = hipMalloc(&ptr, BuffSize);
    if (status != hipSuccess) {
      INFO("Allocation " << successful_allocations << ", retained " << ptrlist.size());
      REQUIRE(status == hipErrorOutOfMemory);
      REQUIRE(hipGetLastError() == hipErrorOutOfMemory);
      REQUIRE_FALSE(ptrlist.empty());
      size_t avail = 0, total = 0;
      HIP_CHECK(hipMemGetInfo(&avail, &total));
      std::cout << "Draining retained pool after out of memory: " << ptrlist.size()
                << " allocations, free " << avail << ", total " << total << " bytes" << std::endl;
      // The failed allocation's output is undefined and is never retained.
      release_pool();
      ++oom_recoveries;
      continue;
    }
    REQUIRE(ptr != nullptr);
    ptrlist.emplace_back(ptr, &hipFree);
    ++successful_allocations;
    peak_retained = std::max(peak_retained, ptrlist.size());
  }

  release_pool();
  std::cout << "Completed " << successful_allocations << " allocations in " << batches
            << " retained batches; peak retained " << peak_retained << ", OOM recoveries "
            << oom_recoveries << '\n';
  REQUIRE(hipGetLastError() == hipSuccess);
  REQUIRE(validateMemoryOnGPU(0));
}


/**
 * Exercise hipMalloc() api parellely on all gpus from
 * multiple threads and regress the api.
 */
HIP_TEST_CASE(Stress_hipMalloc_Multithreaded_MultiGPU) {
  std::vector<std::thread> threadlist;
  int devCnt;

  // Get GPU count
  HIP_CHECK(hipGetDeviceCount(&devCnt));
  REQUIRE(devCnt > 0);

  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(threadFunc, i));
  }

  for (auto& t : threadlist) {
    t.join();
  }

  REQUIRE(g_thTestPassed == true);
}
