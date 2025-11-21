/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// Simple test for Fine Grained GPU-GPU coherency.

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

// Helper function to spin on address until address equals value.
// If the address holds the value of -1, abort because the other thread failed.
__device__ int gpu_spin_loop_or_abort_on_negative_one(unsigned int* address, unsigned int value) {
  unsigned int compare;
  bool check = false;
  do {
    compare = value;
    check =
        __hip_atomic_compare_exchange_strong(address, /*expected=*/&compare,
                                             /*desired=*/value, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                             /*scope=*/__HIP_MEMORY_SCOPE_SYSTEM);
    if (compare == -1) return -1;
  } while (!check);
  return 0;
}

// This kernel requires a single block, single thread dispatch.
__global__ void gpu_cache0(int* A, int* B, int* X, int* Y, size_t N, unsigned int* AA1,
                           unsigned int* AA2, unsigned int* BA1, unsigned int* BA2,
                           unsigned int* cache0_result) {
  for (size_t i = 0; i < N; i++) {
    // Store data into A, system fence, and atomically mark flag.
    // This guarantees this global write is visible by device 1.
    A[i] = X[i];
    __hip_atomic_fetch_add(AA1, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    // Wait on device 1's global write to B.
    if (gpu_spin_loop_or_abort_on_negative_one(BA1, i + 1) == -1) {
      *cache0_result = -1;
      break;
    }

    // Check device 1 properly stored Y into B.
    bool stored_data_matches = (B[i] == Y[i]);
    if (!stored_data_matches) {
      // If the data does not match, alert other thread and abort.
      printf("FAIL: at i=%zu, B[i]=%d, which does not match Y[i]=%d.\n", i, B[i], Y[i]);
      __hip_atomic_exchange(AA2, -1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      *cache0_result = -1;
    }
    // Otherwise tell the other thread to continue.
    __hip_atomic_fetch_add(AA2, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    // Wait on kernel gpu_cache1 to finish checking X is stored in A.
    if (gpu_spin_loop_or_abort_on_negative_one(BA2, i + 1) == -1) {
      *cache0_result = -1;
      break;
    }
  }
  *cache0_result = 0;
}

// This kernel requires a single block, single thread dispatch.
__global__ void gpu_cache1(int* A, int* B, int* X, int* Y, size_t N, unsigned int* AA1,
                           unsigned int* AA2, unsigned int* BA1, unsigned int* BA2,
                           unsigned int* cache1_result) {
  for (size_t i = 0; i < N; i++) {
    B[i] = Y[i];
    __hip_atomic_fetch_add(BA1, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    if (gpu_spin_loop_or_abort_on_negative_one(AA1, i + 1) == -1) {
      *cache1_result = -1;
      break;
    }

    bool stored_data_matches = (A[i] == X[i]);
    if (!stored_data_matches) {
      printf("FAIL: at i=%zu, A[i]=%d, which does not match X[i]=%d.\n", i, A[i], X[i]);
      __hip_atomic_exchange(BA2, -1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      *cache1_result = -1;
    }
    __hip_atomic_fetch_add(BA2, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    if (gpu_spin_loop_or_abort_on_negative_one(AA2, i + 1) == -1) {
      *cache1_result = -1;
      break;
    }
  }
  *cache1_result = 0;
}

static bool gpu_to_gpu_coherency() {
  int *A_d, *B_d, *X_d0, *X_d1, *Y_d0, *Y_d1;
  int *A_h, *B_h, *X_h, *Y_h;
  unsigned int* cache0_result = nullptr;
  unsigned int* cache1_result = nullptr;
  size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int numTestDevices = 2;
  int deviceFineGrain = 0;

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < numTestDevices) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return true;
  }
  SECTION("With device fine grained buffer") {
    HIP_CHECK(hipDeviceGetAttribute(&deviceFineGrain, hipDeviceAttributeFineGrainSupport, 0));
    if (deviceFineGrain == 0) {
      HipTest::HIP_SKIP_TEST("The test skipped due to deviceFineGrain = 0 on device 0");
      return true;
    }
    HIP_CHECK(hipDeviceGetAttribute(&deviceFineGrain, hipDeviceAttributeFineGrainSupport, 1));
    if (deviceFineGrain == 0) {
      HipTest::HIP_SKIP_TEST("The test skipped due to deviceFineGrain = 0 on device 1");
      return true;
    }
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    fprintf(stderr, "info: allocate device mem (%zu bytes) on device 0\n", Nbytes);
    HIP_CHECK(
        hipExtMallocWithFlags(reinterpret_cast<void**>(&A_d), Nbytes, hipDeviceMallocFinegrained));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
    fprintf(stderr, "info: allocate device mem (%zu bytes) on device 1\n", Nbytes);
    HIP_CHECK(
        hipExtMallocWithFlags(reinterpret_cast<void**>(&B_d), Nbytes, hipDeviceMallocFinegrained));
  }
  SECTION("With host(SVM) fine grained buffer") {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipHostMalloc(&A_d, Nbytes));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipHostMalloc(&B_d, Nbytes));
  }
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipHostMalloc(&cache0_result, sizeof(unsigned int)));
  HIP_CHECK(hipHostMalloc(&cache1_result, sizeof(unsigned int)));
  *cache0_result = 0;
  *cache1_result = 0;
  // Allocate Host Side Memory.
  fprintf(stderr, "info: allocate host mem (%zu bytes)\n", Nbytes);
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  B_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(B_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  X_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(X_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  Y_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(Y_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Initialize the arrays and atomic variables.
  for (size_t i = 0; i < N; i++) {
    X_h[i] = 100000000 + i;
    Y_h[i] = 300000000 + i;
  }

  // Initialize shared atomic flags on host coherent memory.
  unsigned int *AA1_h, *AA2_h, *BA1_h, *BA2_h;
  unsigned int *AA1_d, *AA2_d, *BA1_d, *BA2_d;
  HIP_CHECK(hipHostMalloc(&AA1_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&AA1_d), AA1_h, 0));
  *AA1_h = 0;
  HIP_CHECK(hipHostMalloc(&AA2_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&AA2_d), AA2_h, 0));
  *AA2_h = 0;
  HIP_CHECK(hipHostMalloc(&BA1_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&BA1_d), BA1_h, 0));
  *BA1_h = 0;
  HIP_CHECK(hipHostMalloc(&BA2_h, sizeof(unsigned int), hipHostMallocCoherent));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&BA2_d), BA2_h, 0));
  *BA2_h = 0;

  // Skip the first stream.
  hipStream_t stream[3];
  HIP_CHECK(hipStreamCreate(&stream[0]));

  // Set-up Device 0.
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreateWithFlags(&stream[1], hipStreamNonBlocking));

  HIP_CHECK(hipMalloc(&X_d0, Nbytes));
  HIP_CHECK(hipMalloc(&Y_d0, Nbytes));

  // Set-up Device 1.
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipStreamCreateWithFlags(&stream[2], hipStreamNonBlocking));

  HIP_CHECK(hipMalloc(&X_d1, Nbytes));
  HIP_CHECK(hipMalloc(&Y_d1, Nbytes));

  // Transfer initialized data onto the device arrays.
  HIP_CHECK(hipMemcpy(X_d0, X_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(X_d1, X_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Y_d0, Y_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Y_d1, Y_h, Nbytes, hipMemcpyHostToDevice));

  // Prepare and launch the device kernels.
  const unsigned blocks = 1;
  const unsigned threadsPerBlock = 1;
  HIP_CHECK(hipSetDevice(0));
  hipLaunchKernelGGL(gpu_cache0, dim3(blocks), dim3(threadsPerBlock), 0, stream[1], A_d, B_d, X_d0,
                     Y_d0, N, AA1_d, AA2_d, BA1_d, BA2_d, cache0_result);
  // Check if launch failed.
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipSetDevice(1));
  hipLaunchKernelGGL(gpu_cache1, dim3(blocks), dim3(threadsPerBlock), 0, stream[2], A_d, B_d, X_d1,
                     Y_d1, N, AA1_d, AA2_d, BA1_d, BA2_d, cache1_result);
  HIP_CHECK(hipGetLastError());

  // Wait for kernels on both devices.
  HIP_CHECK(hipStreamSynchronize(stream[1]));
  HIP_CHECK(hipStreamSynchronize(stream[2]));
  REQUIRE(*cache0_result == 0);
  REQUIRE(*cache1_result == 0);

  // Evaluate the resultant arrays A and B.
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(B_h, B_d, Nbytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    REQUIRE(A_h[i] == (100000000 + i));
    REQUIRE(B_h[i] == (300000000 + i));
  }

  // Free all the device and host memory allocated.
  if (deviceFineGrain) {
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipFree(B_d));
  } else {
    HIP_CHECK(hipHostFree(A_d));
    HIP_CHECK(hipHostFree(B_d));
  }
  HIP_CHECK(hipFree(X_d0));
  HIP_CHECK(hipFree(Y_d0));
  HIP_CHECK(hipFree(X_d1));
  HIP_CHECK(hipFree(Y_d1));
  HIP_CHECK(hipHostFree(AA1_h));
  HIP_CHECK(hipHostFree(AA2_h));
  HIP_CHECK(hipHostFree(BA1_h));
  HIP_CHECK(hipHostFree(BA2_h));
  HIP_CHECK(hipHostFree(cache0_result));
  HIP_CHECK(hipHostFree(cache1_result));

  free(A_h);
  free(B_h);
  free(X_h);
  free(Y_h);
  for (int i = 0; i < 3; i++) {
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }
  return true;
}

/**
 * Test Description
 * ------------------------
 *    - This test runs on devices where XGMI enables fine-grained communication
 * between GPUs. This performs a message passing test.
 * Array A is allocated on Device 0, and remotely on Device 1.
 * Device 0 also increments atomic ints AA1 and AA2.
 * Array B is allocated on Device 1, and remotely on Device 0.
 * Device 1 also increments atomic ints BA1 and BA2.
 * Kernel 0 will launch on Device 0, and store array X into array A.
 * Kernel 1 will launch on Device 1, and store array Y into array B.
 * Kernel 0 will validate that the correct values of array Y are stored in B.
 * Kernel 1 will validate that the correct values of array X are stored in A.

 * Test source
 * ------------------------
 *    - catch/unit/synchronization/cache_coherency_gpu_gpu.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 *    - Test to be run only on AMD.
 */

TEST_CASE("Unit_cache_coherency_gpu_gpu") {
  bool passed = true;
  // Coherency between GPUs accessing local or remote FB.
  REQUIRE(passed == gpu_to_gpu_coherency());
}
