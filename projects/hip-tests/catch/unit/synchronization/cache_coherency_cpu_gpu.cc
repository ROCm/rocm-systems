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
// Simple test for Fine Grained CPU-GPU coherency.

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
__global__ void gpu_kernel(int* A, int* B, int* X, int* Y, size_t N, unsigned int* AA1,
                           unsigned int* AA2, unsigned int* BA1, unsigned int* BA2,
                           unsigned int* dresult) {
  for (size_t i = 0; i < N; i++) {
    // Store data into A, system fence, and atomically mark flag.
    // This guarantees this global write is visible by device 1.
    A[i] = X[i];
    __hip_atomic_fetch_add(AA1, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    // Wait on device 1's global write to B.
    if (gpu_spin_loop_or_abort_on_negative_one(BA1, i + 1) == -1) {
      *dresult = -1;
      break;
    }

    // Check device 1 properly stored Y into B.
    bool stored_data_matches = (B[i] == Y[i]);
    if (!stored_data_matches) {
      // If the data does not match, alert other thread and abort.
      printf("FAIL: at i=%zu, B[i]=%d, which does not match Y[i]=%d.\n", i, B[i], Y[i]);
      __hip_atomic_exchange(AA2, -1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      *dresult = -1;
    }
    // Otherwise tell the other thread to continue.
    __hip_atomic_fetch_add(AA2, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    // Wait on kernel gpu_cache1 to finish checking X is stored in A.
    if (gpu_spin_loop_or_abort_on_negative_one(BA2, i + 1) == -1) {
      *dresult = -1;
      break;
    }
  }
  *dresult = 0;
}

__host__ int cpu_spin_loop_or_abort_on_negative_one(unsigned int* address, unsigned int value) {
  unsigned int compare;
  bool check = false;
  do {
    compare = value;
    check = __atomic_compare_exchange_n(address, /*expected=*/&compare, /*desired=*/value,
                                        /*weak=*/false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);
    if (compare == -1) return -1;
  } while (!check);
  return 0;
}

// This host thread runs only on a single CPU thread.
__host__ void cpu_thread(int* A, int* B, int* X, int* Y, size_t N, unsigned int* AA1,
                         unsigned int* AA2, unsigned int* BA1, unsigned int* BA2,
                         unsigned int* hresult) {
  for (size_t i = 0; i < N; i++) {
    B[i] = Y[i];
    __atomic_fetch_add(BA1, 1, __ATOMIC_RELEASE);
    if (cpu_spin_loop_or_abort_on_negative_one(AA1, i + 1) == -1) {
      *hresult = -1;
      break;
    }

    bool stored_data_matches = (A[i] == X[i]);
    if (!stored_data_matches) {
      printf("FAIL: at i=%zu, A[i]=%d, which does not match X[i]=%d.\n", i, A[i], X[i]);
      __atomic_exchange_n(BA2, -1, __ATOMIC_RELEASE);
      *hresult = -1;
      break;
    }
    __atomic_fetch_add(BA2, 1, __ATOMIC_RELEASE);
    if (cpu_spin_loop_or_abort_on_negative_one(AA2, i + 1) == -1) {
      *hresult = -1;
      break;
    }
  }
  *hresult = 0;
}

static bool cpu_to_gpu_coherency() {
  int *A_d, *B_d, *X_d, *Y_d;
  int *A_res, *A_h, *B_h, *X_h, *Y_h;
  unsigned int hresult = 0;
  unsigned int* dresult = nullptr;
  size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  int numDevices = 0;
  int deviceFineGrain = 0;

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 1) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 1");
    return true;
  }

  SECTION("With device fine grained buffer") {
    HIP_CHECK(hipDeviceGetAttribute(&deviceFineGrain, hipDeviceAttributeFineGrainSupport, 0));
    if (deviceFineGrain == 0) {
      HipTest::HIP_SKIP_TEST("The test skipped due to deviceFineGrain = 0");
      return true;
    }
    fprintf(stderr, "info: allocate device mem (%zu bytes) on device 0\n", Nbytes);
    HIP_CHECK(
        hipExtMallocWithFlags(reinterpret_cast<void**>(&A_d), Nbytes, hipDeviceMallocFinegrained));
  }
  SECTION("With host(SVM) fine grained buffer") { HIP_CHECK(hipHostMalloc(&A_d, Nbytes)); }
  A_h = A_d;

  HIP_CHECK(hipHostMalloc(&dresult, sizeof(unsigned int)));
  *dresult = 0;
  // Allocate Host Side Memory. Coherent Fine-grained Memory for array B.
  fprintf(stderr, "info: allocate host mem (%zu bytes)\n", Nbytes);
  HIP_CHECK(hipHostMalloc(&B_h, Nbytes, (hipHostMallocCoherent | hipHostMallocMapped)));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&B_d), B_h, 0));
  X_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(X_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  Y_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(Y_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Initialize the arrays and atomic variables.
  for (size_t i = 0; i < N; i++) {
    X_h[i] = 100000000 + i;
    Y_h[i] = 300000000 + i;
  }

  // Initialize shared atomic flags between CPU and GPU.
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

  // Skip the first stream, ensure stream is non-blocking.
  hipStream_t stream;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  HIP_CHECK(hipMalloc(&X_d, Nbytes));
  HIP_CHECK(hipMalloc(&Y_d, Nbytes));

  HIP_CHECK(hipMemcpy(X_d, X_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Y_d, Y_h, Nbytes, hipMemcpyHostToDevice));

  // Launch the GPU kernel.
  const unsigned blocks = 1;
  const unsigned threadsPerBlock = 1;
  hipLaunchKernelGGL(gpu_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d, B_d, X_d, Y_d,
                     N, AA1_d, AA2_d, BA1_d, BA2_d, dresult);
  // Check if launch failed.
  HIP_CHECK(hipGetLastError());

  // Do not sync the launched stream, instead run the cpu_thread.
  std::thread host_thread(cpu_thread, A_h, B_h, X_h, Y_h, N, AA1_h, AA2_h, BA1_h, BA2_h, &hresult);
  // Wait for Device side to finish.
  HIP_CHECK(hipStreamSynchronize(stream));
  host_thread.join();
  REQUIRE(*dresult == 0);
  REQUIRE(hresult == 0);

  // Evaluate the resultant arrays A and B.
  A_res = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_CHECK(A_res == 0 ? hipErrorOutOfMemory : hipSuccess);
  HIP_CHECK(hipMemcpy(A_res, A_d, Nbytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    REQUIRE(A_res[i] == (100000000 + i));
    REQUIRE(B_h[i] == (300000000 + i));
  }

  // Free all the device and host memory allocated.
  if (deviceFineGrain) {
    HIP_CHECK(hipFree(A_d));
  } else {
    HIP_CHECK(hipHostFree(A_d));
  }
  HIP_CHECK(hipFree(X_d));
  HIP_CHECK(hipFree(Y_d));
  HIP_CHECK(hipHostFree(AA1_h));
  HIP_CHECK(hipHostFree(AA2_h));
  HIP_CHECK(hipHostFree(BA1_h));
  HIP_CHECK(hipHostFree(BA2_h));
  HIP_CHECK(hipHostFree(B_h));
  HIP_CHECK(hipHostFree(dresult));
  free(X_h);
  free(Y_h);
  free(A_res);
  HIP_CHECK(hipStreamDestroy(stream));
  return true;
}

/**
 * Test Description
 * ------------------------
 *    - This test runs on devices where XGMI enables fine-grained communication
 * between GPUs. This performs a message passing test.
 * Array A is allocated on Device 0, and remotely on host.
 * Device 0 also increments atomic ints AA1 and AA2.
 * Array B is allocated on host, and remotely on Device 0.
 * Host also increments atomic ints BA1 and BA2.
 * Kernel will launch on Device 0, and store array X into array A.
 * Host Thread will store array Y into array B.
 * Kernel will validate that the correct values of array Y are stored in B.
 * Host Thread will validate that the correct values of array X are stored in A.

 * Test source
 * ------------------------
 *    - catch/unit/synchronization/cache_coherency_cpu_gpu.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 *    - Test to be run only on AMD.
 */

TEST_CASE("Unit_cache_coherency_cpu_gpu") {
  bool passed = true;
  // Coherency between CPU and GPU sharing host and device memory.
  REQUIRE(passed == cpu_to_gpu_coherency());
}
