/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Import side of the Unit_hipGetProcAddress_IPC_Event tests. Opens the
// interprocess event the test process exported through a pointer resolved by
// hipGetProcAddress, then uses it as the completion sync point of a real GPU
// workload to confirm it is a usable event object.

// This runs as a freshly exec'd process, so that the HIP runtime is initialized
// cleanly in this process. A child forked from the test process would inherit a
// KFD connection that libhsakmt refuses to use, and every GPU call here would
// fail.

// Usage: hipGetProcAddressIpcEventImport <handle-hex> [device-id]
// Returns 0 on success, 1 on identifying the failing step otherwise.

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "hipIpcHandleHex.hh"

// hip_test_common.hh's HIP_CHECK is built on Catch2, which this executable does
// not link, so it defines its own.
#define HIP_CHECK(error)                                                                           \
  do {                                                                                             \
    hipError_t local_error = (error);                                                              \
    if (local_error != hipSuccess) {                                                               \
      std::cout << "child: " << #error << " failed with " << hipGetErrorString(local_error)        \
                << " (" << static_cast<int>(local_error) << ")" << std::endl;                      \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

constexpr int N = 40;
constexpr int Nbytes = N * sizeof(int);

static __global__ void addOneKernel(int* arr, int arrSize) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = offset; i < arrSize; i += stride) {
    arr[i] += 1;
  }
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cout << "child: usage: " << argv[0] << " <handle-hex> [device-id]" << std::endl;
    return 1;
  }

  const std::string handleHex(argv[1]);
  if (handleHex.size() != HIP_IPC_HANDLE_SIZE * 2) {
    std::cout << "child: malformed IPC handle argument" << std::endl;
    return 1;
  }
  const hipIpcEventHandle_t handle = hexToIpcHandle<hipIpcEventHandle_t>(handleHex);

  if (argc == 3) {
    HIP_CHECK(hipSetDevice(std::atoi(argv[2])));
  }

  int hipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&hipVersion));

  void* openAddress = nullptr;
  HIP_CHECK(hipGetProcAddress("hipIpcOpenEventHandle", &openAddress, hipVersion, 0, nullptr));
  if (openAddress == nullptr) {
    std::cout << "child: hipGetProcAddress resolved a null pointer" << std::endl;
    return 1;
  }

  auto dynIpcOpenEventHandle =
      reinterpret_cast<hipError_t (*)(hipEvent_t*, hipIpcEventHandle_t)>(openAddress);

  // The call under test: import the handle exported by the test process through
  // the proc-address-resolved pointer.
  hipEvent_t event = nullptr;
  HIP_CHECK(dynIpcOpenEventHandle(&event, handle));
  if (event == nullptr) {
    std::cout << "child: hipIpcOpenEventHandle returned a null pointer" << std::endl;
    return 1;
  }

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<int> hostMem(N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem.data(), Nbytes, hipMemcpyHostToDevice, stream));
  addOneKernel<<<1, 1, 0, stream>>>(devMem, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem, Nbytes, hipMemcpyDeviceToHost, stream));

  // The wait covers the whole pipeline above, so the results are ready to
  // validate and the buffer is safe to read once it returns.
  HIP_CHECK(hipEventRecord(event, stream));
  HIP_CHECK(hipEventSynchronize(event));

  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 11) {
      std::cout << "child: expected 11 at index " << i << ", got " << hostMem[i] << std::endl;
      return 1;
    }
  }

  HIP_CHECK(hipFree(devMem));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(event));

  return 0;
}
