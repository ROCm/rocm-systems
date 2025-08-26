/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_defgroups.hh>
#include <utils.hh>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - memory IPC related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_IPC_Memory") {
  int N = 40;
  int Nbytes = N * sizeof(int);

  int fd[2];
  REQUIRE(pipe(fd) == 0);

  auto pid = fork();

  // Validating hipIpcGetMemHandle API
  if (pid != 0) {  // parent process
    void* hipIpcGetMemHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcGetMemHandle", &hipIpcGetMemHandle_ptr, currentHipVersion, 0,
                                nullptr));

    hipError_t (*dyn_hipIpcGetMemHandle_ptr)(hipIpcMemHandle_t*, void*) =
        reinterpret_cast<hipError_t (*)(hipIpcMemHandle_t*, void*)>(hipIpcGetMemHandle_ptr);

    int* srcHostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(srcHostMem != nullptr);
    fillHostArray(srcHostMem, N, 10);

    int* devMemSrc = nullptr;
    HIP_CHECK(hipMalloc(&devMemSrc, Nbytes));
    REQUIRE(devMemSrc != nullptr);
    HIP_CHECK(hipMemcpy(devMemSrc, srcHostMem, Nbytes, hipMemcpyHostToDevice));

    hipIpcMemHandle_t handle;
    HIP_CHECK(dyn_hipIpcGetMemHandle_ptr(&handle, devMemSrc));

    REQUIRE(write(fd[1], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipFree(devMemSrc));
    free(srcHostMem);
  } else {  // child process
    // Validating hipIpcOpenMemHandle, hipIpcCloseMemHandle API's
    void* hipIpcOpenMemHandle_ptr = nullptr;
    void* hipIpcCloseMemHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcOpenMemHandle", &hipIpcOpenMemHandle_ptr, currentHipVersion,
                                0, nullptr));
    HIP_CHECK(hipGetProcAddress("hipIpcCloseMemHandle", &hipIpcCloseMemHandle_ptr,
                                currentHipVersion, 0, nullptr));

    hipError_t (*dyn_hipIpcOpenMemHandle_ptr)(void**, hipIpcMemHandle_t, unsigned int) =
        reinterpret_cast<hipError_t (*)(void**, hipIpcMemHandle_t, unsigned int)>(
            hipIpcOpenMemHandle_ptr);
    hipError_t (*dyn_hipIpcCloseMemHandle_ptr)(void*) =
        reinterpret_cast<hipError_t (*)(void*)>(hipIpcCloseMemHandle_ptr);

    hipIpcMemHandle_t handle;
    REQUIRE(read(fd[0], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[0]) == 0);

    int* devPtr = nullptr;
    HIP_CHECK(dyn_hipIpcOpenMemHandle_ptr(reinterpret_cast<void**>(&devPtr), handle,
                                          hipIpcMemLazyEnablePeerAccess));
    REQUIRE(devPtr != nullptr);

    addOneKernel<<<1, 1>>>(devPtr, N);

    int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(dstHostMem != nullptr);

    HIP_CHECK(hipMemcpy(dstHostMem, devPtr, Nbytes, hipMemcpyDeviceToHost));
    REQUIRE(validateHostArray(dstHostMem, N, 11) == true);

    HIP_CHECK(dyn_hipIpcCloseMemHandle_ptr(devPtr));

    free(dstHostMem);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Event IPC related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_IPC_Event") {
  int fd[2];
  REQUIRE(pipe(fd) == 0);

  auto pid = fork();

  // Validating hipIpcGetEventHandle API
  if (pid != 0) {  // parent process
    void* hipIpcGetEventHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcGetEventHandle", &hipIpcGetEventHandle_ptr,
                                currentHipVersion, 0, nullptr));

    hipError_t (*dyn_hipIpcGetEventHandle_ptr)(hipIpcEventHandle_t*, hipEvent_t) =
        reinterpret_cast<hipError_t (*)(hipIpcEventHandle_t*, hipEvent_t)>(
            hipIpcGetEventHandle_ptr);

    hipEvent_t start = nullptr;
    HIP_CHECK(hipEventCreateWithFlags(&start, hipEventInterprocess | hipEventDisableTiming));
    REQUIRE(start != nullptr);

    hipIpcEventHandle_t handle;
    HIP_CHECK(dyn_hipIpcGetEventHandle_ptr(&handle, start));

    REQUIRE(write(fd[1], &handle, sizeof(hipIpcEventHandle_t)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipEventDestroy(start));
  } else {  // child process
    // Validating hipIpcOpenMemHandle API
    void* hipIpcOpenEventHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcOpenEventHandle", &hipIpcOpenEventHandle_ptr,
                                currentHipVersion, 0, nullptr));

    hipError_t (*dyn_hipIpcOpenEventHandle_ptr)(hipEvent_t*, hipIpcEventHandle_t) =
        reinterpret_cast<hipError_t (*)(hipEvent_t*, hipIpcEventHandle_t)>(
            hipIpcOpenEventHandle_ptr);

    hipIpcEventHandle_t handle;
    REQUIRE(read(fd[0], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[0]) == 0);

    hipEvent_t start = nullptr;
    HIP_CHECK(dyn_hipIpcOpenEventHandle_ptr(&start, handle));
    REQUIRE(start != nullptr);

    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&stop));
    REQUIRE(stop != nullptr);

    int N = 40;
    int Nbytes = N * sizeof(int);

    int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    fillHostArray(hostMem, N, 10);

    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipEventRecord(start, stream));

    HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes, hipMemcpyHostToDevice, stream));
    addOneKernel<<<1, 1>>>(devMem, N);
    HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    REQUIRE(validateHostArray(hostMem, N, 11) == true);

    float time = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&time, start, stop));
    REQUIRE(time > 0.0f);

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream));
    free(hostMem);
    HIP_CHECK(hipFree(devMem));
  }
}
