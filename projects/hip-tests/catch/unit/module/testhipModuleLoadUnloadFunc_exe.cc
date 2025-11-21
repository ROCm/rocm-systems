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
#include <hip/hip_runtime.h>
#include <fstream>
#include <cstddef>
#include <vector>
#include <iostream>
#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError), localError,      \
             #error, __FUNCTION__, __LINE__);                                                      \
      exit(0);                                                                                     \
    }                                                                                              \
  }
constexpr auto CODEOBJ_FILE = "kernel_composite_test.code";

bool testhipModuleLoadUnloadFunc(const std::vector<char>& buffer, char* globTestID) {
  constexpr auto CODEOBJ_GLOB_KERNEL1 = "testWeightedCopy";
  size_t N = 16 * 16;
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d;
  int *A_h, *B_h;
  int deviceid;
  HIP_CHECK(hipGetDevice(&deviceid));
  // allocate host and device buffer
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));

  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  B_h = reinterpret_cast<int*>(malloc(Nbytes));
  // set host buffers
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = deviceid;
  }
  // Copy buffer from host to device
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  hipModule_t Module;
  hipFunction_t Function;
  int check = atoi(globTestID);
  /**
   * Validates hipModuleLoadUnload if globTestID = 1
   * Validates hipModuleLoadDataUnload if globTestID = 2
   * Validates hipModuleLoadDataExUnload if globTestID = 3
   */
  switch (check) {
    case 1:
      HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    case 2:
      HIP_CHECK(hipModuleLoadData(&Module, &buffer[0]));
    case 3:
      HIP_CHECK(hipModuleLoadDataEx(&Module, &buffer[0], 0, nullptr, nullptr));
  }
  HIP_CHECK(hipModuleGetFunction(&Function, Module, CODEOBJ_GLOB_KERNEL1));
  float deviceGlobalFloatH = 3.14;
  int deviceGlobalInt1H = 100 * deviceid;
  int deviceGlobalInt2H = 50 * deviceid;
  uint32_t deviceGlobalShortH = 25 * deviceid;
  char deviceGlobalCharH = 13 * deviceid;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "deviceGlobalFloat"));
  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &deviceGlobalFloatH, deviceGlobalSize));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "deviceGlobalInt1"));
  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &deviceGlobalInt1H, deviceGlobalSize));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "deviceGlobalInt2"));
  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &deviceGlobalInt2H, deviceGlobalSize));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "deviceGlobalShort"));
  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &deviceGlobalShortH, deviceGlobalSize));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "deviceGlobalChar"));
  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &deviceGlobalCharH, deviceGlobalSize));
  // Launch Function kernel function

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = reinterpret_cast<void*>(A_d);
  args._Bd = reinterpret_cast<void*>(B_d);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, N, 1, 1, 0, stream, NULL,
                                  reinterpret_cast<void**>(&config)));
  // Copy buffer from decice to host
  HIP_CHECK(hipMemcpyAsync(B_h, B_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));

  // Check the results
  for (size_t idx = 0; idx < N; idx++) {
    if (B_h[idx] != (deviceGlobalInt1H * A_h[idx] + deviceGlobalInt2H +
                     static_cast<int>(deviceGlobalShortH) + +static_cast<int>(deviceGlobalCharH) +
                     static_cast<int>(deviceGlobalFloatH * deviceGlobalFloatH))) {
      // exit the current process with failure
      return false;
    }
  }
  HIP_CHECK(hipModuleUnload(Module));
  // free memory
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  free(B_h);
  free(A_h);

  return true;
}
int main(int argc, char* argv[]) {
  if (argc > 0) {
    bool value = false;
    std::ifstream file(CODEOBJ_FILE, std::ios::binary | std::ios::ate);
    std::streamsize fsize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(fsize);
    if (!file.read(buffer.data(), fsize)) {
      value = false;
    }
    file.close();
    value = testhipModuleLoadUnloadFunc(buffer, argv[1]);
    return value;
  }
}
