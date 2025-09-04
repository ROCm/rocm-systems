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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>

#include <stdio.h>
#include <dlfcn.h>
#include <vector>
#include <iostream>
#include <fstream>

#define fileName "bit_extract_kernel.code"

#define LEN 64
#define SIZE LEN * sizeof(float)

/**
* @addtogroup dyn_hipModuleLoad dyn_hipModuleGetFunction dyn_hipModuleLaunchKernel
* @{
* @ingroup DynamicLoading
* ` hipError_t (*dyn_hipModuleLoad)(hipModule_t*, const char*) = reinterpret_cast
             <hipError_t (*)(hipModule_t*, const char*)>(sym_hipModuleLoad)` -
*  Loads code object from file into a module the currrent context
* `hipError_t (*dyn_hipModuleGetFunction)(hipFunction_t*, hipModule_t,
             const char*) = reinterpret_cast < hipError_t (*)(hipFunction_t*,
             hipModule_t, const char*)>(sym_hipModuleGetFunction)` -
* Function with kernelname will be extracted if present in module
* `hipError_t (*dyn_hipModuleLaunchKernel)(hipFunction_t, unsigned int,
             unsigned int, unsigned int, unsigned int, unsigned int,
             unsigned int, unsigned int, hipStream_t, void**, void**)
             = reinterpret_cast<hipError_t(*) (hipFunction_t,
             unsigned int, unsigned int, unsigned int, unsigned int,
             unsigned int, unsigned int, unsigned int, hipStream_t,
             void**, void**)>(sym_hipModuleLaunchKernel)` -
* launches Kernel with launch parameters and shared memory on stream with arguments passed
*/

/**
 * Test Description
 * ------------------------
 * - Test is to load hip runtime using dlopen and get function pointer using dlsym for hip apis.

 * Test source
 * ------------------------
 * - catch/unit/dynamicLoading/hipApiDynamicLoad.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

void test_dynamicLoading(void* sym_hipGetDevice, void* sym_hipMalloc, void* sym_hipMemcpyHtoD,
                         void* sym_hipMemcpyDtoH, void* sym_hipModuleLoad,
                         void* sym_hipGetDeviceProperties, void* sym_hipModuleGetFunction,
                         void* sym_hipModuleLaunchKernel) {
  uint32_t *A_d, *C_d;
  uint32_t *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(uint32_t);
  hipError_t (*dyn_hipGetDevice)(hipDevice_t*, int) =
      reinterpret_cast<hipError_t (*)(hipDevice_t*, int)>(sym_hipGetDevice);

  hipError_t (*dyn_hipMalloc)(void**, uint32_t) =
      reinterpret_cast<hipError_t (*)(void**, uint32_t)>(sym_hipMalloc);

  hipError_t (*dyn_hipMemcpyHtoD)(hipDeviceptr_t, void*, size_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, void*, size_t)>(sym_hipMemcpyHtoD);

  hipError_t (*dyn_hipMemcpyDtoH)(void*, hipDeviceptr_t, size_t) =
      reinterpret_cast<hipError_t (*)(void*, hipDeviceptr_t, size_t)>(sym_hipMemcpyDtoH);

  hipError_t (*dyn_hipModuleLoad)(hipModule_t*, const char*) =
      reinterpret_cast<hipError_t (*)(hipModule_t*, const char*)>(sym_hipModuleLoad);

  hipError_t (*dyn_hipGetDeviceProperties)(hipDeviceProp_t*, int) =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_t*, int)>(sym_hipGetDeviceProperties);

  hipError_t (*dyn_hipModuleGetFunction)(hipFunction_t*, hipModule_t, const char*) =
      reinterpret_cast<hipError_t (*)(hipFunction_t*, hipModule_t, const char*)>(
          sym_hipModuleGetFunction);

  hipError_t (*dyn_hipModuleLaunchKernel)(hipFunction_t, unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, unsigned int, unsigned int,
                                          hipStream_t, void**, void**) =
      reinterpret_cast<hipError_t (*)(hipFunction_t, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int, unsigned int,
                                      hipStream_t, void**, void**)>(sym_hipModuleLaunchKernel);

  hipDevice_t device;
  HIPCHECK(dyn_hipGetDevice(&device, 0));

  hipDeviceProp_t props;
  HIPCHECK(dyn_hipGetDeviceProperties(&props, device));
  A_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  REQUIRE(A_h != NULL);
  C_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  REQUIRE(C_h != NULL);

  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
  }

  HIPCHECK(dyn_hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));
  HIPCHECK(dyn_hipMalloc(reinterpret_cast<void**>(&C_d), Nbytes));

  HIPCHECK(dyn_hipMemcpyHtoD((hipDeviceptr_t)(A_d), A_h, Nbytes));

  struct {
    void* _Cd;
    void* _Ad;
    size_t _N;
  } args;
  args._Cd = reinterpret_cast<void**>(C_d);
  args._Ad = reinterpret_cast<void**>(A_d);
  args._N = static_cast<size_t>(N);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  hipModule_t Module;
  HIPCHECK(dyn_hipModuleLoad(&Module, fileName));

  hipFunction_t Function;
  HIPCHECK(dyn_hipModuleGetFunction(&Function, Module, "bit_extract_kernel"));

  HIPCHECK(dyn_hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL,
                                     reinterpret_cast<void**>(&config)));

  HIPCHECK(dyn_hipMemcpyDtoH(C_h, (hipDeviceptr_t)(C_d), Nbytes));

  for (size_t i = 0; i < N; i++) {
    unsigned Agold = ((A_h[i] & 0xf00) >> 8);
    REQUIRE(C_h[i] == Agold);
  }
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
}
TEST_CASE("Unit_hipApiDynamicLoad_hipGetProcAddress") {
  void* sym_hipGetDevice;
  void* sym_hipMalloc;
  void* sym_hipMemcpyHtoD;
  void* sym_hipMemcpyDtoH;
  void* sym_hipModuleLoad;
  void* sym_hipGetDeviceProperties;
  void* sym_hipModuleGetFunction;
  void* sym_hipModuleLaunchKernel;

  int currentHipVersion = 0;
  HIPCHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIPCHECK(hipGetProcAddress("hipGetDevice", &sym_hipGetDevice, currentHipVersion, 0, nullptr));
  HIPCHECK(hipGetProcAddress("hipMalloc", &sym_hipMalloc, currentHipVersion, 0, nullptr));
  HIPCHECK(hipGetProcAddress("hipMemcpyHtoD", &sym_hipMemcpyHtoD, currentHipVersion, 0, nullptr));
  HIPCHECK(hipGetProcAddress("hipMemcpyDtoH", &sym_hipMemcpyDtoH, currentHipVersion, 0, nullptr));
  HIPCHECK(hipGetProcAddress("hipModuleLoad", &sym_hipModuleLoad, currentHipVersion, 0, nullptr));
  HIPCHECK(hipGetProcAddress("hipGetDeviceProperties", &sym_hipGetDeviceProperties,
                             currentHipVersion, 0, nullptr));
  HIPCHECK(hipGetProcAddress("hipModuleGetFunction", &sym_hipModuleGetFunction, currentHipVersion,
                             0, nullptr));
  HIPCHECK(hipGetProcAddress("hipModuleLaunchKernel", &sym_hipModuleLaunchKernel, currentHipVersion,
                             0, nullptr));

  test_dynamicLoading(sym_hipGetDevice, sym_hipMalloc, sym_hipMemcpyHtoD, sym_hipMemcpyDtoH,
                      sym_hipModuleLoad, sym_hipGetDeviceProperties, sym_hipModuleGetFunction,
                      sym_hipModuleLaunchKernel);
}


TEST_CASE("Unit_hipApiDynamicLoad") {
  void* handle = dlopen("libamdhip64.so", RTLD_LAZY);
  REQUIRE(handle != NULL);

  void* sym_hipGetDevice = dlsym(handle, "hipGetDevice");
  void* sym_hipMalloc = dlsym(handle, "hipMalloc");
  void* sym_hipMemcpyHtoD = dlsym(handle, "hipMemcpyHtoD");
  void* sym_hipMemcpyDtoH = dlsym(handle, "hipMemcpyDtoH");
  void* sym_hipModuleLoad = dlsym(handle, "hipModuleLoad");
  void* sym_hipGetDeviceProperties = dlsym(handle, "hipGetDeviceProperties");
  void* sym_hipModuleGetFunction = dlsym(handle, "hipModuleGetFunction");
  void* sym_hipModuleLaunchKernel = dlsym(handle, "hipModuleLaunchKernel");

  dlclose(handle);

  test_dynamicLoading(sym_hipGetDevice, sym_hipMalloc, sym_hipMemcpyHtoD, sym_hipMemcpyDtoH,
                      sym_hipModuleLoad, sym_hipGetDeviceProperties, sym_hipModuleGetFunction,
                      sym_hipModuleLaunchKernel);
}

/**
 * End doxygen group DynamicLoading.
 * @}
 */
