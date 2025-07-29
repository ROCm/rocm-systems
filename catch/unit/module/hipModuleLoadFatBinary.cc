/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hip_module_common.hh"

TEST_CASE("Unit_hipModuleLoadFatBinary_NegativeTsts") {
  hipModule_t Module;
  SECTION("fatCubin as nullptr") {
    HIP_CHECK_ERROR(hipModuleLoadFatBinary(&Module, nullptr), hipErrorInvalidValue);

  }
  SECTION("Fatbin as Empty String") {
    HIP_CHECK_ERROR(hipModuleLoadFatBinary(&Module, ""), hipErrorInvalidImage);
  }
  SECTION("Load compiled module from file") {
    const auto loaded_module = LoadModuleIntoBuffer("emptyModuleCount.code");
    HIP_CHECK(hipModuleLoadFatBinary(&Module, loaded_module.data()));
    REQUIRE(Module!= nullptr);
    HIP_CHECK(hipModuleUnload(Module));
  }
}
#if HT_AMD
TEST_CASE("Unit_hipModuleLoadFatBinary_PosiiveTsts") {
  /*if (!isGenericTargetSupported()) {
    fprintf(stderr, "Generic target test is skipped\n");
    return;
  }*/
  hipModule_t Module;
  //const auto loaded_module = LoadModuleIntoBuffer("copyKernelGenericTarget.code");
  const auto loaded_module = LoadModuleIntoBuffer("copyKernelCompressed.code");
  HIP_CHECK(hipModuleLoadFatBinary(&Module, loaded_module.data()));
  REQUIRE(Module != nullptr);
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, Module, "copy_ker"));
  REQUIRE(kernel != nullptr);
  constexpr int LEN = 64;
  constexpr int SIZE = LEN << 2;
  float *A, *B, *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = i * 1.0f;
    B[i] = 0.0f;
  }

  HIP_CHECK(hipMalloc(&Ad, SIZE));
  HIP_CHECK(hipMalloc(&Bd, SIZE));

  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
    size_t size;
  } args;
  args._Ad = reinterpret_cast<void*>(Ad);
  args._Bd = reinterpret_cast<void*>(Bd);
  args.size = SIZE;
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, LEN, 1, 1, 0,
               stream, NULL, reinterpret_cast<void**>(&config)));

  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));

  for (uint32_t i = 0; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  delete [] A;
  delete [] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipModuleUnload(Module));
}
#endif
