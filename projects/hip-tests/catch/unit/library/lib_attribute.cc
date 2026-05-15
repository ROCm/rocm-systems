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


TEST_CASE("Unit_hip_library_load_co") {
  constexpr size_t size = 32;
  constexpr size_t size_overwrite = 8;
  std::vector<float> input1, input2;
  input1.resize(size);
  input2.resize(size);
  for (size_t i = 0; i < size; i++) {
    input1[i] = (i + 1) * 2;
    input2[i] = i;
  }

  float *d_in1, *d_in2, *d_out;
  HIP_CHECK(hipMalloc(&d_in1, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_in2, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));

  HIP_CHECK(hipMemset(d_out, 0, sizeof(float) * size));
  HIP_CHECK(hipMemcpy(d_in1, input1.data(), sizeof(float) * size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_in2, input2.data(), sizeof(float) * size, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "library_code_load.code";

  hipLibrary_t library;
  hipFunction_t function;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));
  HIP_CHECK(hipKernelGetFunction(&function, kernel));
  HIP_CHECK(hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES ,sizeof(float) * size_overwrite, kernel, 0));

  void* args[] = {&d_out, &d_in1, &d_in2};

  HIP_CHECK(hipLaunchKernel(function, 1, size, args, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipLibraryUnload(library));

  std::vector<float> out(size, 0);
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < size; i++) {
    float tmp = input1[i] + input2[i];
    INFO("Index: " << i << " cpu res: " << tmp << " gpu res: " << out[i]);
    REQUIRE(out[i] == tmp);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_in1));
  HIP_CHECK(hipFree(d_in2));
  HIP_CHECK(hipFree(d_out));
}

TEST_CASE("Unit_hipKernelSetAttribute_Negative_Parameters") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "library_code_load.code";

  hipLibrary_t library;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  int device_id = 0;

  SECTION("kernel == nullptr") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0, nullptr,
                              device_id),
        hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(static_cast<hipFunction_attribute>(-1), 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute MAX_THREADS_PER_BLOCK") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute NUM_REGS") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_NUM_REGS, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute SHARED_SIZE_BYTES") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute BINARY_VERSION") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_BINARY_VERSION, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("MAX_DYNAMIC_SHARED_SIZE_BYTES negative value") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, -1, kernel,
                              device_id),
        hipErrorInvalidValue);
  }

  SECTION("invalid device id") {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0, kernel,
                              device_count),
        hipErrorInvalidDevice);
  }

  HIP_CHECK(hipLibraryUnload(library));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipKernelGetFunction_Negative_Parameters") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "library_code_load.code";

  hipLibrary_t library;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  SECTION("pFunc == nullptr") {
    HIP_CHECK_ERROR(hipKernelGetFunction(nullptr, kernel), hipErrorInvalidValue);
  }

  SECTION("kernel == nullptr") {
    hipFunction_t function;
    HIP_CHECK_ERROR(hipKernelGetFunction(&function, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipLibraryUnload(library));
  HIP_CHECK(hipStreamDestroy(stream));
}
