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
  std::vector<float> input1, input2;
  input1.reserve(size);
  input2.reserve(size);
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

  SECTION("One Kernel") {
    hipLibrary_t library;
    hipKernel_t function;

    HIP_CHECK(
        hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    HIP_CHECK(hipLibraryGetKernel(&function, library, "add_kernel"));

    unsigned int count = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&count, library));
    REQUIRE(count == 3);

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
  }

  SECTION("Two Kernel") {
    hipLibrary_t library;
    hipKernel_t function;

    HIP_CHECK(
        hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    HIP_CHECK(hipLibraryGetKernel(&function, library, "sub_kernel"));

    unsigned int count = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&count, library));
    REQUIRE(count == 3);

    void* args[] = {&d_out, &d_in1, &d_in2};

    HIP_CHECK(hipLaunchKernel(function, 1, size, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLibraryUnload(library));


    std::vector<float> out(size, 0);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float tmp = input1[i] - input2[i];
      INFO("Index: " << i << " cpu res: " << tmp << " gpu res: " << out[i]);
      REQUIRE(out[i] == tmp);
    }
  }

  SECTION("Three Kernel") {
    hipLibrary_t library;
    hipKernel_t function;

    HIP_CHECK(
        hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    HIP_CHECK(hipLibraryGetKernel(&function, library, "mul_kernel"));

    unsigned int count = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&count, library));
    REQUIRE(count == 3);

    void* args[] = {&d_out, &d_in1, &d_in2};

    HIP_CHECK(hipLaunchKernel(function, 1, size, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLibraryUnload(library));


    std::vector<float> out(size, 0);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float tmp = input1[i] * input2[i];
      INFO("Index: " << i << " cpu res: " << tmp << " gpu res: " << out[i]);
      REQUIRE(out[i] == tmp);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_in1));
  HIP_CHECK(hipFree(d_in2));
  HIP_CHECK(hipFree(d_out));
}