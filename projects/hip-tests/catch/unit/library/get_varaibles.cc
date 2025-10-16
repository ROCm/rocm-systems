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

TEST_CASE("Unit_library_get_global_negative") {
  hipLibrary_t library;
  void* ptr{nullptr};
  size_t bytes{0};
  const char* name = "d_var";
  std::string lib_co = "library_code_load.code";

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  SECTION("invalid ptr and bytes") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(nullptr, nullptr, library, name), hipErrorInvalidValue);
  }

  SECTION("invalid library") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&ptr, &bytes, nullptr, name),
                    hipErrorInvalidResourceHandle);
  }

  SECTION("invalid library") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&ptr, &bytes, library, nullptr), hipErrorInvalidValue);
  }

  SECTION("invalid library") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&ptr, &bytes, library,
                                        "unknown name thats definitely not in the library"),
                    hipErrorNotFound);
  }

  SECTION("should pass sanity") {
    // sanity that all other tests failing is right behavior
    HIP_CHECK(hipLibraryGetGlobal(&ptr, &bytes, library, name));
  }

  SECTION("should pass just size") {
    HIP_CHECK(hipLibraryGetGlobal(nullptr, &bytes, library, name));
  }

  SECTION("should pass just ptr") { HIP_CHECK(hipLibraryGetGlobal(&ptr, nullptr, library, name)); }


  HIP_CHECK(hipLibraryUnload(library));
}

TEST_CASE("Unit_library_get_global_values") {
  hipLibrary_t library;
  void* ptr{nullptr};
  size_t bytes{0};
  const char* name = "d_var";
  hipKernel_t wkernel, rkernel, rmkernel;
  std::string lib_co = "library_code_load.code";
  hipStream_t stream;
  constexpr size_t size = 32;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetGlobal(&ptr, &bytes, library, name));
  REQUIRE(bytes == 32 * sizeof(float));
  HIP_CHECK(hipLibraryGetKernel(&wkernel, library, "write_d_var"));
  HIP_CHECK(hipLibraryGetKernel(&rkernel, library, "read_d_var"));
  HIP_CHECK(hipLibraryGetKernel(&rmkernel, library, "read_modify_d_var"));

  SECTION("write val from kernel") {
    HIP_CHECK(hipLaunchKernel(wkernel, 1, size, nullptr, 0, stream));
    std::vector<float> out(size, 0.0f);
    HIP_CHECK(hipMemcpyAsync(out.data(), ptr, sizeof(float) * size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " got: " << out[i]);
      REQUIRE(out[i] == (i + 1));
    }
  }

  SECTION("read val from kernel") {
    float* dptr;
    HIP_CHECK(hipMalloc(&dptr, sizeof(float) * size));
    std::vector<float> in(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      in[i] = i + 1;
    }

    void* args[] = {&dptr};
    HIP_CHECK(hipMemcpyAsync(ptr, in.data(), sizeof(float) * size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipLaunchKernel(rkernel, 1, size, args, 0, stream));
    std::vector<float> out(size, 0.0f);
    HIP_CHECK(
        hipMemcpyAsync(out.data(), dptr, sizeof(float) * size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(dptr));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " got: " << out[i]);
      REQUIRE(out[i] == (in[i] + 1));
    }
  }

  SECTION("read/modify val from kernel") {
    float* dptr;
    HIP_CHECK(hipMalloc(&dptr, sizeof(float) * size));
    std::vector<float> in(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      in[i] = i + 1;
    }

    void* args[] = {&dptr};
    HIP_CHECK(hipMemcpyAsync(ptr, in.data(), sizeof(float) * size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipLaunchKernel(rmkernel, 1, size, args, 0, stream));
    std::vector<float> out(size, 0.0f), dvarout(size, 0.0f);
    HIP_CHECK(
        hipMemcpyAsync(out.data(), dptr, sizeof(float) * size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(
        hipMemcpyAsync(dvarout.data(), ptr, sizeof(float) * size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(dptr));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " got: " << out[i] << " dptr: " << dvarout[i]);
      REQUIRE(out[i] == (in[i] + 1));
      REQUIRE(out[i] == dvarout[i]);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(library));
}


TEST_CASE("Unit_library_get_managed_negative") {
  hipLibrary_t library;
  void* ptr{nullptr};
  size_t bytes{0};
  const char* name = "m_var";
  std::string lib_co = "library_code_load.code";

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  SECTION("invalid ptr and bytes") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(nullptr, nullptr, library, name), hipErrorInvalidValue);
  }

  SECTION("invalid library") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&ptr, &bytes, nullptr, name),
                    hipErrorInvalidResourceHandle);
  }

  SECTION("invalid library") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&ptr, &bytes, library, nullptr), hipErrorInvalidValue);
  }

  SECTION("invalid library") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&ptr, &bytes, library,
                                         "unknown name thats definitely not in the library"),
                    hipErrorNotFound);
  }

  SECTION("should pass sanity") {
    // sanity that all other tests failing is right behavior
    HIP_CHECK(hipLibraryGetManaged(&ptr, &bytes, library, name));
    REQUIRE(bytes == sizeof(float) * 32);
  }

  SECTION("should pass just size") {
    HIP_CHECK(hipLibraryGetManaged(nullptr, &bytes, library, name));
  }

  SECTION("should pass just ptr") { HIP_CHECK(hipLibraryGetManaged(&ptr, nullptr, library, name)); }

  HIP_CHECK(hipLibraryUnload(library));
}

TEST_CASE("Unit_library_get_managed_values") {
  hipLibrary_t library;
  float* ptr{nullptr};
  size_t bytes{0};
  const char* name = "m_var";
  hipKernel_t wkernel, rkernel, rmkernel;
  std::string lib_co = "library_code_load.code";
  hipStream_t stream;
  constexpr size_t size = 32;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetManaged((void**)&ptr, &bytes, library, name));
  REQUIRE(bytes == 32 * sizeof(float));
  HIP_CHECK(hipLibraryGetKernel(&wkernel, library, "write_m_var"));
  HIP_CHECK(hipLibraryGetKernel(&rkernel, library, "read_m_var"));
  HIP_CHECK(hipLibraryGetKernel(&rmkernel, library, "read_modify_m_var"));

  for (size_t i = 0; i < size; i++) {
    ptr[i] = 0.0f;  // clear it
  }

  SECTION("write val from kernel") {
    HIP_CHECK(hipLaunchKernel(wkernel, 1, size, nullptr, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " got: " << ptr[i]);
      REQUIRE(ptr[i] == (i + 1));
    }
  }

  SECTION("read val from kernel") {
    float* dptr;
    HIP_CHECK(hipMalloc(&dptr, sizeof(float) * size));
    for (size_t i = 0; i < size; i++) {
      ptr[i] = i + 1;
    }

    void* args[] = {&dptr};
    HIP_CHECK(hipLaunchKernel(rkernel, 1, size, args, 0, stream));
    std::vector<float> out(size, 0.0f);
    HIP_CHECK(
        hipMemcpyAsync(out.data(), dptr, sizeof(float) * size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(dptr));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " got: " << out[i]);
      REQUIRE(out[i] == (ptr[i] + 1));
    }
  }

  SECTION("read/modify val from kernel") {
    float* dptr;
    HIP_CHECK(hipMalloc(&dptr, sizeof(float) * size));
    for (size_t i = 0; i < size; i++) {
      ptr[i] = i + 1;
    }

    void* args[] = {&dptr};
    HIP_CHECK(hipLaunchKernel(rmkernel, 1, size, args, 0, stream));
    std::vector<float> out(size, 0.0f);
    HIP_CHECK(
        hipMemcpyAsync(out.data(), dptr, sizeof(float) * size, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(dptr));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " got: " << out[i] << " ptr: " << ptr[i]);
      REQUIRE(out[i] == (i + 2));
      REQUIRE(out[i] == ptr[i]);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(library));
}
