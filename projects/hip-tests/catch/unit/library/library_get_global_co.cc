/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Tests for hipLibraryGetGlobal / hipLibraryGetManaged against an
// ahead-of-time-compiled code object (library_code_load.code, built from
// library_code_load.cc by the CMake custom target). This exercises the
// production user path where natural __device__ / __managed__ syntax works
// because the offline amdclang++ invocation auto-includes
// __clang_hip_runtime_wrapper.h. The HIPRTC counterpart lives in
// library_get_global.cc.
//
// Test cases adapted from upstream PR #1517 (get_varaibles.cc), restructured
// to share a single test fixture and to exercise read-modify-write behavior.

#include <hip_test_common.hh>

#include <string>
#include <vector>

namespace {

constexpr size_t kArrLen = 32;
const std::string kCodeFile = "library_code_load.code";

}  // namespace

HIP_TEST_CASE(Unit_hipLibraryGetGlobal_CO_Negative) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  void* dptr = nullptr;
  size_t bytes = 0;

  SECTION("both dptr and bytes null") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(nullptr, nullptr, lib, "d_var"), hipErrorInvalidValue);
  }
  SECTION("null library") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&dptr, &bytes, nullptr, "d_var"),
                    hipErrorInvalidResourceHandle);
  }
  SECTION("null name") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&dptr, &bytes, lib, nullptr), hipErrorInvalidValue);
  }
  SECTION("empty name") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&dptr, &bytes, lib, ""), hipErrorInvalidValue);
  }
  SECTION("missing symbol") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&dptr, &bytes, lib, "no_such_symbol"), hipErrorNotFound);
  }

  HIP_CHECK(hipLibraryUnload(lib));
}

HIP_TEST_CASE(Unit_hipLibraryGetGlobal_CO_Values) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  void* dptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "d_var"));
  REQUIRE(dptr != nullptr);
  REQUIRE(bytes == sizeof(float) * kArrLen);

  hipKernel_t writer = nullptr;
  hipKernel_t reader = nullptr;
  hipKernel_t rmw = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&writer, lib, "write_d_var"));
  HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_d_var"));
  HIP_CHECK(hipLibraryGetKernel(&rmw, lib, "read_modify_d_var"));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("kernel writes global, host reads via hipMemcpy") {
    HIP_CHECK(hipLaunchKernel(writer, 1, kArrLen, nullptr, 0, stream));
    std::vector<float> out(kArrLen, 0.0f);
    HIP_CHECK(hipMemcpyAsync(out.data(), dptr, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i);
      REQUIRE(out[i] == static_cast<float>(i + 1));
    }
  }

  SECTION("host writes global via hipMemcpy, kernel reads") {
    std::vector<float> in(kArrLen);
    for (size_t i = 0; i < kArrLen; ++i) in[i] = static_cast<float>(i + 1);
    HIP_CHECK(hipMemcpyAsync(dptr, in.data(), bytes, hipMemcpyHostToDevice, stream));

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));
    void* args[] = {&d_out};
    HIP_CHECK(hipLaunchKernel(reader, 1, kArrLen, args, 0, stream));

    std::vector<float> out(kArrLen, 0.0f);
    HIP_CHECK(hipMemcpyAsync(out.data(), d_out, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_out));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i);
      REQUIRE(out[i] == in[i] + 1.0f);
    }
  }

  SECTION("read-modify-write: kernel mutates the global in place") {
    std::vector<float> in(kArrLen);
    for (size_t i = 0; i < kArrLen; ++i) in[i] = static_cast<float>(i + 1);
    HIP_CHECK(hipMemcpyAsync(dptr, in.data(), bytes, hipMemcpyHostToDevice, stream));

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));
    void* args[] = {&d_out};
    HIP_CHECK(hipLaunchKernel(rmw, 1, kArrLen, args, 0, stream));

    std::vector<float> out(kArrLen, 0.0f);
    std::vector<float> after(kArrLen, 0.0f);
    HIP_CHECK(hipMemcpyAsync(out.data(), d_out, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(after.data(), dptr, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_out));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i << " out=" << out[i] << " after=" << after[i]);
      REQUIRE(out[i] == in[i] + 1.0f);  // returned old+1
      REQUIRE(after[i] == out[i]);      // global was incremented to match
    }
  }

  SECTION("partial-null params") {
    // Either dptr or bytes (but not both) may be null.
    void* dptr_only = nullptr;
    size_t bytes_only = 0;
    HIP_CHECK(hipLibraryGetGlobal(&dptr_only, nullptr, lib, "d_var"));
    REQUIRE(dptr_only != nullptr);
    HIP_CHECK(hipLibraryGetGlobal(nullptr, &bytes_only, lib, "d_var"));
    REQUIRE(bytes_only == sizeof(float) * kArrLen);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(lib));
}

HIP_TEST_CASE(Unit_hipLibraryGetManaged_CO_Negative) {
  CHECK_MANAGED_MEMORY_SUPPORT
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  void* host_ptr = nullptr;
  size_t bytes = 0;

  SECTION("both dptr and bytes null") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(nullptr, nullptr, lib, "m_var"), hipErrorInvalidValue);
  }
  SECTION("null library") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&host_ptr, &bytes, nullptr, "m_var"),
                    hipErrorInvalidResourceHandle);
  }
  SECTION("null name") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&host_ptr, &bytes, lib, nullptr), hipErrorInvalidValue);
  }
  SECTION("empty name") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&host_ptr, &bytes, lib, ""), hipErrorInvalidValue);
  }
  SECTION("missing symbol") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&host_ptr, &bytes, lib, "no_such_symbol"),
                    hipErrorNotFound);
  }
  SECTION("rejected when name is a __device__ var, not __managed__") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&host_ptr, &bytes, lib, "d_var"), hipErrorNotFound);
  }

  HIP_CHECK(hipLibraryUnload(lib));
}

HIP_TEST_CASE(Unit_hipLibraryGetManaged_CO_Values) {
  CHECK_MANAGED_MEMORY_SUPPORT
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  float* host_ptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetManaged(reinterpret_cast<void**>(&host_ptr), &bytes, lib, "m_var"));
  REQUIRE(host_ptr != nullptr);
  REQUIRE(bytes == sizeof(float) * kArrLen);

  hipKernel_t writer = nullptr;
  hipKernel_t reader = nullptr;
  hipKernel_t rmw = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&writer, lib, "write_m_var"));
  HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_m_var"));
  HIP_CHECK(hipLibraryGetKernel(&rmw, lib, "read_modify_m_var"));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Zero from the host side — managed pointer is host-dereferenceable.
  for (size_t i = 0; i < kArrLen; ++i) host_ptr[i] = 0.0f;

  SECTION("kernel writes managed, host reads directly") {
    HIP_CHECK(hipLaunchKernel(writer, 1, kArrLen, nullptr, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i);
      REQUIRE(host_ptr[i] == static_cast<float>(i + 1));
    }
  }

  SECTION("host writes managed directly, kernel reads") {
    for (size_t i = 0; i < kArrLen; ++i) host_ptr[i] = static_cast<float>(i + 1);

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));
    void* args[] = {&d_out};
    HIP_CHECK(hipLaunchKernel(reader, 1, kArrLen, args, 0, stream));

    std::vector<float> out(kArrLen, 0.0f);
    HIP_CHECK(hipMemcpyAsync(out.data(), d_out, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_out));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i);
      REQUIRE(out[i] == host_ptr[i] + 1.0f);
    }
  }

  SECTION("read-modify-write: kernel and host agree on final state") {
    for (size_t i = 0; i < kArrLen; ++i) host_ptr[i] = static_cast<float>(i + 1);

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));
    void* args[] = {&d_out};
    HIP_CHECK(hipLaunchKernel(rmw, 1, kArrLen, args, 0, stream));

    std::vector<float> out(kArrLen, 0.0f);
    HIP_CHECK(hipMemcpyAsync(out.data(), d_out, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_out));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i << " out=" << out[i] << " host=" << host_ptr[i]);
      REQUIRE(out[i] == static_cast<float>(i + 2));  // returned old+1
      REQUIRE(out[i] == host_ptr[i]);                // managed mirror updated
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(lib));
}
