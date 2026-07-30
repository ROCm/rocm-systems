/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <string>
#include <vector>

namespace {

constexpr size_t kArrLen = 32;

std::vector<char> CompileSource(const std::string& code, const std::string& gpu_arch) {
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, code.c_str(), "code.cu", 0, nullptr, nullptr));
#ifdef __HIP_PLATFORM_AMD__
  std::string offload_arch = "--offload-arch=" + gpu_arch;
#else
  std::string offload_arch = "--fmad=false";
#endif
  const char* opts[] = {offload_arch.c_str()};
  HIPRTC_CHECK(hiprtcCompileProgram(prog, 1, opts));
  size_t size;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &size));
  std::vector<char> res(size, 0);
  HIPRTC_CHECK(hiprtcGetCode(prog, res.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return res;
}

// Note: variables are *defined* (with `= {0}`) rather than just declared
// `extern`. HIPRTC produces a fully-linked code object via hiprtcGetCode, and
// ld.lld inside HIPRTC rejects unresolved externals at device link time. The
// CUDA equivalent for "extern __device__ resolved at load" requires
// -fgpu-rdc + nvrtcGetBitcode + nvrtcLink*, which is a separate flow and not
// what hipLibraryGetGlobal needs to demonstrate.
const std::string kGlobalSrc =
    "extern \"C\" __device__ float d_var[32] = {0};\n"
    "extern \"C\" __global__ void write_d_var(float v) {\n"
    "  d_var[threadIdx.x] = v + threadIdx.x;\n"
    "}\n"
    "extern \"C\" __global__ void read_d_var(float* out) {\n"
    "  out[threadIdx.x] = d_var[threadIdx.x];\n"
    "}\n";

// HIPRTC's source-compile path does not auto-include
// __clang_hip_runtime_wrapper.h (where the public HIP toolchain defines
// `#define __managed__ __attribute__((managed))`). Including
// <hip/hip_runtime.h> doesn't help either — the public header never defines
// the macro. So in HIPRTC kernels we must spell the underlying attribute
// directly. The offline-compiled .code path (see loadlib_co.cc /
// library_code_load.cc) does have the wrapper and can use natural
// __managed__ syntax.
const std::string kManagedSrc =
    "extern \"C\" __attribute__((managed)) float m_var[32] = {0};\n"
    "extern \"C\" __global__ void write_m_var(float v) {\n"
    "  m_var[threadIdx.x] = v + threadIdx.x;\n"
    "}\n"
    "extern \"C\" __global__ void read_m_var(float* out) {\n"
    "  out[threadIdx.x] = m_var[threadIdx.x];\n"
    "}\n";

std::string GpuArch() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  return prop.gcnArchName;
}

}  // namespace

HIP_TEST_CASE(Unit_hipLibraryGetGlobal_Negative) {
  auto code = CompileSource(kGlobalSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* dptr = nullptr;
  size_t bytes = 0;

  SECTION("null name") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&dptr, &bytes, lib, nullptr), hipErrorInvalidValue);
  }
  SECTION("null library") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&dptr, &bytes, nullptr, "d_var"),
                    hipErrorInvalidResourceHandle);
  }
  SECTION("both dptr and bytes null") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(nullptr, nullptr, lib, "d_var"), hipErrorInvalidValue);
  }
  SECTION("missing symbol") {
    HIP_CHECK_ERROR(hipLibraryGetGlobal(&dptr, &bytes, lib, "no_such_symbol"), hipErrorNotFound);
  }

  HIP_CHECK(hipLibraryUnload(lib));
}

HIP_TEST_CASE(Unit_hipLibraryGetGlobal_Positive) {
  auto code = CompileSource(kGlobalSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* dptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "d_var"));
  REQUIRE(dptr != nullptr);
  REQUIRE(bytes == sizeof(float) * kArrLen);

  hipKernel_t writer = nullptr;
  hipKernel_t reader = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&writer, lib, "write_d_var"));
  HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_d_var"));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float seed = 100.0f;
  void* writer_args[] = {&seed};
  HIP_CHECK(hipLaunchKernel(writer, 1, kArrLen, writer_args, 0, stream));

  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * kArrLen));
  void* reader_args[] = {&d_out};
  HIP_CHECK(hipLaunchKernel(reader, 1, kArrLen, reader_args, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<float> host_out(kArrLen, 0.0f);
  HIP_CHECK(hipMemcpy(host_out.data(), d_out, sizeof(float) * kArrLen, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kArrLen; ++i) {
    INFO("Index: " << i);
    REQUIRE(host_out[i] == seed + static_cast<float>(i));
  }

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(lib));
}

// Verifies the device pointer returned by hipLibraryGetGlobal works with the
// regular memory APIs (hipMemcpy / hipMemset), which is the primary
// CUDA-documented use case. Exercises both H->D and D->H directions and
// confirms a kernel sees the data written via hipMemcpy.
HIP_TEST_CASE(Unit_hipLibraryGetGlobal_HipMemcpyInterop) {
  auto code = CompileSource(kGlobalSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* dptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "d_var"));
  REQUIRE(dptr != nullptr);
  REQUIRE(bytes == sizeof(float) * kArrLen);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("hipMemcpy H->D writes are visible to a kernel reading the global") {
    std::vector<float> host_in(kArrLen);
    for (size_t i = 0; i < kArrLen; ++i) host_in[i] = static_cast<float>(i) * 7.0f + 1.0f;
    HIP_CHECK(hipMemcpy(dptr, host_in.data(), bytes, hipMemcpyHostToDevice));

    hipKernel_t reader = nullptr;
    HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_d_var"));
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));
    void* args[] = {&d_out};
    HIP_CHECK(hipLaunchKernel(reader, 1, kArrLen, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> host_out(kArrLen, 0.0f);
    HIP_CHECK(hipMemcpy(host_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i);
      REQUIRE(host_out[i] == host_in[i]);
    }
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("hipMemcpy D->H reads what a kernel wrote into the global") {
    hipKernel_t writer = nullptr;
    HIP_CHECK(hipLibraryGetKernel(&writer, lib, "write_d_var"));
    float seed = 25.0f;
    void* args[] = {&seed};
    HIP_CHECK(hipLaunchKernel(writer, 1, kArrLen, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> host_out(kArrLen, 0.0f);
    HIP_CHECK(hipMemcpy(host_out.data(), dptr, bytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i);
      REQUIRE(host_out[i] == seed + static_cast<float>(i));
    }
  }

  SECTION("hipMemset on the device pointer zeroes the global") {
    HIP_CHECK(hipMemset(dptr, 0, bytes));

    hipKernel_t reader = nullptr;
    HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_d_var"));
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));
    void* args[] = {&d_out};
    HIP_CHECK(hipLaunchKernel(reader, 1, kArrLen, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> host_out(kArrLen, 1.0f);
    HIP_CHECK(hipMemcpy(host_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kArrLen; ++i) {
      INFO("Index: " << i);
      REQUIRE(host_out[i] == 0.0f);
    }
    HIP_CHECK(hipFree(d_out));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(lib));
}

// Same hipMemcpy/hipMemset interop coverage for managed-var pointers.
// The returned host pointer is a managed allocation, so direct host
// dereference works and hipMemcpy with hipMemcpyDefault works too.
HIP_TEST_CASE(Unit_hipLibraryGetManaged_HipMemcpyInterop) {
  CHECK_MANAGED_MEMORY_SUPPORT
  auto code = CompileSource(kManagedSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* host_ptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetManaged(&host_ptr, &bytes, lib, "m_var"));
  REQUIRE(host_ptr != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Stage: write via hipMemcpy(Default) into the managed buffer, then verify
  // a device kernel sees those values.
  std::vector<float> host_in(kArrLen);
  for (size_t i = 0; i < kArrLen; ++i) host_in[i] = static_cast<float>(i) * 11.0f - 3.0f;
  HIP_CHECK(hipMemcpy(host_ptr, host_in.data(), bytes, hipMemcpyDefault));

  hipKernel_t reader = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_m_var"));
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, bytes));
  void* args[] = {&d_out};
  HIP_CHECK(hipLaunchKernel(reader, 1, kArrLen, args, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<float> dev_out(kArrLen, 0.0f);
  HIP_CHECK(hipMemcpy(dev_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kArrLen; ++i) {
    INFO("Index: " << i);
    REQUIRE(dev_out[i] == host_in[i]);
  }

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(lib));
}

HIP_TEST_CASE(Unit_hipLibraryGetGlobal_OneNullParam) {
  auto code = CompileSource(kGlobalSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  SECTION("dptr null, bytes set") {
    size_t bytes = 0;
    HIP_CHECK(hipLibraryGetGlobal(nullptr, &bytes, lib, "d_var"));
    REQUIRE(bytes == sizeof(float) * kArrLen);
  }
  SECTION("dptr set, bytes null") {
    void* dptr = nullptr;
    HIP_CHECK(hipLibraryGetGlobal(&dptr, nullptr, lib, "d_var"));
    REQUIRE(dptr != nullptr);
  }

  HIP_CHECK(hipLibraryUnload(lib));
}

HIP_TEST_CASE(Unit_hipLibraryGetManaged_Negative) {
  CHECK_MANAGED_MEMORY_SUPPORT
  auto code = CompileSource(kManagedSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* dptr = nullptr;
  size_t bytes = 0;

  SECTION("null name") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&dptr, &bytes, lib, nullptr), hipErrorInvalidValue);
  }
  SECTION("null library") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&dptr, &bytes, nullptr, "m_var"),
                    hipErrorInvalidResourceHandle);
  }
  SECTION("both dptr and bytes null") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(nullptr, nullptr, lib, "m_var"), hipErrorInvalidValue);
  }
  SECTION("missing symbol") {
    HIP_CHECK_ERROR(hipLibraryGetManaged(&dptr, &bytes, lib, "no_such_symbol"), hipErrorNotFound);
  }

  HIP_CHECK(hipLibraryUnload(lib));
}

HIP_TEST_CASE(Unit_hipLibraryGetManaged_Positive) {
  CHECK_MANAGED_MEMORY_SUPPORT
  auto code = CompileSource(kManagedSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* host_ptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetManaged(&host_ptr, &bytes, lib, "m_var"));
  REQUIRE(host_ptr != nullptr);
  REQUIRE(bytes == sizeof(float) * kArrLen);

  hipKernel_t writer = nullptr;
  hipKernel_t reader = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&writer, lib, "write_m_var"));
  HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_m_var"));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Regression test for PR #1517 lazy-allocation bug:
  // immediately after hipLibraryGetManaged, launch a kernel that reads m_var.
  // The device-side managed-pointer slot must already be patched at library
  // build time (DynCO::initDynManagedVars). If it is null the kernel will fault.
  float seed = 50.0f;
  void* writer_args[] = {&seed};
  HIP_CHECK(hipLaunchKernel(writer, 1, kArrLen, writer_args, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Read managed memory directly from host after kernel finished.
  auto* host_view = static_cast<float*>(host_ptr);
  for (size_t i = 0; i < kArrLen; ++i) {
    INFO("Host read after device write, index: " << i);
    REQUIRE(host_view[i] == seed + static_cast<float>(i));
  }

  // Also exercise host-write → device-read direction.
  for (size_t i = 0; i < kArrLen; ++i) {
    host_view[i] = static_cast<float>(i) * 3.0f;
  }
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * kArrLen));
  void* reader_args[] = {&d_out};
  HIP_CHECK(hipLaunchKernel(reader, 1, kArrLen, reader_args, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<float> dev_out(kArrLen, 0.0f);
  HIP_CHECK(hipMemcpy(dev_out.data(), d_out, sizeof(float) * kArrLen, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kArrLen; ++i) {
    INFO("Device read after host write, index: " << i);
    REQUIRE(dev_out[i] == static_cast<float>(i) * 3.0f);
  }

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipLibraryUnload(lib));
}

// Coverage for hipLibraryGetKernel / hipLibraryGetKernelCount /
// hipLibraryEnumerateKernels exercised against a code object that also
// contains __device__ globals — guards the LibraryContainer refactor that
// now sources kernels from the underlying DynCO instead of an internal
// FatBinaryInfo + Function map.
HIP_TEST_CASE(Unit_hipLibraryGetKernel_CountAndEnumerate) {
  auto code = CompileSource(kGlobalSrc, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  // kGlobalSrc defines two kernels (write_d_var, read_d_var).
  unsigned int count = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&count, lib));
  REQUIRE(count == 2);

  hipKernel_t writer = nullptr;
  hipKernel_t reader = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&writer, lib, "write_d_var"));
  HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_d_var"));
  REQUIRE(writer != nullptr);
  REQUIRE(reader != nullptr);
  REQUIRE(writer != reader);

  // Cached lookup: asking for the same name again must return the same handle.
  hipKernel_t writer_again = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&writer_again, lib, "write_d_var"));
  REQUIRE(writer_again == writer);

  // Missing name must report not found.
  hipKernel_t missing = nullptr;
  HIP_CHECK_ERROR(hipLibraryGetKernel(&missing, lib, "nonexistent_kernel"), hipErrorNotFound);

  // EnumerateKernels must return both handles (order is implementation-defined).
  hipKernel_t enumerated[2] = {nullptr, nullptr};
  HIP_CHECK(hipLibraryEnumerateKernels(enumerated, 2, lib));
  REQUIRE(enumerated[0] != nullptr);
  REQUIRE(enumerated[1] != nullptr);
  REQUIRE(enumerated[0] != enumerated[1]);
  const bool covers_writer = (enumerated[0] == writer) || (enumerated[1] == writer);
  const bool covers_reader = (enumerated[0] == reader) || (enumerated[1] == reader);
  REQUIRE(covers_writer);
  REQUIRE(covers_reader);

  HIP_CHECK(hipLibraryUnload(lib));
}

// Confirms hipLibraryGetGlobal and hipLibraryGetKernel are interchangeable in
// either order — both trigger the lazy BuildIt, and DynCO populates kernels
// and globals together in loadCodeObject. Either path must leave the other
// operational.
HIP_TEST_CASE(Unit_hipLibraryGetKernel_OrderingWithGetGlobal) {
  SECTION("GetGlobal first, then GetKernel") {
    auto code = CompileSource(kGlobalSrc, GpuArch());
    hipLibrary_t lib = nullptr;
    HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

    void* dptr = nullptr;
    size_t bytes = 0;
    HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "d_var"));

    hipKernel_t reader = nullptr;
    HIP_CHECK(hipLibraryGetKernel(&reader, lib, "read_d_var"));
    REQUIRE(reader != nullptr);

    HIP_CHECK(hipLibraryUnload(lib));
  }
  SECTION("GetKernel first, then GetGlobal") {
    auto code = CompileSource(kGlobalSrc, GpuArch());
    hipLibrary_t lib = nullptr;
    HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

    hipKernel_t writer = nullptr;
    HIP_CHECK(hipLibraryGetKernel(&writer, lib, "write_d_var"));
    REQUIRE(writer != nullptr);

    void* dptr = nullptr;
    size_t bytes = 0;
    HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "d_var"));
    REQUIRE(dptr != nullptr);
    REQUIRE(bytes == sizeof(float) * kArrLen);

    HIP_CHECK(hipLibraryUnload(lib));
  }
}

HIP_TEST_CASE(Unit_hipLibraryGet_CrossKindRejection) {
  CHECK_MANAGED_MEMORY_SUPPORT
  // Source has both a __device__ var (d_var) and a __managed__ var (m_var).
  const std::string src = kGlobalSrc + kManagedSrc;
  auto code = CompileSource(src, GpuArch());
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* dptr = nullptr;
  size_t bytes = 0;
  // GetManaged on a __device__ variable must fail.
  HIP_CHECK_ERROR(hipLibraryGetManaged(&dptr, &bytes, lib, "d_var"), hipErrorNotFound);

  // GetGlobal on a __managed__ variable returns the managed host pointer
  // (matching CUDA's cuLibraryGetGlobal behavior on managed symbols).
  HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "m_var"));
  REQUIRE(dptr != nullptr);
  REQUIRE(bytes == sizeof(float) * kArrLen);

  HIP_CHECK(hipLibraryUnload(lib));
}
