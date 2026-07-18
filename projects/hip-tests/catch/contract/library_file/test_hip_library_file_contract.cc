/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// The library-from-file and managed-symbol APIs (hipLibraryLoadFromFile,
// hipLibraryGetManaged) are exercised on both backends: on NVIDIA they map to
// the CUDA driver cuLibrary* entry points.
namespace {
constexpr char const kWriteKernelName[] = "write_value";

constexpr char const kLibrarySource[] =
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    out[0] = value;\n"
    "  }\n"
    "}\n";

// Compiles kLibrarySource with HIPRTC for device 0. A compile failure is a
// contract violation rather than an unsupported-capability skip: it surfaces the
// build log and aborts through HIPRTC_CHECK.
bool CompileLibrarySource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kLibrarySource, "library_file_contract.cu", 0, nullptr,
                                   nullptr));

#ifdef __HIP_PLATFORM_AMD__
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  const std::string offload_arch = std::string("--offload-arch=") + properties.gcnArchName;
  const char* options[] = {offload_arch.c_str()};
  const int num_options = 1;
#else
  const std::string fmad = "--fmad=false";
  const char* options[] = {fmad.c_str()};
  const int num_options = 1;
#endif

  const hiprtcResult compile_result = hiprtcCompileProgram(program, num_options, options);
  if (compile_result != HIPRTC_SUCCESS) {
    size_t log_size = 0;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
    std::string log(log_size, '\0');
    if (log_size > 0) {
      HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
    }
    INFO("HIPRTC compile log:\n" << log);
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    HIPRTC_CHECK(compile_result);
    return false;
  }

  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(program, &code_size));
  code.assign(code_size, 0);
  HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  return true;
}

// Compiles the library source and writes it to a unique per-test file in the
// current working directory, returning the path. Skips when HIPRTC is
// unavailable. The caller must remove the file when done.
std::string WriteCodeObjectFile(const char* suffix) {
  std::vector<char> code;
  if (!CompileLibrarySource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }
  const std::string path = std::string("hip-contract-library-file-") + suffix + ".code";
  std::ofstream out(path, std::ios::binary);
  REQUIRE(out.is_open());
  out.write(code.data(), static_cast<std::streamsize>(code.size()));
  out.close();
  REQUIRE(out.good());
  return path;
}
}  // namespace

HIP_TEST_CASE(Contract_LibraryFile_LoadFromFile_ResolvesKnownKernel) {
  const std::string path = WriteCodeObjectFile("resolve");

  // Loading a HIPRTC-produced code object from a file must yield a non-null
  // library handle whose known kernel resolves to a non-null kernel handle.
  hipLibrary_t library = nullptr;
  HIP_CHECK(
      hipLibraryLoadFromFile(&library, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  REQUIRE(library != nullptr);

  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kWriteKernelName));
  REQUIRE(kernel != nullptr);

  HIP_CHECK(hipLibraryUnload(library));
  std::remove(path.c_str());
}

HIP_TEST_CASE(Contract_LibraryFile_LoadFromFile_MissingFile_IsRejected) {
  // Loading from a path that does not exist must not silently succeed. The exact
  // error code is backend-specific, so only a non-success status is required.
  hipLibrary_t library = nullptr;
  const hipError_t status = hipLibraryLoadFromFile(
      &library, "hip-contract-library-file-missing.code", nullptr, nullptr, 0, nullptr, nullptr, 0);
  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_LibraryFile_GetManaged_UnknownSymbol_IsRejected) {
  const std::string path = WriteCodeObjectFile("managed");

  hipLibrary_t library = nullptr;
  HIP_CHECK(
      hipLibraryLoadFromFile(&library, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  REQUIRE(library != nullptr);

  // Requesting a managed symbol that the library does not define must fail
  // rather than return a bogus device pointer. The source declares no managed
  // variables, so any name is unknown.
  void* device_address = nullptr;
  size_t byte_count = 0;
  const hipError_t status =
      hipLibraryGetManaged(&device_address, &byte_count, library, "no_such_managed_symbol");
  REQUIRE(status != hipSuccess);

  HIP_CHECK(hipLibraryUnload(library));
  std::remove(path.c_str());
}
