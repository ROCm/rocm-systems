/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// The HIP library/kernel object APIs (hipLibrary*, hipKernel*) are AMD-side in
// this tree, so the whole domain is gated like the AMD-only extension contracts.
#if HT_AMD

namespace {
constexpr int kExpectedValue = 0x1234;
constexpr char const kWriteKernelName[] = "write_value";
constexpr char const kGlobalName[] = "g_value";

// Magic that prefixes an uncompressed clang offload bundle. Kept as a local
// literal so the contract does not depend on hipamd-internal headers; it must
// stay in sync with symbols::kOffloadBundleUncompressedMagicStr in
// clr/hipamd/src/hip_code_object.hpp.
constexpr char const kUncompressedBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";

// In-source device code compiled at runtime with HIPRTC. It exposes two named
// kernels plus a resolvable device global so the library lookup, enumeration,
// launch, and global contracts have portable symbols to resolve without any
// external per-arch code-object fixture. The `write_value` kernel publishes a
// value through a device pointer so the launch contract can observe a write;
// `noop_kernel` gives the enumeration and count contracts a second symbol.
constexpr char const kLibrarySource[] =
    "__device__ int g_value = 0;\n"
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    g_value = value;\n"
    "    out[0] = value;\n"
    "  }\n"
    "}\n"
    "extern \"C\" __global__ void noop_kernel(int* out) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    out[0] = out[0];\n"
    "  }\n"
    "}\n";

// Compiles kLibrarySource with HIPRTC for the current device and returns the
// resulting code object. A false return signals that HIPRTC compilation is not
// available on this device/runtime path so the caller can skip cleanly; any
// other HIPRTC failure is an unexpected contract violation and aborts through
// HIPRTC_CHECK.
bool CompileLibrarySource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kLibrarySource, "library_contract.cu", 0, nullptr,
                                   nullptr));

  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  const std::string offload_arch = std::string("--offload-arch=") + properties.gcnArchName;
  const char* options[] = {offload_arch.c_str()};
  const int num_options = 1;

  const hiprtcResult compile_result = hiprtcCompileProgram(program, num_options, options);
  if (compile_result != HIPRTC_SUCCESS) {
    // A compilation failure here indicates that the in-process HIPRTC path is
    // not supported by this device/runtime rather than a broken contract, so
    // release the program and let the caller skip.
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    return false;
  }

  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(program, &code_size));
  code.assign(code_size, 0);
  HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  return true;
}

// Compiles the library source into the caller-owned `code` buffer or skips the
// test when HIPRTC is unavailable. On a successful return the library is loaded
// and ready for the per-test contract. The runtime borrows the image pointer and
// parses it lazily on the first accessor, so `code` must outlive the returned
// `hipLibrary_t` and all of its accessors; the caller owns `code` and must keep
// it alive until after hipLibraryUnload.
void LoadContractLibrary(std::vector<char>& code, hipLibrary_t& library) {
  if (!CompileLibrarySource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }
  HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  REQUIRE(library != nullptr);
}
}  // namespace

HIP_TEST_CASE(Contract_Library_LoadData_FromRtc_Succeeds) {
  std::vector<char> code;
  if (!CompileLibrarySource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  // A HIPRTC-produced code object must load into a non-null library handle and
  // unload again without error.
  hipLibrary_t library = nullptr;
  HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  REQUIRE(library != nullptr);
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_LoadData_NullImage_IsRejected) {
  // Loading a library from a null image must not silently succeed. Backends may
  // report hipErrorInvalidValue or hipErrorInvalidImage; the contract only
  // requires a non-success status so the exact code is not pinned.
  hipLibrary_t library = nullptr;
  const hipError_t status =
      hipLibraryLoadData(&library, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0);
  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_Library_LoadData_InvalidImage_IsRejected) {
  // A buffer whose leading bytes match none of the recognized code-object
  // headers (compressed/uncompressed clang offload bundle, or a bare AMDGPU
  // ELF) must be rejected rather than copied as if it were a valid image. The
  // runtime prefers hipErrorInvalidImage, but the contract only pins a
  // non-success status so it does not overfit to a single backend code.
  static const unsigned char kJunk[] = {0x7F, 0x21, 0x00, 0x13, 0x37, 0xAB,
                                        0xCD, 0xEF, 0x00, 0x42, 0x99, 0x01};
  hipLibrary_t library = nullptr;
  const hipError_t status =
      hipLibraryLoadData(&library, kJunk, nullptr, nullptr, 0, nullptr, nullptr, 0);
  REQUIRE(status != hipSuccess);
  REQUIRE(library == nullptr);
}

HIP_TEST_CASE(Contract_Library_LoadData_TruncatedBundle_IsRejected) {
  // A buffer that begins with the uncompressed clang offload bundle magic but
  // is too short to hold the descriptor table it claims must be rejected
  // without over-reading past the buffer. The header claims a hostile number of
  // code objects while carrying none of the descriptor records that count
  // implies, so a naive walk driven by the untrusted count would read far past
  // the allocation. The contract requires a non-success status and no crash.
  //
  // Layout mirrors symbols::ClangOffloadBundleUncompressedHeader: the magic,
  // then a little-endian uint64 numOfCodeObjects, then (normally) the
  // descriptor table. We deliberately stop right after the count so the
  // descriptors are absent.
  const size_t magic_len = std::strlen(kUncompressedBundleMagic);
  std::vector<char> truncated(kUncompressedBundleMagic, kUncompressedBundleMagic + magic_len);

  // numOfCodeObjects = 0xFFFFFFFF: absurdly larger than any real fat binary, so
  // the runtime must reject on the count alone instead of walking descriptors
  // that do not exist.
  const uint64_t hostile_count = 0xFFFFFFFFull;
  for (size_t i = 0; i < sizeof(hostile_count); ++i) {
    truncated.push_back(static_cast<char>((hostile_count >> (8 * i)) & 0xFF));
  }
  // No descriptor records follow: the buffer ends here, truncated.

  hipLibrary_t library = nullptr;
  const hipError_t status = hipLibraryLoadData(&library, truncated.data(), nullptr, nullptr, 0,
                                               nullptr, nullptr, 0);
  REQUIRE(status != hipSuccess);
  REQUIRE(library == nullptr);
}

HIP_TEST_CASE(Contract_Library_LoadData_CopiesImageForLaterAccess) {
  // hipLibraryLoadData must take ownership of (copy) the code image so that the
  // caller's buffer can be freed once the call returns, mirroring CUDA's
  // cuLibraryLoadData which copies the image by default. The runtime parses the
  // image lazily on the first accessor, so if it merely borrowed the caller
  // pointer this accessor would read freed memory and fail with
  // hipErrorInvalidImage. This test deliberately does not use
  // LoadContractLibrary because that helper keeps the caller buffer alive and
  // would mask a borrowed-image bug.
  hipLibrary_t library = nullptr;
  {
    // Compile into a buffer that is destroyed before the first accessor runs.
    std::vector<char> code;
    if (!CompileLibrarySource(code)) {
      HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
    }
    HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    REQUIRE(library != nullptr);
  }  // `code` is freed here, before any accessor touches the image.

  // The first accessor forces the lazy build of the code object. If the runtime
  // owns the image this resolves normally; if it borrowed the freed buffer it
  // reports hipErrorInvalidImage.
  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kWriteKernelName));
  REQUIRE(kernel != nullptr);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_GetKernel_ResolvesKnownSymbol) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  // A symbol that exists in the loaded library must resolve to a non-null
  // kernel handle.
  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kWriteKernelName));
  REQUIRE(kernel != nullptr);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_GetKernel_UnknownSymbol_IsRejected) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  // Resolving a symbol that the library does not define must fail rather than
  // return a bogus handle. The exact error code is backend-specific, so only a
  // non-success status is required.
  hipKernel_t kernel = nullptr;
  const hipError_t status = hipLibraryGetKernel(&kernel, library, "no_such_kernel");
  REQUIRE(status != hipSuccess);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_GetKernelCount_MatchesLowerBound) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  // The source defines two kernels, so the reported count must be at least two.
  // The exact value is not pinned because the runtime may inject helper kernels
  // into the code object.
  unsigned int count = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&count, library));
  REQUIRE(count >= 2);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_EnumerateKernels_ZeroMax_LeavesBufferUntouched) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  // Enumerating with a maximum of zero must not write into the caller's buffer.
  // A sentinel guard slot is used to detect an out-of-contract write.
  hipKernel_t guard = reinterpret_cast<hipKernel_t>(static_cast<uintptr_t>(0xDEADBEEF));
  hipKernel_t buffer[1] = {guard};
  HIP_CHECK(hipLibraryEnumerateKernels(buffer, 0, library));
  REQUIRE(buffer[0] == guard);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_EnumerateKernels_HandlesResolveToFunctions) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  unsigned int count = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&count, library));
  REQUIRE(count >= 2);

  // Every enumerated kernel handle must be usable: it resolves to a non-null
  // launchable function through the public kernel-to-function accessor.
  std::vector<hipKernel_t> kernels(count, nullptr);
  HIP_CHECK(hipLibraryEnumerateKernels(kernels.data(), count, library));
  for (unsigned int i = 0; i < count; ++i) {
    REQUIRE(kernels[i] != nullptr);
    hipFunction_t function = nullptr;
    HIP_CHECK(hipKernelGetFunction(&function, kernels[i]));
    REQUIRE(function != nullptr);
  }

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_GetKernel_RepeatedLookupIsStable) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  // Repeated lookups of the same symbol against the same library must return the
  // same stable kernel handle.
  hipKernel_t first = nullptr;
  hipKernel_t second = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&first, library, kWriteKernelName));
  HIP_CHECK(hipLibraryGetKernel(&second, library, kWriteKernelName));
  REQUIRE(first != nullptr);
  REQUIRE(first == second);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_KernelGetName_ReturnsRequestedSymbol) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kWriteKernelName));
  REQUIRE(kernel != nullptr);

  // The name reported for a resolved kernel must be non-null and match the
  // symbol that was requested.
  const char* name = nullptr;
  HIP_CHECK(hipKernelGetName(&name, kernel));
  REQUIRE(name != nullptr);
  REQUIRE(std::strcmp(name, kWriteKernelName) == 0);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_KernelGetFunction_LaunchesAndWrites) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kWriteKernelName));
  REQUIRE(kernel != nullptr);

  // The kernel object must expose a launchable driver-style function handle.
  hipFunction_t function = nullptr;
  HIP_CHECK(hipKernelGetFunction(&function, kernel));
  REQUIRE(function != nullptr);

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  // Launch the resolved function through the driver-style module launch entry
  // point with a single-thread grid so the write is deterministic.
  int value = kExpectedValue;
  void* kernel_args[] = {&device_value, &value};
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, device_value, sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == kExpectedValue);

  HIP_CHECK(hipFree(device_value));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_GetGlobal_ReturnsAddressAndSize) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  // A device global defined in the library must resolve to a non-null device
  // address with a size that covers the declared type. The exact address is not
  // asserted; only its structural validity is part of the contract.
  void* device_address = nullptr;
  size_t byte_count = 0;
  HIP_CHECK(hipLibraryGetGlobal(&device_address, &byte_count, library, kGlobalName));
  REQUIRE(device_address != nullptr);
  REQUIRE(byte_count >= sizeof(int));

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Contract_Library_GetGlobal_MatchesModuleGetGlobal) {
  std::vector<char> code;
  if (!CompileLibrarySource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  // Loading the same code object through the module and library entry points
  // must resolve the same global to structurally equivalent results: both a
  // non-null/non-zero device pointer and an identical reported size. Pointer
  // identity across two independent loads is not part of the contract.
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  REQUIRE(module != nullptr);

  hipDeviceptr_t module_address = 0;
  size_t module_bytes = 0;
  HIP_CHECK(hipModuleGetGlobal(&module_address, &module_bytes, module, kGlobalName));
  REQUIRE(module_address != 0);

  hipLibrary_t library = nullptr;
  HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  REQUIRE(library != nullptr);

  void* library_address = nullptr;
  size_t library_bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&library_address, &library_bytes, library, kGlobalName));
  REQUIRE(library_address != nullptr);

  REQUIRE(module_bytes == library_bytes);

  HIP_CHECK(hipLibraryUnload(library));
  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Contract_Library_KernelGetLibrary_RoundTrips) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  LoadContractLibrary(code, library);

  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kWriteKernelName));
  REQUIRE(kernel != nullptr);

  // The library a kernel object was obtained from must round-trip back to the
  // originating library handle.
  hipLibrary_t resolved = nullptr;
  HIP_CHECK(hipKernelGetLibrary(&resolved, kernel));
  REQUIRE(resolved == library);

  HIP_CHECK(hipLibraryUnload(library));
}

#endif  // HT_AMD
