/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>

#include <string>
#include <vector>

namespace {
constexpr int kExpectedValue = 0x1234;

// In-source device code compiled at runtime with HIPRTC. It exposes a tiny
// kernel that publishes a value through a device pointer plus a device global so
// the module lookup, launch, and global contracts have portable symbols to
// resolve without any external per-arch code-object fixture.
constexpr char const kModuleSource[] =
    "__device__ int g_value = 0;\n"
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    g_value = value;\n"
    "    out[0] = value;\n"
    "  }\n"
    "}\n";

// Compiles kModuleSource with HIPRTC for the current device and returns the
// resulting code object. A false return signals that HIPRTC compilation is not
// available on this device/runtime path so the caller can skip cleanly; any
// other HIPRTC failure is an unexpected contract violation and aborts through
// HIPRTC_CHECK.
bool CompileModuleSource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource, "module_contract.cu", 0, nullptr,
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

// Compiles the module source or skips the test when HIPRTC is unavailable. On a
// successful return the module is loaded and ready for the per-test contract.
void LoadContractModule(hipModule_t& module) {
  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  REQUIRE(module != nullptr);
}
}  // namespace

HIP_TEST_CASE(Contract_Module_LoadData_FromRtc_Succeeds) {
  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  // A HIPRTC-produced code object must load into a non-null module handle and
  // unload again without error.
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Contract_Module_LoadData_NullImage_IsRejected) {
  // Loading a module from a null image must not silently succeed. Backends may
  // report hipErrorInvalidValue or hipErrorInvalidImage; the contract only
  // requires a non-success status so the exact code is not pinned.
  hipModule_t module = nullptr;
  const hipError_t status = hipModuleLoadData(&module, nullptr);
  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_Module_GetFunction_ResolvesKnownSymbol) {
  hipModule_t module = nullptr;
  LoadContractModule(module);

  // A symbol that exists in the loaded module must resolve to a non-null
  // function handle.
  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Contract_Module_GetFunction_UnknownSymbol_IsRejected) {
  hipModule_t module = nullptr;
  LoadContractModule(module);

  // Resolving a symbol that the module does not define must fail rather than
  // return a bogus handle. The exact error code is backend-specific, so only a
  // non-success status is required.
  hipFunction_t function = nullptr;
  const hipError_t status = hipModuleGetFunction(&function, module, "no_such_kernel");
  REQUIRE(status != hipSuccess);

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Contract_Module_LaunchKernel_WritesExpectedValue) {
  hipModule_t module = nullptr;
  LoadContractModule(module);

  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
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
  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Contract_Module_GetGlobal_ReturnsAddressAndSize) {
  hipModule_t module = nullptr;
  LoadContractModule(module);

  // A device global defined in the module must resolve to a non-null device
  // address with a size that covers the declared type. The exact address is not
  // asserted; only its structural validity is part of the contract.
  hipDeviceptr_t device_address = nullptr;
  size_t byte_count = 0;
  HIP_CHECK(hipModuleGetGlobal(&device_address, &byte_count, module, "g_value"));
  REQUIRE(device_address != nullptr);
  REQUIRE(byte_count >= sizeof(int));

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Contract_Module_FuncGetAttribute_ReturnsSaneValues) {
  hipModule_t module = nullptr;
  LoadContractModule(module);

  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);

  // The maximum thread count for a launchable function must be positive.
  int max_threads_per_block = 0;
  HIP_CHECK(hipFuncGetAttribute(&max_threads_per_block,
                                HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
  REQUIRE(max_threads_per_block > 0);

  // The register count is a non-negative resource usage figure.
  int num_registers = 0;
  HIP_CHECK(hipFuncGetAttribute(&num_registers, HIP_FUNC_ATTRIBUTE_NUM_REGS, function));
  REQUIRE(num_registers >= 0);

  HIP_CHECK(hipModuleUnload(module));
}
