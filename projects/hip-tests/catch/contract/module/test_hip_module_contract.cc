/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

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
// resulting code object. This always returns true on success; a compile failure
// is treated as a contract violation rather than an unsupported-capability skip,
// so the HIPRTC log is emitted through INFO and the failure aborts through
// HIPRTC_CHECK. The bool return is retained only so LoadContractModule keeps its
// familiar `if (!Compile...())` shape; the false branch is unreachable because
// any real failure aborts first.
bool CompileModuleSource(std::vector<char>& code) {
  // Ensure a device context exists before the module is loaded below. On the
  // NVIDIA backend hipModuleLoadData maps to the driver-API cuModuleLoadData,
  // which requires a bound primary context; a test that loads a module before
  // any allocation would otherwise fail with "invalid device context". hipFree(0)
  // is the canonical no-op that forces primary-context initialization, and is a
  // harmless success on AMD where the runtime already auto-initializes.
  HIP_CHECK(hipFree(0));

  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource, "module_contract.cu", 0, nullptr,
                                   nullptr));

#if HT_AMD
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
    // A compilation failure is a contract violation, not an unsupported path:
    // surface the HIPRTC build log so compiler/source/option regressions are
    // diagnosable, release the program, then fail through HIPRTC_CHECK.
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

// @asserts: hipModuleLoadData - a HIPRTC-produced code object loads into a non-null module handle and unloads without error
HIP_TEST_CASE(Contract_Module_LoadData_FromRtc_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  // A HIPRTC-produced code object must load into a non-null module handle and
  // unload again without error.
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  cleanup.Add([module] { (void)hipModuleUnload(module); });
  REQUIRE(module != nullptr);
}

// @asserts: hipModuleLoadData - loading from a null image is rejected with a non-success status
HIP_TEST_CASE(Contract_Module_LoadData_NullImage_IsRejected) {
  // Loading a module from a null image must not silently succeed. Backends may
  // report hipErrorInvalidValue or hipErrorInvalidImage; the contract only
  // requires a non-success status so the exact code is not pinned.
  hipModule_t module = nullptr;
  const hipError_t status = hipModuleLoadData(&module, nullptr);
  REQUIRE(status != hipSuccess);
}

// @asserts: hipModuleGetFunction - a symbol present in the module resolves to a non-null function handle
HIP_TEST_CASE(Contract_Module_GetFunction_ResolvesKnownSymbol) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  // A symbol that exists in the loaded module must resolve to a non-null
  // function handle.
  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);
}

// @asserts: hipModuleGetFunction - resolving a symbol the module does not define fails rather than returning a bogus handle
HIP_TEST_CASE(Contract_Module_GetFunction_UnknownSymbol_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  // Resolving a symbol that the module does not define must fail rather than
  // return a bogus handle. The exact error code is backend-specific, so only a
  // non-success status is required.
  hipFunction_t function = nullptr;
  const hipError_t status = hipModuleGetFunction(&function, module, "no_such_kernel");
  REQUIRE(status != hipSuccess);
}

// @asserts: hipModuleLaunchKernel - launching a resolved module function with a single-thread grid deterministically writes the expected value
HIP_TEST_CASE(Contract_Module_LaunchKernel_WritesExpectedValue) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
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
}

// @asserts: hipModuleGetGlobal - a device global resolves to a non-null address with a size covering its declared type
HIP_TEST_CASE(Contract_Module_GetGlobal_ReturnsAddressAndSize) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  // A device global defined in the module must resolve to a non-null device
  // address with a size that covers the declared type. The exact address is not
  // asserted; only its structural validity is part of the contract.
  hipDeviceptr_t device_address = 0;
  size_t byte_count = 0;
  HIP_CHECK(hipModuleGetGlobal(&device_address, &byte_count, module, "g_value"));
  REQUIRE(device_address != 0);
  REQUIRE(byte_count >= sizeof(int));
}

// @asserts: hipFuncGetAttribute - a module function reports positive max-threads-per-block and a non-negative register count
HIP_TEST_CASE(Contract_Module_FuncGetAttribute_ReturnsSaneValues) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

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
}
