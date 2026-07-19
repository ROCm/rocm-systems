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
// kernel that publishes a value through a device pointer so the module
// function-count, occupancy-query, and cooperative-launch contracts have a
// portable symbol to resolve without any external per-arch code-object fixture.
constexpr char const kModuleSource[] =
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
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
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource, "module_exec_contract.cu", 0, nullptr,
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

// Resolves the write_value symbol from a loaded module into a non-null function
// handle so the occupancy and cooperative-launch contracts share one lookup.
void ResolveWriteValue(hipModule_t module, hipFunction_t& function) {
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);
}

// Reports whether the current device advertises cooperative launch support so
// the cooperative-launch contracts can skip cleanly on paths that lack it.
bool CooperativeLaunchSupported() {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));
  int cooperative_launch = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cooperative_launch, hipDeviceAttributeCooperativeLaunch,
                                  current_device));
  return cooperative_launch != 0;
}
}  // namespace

// @asserts: hipModuleGetFunctionCount - a module defining at least one kernel reports a function count of at least one
HIP_TEST_CASE(Contract_ModuleExec_GetFunctionCount_ReturnsPositiveCount) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  // A module that defines at least one kernel must report a positive function
  // count. The exact count is backend-specific (backends may expose helper
  // symbols), so only the lower bound is part of the contract.
  unsigned int count = 0;
  HIP_CHECK(hipModuleGetFunctionCount(&count, module));
  REQUIRE(count >= 1);
}

// @asserts: hipModuleGetFunctionCount - a null count out-pointer is rejected with a non-success status
HIP_TEST_CASE(Contract_ModuleExec_GetFunctionCount_NullCount_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  // Querying the function count into a null out pointer must not silently
  // succeed. The exact error code is backend-specific, so only a non-success
  // status is required.
  const hipError_t status = hipModuleGetFunctionCount(nullptr, module);
  REQUIRE(status != hipSuccess);
}

// @asserts: hipModuleOccupancyMaxPotentialBlockSize - returns a positive block size and non-negative minimum grid size for a module function
HIP_TEST_CASE(Contract_ModuleExec_OccupancyMaxPotentialBlockSize_ReturnsUsableValues) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  ResolveWriteValue(module, function);

  // The potential-block-size query for a launchable module function must return
  // a positive block size and a non-negative minimum grid size. The exact
  // values are device-specific and are not pinned.
  int min_grid_size = 0;
  int block_size = 0;
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, function, 0, 0));
  REQUIRE(block_size > 0);
  REQUIRE(min_grid_size >= 0);
}

// @asserts: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor - returns non-negative active-blocks occupancy for a concrete block size
HIP_TEST_CASE(Contract_ModuleExec_OccupancyMaxActiveBlocks_ReturnsNonNegativeValue) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  ResolveWriteValue(module, function);

  // The active-blocks-per-multiprocessor query for a module function must return
  // a non-negative occupancy for a concrete block size.
  int num_blocks = -1;
  HIP_CHECK(
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, function, 64, 0));
  REQUIRE(num_blocks >= 0);
}

// @asserts: hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags - default-flags active-blocks query matches the non-flags query
HIP_TEST_CASE(Contract_ModuleExec_OccupancyWithFlags_MatchesDefault) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  ResolveWriteValue(module, function);

  // The default-flags occupancy query must agree with the non-flags query for
  // the same function and block size. The runtime documents that only the
  // default occupancy flag is supported, so the two entry points must be
  // consistent.
  int num_blocks_default = -1;
  HIP_CHECK(
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_default, function, 64, 0));
  REQUIRE(num_blocks_default >= 0);

  int num_blocks_with_flags = -1;
  HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &num_blocks_with_flags, function, 64, 0, hipOccupancyDefault));
  REQUIRE(num_blocks_with_flags >= 0);
  REQUIRE(num_blocks_with_flags == num_blocks_default);
}

// @asserts: hipModuleOccupancyMaxPotentialBlockSizeWithFlags - default-flags potential-block-size query matches the non-flags grid/block suggestion
HIP_TEST_CASE(Contract_ModuleExec_OccupancyPotentialBlockSizeWithFlags_MatchesDefault) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  ResolveWriteValue(module, function);

  // The default-flags potential-block-size query must agree with the non-flags
  // query for the same function. The runtime documents that only the default
  // occupancy flag is supported, so the two entry points must produce identical
  // grid/block suggestions.
  int grid_default = -1;
  int block_default = 0;
  HIP_CHECK(
      hipModuleOccupancyMaxPotentialBlockSize(&grid_default, &block_default, function, 0, 0));
  REQUIRE(block_default > 0);
  REQUIRE(grid_default >= 0);

#ifdef hipOccupancyDefault
  const unsigned int default_flags = hipOccupancyDefault;
#else
  const unsigned int default_flags = 0u;
#endif

  int grid_flags = -1;
  int block_flags = 0;
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&grid_flags, &block_flags, function, 0,
                                                             0, default_flags));
  REQUIRE(block_flags > 0);
  REQUIRE(grid_flags >= 0);
  REQUIRE(block_flags == block_default);
  REQUIRE(grid_flags == grid_default);
}

// @asserts: hipModuleLaunchCooperativeKernel - a cooperative launch of a module function executes and deterministically publishes the expected value
HIP_TEST_CASE(Contract_ModuleExec_LaunchCooperativeKernel_WritesExpectedValue) {
  if (!CooperativeLaunchSupported()) {
    HIP_SKIP_TEST("This device does not support cooperative kernel launch.");
  }
  hip::contract::ContractCleanup cleanup;

  hipModule_t module = nullptr;
  LoadContractModule(module);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  ResolveWriteValue(module, function);

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  // A cooperative launch of the resolved function with a single-thread grid must
  // execute and publish the expected value deterministically.
  int value = kExpectedValue;
  void* kernel_args[] = {&device_value, &value};
  HIP_CHECK(hipModuleLaunchCooperativeKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args));
  HIP_CHECK(hipDeviceSynchronize());

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, device_value, sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == kExpectedValue);
}

// @asserts: hipModuleLaunchCooperativeKernel - a null function handle is rejected with a non-success status
HIP_TEST_CASE(Contract_ModuleExec_LaunchCooperativeKernel_NullFunction_IsRejected) {
  if (!CooperativeLaunchSupported()) {
    HIP_SKIP_TEST("This device does not support cooperative kernel launch.");
  }

  // Launching a cooperative kernel with a null function handle must not silently
  // succeed. The exact error code is backend-specific, so only a non-success
  // status is required.
  const hipError_t status =
      hipModuleLaunchCooperativeKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr);
  REQUIRE(status != hipSuccess);
}
