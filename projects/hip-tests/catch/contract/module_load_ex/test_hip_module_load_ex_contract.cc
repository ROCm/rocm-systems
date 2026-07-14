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
// kernel that publishes a value through a device pointer so the load-with-options
// lookup and launch contracts have a portable symbol to resolve without any
// external per-arch code-object fixture.
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
// HIPRTC_CHECK. The bool return is retained only so LoadContractModuleEx keeps
// its familiar `if (!Compile...())` shape; the false branch is unreachable
// because any real failure aborts first.
bool CompileModuleSource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource, "module_load_ex_contract.cu", 0, nullptr,
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
// successful return the module is loaded via hipModuleLoadDataEx with zero
// options and ready for the per-test contract.
void LoadContractModuleEx(hipModule_t& module) {
  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }
  HIP_CHECK(hipModuleLoadDataEx(&module, code.data(), 0, nullptr, nullptr));
  REQUIRE(module != nullptr);
}
}  // namespace

HIP_TEST_CASE(Contract_ModuleLoadEx_ZeroOptions_LoadsAndUnloads) {
  hip::contract::ContractCleanup cleanup;
  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  // Loading a HIPRTC-produced code object through the load-with-options entry
  // point with zero options must behave like the plain load: it must produce a
  // non-null module handle and unload again without error.
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadDataEx(&module, code.data(), 0, nullptr, nullptr));
  cleanup.Add([&] { (void)hipModuleUnload(module); });
  REQUIRE(module != nullptr);
}

HIP_TEST_CASE(Contract_ModuleLoadEx_WithJitOptions_ResolvesSymbol) {
  hip::contract::ContractCleanup cleanup;
  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  // Passing a benign, portable JIT option pair (an error-log buffer plus its
  // size) must not prevent the module from loading. The contract only requires
  // that supplying options is accepted and that a known symbol still resolves;
  // the exact log contents are backend-specific and are not inspected. If the
  // option path is genuinely unsupported the test skips rather than fails.
  std::vector<char> log_buffer(1024, '\0');
  hipJitOption options[] = {hipJitOptionErrorLogBuffer, hipJitOptionErrorLogBufferSizeBytes};
  void* option_values[] = {reinterpret_cast<void*>(log_buffer.data()),
                           reinterpret_cast<void*>(static_cast<size_t>(log_buffer.size()))};

  hipModule_t module = nullptr;
  const hipError_t status =
      hipModuleLoadDataEx(&module, code.data(), 2, options, option_values);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("This runtime path does not support the requested JIT option.");
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipModuleUnload(module); });
  REQUIRE(module != nullptr);

  // A symbol that exists in the loaded module must resolve to a non-null
  // function handle regardless of the JIT options that were supplied.
  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);
}

HIP_TEST_CASE(Contract_ModuleLoadEx_NullImage_IsRejected) {
  // Loading a module from a null image through the load-with-options entry point
  // must not silently succeed. Backends may report hipErrorInvalidValue or
  // hipErrorInvalidImage; the contract only requires a non-success status so the
  // exact code is not pinned.
  hipModule_t module = nullptr;
  const hipError_t status = hipModuleLoadDataEx(&module, nullptr, 0, nullptr, nullptr);
  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_ModuleLoadEx_LaunchWritesExpectedValue) {
  hip::contract::ContractCleanup cleanup;
  hipModule_t module = nullptr;
  LoadContractModuleEx(module);
  cleanup.Add([&] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([&] { (void)hipFree(device_value); });
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
