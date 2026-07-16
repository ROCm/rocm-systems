/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// Driver-style extended launch and SM-resource group-split contracts.
//
// hipLaunchKernelExC launches a kernel through a driver-style hipLaunchConfig_t
// and is exercised as a functional round-trip: a single-thread kernel writes a
// value through a device pointer and the value is read back after the launch.
//
// hipDrvLaunchKernelEx is the driver-API extended launch: it takes a driver
// function handle (hipFunction_t) resolved from a loaded module plus a
// HIP_LAUNCH_CONFIG, and is exercised as the same functional write-value
// round-trip through an HIPRTC-compiled kernel.
//
// hipDevSmResourceSplit partitions a device's SM resource into caller-sized groups
// (as opposed to the count-based hipDevSmResourceSplitByCount covered by the green
// context domain); the produced group must contain a positive SM count bounded by
// the whole-device SM count.
//
// Both are gated to discrete GPUs. On integrated devices these device-resource and
// driver-launch paths are not meaningfully exercised, so the tests skip.

namespace {
int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

void SkipIfIntegratedDevice() {
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, CurrentDevice()));
  if (props.integrated != 0) {
    HIP_SKIP_TEST("driver-style extended launch and SM resource split are only "
                  "exercised on discrete GPUs.");
  }
}

// Owns a device allocation and frees it on destruction, so a failing REQUIRE
// (which throws and unwinds) does not leak the buffer into sibling tests.
class ScopedDeviceInt {
 public:
  explicit ScopedDeviceInt(int initial_value) {
    HIP_CHECK(hipMalloc(&ptr_, sizeof(int)));
    HIP_CHECK(hipMemset(ptr_, initial_value, sizeof(int)));
  }
  ~ScopedDeviceInt() {
    if (ptr_ != nullptr) {
      (void)hipFree(ptr_);
    }
  }
  ScopedDeviceInt(const ScopedDeviceInt&) = delete;
  ScopedDeviceInt& operator=(const ScopedDeviceInt&) = delete;

  int* get() { return ptr_; }
  int** address() { return &ptr_; }

 private:
  int* ptr_ = nullptr;
};

__global__ void WriteValueKernel(int* out, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = value;
  }
}

// HIPRTC module source providing a driver-launchable write_value kernel so
// hipDrvLaunchKernelEx has a hipFunction_t to launch without an external code
// object fixture.
constexpr char const kModuleSource[] =
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    out[0] = value;\n"
    "  }\n"
    "}\n";

// Compiles kModuleSource with HIPRTC for the current device. Returns false only
// when HIPRTC is unavailable (the caller then skips); a real compile failure
// aborts through HIPRTC_CHECK.
bool CompileModuleSource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource, "driver_launch_ex_contract.cu", 0,
                                   nullptr, nullptr));
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
}  // namespace

// hipLaunchKernelExC submits a kernel via a driver-style launch configuration.
// The launched kernel must publish its argument value, observable after a device
// synchronize.
HIP_TEST_CASE(Contract_DriverLaunchEx_LaunchKernelExC_WritesExpectedValue) {
  SkipIfIntegratedDevice();

  ScopedDeviceInt device_value(0);

  constexpr int kExpected = 0x5A5A;
  int value = kExpected;
  void* args[] = {device_value.address(), &value};

  hipLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(1);
  config.dynamicSmemBytes = 0;
  config.stream = nullptr;
  config.attrs = nullptr;
  config.numAttrs = 0;

  const hipError_t status =
      hipLaunchKernelExC(&config, reinterpret_cast<const void*>(WriteValueKernel), args);
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipLaunchKernelExC is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipDeviceSynchronize());

  int observed = -1;
  HIP_CHECK(hipMemcpy(&observed, device_value.get(), sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(observed == kExpected);
}

// hipDevSmResourceSplit partitions the device SM resource into caller-specified
// groups. Requesting one group must yield a group whose SM count is within
// (0, device SM count]; the split cannot invent SMs the device lacks.
HIP_TEST_CASE(Contract_DriverLaunchEx_DevSmResourceSplit_ProducesBoundedGroup) {
  SkipIfIntegratedDevice();

  hipDevResource device_resource{};
  const hipError_t query_status =
      hipDeviceGetDevResource(CurrentDevice(), &device_resource, hipDevResourceTypeSm);
  if (query_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Device resource queries are not supported by this runtime path.");
  }
  HIP_CHECK(query_status);
  REQUIRE(device_resource.type == hipDevResourceTypeSm);
  REQUIRE(device_resource.sm.smCount > 0);

  hipDevResource group{};
  hipDevResource remainder{};
  hipDevSmResourceGroupParams params{};
  params.smCount = 1;

  const hipError_t status =
      hipDevSmResourceSplit(&group, 1, &device_resource, &remainder, 0, &params);
  // The group-parameter split is a configuration some backends do not accept:
  // RDNA GPUs support the count-based split (hipDevSmResourceSplitByCount) but
  // reject the group-params variant with hipErrorInvalidResourceConfiguration,
  // while all CDNA arches accept it. Treat both "not supported" and "invalid
  // resource configuration" as a capability skip rather than a contract
  // violation; the split is only asserted where the backend accepts the config.
  if (status == hipErrorNotSupported || status == hipErrorInvalidResourceConfiguration) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("SM resource group splitting is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  REQUIRE(group.sm.smCount > 0);
  REQUIRE(group.sm.smCount <= device_resource.sm.smCount);
}

// hipDrvLaunchKernelEx submits a kernel through the driver-API extended launch:
// a driver function handle (resolved from an HIPRTC-compiled module) plus a
// HIP_LAUNCH_CONFIG. The launched kernel must publish its argument value,
// observable after a device synchronize.
HIP_TEST_CASE(Contract_DriverLaunchEx_DrvLaunchKernelEx_WritesExpectedValue) {
  SkipIfIntegratedDevice();
  hip::contract::ContractCleanup cleanup;

  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  REQUIRE(module != nullptr);
  cleanup.Add([&] { (void)hipModuleUnload(module); });

  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
  REQUIRE(function != nullptr);

  ScopedDeviceInt device_value(0);
  constexpr int kExpected = 0x3C3C;
  int value = kExpected;
  void* params[] = {device_value.address(), &value};

  HIP_LAUNCH_CONFIG config{};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = nullptr;
  config.attrs = nullptr;
  config.numAttrs = 0;

  const hipError_t status = hipDrvLaunchKernelEx(&config, function, params, nullptr);
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipDrvLaunchKernelEx is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipDeviceSynchronize());

  int observed = -1;
  HIP_CHECK(hipMemcpy(&observed, device_value.get(), sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(observed == kExpected);
}
