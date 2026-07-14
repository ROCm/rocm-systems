/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>

// In-source kernel used by the host-symbol multi-device launches. It publishes a
// value through a device pointer so the cooperative and extended multi-device
// launch contracts have a portable symbol to submit on every device.
__global__ void MultiDeviceWriteValue(int* out, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = value;
  }
}

// Cooperative multi-device launch contracts. hipLaunchCooperativeKernelMultiDevice,
// hipExtLaunchMultiKernelMultiDevice, and hipModuleLaunchCooperativeKernelMultiDevice
// each submit a per-device kernel across a launch-parameter array and are exercised
// as functional round-trips: every participating device writes a distinct value
// through its own device pointer, and each value is read back after synchronization.
//
// These require at least two discrete GPUs that advertise cooperative multi-device
// launch. On integrated devices (APUs/iGPUs), single-GPU hosts, or paths without the
// capability, the tests skip cleanly rather than reporting a contract violation.

namespace {
constexpr int kNumDevices = 2;

int CurrentDevice() {
  int device = -1;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

bool IsDiscreteDevice(int device) {
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  return props.integrated == 0;
}

bool CooperativeMultiDeviceLaunchSupported(int device) {
  int value = 0;
  HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeCooperativeMultiDeviceLaunch, device));
  return value != 0;
}

// Skips unless at least kNumDevices discrete GPUs are present and every device in
// [0, kNumDevices) advertises cooperative multi-device launch. Gating on the
// discrete-device property keeps the multi-device launch path off integrated
// runtimes where a stack launch-parameter array is not a meaningful submission.
void RequireCooperativeMultiDeviceLaunch() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < kNumDevices) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  for (int device = 0; device < kNumDevices; ++device) {
    if (!IsDiscreteDevice(device) || !CooperativeMultiDeviceLaunchSupported(device)) {
      HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    }
  }
}

// Per-device state for a multi-device launch: a device allocation to publish into,
// its own stream, and the expected value the kernel should write.
struct DeviceLaunchTarget {
  int* device_ptr = nullptr;
  hipStream_t stream = nullptr;
  int expected_value = 0;
};

void AllocateTargets(std::vector<DeviceLaunchTarget>& targets) {
  targets.resize(kNumDevices);
  for (int device = 0; device < kNumDevices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipMalloc(&targets[device].device_ptr, sizeof(int)));
    HIP_CHECK(hipMemset(targets[device].device_ptr, 0, sizeof(int)));
    HIP_CHECK(hipStreamCreate(&targets[device].stream));
    targets[device].expected_value = 100 + device;
  }
}

void VerifyAndRelease(std::vector<DeviceLaunchTarget>& targets) {
  for (int device = 0; device < kNumDevices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipDeviceSynchronize());
    int observed = -1;
    HIP_CHECK(hipMemcpy(&observed, targets[device].device_ptr, sizeof(int),
                        hipMemcpyDeviceToHost));
    REQUIRE(observed == targets[device].expected_value);
    HIP_CHECK(hipStreamDestroy(targets[device].stream));
    HIP_CHECK(hipFree(targets[device].device_ptr));
  }
  targets.clear();
}

// ---------------------------------------------------------------------------
// HIPRTC module fixture for the module-based multi-device launch. Each device
// loads its own module instance because module handles are context-bound.
// ---------------------------------------------------------------------------
constexpr char const kModuleSource[] =
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    out[0] = value;\n"
    "  }\n"
    "}\n";

bool CompileModuleSource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource, "multi_device_launch_contract.cu", 0,
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

// hipLaunchCooperativeKernelMultiDevice submits one host-symbol kernel per device
// through a hipLaunchParams array; each participating device must observe its own
// written value after synchronization.
HIP_TEST_CASE(Contract_MultiDeviceLaunch_CooperativeKernel_WritesPerDeviceValue) {
  RequireCooperativeMultiDeviceLaunch();

  std::vector<DeviceLaunchTarget> targets;
  AllocateTargets(targets);

  std::vector<hipLaunchParams> launch_params(kNumDevices);
  std::memset(launch_params.data(), 0, launch_params.size() * sizeof(hipLaunchParams));
  std::vector<std::array<void*, 2>> kernel_args(kNumDevices);
  for (int device = 0; device < kNumDevices; ++device) {
    kernel_args[device] = {&targets[device].device_ptr, &targets[device].expected_value};
    launch_params[device].func = reinterpret_cast<void*>(MultiDeviceWriteValue);
    launch_params[device].gridDim = dim3(1);
    launch_params[device].blockDim = dim3(1);
    launch_params[device].sharedMem = 0;
    launch_params[device].stream = targets[device].stream;
    launch_params[device].args = kernel_args[device].data();
  }

  const int original_device = CurrentDevice();
  HIP_CHECK(hipSetDevice(0));
  const hipError_t status =
      hipLaunchCooperativeKernelMultiDevice(launch_params.data(), kNumDevices, 0);
  HIP_CHECK(hipSetDevice(original_device));
  HIP_CHECK(status);

  VerifyAndRelease(targets);
}

// hipExtLaunchMultiKernelMultiDevice is the AMD extended multi-device launch entry
// point over the same hipLaunchParams array; it must produce the same per-device
// observable writes.
HIP_TEST_CASE(Contract_MultiDeviceLaunch_ExtMultiKernel_WritesPerDeviceValue) {
  RequireCooperativeMultiDeviceLaunch();

  std::vector<DeviceLaunchTarget> targets;
  AllocateTargets(targets);

  std::vector<hipLaunchParams> launch_params(kNumDevices);
  std::memset(launch_params.data(), 0, launch_params.size() * sizeof(hipLaunchParams));
  std::vector<std::array<void*, 2>> kernel_args(kNumDevices);
  for (int device = 0; device < kNumDevices; ++device) {
    kernel_args[device] = {&targets[device].device_ptr, &targets[device].expected_value};
    launch_params[device].func = reinterpret_cast<void*>(MultiDeviceWriteValue);
    launch_params[device].gridDim = dim3(1);
    launch_params[device].blockDim = dim3(1);
    launch_params[device].sharedMem = 0;
    launch_params[device].stream = targets[device].stream;
    launch_params[device].args = kernel_args[device].data();
  }

  const int original_device = CurrentDevice();
  HIP_CHECK(hipSetDevice(0));
  const hipError_t status =
      hipExtLaunchMultiKernelMultiDevice(launch_params.data(), kNumDevices, 0);
  HIP_CHECK(hipSetDevice(original_device));
  HIP_CHECK(status);

  VerifyAndRelease(targets);
}

// hipModuleLaunchCooperativeKernelMultiDevice submits a module-resolved function per
// device through a hipFunctionLaunchParams array. Each device loads its own module
// instance (module handles are context-bound) and must observe its written value.
HIP_TEST_CASE(Contract_MultiDeviceLaunch_ModuleCooperativeKernel_WritesPerDeviceValue) {
  RequireCooperativeMultiDeviceLaunch();

  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  std::vector<DeviceLaunchTarget> targets;
  AllocateTargets(targets);

  std::vector<hipModule_t> modules(kNumDevices, nullptr);
  std::vector<hipFunctionLaunchParams> launch_params(kNumDevices);
  std::memset(launch_params.data(), 0, launch_params.size() * sizeof(hipFunctionLaunchParams));
  std::vector<std::array<void*, 2>> kernel_args(kNumDevices);
  for (int device = 0; device < kNumDevices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipModuleLoadData(&modules[device], code.data()));
    REQUIRE(modules[device] != nullptr);
    hipFunction_t function = nullptr;
    HIP_CHECK(hipModuleGetFunction(&function, modules[device], "write_value"));
    REQUIRE(function != nullptr);

    kernel_args[device] = {&targets[device].device_ptr, &targets[device].expected_value};
    launch_params[device].function = function;
    launch_params[device].gridDimX = 1;
    launch_params[device].gridDimY = 1;
    launch_params[device].gridDimZ = 1;
    launch_params[device].blockDimX = 1;
    launch_params[device].blockDimY = 1;
    launch_params[device].blockDimZ = 1;
    launch_params[device].sharedMemBytes = 0;
    launch_params[device].hStream = targets[device].stream;
    launch_params[device].kernelParams = kernel_args[device].data();
  }

  const int original_device = CurrentDevice();
  HIP_CHECK(hipSetDevice(0));
  const hipError_t status =
      hipModuleLaunchCooperativeKernelMultiDevice(launch_params.data(), kNumDevices, 0);
  HIP_CHECK(hipSetDevice(original_device));
  HIP_CHECK(status);

  VerifyAndRelease(targets);

  for (int device = 0; device < kNumDevices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipModuleUnload(modules[device]));
  }
  HIP_CHECK(hipSetDevice(original_device));
}
