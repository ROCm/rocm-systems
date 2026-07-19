/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>

// BACKEND-DIFF: The cooperative multi-device launch family exercised here is
// AMD-only on the NVIDIA backend: hipExtLaunchMultiKernelMultiDevice has no
// NVIDIA entry point, and hipLaunchParams maps to CUDA's cudaLaunchParams, which
// the NVIDIA-backend headers leave incomplete. These contracts also require two
// or more cooperative GPUs, so they build only on AMD. Parity would require the
// NVIDIA headers to complete cudaLaunchParams and provide the ext launch entry.
#if HT_AMD

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

// Restores the process current device when it leaves scope, so a test that
// switches devices (or throws mid-switch through a failing REQUIRE) cannot leave
// a sibling test running on the wrong device.
class ScopedDevice {
 public:
  ScopedDevice() { HIP_CHECK(hipGetDevice(&saved_device_)); }
  ~ScopedDevice() { (void)hipSetDevice(saved_device_); }
  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

 private:
  int saved_device_ = 0;
};

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

// True when every device in [0, kNumDevices) reports the same architecture name.
// The module-cooperative test compiles a single code object (for the current
// device's arch) and loads it on every participating device, so a mixed-arch
// multi-GPU host would fail the load even though each device supports cooperative
// multi-device launch. Callers skip when the participating devices differ.
bool ParticipatingDevicesShareArch() {
  hipDeviceProp_t first{};
  HIP_CHECK(hipGetDeviceProperties(&first, 0));
  for (int device = 1; device < kNumDevices; ++device) {
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    if (std::strcmp(props.gcnArchName, first.gcnArchName) != 0) {
      return false;
    }
  }
  return true;
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
// its own stream, and the expected value the kernel should write. The allocation
// and stream are released in the destructor so a failing REQUIRE (which throws
// and unwinds) cannot leak device memory or streams into sibling tests that share
// the process. Each target remembers its owning device so teardown frees on the
// correct context regardless of the currently-selected device.
class DeviceLaunchTarget {
 public:
  DeviceLaunchTarget() = default;

  // Allocates a one-int device buffer and a stream on `device`, zeroing the
  // buffer so a later readback proves the launched kernel wrote the value.
  void Allocate(int device) {
    device_ = device;
    expected_value_ = 100 + device;
    HIP_CHECK(hipSetDevice(device_));
    HIP_CHECK(hipMalloc(&device_ptr_, sizeof(int)));
    HIP_CHECK(hipMemset(device_ptr_, 0, sizeof(int)));
    HIP_CHECK(hipStreamCreate(&stream_));
  }

  ~DeviceLaunchTarget() {
    if (device_ptr_ == nullptr && stream_ == nullptr) {
      return;
    }
    (void)hipSetDevice(device_);
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
    if (device_ptr_ != nullptr) {
      (void)hipFree(device_ptr_);
    }
  }

  DeviceLaunchTarget(const DeviceLaunchTarget&) = delete;
  DeviceLaunchTarget& operator=(const DeviceLaunchTarget&) = delete;

  int* device_ptr() { return device_ptr_; }
  int** device_ptr_address() { return &device_ptr_; }
  hipStream_t stream() const { return stream_; }
  int device() const { return device_; }
  int expected_value() const { return expected_value_; }
  int* expected_value_address() { return &expected_value_; }

 private:
  int* device_ptr_ = nullptr;
  hipStream_t stream_ = nullptr;
  int device_ = 0;
  int expected_value_ = 0;
};

// Allocates per-device targets. Uses unique_ptr so the vector owns each target
// by pointer: reallocation during growth cannot invoke a (deleted) move that
// would double-free, and any target already constructed is destroyed (freeing
// its resources) if a later device's allocation throws.
void AllocateTargets(std::vector<std::unique_ptr<DeviceLaunchTarget>>& targets) {
  for (int device = 0; device < kNumDevices; ++device) {
    targets.push_back(std::make_unique<DeviceLaunchTarget>());
    targets.back()->Allocate(device);
  }
}

// Reads back each device's value and asserts it matches. Cleanup is left to the
// targets' destructors, so a failing REQUIRE here still releases every device
// allocation and stream as the vector unwinds.
void VerifyTargets(std::vector<std::unique_ptr<DeviceLaunchTarget>>& targets) {
  for (auto& target : targets) {
    HIP_CHECK(hipSetDevice(target->device()));
    HIP_CHECK(hipDeviceSynchronize());
    int observed = -1;
    HIP_CHECK(hipMemcpy(&observed, target->device_ptr(), sizeof(int),
                        hipMemcpyDeviceToHost));
    REQUIRE(observed == target->expected_value());
  }
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

// Owns a module loaded on a specific device and unloads it on destruction, so a
// failing REQUIRE partway through the per-device load loop does not leak the
// context-bound modules already loaded.
class ScopedModule {
 public:
  ScopedModule(hipModule_t module, int device) : module_(module), device_(device) {}
  ~ScopedModule() {
    if (module_ != nullptr) {
      (void)hipSetDevice(device_);
      (void)hipModuleUnload(module_);
    }
  }
  ScopedModule(ScopedModule&& other) noexcept
      : module_(other.module_), device_(other.device_) {
    other.module_ = nullptr;
  }
  ScopedModule& operator=(const ScopedModule&) = delete;
  ScopedModule(const ScopedModule&) = delete;

  hipModule_t get() const { return module_; }

 private:
  hipModule_t module_ = nullptr;
  int device_ = 0;
};

bool CompileModuleSource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource, "multi_device_launch_contract.cu", 0,
                                   nullptr, nullptr));

#if HT_AMD
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, current_device));
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
// @asserts: hipLaunchCooperativeKernelMultiDevice - each device runs its per-device kernel and observes its own written value
HIP_TEST_CASE(Contract_MultiDeviceLaunch_CooperativeKernel_WritesPerDeviceValue) {
  RequireCooperativeMultiDeviceLaunch();

  ScopedDevice restore_device;
  std::vector<std::unique_ptr<DeviceLaunchTarget>> targets;
  AllocateTargets(targets);

  std::vector<hipLaunchParams> launch_params(kNumDevices);
  std::memset(launch_params.data(), 0, launch_params.size() * sizeof(hipLaunchParams));
  std::vector<std::array<void*, 2>> kernel_args(kNumDevices);
  for (int device = 0; device < kNumDevices; ++device) {
    kernel_args[device] = {targets[device]->device_ptr_address(),
                           targets[device]->expected_value_address()};
    launch_params[device].func = reinterpret_cast<void*>(MultiDeviceWriteValue);
    launch_params[device].gridDim = dim3(1);
    launch_params[device].blockDim = dim3(1);
    launch_params[device].sharedMem = 0;
    launch_params[device].stream = targets[device]->stream();
    launch_params[device].args = kernel_args[device].data();
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launch_params.data(), kNumDevices, 0));

  VerifyTargets(targets);
}

// hipExtLaunchMultiKernelMultiDevice is the AMD extended multi-device launch entry
// point over the same hipLaunchParams array; it must produce the same per-device
// observable writes.
// @asserts: hipExtLaunchMultiKernelMultiDevice - extended multi-device launch produces the same per-device observable writes
HIP_TEST_CASE(Contract_MultiDeviceLaunch_ExtMultiKernel_WritesPerDeviceValue) {
  RequireCooperativeMultiDeviceLaunch();

  ScopedDevice restore_device;
  std::vector<std::unique_ptr<DeviceLaunchTarget>> targets;
  AllocateTargets(targets);

  std::vector<hipLaunchParams> launch_params(kNumDevices);
  std::memset(launch_params.data(), 0, launch_params.size() * sizeof(hipLaunchParams));
  std::vector<std::array<void*, 2>> kernel_args(kNumDevices);
  for (int device = 0; device < kNumDevices; ++device) {
    kernel_args[device] = {targets[device]->device_ptr_address(),
                           targets[device]->expected_value_address()};
    launch_params[device].func = reinterpret_cast<void*>(MultiDeviceWriteValue);
    launch_params[device].gridDim = dim3(1);
    launch_params[device].blockDim = dim3(1);
    launch_params[device].sharedMem = 0;
    launch_params[device].stream = targets[device]->stream();
    launch_params[device].args = kernel_args[device].data();
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipExtLaunchMultiKernelMultiDevice(launch_params.data(), kNumDevices, 0));

  VerifyTargets(targets);
}

// hipModuleLaunchCooperativeKernelMultiDevice submits a module-resolved function per
// device through a hipFunctionLaunchParams array. Each device loads its own module
// instance (module handles are context-bound) and must observe its written value.
// @asserts: hipModuleLaunchCooperativeKernelMultiDevice - each device runs its module-resolved function and observes its own written value
HIP_TEST_CASE(Contract_MultiDeviceLaunch_ModuleCooperativeKernel_WritesPerDeviceValue) {
  RequireCooperativeMultiDeviceLaunch();

  // The single compiled code object is loaded on every participating device, so
  // require a homogeneous architecture set: a mixed-arch host would fail the
  // per-device module load even though cooperative multi-device launch is
  // supported.
  if (!ParticipatingDevicesShareArch()) {
    HIP_SKIP_TEST("Module-cooperative multi-device launch requires all devices to share an "
                  "architecture; this host is mixed-arch.");
  }

  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  ScopedDevice restore_device;
  std::vector<std::unique_ptr<DeviceLaunchTarget>> targets;
  AllocateTargets(targets);

  // Each loaded module is owned by a ScopedModule so a failing REQUIRE mid-loop
  // unloads the modules already loaded on earlier devices.
  std::vector<ScopedModule> modules;
  std::vector<hipFunctionLaunchParams> launch_params(kNumDevices);
  std::memset(launch_params.data(), 0, launch_params.size() * sizeof(hipFunctionLaunchParams));
  std::vector<std::array<void*, 2>> kernel_args(kNumDevices);
  for (int device = 0; device < kNumDevices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    hipModule_t module = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    modules.emplace_back(module, device);
    REQUIRE(module != nullptr);
    hipFunction_t function = nullptr;
    HIP_CHECK(hipModuleGetFunction(&function, module, "write_value"));
    REQUIRE(function != nullptr);

    kernel_args[device] = {targets[device]->device_ptr_address(),
                           targets[device]->expected_value_address()};
    launch_params[device].function = function;
    launch_params[device].gridDimX = 1;
    launch_params[device].gridDimY = 1;
    launch_params[device].gridDimZ = 1;
    launch_params[device].blockDimX = 1;
    launch_params[device].blockDimY = 1;
    launch_params[device].blockDimZ = 1;
    launch_params[device].sharedMemBytes = 0;
    launch_params[device].hStream = targets[device]->stream();
    launch_params[device].kernelParams = kernel_args[device].data();
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipModuleLaunchCooperativeKernelMultiDevice(launch_params.data(), kNumDevices, 0));

  VerifyTargets(targets);
}
#endif  // HT_AMD
