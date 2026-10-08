/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <string>
#include <vector>

// The hipKernel_t object accessors (hipKernelGetAttribute, hipKernelSetAttribute,
// hipKernelGetParamInfo) and the hipLibrary* loader used to obtain a kernel are
// exercised on both backends: on NVIDIA they map to the CUDA driver cuKernel*/
// cuLibrary* entry points.
namespace {
constexpr char const kWriteValueKernelName[] = "write_value";

// In-source device code compiled at runtime with HIPRTC. The kernel takes a
// pointer and an int so the parameter-info contract has a known two-argument
// layout to inspect.
constexpr char const kWriteValueKernelSource[] =
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    out[0] = value;\n"
    "  }\n"
    "}\n";

constexpr char const kUseMaxSharedMemoryKernelName[] = "use_max_shared_memory";
constexpr char const kUseMaxSharedMemoryKernelSource[] =
R"%(
extern "C" __global__ void use_max_shared_memory(int* result, int n)
{
  extern __shared__ int myArray[];

  unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < n) {
    myArray[pos] = pos;
  }

  __syncthreads();

  if (pos < n) {
    atomicAdd(result, pos);
  }
};
)%";

// Compiles kKernelSource with HIPRTC for device 0. A compile failure is a
// contract violation rather than an unsupported-capability skip: it surfaces the
// build log and aborts through HIPRTC_CHECK. The bool return keeps the familiar
// `if (!Compile...())` shape at the call sites.
bool CompileKernelSource(std::vector<char>& code, const char* source) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, source, "kernel_object_attributes_contract.cu",
                                   0, nullptr, nullptr));

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

// Compiles the kernel source and loads it as a library, resolving the known
// kernel into `kernel`. Skips when HIPRTC is unavailable. `code` must stay alive
// only until the library is loaded; callers keep it for simplicity.
void LoadContractKernel(std::vector<char>& code,
                        hipLibrary_t& library,
                        hipKernel_t& kernel,
                        const char* kernelName,
                        const char* kernelSource) {
  // Establish a device context before the driver-style library/kernel entry
  // points run. On NVIDIA these map to the CUDA driver API (cuLibrary*/cuKernel*),
  // which requires a bound primary context; hipFree(0) is the canonical no-op
  // that forces primary-context initialization and is a harmless success on AMD.
  HIP_CHECK(hipFree(0));
  if (!CompileKernelSource(code, kernelSource)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }
  HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  REQUIRE(library != nullptr);
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kernelName));
  REQUIRE(kernel != nullptr);
}
}  // namespace

// @asserts: hipKernelGetAttribute - reports a positive max-threads-per-block and non-negative static resource usage for a loaded kernel
HIP_TEST_CASE(Contract_KernelObjectAttributes_HipKernelGetAttribute_Default_ReturnsSaneValues) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadContractKernel(code, library, kernel, kWriteValueKernelName, kWriteValueKernelSource);

  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));

  // The maximum thread count reported for a launchable kernel must be positive,
  // and its static resource usages (register and shared-memory counts) must be
  // non-negative. Exact values are device-specific, so only structural validity
  // is asserted.
  int max_threads_per_block = 0;
  HIP_CHECK(hipKernelGetAttribute(&max_threads_per_block, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  kernel, device));
  REQUIRE(max_threads_per_block > 0);

  int num_registers = -1;
  HIP_CHECK(hipKernelGetAttribute(&num_registers, HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel, device));
  REQUIRE(num_registers >= 0);

  HIP_CHECK(hipLibraryUnload(library));
}

// @asserts: hipKernelSetAttribute - setting max dynamic shared memory to zero is either accepted or reported as unsupported, never another error
HIP_TEST_CASE(Contract_KernelObjectAttributes_HipKernelSetAttribute_SetMaxDynamicSharedMemory_IsAcceptedOrUnsupported) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadContractKernel(code, library, kernel, kWriteValueKernelName, kWriteValueKernelSource);

  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));

  // Setting the max dynamic shared memory to zero is a benign, portable request:
  // the runtime must either accept it or report that the attribute cannot be set
  // on this path. Any other failure is a contract violation.
  const hipError_t status = hipKernelSetAttribute(
      HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0, kernel, device);
  if (status != hipSuccess && status != hipErrorNotSupported) {
    HIP_CHECK(status);
  }

  HIP_CHECK(hipLibraryUnload(library));
}

#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 12080)
// @asserts: hipKernelSetAttributeForDevice - preserves current device and target value
HIP_TEST_CASE(
    Contract_KernelObjectAttributes_HipKernelSetAttributeForDevice_NonCurrentDevice_PreservesCurrentAndReadsBack) {
  int device_count = 0;
  int original_device = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  HIP_CHECK(hipGetDevice(&original_device));
  if (device_count < 2) {
    HIP_SKIP_TEST("The non-current-device contract requires at least two devices.");
  }

  // Load the kernel for device 0, then make device 1 current so
  // the explicit target differs from the calling thread's current device.
  HIP_CHECK(hipSetDevice(0));
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadContractKernel(code, library, kernel, kWriteValueKernelName, kWriteValueKernelSource);

  constexpr int current_device_for_call = 1;
  HIP_CHECK(hipSetDevice(current_device_for_call));

  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0));

  int current_device_after = -1;
  HIP_CHECK(hipGetDevice(&current_device_after));
  REQUIRE(current_device_after == current_device_for_call);

  int observed = -1;
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  REQUIRE(observed == 0);

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
  HIP_CHECK(hipSetDevice(original_device));
}
#else
// @asserts: hipKernelSetAttributeForDevice - unavailable before CUDA 12.8
HIP_TEST_CASE(
    Contract_KernelObjectAttributes_HipKernelSetAttributeForDevice_NvidiaPre128Unsupported_IsSkipped) {
  HIP_SKIP_TEST("hipKernelSetAttributeForDevice requires CUDA 12.8 or newer.");
}
#endif

#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
// @asserts: hipKernelSetCacheConfig - all hipFuncCache_t values are supported and kernel
//           launches execute correctly, regardless of the carveout (which is a hint and the kernel
//           must work even if the user choose a carveout that does not provide enough shared memory).
//           hiprtc launches too.
HIP_TEST_CASE(Contract_KernelObjectAttributes_HipKernelSetCacheConfig_Values_AreSupported)
{
  std::vector<char> code;
  hipFuncCache_t carveouts[] = { hipFuncCachePreferNone,
                                 hipFuncCachePreferShared,
                                 hipFuncCachePreferL1,
                                 hipFuncCachePreferEqual };
  dim3 gridDim = {};
  dim3 blockDim = { 128 };
  int maxSharedMemPerBlock = 0;
  int currentDevice = -1;
  hipDevice_t device;
  int h_result;
  LinearAllocGuard<int> d_result(LinearAllocs::hipMalloc, sizeof(h_result));
  int n;
  hipKernel_t kernel;
  hipLibrary_t library;
  hipFunction_t useMaxSharedMemoryFunc;
  int* devicePtr = d_result.ptr();
  void* args[] = { &devicePtr, &n };

  HIP_CHECK(hipGetDevice(&currentDevice));
  HIP_CHECK(hipDeviceGet(&device, currentDevice));
  HIP_CHECK(hipDeviceGetAttribute(&maxSharedMemPerBlock,
                                  hipDeviceAttributeMaxSharedMemoryPerBlock,
                                  currentDevice));
  n = maxSharedMemPerBlock / sizeof(int);
  gridDim.x = (n + blockDim.x - 1) / blockDim.x;
  LoadContractKernel(code,
                     library,
                     kernel,
                     kUseMaxSharedMemoryKernelName,
                     kUseMaxSharedMemoryKernelSource);
  HIP_CHECK(hipKernelGetFunction(&useMaxSharedMemoryFunc, kernel));

  for (const auto& carveout : carveouts) {
    // launch kernel
    HIP_CHECK(hipMemset(d_result.ptr(), 0, d_result.size_bytes()));
    HIP_CHECK(hipKernelSetCacheConfig(kernel, carveout, device));
    HIP_CHECK(hipModuleLaunchKernel(useMaxSharedMemoryFunc,
                       gridDim.x, gridDim.y, gridDim.z,
                       blockDim.x, blockDim.y, blockDim.z,
                       maxSharedMemPerBlock, 0,
                       args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&h_result, d_result.ptr(), sizeof(h_result), hipMemcpyDeviceToHost));
    REQUIRE(h_result == ((n * (n - 1)) / 2));
  }

  HIP_CHECK(hipLibraryUnload(library));
}
#else
// @asserts: hipKernelSetAttributeForDevice - unavailable before CUDA 12.8
HIP_TEST_CASE(Contract_KernelObjectAttributes_HipKernelSetCacheConfig_Values_AreSupported)
{
   HIP_SKIP_TEST("hipKernelSetCacheConfig requires CUDA 12.0 or later")
}
#endif

// @asserts: hipKernelGetParamInfo - reports the first parameter at offset zero with size at least that of a device pointer
HIP_TEST_CASE(Contract_KernelObjectAttributes_HipKernelGetParamInfo_Default_ReturnsFirstParamLayout) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;

  LoadContractKernel(code, library, kernel, kWriteValueKernelName, kWriteValueKernelSource);

  // The first parameter of write_value is an int* pointer. Its reported offset
  // must be zero (first on the argument stack) and its size must be large enough
  // to hold a device pointer. Exact size is ABI-specific, so a lower bound is
  // used.
  size_t param_offset = static_cast<size_t>(-1);
  size_t param_size = 0;
  HIP_CHECK(hipKernelGetParamInfo(kernel, 0, &param_offset, &param_size));

  REQUIRE(param_offset == 0);
  REQUIRE(param_size >= sizeof(void*));

  HIP_CHECK(hipLibraryUnload(library));
}
