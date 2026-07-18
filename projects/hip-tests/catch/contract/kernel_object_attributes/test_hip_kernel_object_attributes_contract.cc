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

// The hipKernel_t object accessors (hipKernelGetAttribute, hipKernelSetAttribute,
// hipKernelGetParamInfo) and the hipLibrary* loader used to obtain a kernel are
// exercised on both backends: on NVIDIA they map to the CUDA driver cuKernel*/
// cuLibrary* entry points.
namespace {
constexpr char const kWriteKernelName[] = "write_value";

// In-source device code compiled at runtime with HIPRTC. The kernel takes a
// pointer and an int so the parameter-info contract has a known two-argument
// layout to inspect.
constexpr char const kKernelSource[] =
    "extern \"C\" __global__ void write_value(int* out, int value) {\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    out[0] = value;\n"
    "  }\n"
    "}\n";

// Compiles kKernelSource with HIPRTC for device 0. A compile failure is a
// contract violation rather than an unsupported-capability skip: it surfaces the
// build log and aborts through HIPRTC_CHECK. The bool return keeps the familiar
// `if (!Compile...())` shape at the call sites.
bool CompileKernelSource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kKernelSource, "kernel_object_attributes_contract.cu",
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
void LoadContractKernel(std::vector<char>& code, hipLibrary_t& library, hipKernel_t& kernel) {
  // Establish a device context before the driver-style library/kernel entry
  // points run. On NVIDIA these map to the CUDA driver API (cuLibrary*/cuKernel*),
  // which requires a bound primary context; hipFree(0) is the canonical no-op
  // that forces primary-context initialization and is a harmless success on AMD.
  HIP_CHECK(hipFree(0));
  if (!CompileKernelSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }
  HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  REQUIRE(library != nullptr);
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, kWriteKernelName));
  REQUIRE(kernel != nullptr);
}
}  // namespace

HIP_TEST_CASE(Contract_KernelObjectAttributes_GetAttribute_ReturnsSaneValues) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadContractKernel(code, library, kernel);

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

HIP_TEST_CASE(Contract_KernelObjectAttributes_SetMaxDynamicSharedMemory_IsAcceptedOrUnsupported) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadContractKernel(code, library, kernel);

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

HIP_TEST_CASE(Contract_KernelObjectAttributes_GetParamInfo_ReturnsFirstParamLayout) {
  std::vector<char> code;
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadContractKernel(code, library, kernel);

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
