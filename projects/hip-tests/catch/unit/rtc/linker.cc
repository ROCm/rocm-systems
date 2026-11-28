#include <hip_test_common.hh>
#include <hip_test_filesystem.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>


#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

#pragma clang diagnostic ignored "-Wuninitialized"

static constexpr auto src{
    R"(
extern "C"
__global__
void saxpy(float a, float* x, float* y, float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
           out[tid] = a * x[tid] + y[tid];
    }
}
)"};

HIP_TEST_CASE(Unit_RTC_LinkerAPI_Negative) {
  SECTION("get bitcode - nullptr size and code") {
    hiprtcProgram program;
    REQUIRE(hiprtcGetBitcodeSize(program, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
    REQUIRE(hiprtcGetBitcode(program, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
  }

  SECTION("link create - nullptr image and input type") {
    hiprtcLinkState linkstate;
    REQUIRE(hiprtcLinkAddData(linkstate, HIPRTC_JIT_INPUT_LLVM_BITCODE, nullptr, 0, "code", 0,
                              nullptr, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
    REQUIRE(hiprtcLinkAddData(linkstate, HIPRTC_JIT_INPUT_CUBIN, &linkstate, 1, "code", 0, nullptr,
                              nullptr) == HIPRTC_ERROR_INVALID_INPUT);
  }

  SECTION("link complete - nullptr output and size") {
    hiprtcLinkState linkstate;
    REQUIRE(hiprtcLinkComplete(linkstate, nullptr, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
  }
}

HIP_TEST_CASE(Unit_RTC_LinkDestroy_Negative) {
  SECTION("link destroy - incorrect hiprtcLinkState") {
    hiprtcLinkState linkstate;
    REQUIRE(hiprtcLinkDestroy(linkstate) == HIPRTC_ERROR_INVALID_INPUT);
  }
}

HIP_TEST_CASE(Unit_RTC_LinkDestroy_Default) {
  SECTION("link destroy - nullptr") {
    REQUIRE(hiprtcLinkDestroy(nullptr) == HIPRTC_ERROR_INVALID_INPUT);
  }
}

std::vector<char> createBitcodeFromSource(const char* src, const char* name, int num_options,
                                          const char** options) {
  hiprtcProgram program;
  HIPRTC_CHECK(hiprtcCreateProgram(&program, src, name, 0, nullptr, nullptr));

  HIPRTC_CHECK(hiprtcCompileProgram(program, num_options, options));

  size_t codesize = 0;
  HIPRTC_CHECK(hiprtcGetBitcodeSize(program, &codesize));

  std::vector<char> code(codesize, '\0');
  HIPRTC_CHECK(hiprtcGetBitcode(program, &code[0]));

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));

  return code;
}

HIP_TEST_CASE(Unit_RTC_LinkerAPI) {
  const char* options[]{"-fgpu-rdc"};
  std::vector<char> code = createBitcodeFromSource(src, "saxpy", 1, options);

  const char* isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm", "-inlinehint-threshold=1"};
  std::vector<hiprtcJIT_option> jit_options = {HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
                                               HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
  size_t isaoptssize = 4;
  const void* lopts[] = {(void*)isaopts, (void*)(isaoptssize)};
  hiprtcLinkState linkstate;
  HIPRTC_CHECK(hiprtcLinkCreate(jit_options.size(), jit_options.data(), (void**)lopts, &linkstate));
  HIPRTC_CHECK(hiprtcLinkAddData(linkstate, HIPRTC_JIT_INPUT_LLVM_BITCODE, code.data(), code.size(),
                                 "LinkISA", 0, nullptr, nullptr));

  void* finaldata;
  size_t finalsize = 0;
  HIPRTC_CHECK(hiprtcLinkComplete(linkstate, &finaldata, &finalsize));

  size_t n = 128 * 32;
  size_t bufferSize = n * sizeof(float);

  float *dX, *dY, *dOut;
  HIP_CHECK(hipMalloc(&dX, bufferSize));
  HIP_CHECK(hipMalloc(&dY, bufferSize));
  HIP_CHECK(hipMalloc(&dOut, bufferSize));

  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, finaldata));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "saxpy"));

  float a = 5.1f;
  std::unique_ptr<float[]> hX{new float[n]};
  std::unique_ptr<float[]> hY{new float[n]};
  std::unique_ptr<float[]> hOut{new float[n]};
  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }

  HIP_CHECK(hipMemcpy(dX, hX.get(), bufferSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dY, hY.get(), bufferSize, hipMemcpyHostToDevice));

  struct {
    float a_;
    float* b_;
    float* c_;
    float* d_;
    size_t e_;
  } args{a, dX, dY, dOut, n};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel, 32, 1, 1, 128, 1, 1, 0, nullptr, nullptr, config));

  HIP_CHECK(hipMemcpy(hOut.get(), dOut, bufferSize, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(dX));
  HIP_CHECK(hipFree(dY));
  HIP_CHECK(hipFree(dOut));

  HIPRTC_CHECK(hiprtcLinkDestroy(linkstate));
  HIP_CHECK(hipModuleUnload(module));

  for (size_t i = 0; i < n; ++i) {
    REQUIRE(fabs(a * hX[i] + hY[i] - hOut[i]) <= fabs(hOut[i]) * 1e-6);
  }
}

HIP_TEST_CASE(Unit_RTC_LinkAddFile_Negative) {
  static constexpr hiprtcJITInputType input_type = HIPRTC_JIT_INPUT_LLVM_BITCODE;
  static constexpr const char* file_name = "bitcode_file";

  SECTION("link add file - incorrect hiprtcLinkState") {
    hiprtcLinkState linkstate;
    REQUIRE(hiprtcLinkAddFile(linkstate, input_type, file_name, 0, nullptr, nullptr) ==
            HIPRTC_ERROR_INVALID_INPUT);
  }

  SECTION("link add file - nullptr hiprtcLinkState") {
    REQUIRE(hiprtcLinkAddFile(nullptr, input_type, file_name, 0, nullptr, nullptr) ==
            HIPRTC_ERROR_INVALID_INPUT);
  }

  SECTION("link add file - incorrect input type") {
    hiprtcLinkState linkstate{};
    HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &linkstate));
    hiprtcJITInputType incorrect_input_type = HIPRTC_JIT_INPUT_NVVM;
    REQUIRE(hiprtcLinkAddFile(linkstate, incorrect_input_type, file_name, 0, nullptr, nullptr) ==
            HIPRTC_ERROR_INVALID_INPUT);
    HIPRTC_CHECK(hiprtcLinkDestroy(linkstate));
  }

  SECTION("link add file - file does not exists") {
    hiprtcLinkState linkstate{};
    HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &linkstate));
    REQUIRE(hiprtcLinkAddFile(linkstate, input_type, file_name, 0, nullptr, nullptr) ==
            HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);
    HIPRTC_CHECK(hiprtcLinkDestroy(linkstate));
  }
}

HIP_TEST_CASE(Unit_RTC_LinkAddFile_Default) {
  // Create bitcode and save it to file
  const char* options[]{"-fgpu-rdc"};
  std::vector<char> code = createBitcodeFromSource(src, "saxpy", 1, options);
  static constexpr const char* file_name = "bitcode_file";
  std::ofstream file(file_name, std::ios::binary);
  REQUIRE(file.is_open());
  file.write(code.data(), code.size());
  file.close();

  // Create link with options
  const char* isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm", "-inlinehint-threshold=1"};
  std::vector<hiprtcJIT_option> jit_options = {HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
                                               HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
  size_t isaoptssize = 4;
  const void* lopts[] = {(void*)isaopts, (void*)(isaoptssize)};
  hiprtcLinkState linkstate;
  HIPRTC_CHECK(hiprtcLinkCreate(jit_options.size(), jit_options.data(), (void**)lopts, &linkstate));
  REQUIRE(hiprtcLinkAddFile(linkstate, HIPRTC_JIT_INPUT_LLVM_BITCODE, file_name, 0, nullptr,
                            nullptr) == HIPRTC_SUCCESS);

  // Cleanup
  HIPRTC_CHECK(hiprtcLinkDestroy(linkstate));
  if (fs::exists(file_name)) {
    REQUIRE(fs::remove(file_name));
  }
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Links an LLVM bitcode bundle (HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE) built at run time: the
 *    -fgpu-rdc bitcode of a kernel wrapped in a clang offload bundle with a host entry and an
 *    entry for the device's target. The runtime must unbundle the device entry, so the linked
 *    code object holds the kernel and the kernel runs.
 * Test source
 * ------------------------
 *  - unit/rtc/linker.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_RTC_LinkAddData_BundledBitcode) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string arch = prop.gcnArchName;

  static constexpr const char* kSource = R"(extern "C" __global__ void k(int* a) { a[0] = 4242; })";
  hiprtcProgram program;
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kSource, "bundled.cu", 0, nullptr, nullptr));
  const std::string archOption = "--offload-arch=" + arch;
  const char* options[] = {"-fgpu-rdc", archOption.c_str()};
  HIPRTC_CHECK(hiprtcCompileProgram(program, 2, options));
  size_t bitcodeSize = 0;
  HIPRTC_CHECK(hiprtcGetBitcodeSize(program, &bitcodeSize));
  std::vector<char> bitcode(bitcodeSize);
  HIPRTC_CHECK(hiprtcGetBitcode(program, bitcode.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&program));

  // Clang offload bundle: magic, entry count, {offset, size, id size, id} per entry, payloads.
  const std::vector<std::string> ids = {"host-x86_64-unknown-linux-gnu-",
                                        "hip-amdgcn-amd-amdhsa--" + arch};
  auto put64 = [](std::vector<char>& out, uint64_t value) {
    const char* bytes = reinterpret_cast<const char*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
  };
  static constexpr char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  std::vector<char> bundle(kMagic, kMagic + sizeof(kMagic) - 1);
  put64(bundle, ids.size());
  size_t headerSize = bundle.size();
  for (const auto& id : ids) headerSize += 3 * sizeof(uint64_t) + id.size();
  const size_t payloadOffset = (headerSize + 4095) & ~size_t{4095};
  const uint64_t sizes[] = {0, bitcode.size()};
  for (size_t i = 0; i < ids.size(); ++i) {
    put64(bundle, payloadOffset);
    put64(bundle, sizes[i]);
    put64(bundle, ids[i].size());
    bundle.insert(bundle.end(), ids[i].begin(), ids[i].end());
  }
  bundle.resize(payloadOffset, 0);
  bundle.insert(bundle.end(), bitcode.begin(), bitcode.end());

  hiprtcLinkState linkState;
  HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &linkState));
  HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE, bundle.data(),
                                 bundle.size(), "bundle", 0, nullptr, nullptr));
  void* code = nullptr;
  size_t codeSize = 0;
  HIPRTC_CHECK(hiprtcLinkComplete(linkState, &code, &codeSize));

  hipModule_t module;
  HIP_CHECK(hipModuleLoadData(&module, code));
  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, "k"));
  int* result_d = nullptr;
  HIP_CHECK(hipMalloc(&result_d, sizeof(int)));
  void* args[] = {&result_d};
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  int result = 0;
  HIP_CHECK(hipMemcpy(&result, result_d, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(result == 4242);

  HIP_CHECK(hipFree(result_d));
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcLinkDestroy(linkState));
}
#endif
