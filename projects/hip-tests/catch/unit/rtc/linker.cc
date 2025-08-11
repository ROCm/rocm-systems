#include <hip_test_common.hh>
#include <hip_test_filesystem.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>


#include <cassert>
#include <cstddef>
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

TEST_CASE("Unit_RTC_LinkerAPI_Negative") {
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

TEST_CASE("Unit_RTC_LinkDestroy_Negative") {
  SECTION("link destroy - incorrect hiprtcLinkState") {
    hiprtcLinkState linkstate;
    REQUIRE(hiprtcLinkDestroy(linkstate) == HIPRTC_ERROR_INVALID_INPUT);
  }
}

TEST_CASE("Unit_RTC_LinkDestroy_Default") {
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

TEST_CASE("Unit_RTC_LinkerAPI") {
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

TEST_CASE("Unit_RTC_LinkAddFile_Negative") {
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

TEST_CASE("Unit_RTC_LinkAddFile_Default") {
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
