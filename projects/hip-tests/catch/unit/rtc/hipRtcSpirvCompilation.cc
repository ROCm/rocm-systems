#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>

static constexpr auto globalfunc{
    R"(
extern "C" __device__ int load_callback(const int* ptr, size_t offset, void* cbdata, void* lds);
extern "C" __device__ void
    store_callback(int* ptr, size_t offset, int data, void* cbdata, void* lds);
extern "C" __global__ void kernelfunc_int(const int* ptr_in, int* ptr_out)
{
    auto val = load_callback(ptr_in, threadIdx.x, nullptr, nullptr);
    store_callback(ptr_out, threadIdx.x, val, nullptr, nullptr);
}
)"};

static constexpr auto devicefunc{
    R"(
extern "C" __device__ int load_callback(const int* ptr, size_t offset, void* cbdata, void* lds)
{
    return ptr[offset] * 2;
}
extern "C" __device__ void store_callback(int* ptr, size_t offset, int data, void* cbdata, void* lds)
{
    ptr[offset] = data + 1;
}
)"};


static constexpr auto testfunc{
    R"(
__forceinline__ __device__ float f() { return 123.4f; }
extern "C"
__global__ void testinline()
{
 f();
}
)"};

std::vector<char> compile_prog(const char* src) {
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, src, nullptr, 0, nullptr, nullptr));
  
  std::vector<const char *> options;
  options.push_back("-x");
  options.push_back("hip");
  options.push_back("--offload-arch=amdgcnspirv");

  hiprtcResult compileResult{
      hiprtcCompileProgram(prog, static_cast<int>(options.size()), options.data())};

  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    INFO(log);
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  
  std::vector<char> code(codeSize);
  if (codeSize > 0) {
    HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  } else {
    HIPRTC_CHECK(hiprtcGetBitcodeSize(prog, &codeSize));
    REQUIRE(codeSize > 0);
    code.resize(codeSize);
    HIPRTC_CHECK(hiprtcGetBitcode(prog, code.data()));
  }
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return code;
}

void* link_prog(hipLinkState_t* state,const std::vector<char>& global_obj, const std::vector<char>& device_obj) {
  HIP_CHECK(hipLinkCreate(0, nullptr, nullptr, state));

  if (global_obj.size() > 0) {
  HIP_CHECK(hipLinkAddData(*state, hipJitInputSpirv,
                           const_cast<void*>(static_cast<const void*>(global_obj.data())),
                           global_obj.size(), "globalfunc.spv", 0, nullptr,
                           nullptr));
  }

  if (device_obj.size() > 0) {
  HIP_CHECK(hipLinkAddData(*state, hipJitInputSpirv,
                           const_cast<void*>(static_cast<const void*>(device_obj.data())),
                           device_obj.size(), "devicefunc.spv", 0, nullptr,
                           nullptr));
  }

  void *bin = nullptr;
  size_t binSize = 0;
  HIP_CHECK(hipLinkComplete(*state, &bin, &binSize));
  REQUIRE(bin != nullptr);

  return bin;
}

HIP_TEST_CASE(Unit_hiprtc_spirv_compilation) {
  std::vector<char> code = compile_prog(testfunc);
  hipLinkState_t state;
  void* bin = link_prog(&state, code, {});

  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, bin));
  HIP_CHECK(hipModuleGetFunction(&function, module, "testinline"));

  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, 0, nullptr, 0));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipModuleUnload(module));

  HIP_CHECK(hipLinkDestroy(state));
}

HIP_TEST_CASE(Unit_hiprtc_spirv_linker) {
  std::vector<char> globalcode = compile_prog(globalfunc);
  std::vector<char> devicecode = compile_prog(devicefunc);
  hipLinkState_t state;

  void* bin = link_prog(&state, globalcode, devicecode);
  
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, bin));

  HIP_CHECK(hipModuleGetFunction(&kernel, module, "kernelfunc_int"));

  // allocate input and output buffers
  static const size_t N = 10;
  std::vector<int> in_host(N);
  std::iota(in_host.begin(), in_host.end(), 1);
  std::vector<int> out_host(N);
  std::fill(out_host.begin(), out_host.end(), 0);

  int *in_device = nullptr;
  HIP_CHECK(hipMalloc(&in_device, N * sizeof(int)));
  int *out_device = nullptr;
  HIP_CHECK(hipMalloc(&out_device, N * sizeof(int)));

  HIP_CHECK(hipMemcpy(in_device, in_host.data(), N * sizeof(int),
                hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(out_device, out_host.data(), N * sizeof(int),
                hipMemcpyHostToDevice));
  
  std::vector<void *> kargs(2);
  kargs[0] = in_device;
  kargs[1] = out_device;

  auto size = kargs.size() * sizeof(void *);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, N, 1, 1, 0, nullptr, nullptr, config));

  HIP_CHECK(hipMemcpy(out_host.data(), out_device, N * sizeof(int), hipMemcpyDeviceToHost));

  for(size_t i = 0; i < N; i++) {
    REQUIRE(out_host[i]  == in_host[i] * 2 + 1);
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(in_device));
  HIP_CHECK(hipFree(out_device));
  HIP_CHECK(hipLinkDestroy(state));
}