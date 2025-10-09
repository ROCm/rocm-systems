#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

static constexpr auto managed_code{
    R"(
extern "C" __managed__ int mid[32];
extern "C"
__global__
void saxpy(float* out)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = 2.0f * tid;
    mid[tid] = out[tid];
}
)"};

TEST_CASE("Unit_hiprtc_managed_varaible") {
  using namespace std;
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog,              // prog
                      managed_code,       // buffer
                      "managed_code.cu",  // name
                      0, nullptr, nullptr);
  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
#else
  std::string sarg = std::string("--fmad=false");
#endif
  const char* options[] = {sarg.c_str()};
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  // Do hip malloc first so that we donot need to do a cuInit manually before calling hipModule APIs
  size_t n = 32;
  size_t bufferSize = n * sizeof(float);

  float *dOut;
  HIP_CHECK(hipMalloc(&dOut, bufferSize));

  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "managed_code"));
  hipDeviceptr_t p = nullptr;
  size_t p_s = 0;
  HIP_CHECK(hipModuleGetGlobal(&p, &p_s, module, "mid"));

  struct {
    float* out;
  } args{dOut};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, nullptr,
                                  config));

  std::vector<float> hOut(32, 0.0f);
  HIP_CHECK(hipMemcpy(hOut.data(), dOut, bufferSize, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(dOut));

  HIP_CHECK(hipModuleUnload(module));

  for (size_t i = 0; i < n; ++i) {
    //
  }
}