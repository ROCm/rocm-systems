/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include <hip_test_common.hh>

#include <vector>
#include <string>

#include <hip/hiprtc.h>
#include <hip/hip_fp6.h>

// FP6 has two OCP encodings: E2M3 and E3M2.
// Inputs below are exactly representable in BOTH encodings (and also in
// FP4 E2M1, in case the test is reused as a reference). This eliminates
// rounding ambiguity, so the round-trip is equal regardless of whether the
// kernel takes the gfx950 hardware path or the fcbx::* software fallback.
TEST_CASE("Unit_hiprtc_fp6_simple") {
  constexpr const char* source = R"(
extern "C" __global__ void float_to_fp6_to_float(float* out, float* in, bool e2m3, size_t size) {
  size_t i = threadIdx.x;
  if (i < size) {
    if (e2m3) {
      __hip_fp6_e2m3 tmp = __hip_fp6_e2m3(in[i]);
      out[i] = float(tmp);
    } else {
      __hip_fp6_e3m2 tmp = __hip_fp6_e3m2(in[i]);
      out[i] = float(tmp);
    }
  }
}
)";

  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, source, "fp6.cu", 0, nullptr, nullptr));
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
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);

  size_t codeSize{};
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  std::vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  // Values exactly representable in both FP6 E2M3 and E3M2.
  std::vector<float> in = {0.0f,  0.5f, -0.5f, 1.0f, -1.0f,
                           1.5f, -1.5f,  2.0f, 3.0f,  4.0f};
  const size_t size = in.size();

  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, size * sizeof(float)));

  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "float_to_fp6_to_float"));

  HIP_CHECK(hipMemcpy(d_in, in.data(), size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, size * sizeof(float)));

  struct {
    float* out;
    float* in;
    bool e2m3;
    size_t size;
  } args{d_out, d_in, true, size};

  auto arg_size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                    HIP_LAUNCH_PARAM_END};

  // ---- E2M3 ----
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, size, 1, 1, 0, nullptr, nullptr, config));

  std::vector<float> out(size, 0.0f);
  HIP_CHECK(hipMemcpy(out.data(), d_out, size * sizeof(float), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < size; i++) {
    __hip_fp6_e2m3 tmp = __hip_fp6_e2m3(in[i]);
    float cpu_out = float(tmp);
    INFO("E2M3 Index: " << i << " in: " << in[i] << " GPU: " << out[i] << " cpu: " << cpu_out);
    REQUIRE(cpu_out == out[i]);
  }

  // ---- E3M2 ----
  args.e2m3 = false;
  HIP_CHECK(hipMemset(d_out, 0, size * sizeof(float)));
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, size, 1, 1, 0, nullptr, nullptr, config));

  HIP_CHECK(hipMemcpy(out.data(), d_out, size * sizeof(float), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < size; i++) {
    __hip_fp6_e3m2 tmp = __hip_fp6_e3m2(in[i]);
    float cpu_out = float(tmp);
    INFO("E3M2 Index: " << i << " in: " << in[i] << " GPU: " << out[i] << " cpu: " << cpu_out);
    REQUIRE(cpu_out == out[i]);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));

  HIP_CHECK(hipModuleUnload(module));
}
