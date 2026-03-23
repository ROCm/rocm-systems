/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_features.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <cmath>

static constexpr auto kernel{
    R"(
extern "C"
__global__
void unsafeAdd_f(float *p, float v)
{
    auto val = unsafeAtomicAdd(p, v);
}

extern "C"
__global__
void unsafeAdd_d(double *p, double v)
{
    auto val = unsafeAtomicAdd(p, v);
}
)"};


HIP_TEST_CASE(Unit_unsafeAtomicAdd) {
  using namespace std;
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if (CheckIfFeatSupported(CTFeatures::CT_FEATURE_FINEGRAIN_HWSUPPORT, gfxName)) {
    hiprtcProgram prog;
    hiprtcCreateProgram(&prog,        // prog
                        kernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
    const char* options[] = {sarg.c_str()};
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
    if (logSize) {
      string log(logSize, '\0');
      HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
      INFO(log);
    }

    REQUIRE(compileResult == HIPRTC_SUCCESS);
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

    vector<char> code(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    float* fX;
    double* dX;
    HIP_CHECK(hipMalloc(&fX, sizeof(float)));
    HIP_CHECK(hipMalloc(&dX, sizeof(double)));

    hipModule_t module;
    hipFunction_t f_kernel, d_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "unsafeAdd_f"));
    HIP_CHECK(hipModuleGetFunction(&d_kernel, module, "unsafeAdd_d"));

    float f_val = 10.1f;
    double d_val = 10.1;

    HIP_CHECK(hipMemcpy(fX, &f_val, sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, &d_val, sizeof(double), hipMemcpyHostToDevice));

    struct {
      float* p;
      float val;
    } args_f{fX, f_val};

    struct {
      double* p;
      double val;
    } args_d{dX, d_val};

    auto size_f = sizeof(args_f);
    void* config_f[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                        &size_f, HIP_LAUNCH_PARAM_END};

    auto size_d = sizeof(args_d);
    void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_d, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                        &size_d, HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(f_kernel, 10, 1, 1, 100, 1, 1, 0, nullptr, nullptr, config_f));
    HIP_CHECK(hipModuleLaunchKernel(d_kernel, 10, 1, 1, 100, 1, 1, 0, nullptr, nullptr, config_d));

    float res_f = 0.0f;
    double res_d = 0.0;
    HIP_CHECK(hipMemcpy(&res_f, fX, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&res_d, dX, sizeof(double), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(fX));

    HIP_CHECK(hipModuleUnload(module));

    REQUIRE(fabs((res_f / 1000) - f_val) <= 0.2f);
    REQUIRE(fabs((res_d / 1000) - d_val) <= 0.2);
  }
}
