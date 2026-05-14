/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip_helper.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

static constexpr auto NUM_THREADS{128};
static constexpr auto NUM_BLOCKS{32};

static constexpr auto saxpy{
    R"(
#include "test_header.h"
#include "test_header1.h"
extern "C"
__global__
void saxpy(real a, realptr x, realptr y, realptr out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
       out[tid] = a * x[tid] + y[tid] ;
    }
}
)"};

int main() {
  using namespace std;

  hiprtcProgram prog;
  int num_headers = 2;
  vector<const char*> header_names;
  vector<const char*> header_sources;
  header_names.push_back("test_header.h");
  header_names.push_back("test_header1.h");
  header_sources.push_back(
      "#ifndef HIPRTC_TEST_HEADER_H\n#define HIPRTC_TEST_HEADER_H\ntypedef float real;\n#endif "
      "//HIPRTC_TEST_HEADER_H\n");
  header_sources.push_back(
      "#ifndef HIPRTC_TEST_HEADER1_H\n#define HIPRTC_TEST_HEADER1_H\ntypedef float* "
      "realptr;\n#endif //HIPRTC_TEST_HEADER1_H\n");
  hiprtcCreateProgram(&prog,               // prog
                      saxpy,               // buffer
                      "saxpy.cu",          // name
                      num_headers,         // numHeaders
                      &header_sources[0],  // headers
                      &header_names[0]);   // includeNames

  hipDeviceProp_t props;
  int device = 0;
  checkHipErrors(hipGetDeviceProperties(&props, device));

  const char* options[] = {};

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 0, options)};

  size_t logSize;
  hiprtcGetProgramLogSize(prog, &logSize);

  if (logSize) {
    string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);

    cout << log << '\n';
  }

  if (compileResult != HIPRTC_SUCCESS) {
    cout << "Compilation failed." << endl;
  }

  size_t codeSize;
  hiprtcGetCodeSize(prog, &codeSize);

  vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());

  hiprtcDestroyProgram(&prog);

  hipModule_t module;
  hipFunction_t kernel;

  checkHipErrors(hipModuleLoadData(&module, code.data()));
  checkHipErrors(hipModuleGetFunction(&kernel, module, "saxpy"));

  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);

  float a = 5.1f;
  unique_ptr<float[]> hX{new float[n]};
  unique_ptr<float[]> hY{new float[n]};
  unique_ptr<float[]> hOut{new float[n]};

  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }

  hipDeviceptr_t dX, dY, dOut;
  checkHipErrors(hipMalloc((void**)&dX, bufferSize));
  checkHipErrors(hipMalloc((void**)&dY, bufferSize));
  checkHipErrors(hipMalloc((void**)&dOut, bufferSize));
  checkHipErrors(hipMemcpyHtoD(dX, hX.get(), bufferSize));
  checkHipErrors(hipMemcpyHtoD(dY, hY.get(), bufferSize));

  struct {
    float a_;
    hipDeviceptr_t b_;
    hipDeviceptr_t c_;
    hipDeviceptr_t d_;
    size_t e_;
  } args{a, dX, dY, dOut, n};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  checkHipErrors(hipModuleLaunchKernel(kernel, NUM_BLOCKS, 1, 1, NUM_THREADS, 1, 1, 0, nullptr,
                                       nullptr, config));
  checkHipErrors(hipMemcpyDtoH(hOut.get(), dOut, bufferSize));

  for (size_t i = 0; i < n; ++i) {
    if (fabs(a * hX[i] + hY[i] - hOut[i]) > fabs(hOut[i]) * 1e-6) {
      cout << "Validation failed." << endl;
    }
  }

  checkHipErrors(hipFree((void*)dX));
  checkHipErrors(hipFree((void*)dY));
  checkHipErrors(hipFree((void*)dOut));

  checkHipErrors(hipModuleUnload(module));

  cout << "SAXPY test completed" << endl;
}
