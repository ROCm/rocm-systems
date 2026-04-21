/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <math.h>
#include <vector>

static constexpr auto kernel_src{
    R"_KERN_EMBED_(
    extern "C" __global__ void kernel_func(float* f)
    {
      f[0] = 1.0;
    }
  )_KERN_EMBED_"};


HIP_TEST_CASE(Unit_hipStreamCaptureRtc) {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  float data_h = 0.0;
  float* data_d = nullptr;

  // Init data
  HIPCHECK(hipMalloc(&data_d, sizeof(float)));
  HIPCHECK(hipMemcpy(data_d, &data_h, sizeof(float), hipMemcpyHostToDevice));

  // Compile kernel
  std::vector<char> code;
  hiprtcProgram prog;
  HIPRTC_CHECK(
      hiprtcCreateProgram(&prog, kernel_src, "hipStreamCaptureRtc.cu", 0, nullptr, nullptr));

  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
#else
  std::string sarg = std::string("--fmad=false");
#endif

  std::vector<const char*> options = {sarg.c_str()};

  auto compileResult = hiprtcCompileProgram(prog, options.size(), options.data());
  if (compileResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    if (logSize) {
      std::vector<char> log(logSize, '\0');
      if (hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS) {
        FAIL("hiprtcCompileProgram failed with log" << log.data());
        return;
      }
    }
    FAIL("hiprtcCompileProgram failed without log");
    return;
  }

  size_t codeSize = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  code.resize(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
#if HT_NVIDIA
  HIPCHECK(hipInit(0));
  hipCtx_t ctx;
  HIPCHECK(hipCtxCreate(&ctx, 0, device));
#endif

  HIPCHECK(hipModuleLoadData(&module, code.data()));

  HIPCHECK(hipModuleGetFunction(&kernel, module, "kernel_func"));

  // Start capture
  HIPCHECK(hipStreamCreate(&stream));
  HIPCHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  // Launch kernel
  auto size = sizeof(float*);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &data_d, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIPCHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, config));
  HIPCHECK(hipStreamEndCapture(stream, &graph));

  size_t numNodes = 0;
  HIPCHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  INFO("Num of nodes returned by GetNodes : " << numNodes);
  REQUIRE(numNodes == 1);

  // Ensure that no actual work has been done for the captured
  // stream before graph execution
  float tmp = 2.0;
  HIPCHECK(hipMemcpy(&tmp, data_d, sizeof(float), hipMemcpyDeviceToHost));
  REQUIRE(tmp == 0.0);

  HIPCHECK(hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
  HIPCHECK(hipGraphDestroy(graph));

  HIPCHECK(hipGraphLaunch(graph_exec, stream));

  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipGraphExecDestroy(graph_exec));
  HIPCHECK(hipStreamDestroy(stream));

  // Check that the work was done
  HIPCHECK(hipMemcpy(&tmp, data_d, sizeof(float), hipMemcpyDeviceToHost));
  HIPCHECK(hipFree(data_d));
  HIP_CHECK(hipModuleUnload(module));

  REQUIRE(tmp == 1.0);
#if HT_NVIDIA
  HIPCHECK(hipCtxDestroy(ctx));
#endif
}
