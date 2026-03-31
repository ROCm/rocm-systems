/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cstdlib>

// Helper macro for HIPRTC error checking
#define CHECK_HIPRTC(cmd) \
  do { \
    hiprtcResult result = cmd; \
    if (result != HIPRTC_SUCCESS) { \
      fprintf(stderr, "error: %s (%d) at %s:%d\n", \
              hiprtcGetErrorString(result), result, __FILE__, __LINE__); \
      abort(); \
    } \
  } while(0)

// Kernel source that defines ROCSHMEM_CTX_DEFAULT
// This kernel mimics the rocshmem device context definition to ensure 
// the device global symbol exists in the module's device symbol table and 
// can be queried later in rocshmem_hipmodule_init() host API for verification.
const char* test_kernel_src = R"(
#include <hip/hip_runtime.h>

typedef struct rocshmem_ctx {
  void *ctx_opaque;
  void *team_opaque;
} rocshmem_ctx_t;

typedef unsigned long long* rocshmem_team_t;

// Define the ROCSHMEM_CTX_DEFAULT symbol. rocshmem_hipmodule_init() looks for
// this device symbol in the given kernel module
extern "C" __device__ rocshmem_ctx_t __attribute__((visibility("default"))) ROCSHMEM_CTX_DEFAULT{};

// Define ROCSHMEM_TEAM_WORLD so rocshmem_hipmodule_init() can copy the team world into this module
extern "C" __constant__ rocshmem_team_t __attribute__((visibility("default"))) ROCSHMEM_TEAM_WORLD {nullptr};

// stub kernel function used for module verification
extern "C" __global__ void simple_test_kernel(int *result) {
  // Simple test kernel that has the ROCSHMEM_CTX_DEFAULT symbol compiled in its module
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Just write a test value
    *result = 42;
  }
}
)";

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
HipModuleInitTester::HipModuleInitTester(TesterArguments args)
    : Tester(args),
      test_module(nullptr),
      kernel_func(nullptr),
      device_result(nullptr) {
  my_pe = rocshmem_my_pe();
  n_pes = rocshmem_n_pes();

  // Allocate device memory for test result
  CHECK_HIP(hipMalloc(&device_result, sizeof(int)));

  // Compile the kernel using HIPRTC with rocshmem headers
  hiprtcProgram prog;
  CHECK_HIPRTC(hiprtcCreateProgram(&prog, test_kernel_src, "simple_test_kernel.cpp", 0, nullptr, nullptr));

  // Get device architecture for compilation
  hipDeviceProp_t props;
  CHECK_HIP(hipGetDeviceProperties(&props, 0));
  std::string arch = std::string("--offload-arch=") + props.gcnArchName;

  // Add include paths for rocshmem headers
  std::string include_path = "-I" + std::string(CMAKE_SOURCE_DIR) + "/include";
  std::string build_include = "-I" + std::string(CMAKE_BINARY_DIR) + "/include";
  std::string src_include = "-I" + std::string(CMAKE_SOURCE_DIR) + "/src";

  const char* rocm_path = std::getenv("ROCM_PATH");
  std::string hip_include = rocm_path ?
    "-I" + std::string(rocm_path) + "/include" :
    "-I/opt/rocm/include";

  const char* options[] = {
    arch.c_str(),
    hip_include.c_str(),
    include_path.c_str(),
    build_include.c_str(),
    src_include.c_str(),
    "-D__HIP_PLATFORM_AMD__"
  };

  hiprtcResult compileResult = hiprtcCompileProgram(prog, 6, options);

  // Check compilation result
  if (compileResult != HIPRTC_SUCCESS) {
    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);
    if (logSize) {
      char* log = new char[logSize];
      hiprtcGetProgramLog(prog, log);
      fprintf(stderr, "HIPRTC compilation failed:\n%s\n", log);
      delete[] log;
    }
    CHECK_HIPRTC(compileResult);
  }

  // Get compiled code
  size_t codeSize;
  CHECK_HIPRTC(hiprtcGetCodeSize(prog, &codeSize));
  char* code = new char[codeSize];
  CHECK_HIPRTC(hiprtcGetCode(prog, code));

  // Load the compiled module
  // Note: This creates a HIP module that we'll pass to rocshmem_hipmodule_init
  // this HIP module has device global symbol ROCSHMEM_CTX_DEFAULT in device symbol table.
  CHECK_HIP(hipModuleLoadData(&test_module, code));

  // Clean up
  delete[] code;
  CHECK_HIPRTC(hiprtcDestroyProgram(&prog));

  // Get the kernel function from the module
  CHECK_HIP(hipModuleGetFunction(&kernel_func, test_module, "simple_test_kernel"));
}

HipModuleInitTester::~HipModuleInitTester() {
  if (device_result) {
    CHECK_HIP(hipFree(device_result));
  }
  if (test_module) {
    CHECK_HIP(hipModuleUnload(test_module));
  }
}

void HipModuleInitTester::resetBuffers([[maybe_unused]] size_t size) {
  // Reset device result to 0
  CHECK_HIP(hipMemset(device_result, 0, sizeof(int)));
}

void HipModuleInitTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                        int loop, size_t size) {
  // Note: This test intentionally ignores gridSize/blockSize parameters and uses
  // a simple 1x1x1 launch.
  (void)gridSize;
  (void)blockSize;
  (void)loop;
  (void)size;

  if (my_pe == 0) {
    printf("\n=== HIP graph capture test ===\n");
  }

  // Reset result buffer
  CHECK_HIP(hipMemset(device_result, 0, sizeof(int)));

  // Create a stream for graph capture
  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  // Begin graph capture
  hipGraph_t graph;
  CHECK_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  // Call rocshmem_hipmodule_init within graph capture
  // verifies device-to-device copy for device global symbols work in graph.
  // Make sure this API does not issue any operation on legacy/default stream
  // while HIP graph capturing is active on another stream.
  int ret = rocshmem_hipmodule_init(test_module, stream);

  if (ret != 0) {
    fprintf(stderr, "[Rank %d] FAIL: rocshmem_hipmodule_init in graph failed with code %d\n", my_pe, ret);
    CHECK_HIP(hipStreamEndCapture(stream, &graph));  // Clean up
    rocshmem_global_exit(1);
  }

  // Launch kernel within the graph
  void *args[] = {&device_result};
  CHECK_HIP(hipModuleLaunchKernel(
      kernel_func,
      1, 1, 1,    // gridDim
      1, 1, 1,    // blockDim
      0,          // sharedMem
      stream,     // Use the capturing stream
      args,
      nullptr));

  // End graph capture
  CHECK_HIP(hipStreamEndCapture(stream, &graph));
  if (my_pe == 0) {
    printf("PASS: Graph capture completed successfully\n");
  }

  // Instantiate the captured graph
  hipGraphExec_t graph_exec;
  CHECK_HIP(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  if (my_pe == 0) {
    printf("PASS: Graph instantiated successfully\n");
  }

  // Execute the graph
  CHECK_HIP(hipGraphLaunch(graph_exec, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  if (my_pe == 0) {
    printf("PASS: Graph executed successfully\n");
  }

  // Verify result from graph execution
  int host_result = 0;
  CHECK_HIP(hipMemcpy(&host_result, device_result, sizeof(int),
                      hipMemcpyDeviceToHost));

  if (host_result != 42) {
    fprintf(stderr, "[Rank %d] FAIL: Graph execution failed (expected 42, got %d)\n",
            my_pe, host_result);
    rocshmem_global_exit(1);
  }

  if (my_pe == 0) {
    printf("PASS: Graph execution verified (result = %d)\n", host_result);
    printf("PASS: rocshmem_hipmodule_init is CUDA graph compatible!\n");
  }

  // Cleanup graph resources
  CHECK_HIP(hipGraphExecDestroy(graph_exec));
  CHECK_HIP(hipGraphDestroy(graph));
  CHECK_HIP(hipStreamDestroy(stream));
}

void HipModuleInitTester::verifyResults(size_t size) {
  (void)size;  // Not used in this verification test

  // Verify the kernel executed correctly
  int host_result = 0;
  CHECK_HIP(hipMemcpy(&host_result, device_result, sizeof(int),
                      hipMemcpyDeviceToHost));

  if (host_result == 42) {
    if (my_pe == 0) {
      printf("PASS: Kernel execution verified (result = %d)\n", host_result);
    }
  } else {
    // Explicitly fail the test when verification fails
    fprintf(stderr, "[Rank %d] FAIL: Kernel verification failed (expected 42, got %d)\n",
            my_pe, host_result);
    // Exit with error to ensure test is reported as failed
    rocshmem_global_exit(1);
  }
}
