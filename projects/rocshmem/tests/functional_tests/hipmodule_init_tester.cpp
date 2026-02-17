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
#ifndef USE_PRECOMPILED_HSACO
#include <hip/hiprtc.h>
#endif
#include <cstring>
#include <cassert>
#include <fstream>

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
// This kernel defines the rocshmem device context to ensure the symbol exists in the module
const char* test_kernel_src = R"(
#include <hip/hip_runtime.h>

// Forward declare the rocshmem context type (16 bytes as expected by rocshmem)
typedef struct rocshmem_ctx {
  char placeholder[16];  // Must be exactly 16 bytes
} rocshmem_ctx_t;

// Define the ROCSHMEM_CTX_DEFAULT symbol that rocshmem_hipmodule_init looks for
// This must match what's in rocshmem_gpu.cpp:76
extern "C" __device__ rocshmem_ctx_t __attribute__((visibility("default"))) ROCSHMEM_CTX_DEFAULT{};

extern "C" __global__ void simple_test_kernel(int *result, int *shmem_buf) {
  // Simple test kernel that has the ROCSHMEM_CTX_DEFAULT symbol in its module
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Just write a test value
    *result = 42;

    // The key is that this module contains the ROCSHMEM_CTX_DEFAULT symbol
    // which rocshmem_hipmodule_init needs to find
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
      device_result(nullptr),
      shmem_buf(nullptr) {
  my_pe = rocshmem_my_pe();
  n_pes = rocshmem_n_pes();

  // Allocate device memory for test result
  CHECK_HIP(hipMalloc(&device_result, sizeof(int)));

  // Allocate symmetric memory for rocshmem operations
  shmem_buf = (int *)rocshmem_malloc(sizeof(int));

#ifdef USE_PRECOMPILED_HSACO
  // Load pre-compiled HSACO file
  // Get device architecture
  hipDeviceProp_t props;
  CHECK_HIP(hipGetDeviceProperties(&props, 0));

  // Extract base architecture (strip :xnack-, :sramecc+, etc.)
  std::string arch_full(props.gcnArchName);
  std::string base_arch = arch_full.substr(0, arch_full.find(':'));

  // Construct path to HSACO file based on GPU architecture
  std::string hsaco_path = std::string(CMAKE_BINARY_DIR) +
                           "/tests/functional_tests/test_kernel_" +
                           base_arch + ".hsaco";

  // Read HSACO file
  std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
  if (!hsaco_file.is_open()) {
    fprintf(stderr, "Failed to open HSACO file: %s (arch=%s)\n", hsaco_path.c_str(), arch_full.c_str());
    fprintf(stderr, "Falling back to first available HSACO file\n");
    // Try common architectures
    const char* common_archs[] = {"gfx942", "gfx90a", "gfx950", "gfx1100"};
    bool found = false;
    for (const char* arch : common_archs) {
      hsaco_path = std::string(CMAKE_BINARY_DIR) + "/tests/functional_tests/test_kernel_" + arch + ".hsaco";
      hsaco_file.open(hsaco_path, std::ios::binary | std::ios::ate);
      if (hsaco_file.is_open()) {
        fprintf(stderr, "Using HSACO for architecture: %s\n", arch);
        found = true;
        break;
      }
    }
    if (!found) {
      fprintf(stderr, "Could not find any HSACO file\n");
      abort();
    }
  }

  std::streamsize hsaco_size = hsaco_file.tellg();
  hsaco_file.seekg(0, std::ios::beg);

  char* hsaco_data = new char[hsaco_size];
  if (!hsaco_file.read(hsaco_data, hsaco_size)) {
    fprintf(stderr, "Failed to read HSACO file\n");
    delete[] hsaco_data;
    abort();
  }
  hsaco_file.close();

  // Load the compiled module from HSACO
  CHECK_HIP(hipModuleLoadData(&test_module, hsaco_data));
  delete[] hsaco_data;

#else  // Use HIPRTC for runtime compilation
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

  const char* options[] = {
    arch.c_str(),
    include_path.c_str(),
    build_include.c_str(),
    src_include.c_str(),
    "-D__HIP_PLATFORM_AMD__"
  };

  hiprtcResult compileResult = hiprtcCompileProgram(prog, 5, options);

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
  CHECK_HIP(hipModuleLoadData(&test_module, code));

  // Clean up
  delete[] code;
  CHECK_HIPRTC(hiprtcDestroyProgram(&prog));
#endif

  // Get the kernel function from the module
  CHECK_HIP(hipModuleGetFunction(&kernel_func, test_module, "simple_test_kernel"));
}

HipModuleInitTester::~HipModuleInitTester() {
  if (device_result) {
    CHECK_HIP(hipFree(device_result));
  }
  if (shmem_buf) {
    rocshmem_free(shmem_buf);
  }
  if (test_module) {
    CHECK_HIP(hipModuleUnload(test_module));
  }
}

void HipModuleInitTester::resetBuffers(size_t size) {
  // Reset device result to 0
  CHECK_HIP(hipMemset(device_result, 0, sizeof(int)));
  // Reset symmetric buffer
  if (shmem_buf) {
    memset(shmem_buf, 0, sizeof(int));
  }
}

void HipModuleInitTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                        int loop, size_t size) {
  // Test the rocshmem_hipmodule_init API
  // This is the core API we're testing
  int ret = rocshmem_hipmodule_init(test_module, nullptr);

  if (ret != 0) {
    if (my_pe == 0) {
      fprintf(stderr, "❌ rocshmem_hipmodule_init failed with code %d\n", ret);
    }
    return;
  }

  if (my_pe == 0) {
    printf("✅ rocshmem_hipmodule_init succeeded\n");
  }

  // Launch the test kernel to verify the module works
  void *args[] = {&device_result, &shmem_buf};
  CHECK_HIP(hipModuleLaunchKernel(
      kernel_func,
      1, 1, 1,    // gridDim
      1, 1, 1,    // blockDim
      0,          // sharedMem
      nullptr,    // stream
      args,       // kernel arguments
      nullptr));  // extra

  CHECK_HIP(hipDeviceSynchronize());
}

void HipModuleInitTester::verifyResults(size_t size) {
  // Verify the kernel executed correctly
  int host_result = 0;
  CHECK_HIP(hipMemcpy(&host_result, device_result, sizeof(int),
                      hipMemcpyDeviceToHost));

  if (host_result == 42) {
    if (my_pe == 0) {
      printf("✅ Kernel execution verified (result = %d)\n", host_result);
    }
  } else {
    if (my_pe == 0) {
      fprintf(stderr, "❌ Kernel verification failed (expected 42, got %d)\n",
              host_result);
    }
  }
}
