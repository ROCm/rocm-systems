/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include "hip_module_common.hh"
#include <hip_test_helper.hh>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>

// launch with 'extra' buffer
static hipError_t launchWithExtra(hipFunction_t f, void* kernargs, size_t kernargs_size) {
  void* extra[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kernargs_size,
    HIP_LAUNCH_PARAM_END
  };
  return hipModuleLaunchKernel(f,
                               1,1,1,    // grid
                               1,1,1,    // block
                               0,        // shared
                               0,        // stream
                               nullptr,  // kernelParams
                               extra);   // extra
}

TEST_CASE("hipModuleLaunchKernel extra-buffer behavior matches CUDA", "[module]") {
  //      Load a code object that contains both kernels:
  //      helloKernelWithArgs(int, float, const char*)
  //      helloKernel()
  hipModule_t module;
  REQUIRE(hipModuleLoad(&module, "hipKernel.code") == hipSuccess);

  hipFunction_t fWithArgs, fNoArgs;
  REQUIRE(hipModuleGetFunction(&fWithArgs, module, "helloKernelWithArgs") == hipSuccess );
  REQUIRE(hipModuleGetFunction(&fNoArgs,    module, "helloKernel")             == hipSuccess );

  // Prepare a args-struct for the 3 parameter kernel
  struct KernelArgs {int a; float b; const char* msg; };
  KernelArgs args = {123, 4.56f, "test" };
  size_t sizeInt = sizeof(int), zero = 0;

  SECTION("hasArgs, nullptr, size > 0") {
    REQUIRE(launchWithExtra(fWithArgs, nullptr, sizeInt)
             == hipErrorInvalidValue );
  }
  SECTION("hasArgs, valid ptr, size = 0") {
    REQUIRE(launchWithExtra(fWithArgs, &args, zero)
             == hipErrorInvalidValue );
  }
  SECTION("hasArgs, valid ptr, size = sizeof(int)") {
    REQUIRE(launchWithExtra(fWithArgs, &args, sizeInt)
             == hipSuccess );
  }

  SECTION("no args, nullptr, size = 0") {
    REQUIRE(launchWithExtra(fNoArgs, nullptr, zero)
             == hipSuccess );
  }
  SECTION("no args, valid ptr, size = 0") {
    REQUIRE(launchWithExtra(fNoArgs, &args, zero)
             == hipSuccess );
  }
  SECTION("no args, nullptr, size > 0") {
    REQUIRE(launchWithExtra(fNoArgs, nullptr, sizeInt)
             == hipErrorInvalidValue );
  }
  SECTION("no args, valid ptr, size > 0") {
    REQUIRE(launchWithExtra(fNoArgs, &args, sizeInt)
             == hipErrorInvalidConfiguration );
  }

  REQUIRE(hipModuleUnload(module) == hipSuccess );
}