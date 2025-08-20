/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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
 THE SOFTWARE. */

#include "OCLMemObjs.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>

const char* OCLMemObjs::kernel_src = "";

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

OCLMemObjs::OCLMemObjs() { _numSubTests = 1; }

OCLMemObjs::~OCLMemObjs() {}

void OCLMemObjs::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
}

int OCLMemObjs::test(void) {
  cl_int err;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    std::cerr << "Platform::get() failed \n";
    return EXIT_FAILURE;
  }
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)(platforms[0])(), 0};
  cl::Context context(CL_DEVICE_TYPE_ALL, properties, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Context::Context() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }

  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  if (err != CL_SUCCESS) {
    std::cerr << "Context::getInfo() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }
  if (devices.size() == 0) {
    std::cerr << "No device available\n";
    return EXIT_FAILURE;
  }

  const char source[] = "__kernel void test_memobjs(__global int* ptr) {}";
  cl::Program::Sources sources(1, std::make_pair(source, 0));

  cl::Program program(context, sources, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Program::Program() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }
  err = program.build(devices);
  if (err != CL_SUCCESS) {
    std::cerr << "Program::build() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }

  cl::Kernel kernel(program, "test_memobjs", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Kernel::Kernel() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }
  if (err != CL_SUCCESS) {
    std::cerr << "Kernel::setArg() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }

  cl::CommandQueue queue(context, devices[0], 0, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "CommandQueue::CommandQueue() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }

  cl::Buffer buffer(context, (cl_mem_flags)0, 1024, NULL, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Buffer::Buffer() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }

  err = kernel.setArg(0, buffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Kernel::setArg() failed (" << err << ")\n";
    return EXIT_FAILURE;
  }

  err = queue.enqueueTask(kernel);
  if (err != CL_SUCCESS) {
    std::cerr << "CommandQueue::enqueueTask() failed (" << err << ")\n";
  }

  // Force a clReleaseMemoryObject on buffer before dispatch.
  buffer = cl::Buffer();

  err = queue.finish();
  if (err != CL_SUCCESS) {
    std::cerr << "CommandQueue::finish() failed (" << err << ")\n";
  }

  // std::cout << " Test: Pass!\n";
  return EXIT_SUCCESS;
}

void OCLMemObjs::run(void) { CHECK_RESULT((test() != EXIT_SUCCESS), "test failed"); }

unsigned int OCLMemObjs::close(void) { return _crcword; }
