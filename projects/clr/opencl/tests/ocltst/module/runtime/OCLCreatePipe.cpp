/* Copyright (c) 2025 Advanced Micro Devices, Inc.

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

#include "OCLCreatePipe.h"

#include <vector>
#include <array>

#include "CL/cl.h"

OCLCreatePipe::OCLCreatePipe() { _numSubTests = 1; }

OCLCreatePipe::~OCLCreatePipe() {}

void OCLCreatePipe::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  _deviceId = deviceId;
}

void OCLCreatePipe::run(void) {
  std::vector<cl_device_id> devices(_deviceId + 1, nullptr);
  cl_platform_id platform = nullptr;
  cl_context context = nullptr;
  cl_int err = 0;

  err = _wrapper->clGetPlatformIDs(1, &platform, nullptr);
  CHECK_RESULT(err, "clGetPlatformIDs failed");

  err = _wrapper->clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, devices.size(), devices.data(),
                                 nullptr);
  CHECK_RESULT(err, "clGetDeviceIDs failed");

  context = _wrapper->clCreateContext(nullptr, 1, &devices[_deviceId], nullptr, nullptr, &err);
  CHECK_RESULT(err, "clCreateContext failed");

  constexpr std::array<cl_mem_flags, 3> valid_flags = {
      CL_MEM_READ_WRITE,
      CL_MEM_HOST_NO_ACCESS,

      CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
  };

  constexpr cl_uint pipe_packet_size = sizeof(int);
  constexpr cl_uint pipe_max_packets = 16;

  for (cl_mem_flags flags : valid_flags) {
    cl_mem pipe = nullptr;

    pipe =
        _wrapper->clCreatePipe(context, flags, pipe_packet_size, pipe_max_packets, nullptr, &err);

    CHECK_RESULT(err, "clCreatePipe failed with flag %lu", flags);

    if (!pipe) {
      break;
    }

    err = _wrapper->clReleaseMemObject(pipe);
    CHECK_RESULT(err, "clReleaseMemObject failed");

    if (err) {
      break;
    }
  }

  if (err) {
    err = _wrapper->clReleaseContext(context);
    CHECK_RESULT(err, "clReleaseContext failed");

    return;
  }

  constexpr std::array<cl_mem_flags, 5> invalid_flags = {
      CL_MEM_READ_ONLY,
      CL_MEM_WRITE_ONLY,

      CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY,

      0,
      ~0UL,
  };

  for (cl_mem_flags flags : valid_flags) {
    cl_mem pipe = nullptr;

    pipe =
        _wrapper->clCreatePipe(context, flags, pipe_packet_size, pipe_max_packets, nullptr, &err);

    CHECK_RESULT(err, "clCreatePipe passed when it shouldn't with flag %lu", flags);

    if (err) {
      break;
    }
  }

  err = _wrapper->clReleaseContext(context);
  CHECK_RESULT(err, "clReleaseContext failed");
}
