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

#include "OCLAsyncTransfer.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

#if EMU_ENV
static const size_t Iterations = 1;
static const size_t IterationDivider = 1;
static const size_t BufSize = 10;
#else
static const size_t Iterations = 0x100;
static const size_t IterationDivider = 2;
static const size_t BufSize = 0x800000;
#endif  // EMU_ENV

static const size_t MaxBuffers = IterationDivider;

const static char* strKernel =
    "__kernel void factorial(__global uint* out)                        \n"
    "{                                                                  \n"
    "   uint id = get_global_id(0);                                     \n"
    "   uint factorial = 1;                                             \n"
#if EMU_ENV
    "   for (uint i = 1; i < id; ++i)                                   \n"
#else
    "   for (uint i = 1; i < (id / 0x10000); ++i)                       \n"
#endif  // EMU_ENV
    "   {                                                               \n"
    "       factorial *= i;                                             \n"
    "   }                                                               \n"
    "	out[id] = factorial;                                            \n"
    "}                                                                  \n";

OCLAsyncTransfer::OCLAsyncTransfer() { _numSubTests = 1; }

OCLAsyncTransfer::~OCLAsyncTransfer() {}

void OCLAsyncTransfer::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "factorial", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  for (size_t i = 0; i < MaxBuffers; ++i) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, BufSize * sizeof(cl_uint), NULL,
                                      &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_ALLOC_HOST_PTR, BufSize * sizeof(cl_uint),
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLAsyncTransfer::run(void) {
  void* values;
  CPerfCounter timer;
  cl_mem mapBuffer = buffers()[MaxBuffers];

  values = _wrapper->clEnqueueMapBuffer(cmdQueues_[_deviceId], mapBuffer, true,
                                        (CL_MAP_READ | CL_MAP_WRITE), 0, BufSize * sizeof(cl_uint),
                                        0, NULL, NULL, &error_);

  timer.Reset();
  timer.Start();
  size_t x;
  for (x = 0; x < Iterations / IterationDivider; x++) {
    for (size_t y = 0; y < IterationDivider; ++y) {
      cl_mem buffer = buffers()[y];

      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

      size_t gws[1] = {BufSize};
      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL,
                                                0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }

    cl_mem readBuffer = buffers()[0];
    error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], readBuffer, false, 0,
                                           BufSize * sizeof(cl_uint), values, 0, NULL, NULL);
    _wrapper->clFlush(cmdQueues_[_deviceId]);

    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();

  double sec = timer.GetElapsedTime();
  // Buffer read bandwidth in GB/s
  double perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

  printf(" Time: %.2f sec, BW: %.2f GB/s   ", sec, perf);

  error_ =
      _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], mapBuffer, values, 0, NULL, NULL);
  _wrapper->clFinish(cmdQueues_[_deviceId]);
}

unsigned int OCLAsyncTransfer::close(void) { return OCLTestImp::close(); }
