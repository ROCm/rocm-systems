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

#include "OCLGlobalOffset.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static cl_uint ThreadsForCheck = 2;
const static cl_uint GlobalOffset = 64;

const static char* strKernel =
    "__kernel void global_offset_test(                                      \n"
    "   global uint* out_val)                                               \n"
    "{                                                                      \n"
    "   // Check the first thread                                           \n"
    "   if (get_global_id(0) == get_global_offset(0)) {                     \n"
    "       out_val[0] = (uint)get_global_offset(0);                        \n"
    "   }                                                                   \n"
    "   // Check the last thread                                            \n"
    "   if (get_global_id(0) == (get_global_size(0) + get_global_offset(0) - "
    "1)) {  \n"
    "       out_val[1] = (uint)get_global_offset(0);                        \n"
    "   }                                                                   \n"
    "}                                                                      \n";

OCLGlobalOffset::OCLGlobalOffset() { _numSubTests = 1; }

OCLGlobalOffset::~OCLGlobalOffset() {}

void OCLGlobalOffset::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  char dbuffer[1024] = {0};
  _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_VERSION, 1024, dbuffer, NULL);
  if (strstr(dbuffer, "OpenCL 1.0")) {
    return;
  }
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

  kernel_ = _wrapper->clCreateKernel(program_, "global_offset_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, ThreadsForCheck * sizeof(cl_uint),
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLGlobalOffset::run(void) {
  char dbuffer[1024] = {0};
  _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, 1024, dbuffer, NULL);
  if (strstr(dbuffer, "OpenCL 1.0")) {
    return;
  }
  cl_uint offsetValues[ThreadsForCheck] = {0xffffffff, 0xffffffff};
  cl_mem buffer = buffers()[0];
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, true, 0,
                                          ThreadsForCheck * sizeof(cl_uint), offsetValues, 0, NULL,
                                          NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t gws[1] = {0x0800000};
  size_t gwo[1] = {GlobalOffset};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, gwo, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  error_ =
      _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, true, 0,
                                    ThreadsForCheck * sizeof(cl_uint), offsetValues, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
  for (cl_uint i = 0; i < ThreadsForCheck; ++i) {
    if (offsetValues[i] != GlobalOffset) {
      printf("%d != %d", GlobalOffset, offsetValues[i]);
      CHECK_RESULT(true, " - Incorrect result for global offset!\n");
    }
  }
}

unsigned int OCLGlobalOffset::close(void) { return OCLTestImp::close(); }
