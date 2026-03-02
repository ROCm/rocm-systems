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

#include "OCLMemDependency.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static cl_uint Stages = 4;
const static cl_uint ThreadsForCheck = 1 << Stages;

#define KERNEL_CODE(...) #__VA_ARGS__

const static char* strKernel = KERNEL_CODE(
\n __kernel void bitonicSort(__global uint2* keys, uint stage, uint pass) {
  const uint thread = get_global_id(0);

  const uint pairDistance = 1 << (stage - pass);

  /* The purpose of this is to introduce an additional zero at stage - pass
   * bit*/
  const uint leftID = (thread & (pairDistance - 1)) |
                      ((thread & ~(pairDistance - 1)) << 1); /* Is the same as below */

  const uint direction = ((thread >> stage) & 1) == 1 ? 0 : 1;

  const uint rightID = leftID + pairDistance;
  const uint2 left = keys[leftID];
  const uint2 right = keys[rightID];

  const uint2 larger = left.x > right.x ? left : right;
  const uint2 smaller = left.x > right.x ? right : left;

  keys[leftID] = direction ? smaller : larger;
  keys[rightID] = direction ? larger : smaller;
}
\n);

OCLMemDependency::OCLMemDependency() { _numSubTests = 1; }

OCLMemDependency::~OCLMemDependency() {}

void OCLMemDependency::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  char dbuffer[1024] = {0};
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

  kernel_ = _wrapper->clCreateKernel(program_, "bitonicSort", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, ThreadsForCheck * sizeof(cl_uint2),
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
  cl_buffer_region reg = {0, ThreadsForCheck * sizeof(cl_uint2)};
  buffer = _wrapper->clCreateSubBuffer(buffers()[0], CL_MEM_READ_WRITE,
                                       CL_BUFFER_CREATE_TYPE_REGION, &reg, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLMemDependency::run(void) {
  cl_uint2 values[ThreadsForCheck] = {
      {{3, 0}},  {{1, 5}}, {{4, 6}},  {{2, 4}},  {{0, 3}},   {{5, 10}},  {{15, 7}}, {{13, 8}},
      {{10, 2}}, {{9, 1}}, {{7, 11}}, {{11, 9}}, {{14, 12}}, {{12, 14}}, {{6, 13}}, {{8, 15}}};
  cl_uint2 reference[ThreadsForCheck] = {
      {{0, 3}},  {{1, 5}}, {{3, 0}},  {{2, 4}},  {{4, 6}},   {{5, 10}},  {{6, 13}}, {{8, 15}},
      {{7, 11}}, {{9, 1}}, {{10, 2}}, {{11, 9}}, {{14, 12}}, {{12, 14}}, {{15, 7}}, {{13, 8}}};
  cl_uint2 results[ThreadsForCheck];

  cl_mem buffer = buffers()[0];
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, true, 0, sizeof(values),
                                          values, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  size_t gws[1] = {ThreadsForCheck};

  for (unsigned int i = 0; i < Stages; ++i) {
    buffer = buffers()[i % 2];
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
    for (unsigned int j = 0; j < i; ++j) {
      error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(unsigned int), &i);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
      error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(unsigned int), &j);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL,
                                                0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
  }

  buffer = buffers()[0];
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, true, 0, sizeof(results),
                                         results, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
  for (unsigned int i = 0; i < ThreadsForCheck; ++i) {
    if ((results[i].s[0] != reference[i].s[0]) || (results[i].s[1] != reference[i].s[1])) {
      CHECK_RESULT(true, "Incorrect result for dependency!\n");
    }
  }
}

unsigned int OCLMemDependency::close(void) { return OCLTestImp::close(); }
