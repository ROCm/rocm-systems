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

#include "OCLAtomicCounter.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static unsigned int MaxCounters = 2;
const static char* strKernel =
    "#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable            \n"
    "__kernel void atomic_test(                                             \n"
    "   counter32_t counter0, counter32_t counter1, global uint* out_val)   \n"
    "{                                                                      \n"
    "   if (!get_global_id(0)) {                                            \n"
    "       uint val0 = atomic_inc(counter0);                               \n"
    "       uint val1 = atomic_dec(counter1);                               \n"
    "       out_val[0] = val0;                                              \n"
    "       out_val[1] = val1;                                              \n"
    "   }                                                                   \n"
    "}                                                                      \n";

OCLAtomicCounter::OCLAtomicCounter() {
  _numSubTests = 1;
  failed_ = false;
}

OCLAtomicCounter::~OCLAtomicCounter() {}

void OCLAtomicCounter::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening");

  char name[1024] = {0};
  size_t size = 0;

  if (deviceId >= deviceCount_) {
    failed_ = true;
    return;
  }

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 1024, name, &size);
  if (!strstr(name, "cl_ext_atomic_counter")) {
    printf("Atomic counter extension is required for this test!\n");
    failed_ = true;
    return;
  }

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], "-legacy", NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "atomic_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  for (unsigned int i = 0; i < MaxCounters; ++i) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, MaxCounters * sizeof(cl_uint),
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLAtomicCounter::run(void) {
  if (failed_) {
    return;
  }
  cl_uint initVal[2] = {5, 10};
  for (unsigned int i = 0; i < MaxCounters; ++i) {
    error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers()[i], true, 0,
                                            sizeof(cl_uint), &initVal[i], 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
  }

  for (unsigned int i = 0; i < MaxCounters + 1; ++i) {
    cl_mem buffer = buffers()[i];
    error_ = _wrapper->clSetKernelArg(kernel_, i, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
  }

  size_t gws[1] = {64};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  cl_uint outputV[MaxCounters] = {0};

  // Find the new counter value
  initVal[0]++;
  initVal[1]--;

  for (unsigned int i = 0; i < MaxCounters; ++i) {
    cl_mem buffer = buffers()[i];
    error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers()[i], true, 0,
                                           sizeof(cl_uint), &outputV[i], 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
    if (initVal[i] != outputV[i]) {
      printf("%d != %d", initVal[i], outputV[i]);
      CHECK_RESULT(true, " - Incorrect result for counter!\n");
    }
  }

  // Restore the original value to check the returned result in the kernel
  initVal[0]--;
  initVal[1]++;

  cl_mem buffer = buffers()[MaxCounters];
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers()[MaxCounters], true, 0,
                                         MaxCounters * sizeof(cl_uint), outputV, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
  for (unsigned int i = 0; i < MaxCounters; ++i) {
    if (initVal[i] != outputV[i]) {
      printf("%d != %d", initVal[i], outputV[i]);
      CHECK_RESULT(true,
                   " - Incorrect result for counter inside kernel. Returned "
                   "value != original.\n");
    }
  }
}

unsigned int OCLAtomicCounter::close(void) { return OCLTestImp::close(); }
