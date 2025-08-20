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

#include "OCLPerfMemCreate.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <sstream>
#include <string>

#include "CL/cl.h"
#include "CL/cl_ext.h"

#if defined(_WIN32) && !defined(_WIN64)
static const size_t BufSize = 0x200000;
static const size_t BufSizeC = 0x100000;
#else
static const size_t BufSize = 0x400000;
static const size_t BufSizeC = 0x200000;
#endif

static const size_t Iterations = 0x100;
static const size_t IterationsC = 0x1000;

static const char* strKernel =
    "__kernel void dummy(__global uint* out)    \n"
    "{                                          \n"
    "   uint id = get_global_id(0);             \n"
    "   uint value = 1;                         \n"
    "   if ((int)get_local_id(0) < 0)           \n"
    "       out[id] = value;                    \n"
    "}                                          \n";

#define NUM_TESTS 5
OCLPerfMemCreate::OCLPerfMemCreate() {
  _numSubTests = NUM_TESTS * 2;
  failed_ = false;
}

OCLPerfMemCreate::~OCLPerfMemCreate() {}

void OCLPerfMemCreate::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  _deviceId = deviceId;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  test_ = test % NUM_TESTS;
  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  useSubBuf_ = (test >= NUM_TESTS);

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
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
  kernel_ = _wrapper->clCreateKernel(program_, "dummy", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfMemCreate::run(void) {
  if (failed_) {
    return;
  }
  cl_mem buffer, subBuf;
  cl_mem* bufptr;
  unsigned int* values;
  values = reinterpret_cast<unsigned int*>(new cl_int4[BufSize]);
  CPerfCounter timer;
  cl_mem_flags flags = CL_MEM_READ_ONLY;
  void* hostPtr = NULL;

  // Clear destination buffer
  memset(values, 0, BufSize * sizeof(cl_int4));

  size_t bufSize = ((test_ % 2) == 0) ? BufSize * sizeof(cl_int4) : BufSizeC * sizeof(cl_int4);
  size_t iter = ((test_ % 2) == 0) ? Iterations : IterationsC;

  if (test_ == 4) {
    hostPtr = values;
    bufSize = 0x100000;
    flags = CL_MEM_USE_HOST_PTR;
  } else if ((test_ / 2) > 0) {
    iter = ((test_ % 2) == 0) ? Iterations / 10 : IterationsC;
    flags |= CL_MEM_ALLOC_HOST_PTR;
  }
  timer.Reset();
  timer.Start();

  for (size_t i = 0; i < iter; ++i) {
    buffer = _wrapper->clCreateBuffer(context_, flags, bufSize, hostPtr, &error_);
    bufptr = &buffer;
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    if (useSubBuf_) {
      cl_buffer_region reg;
      reg.origin = 0;
      reg.size = bufSize;
      subBuf =
          _wrapper->clCreateSubBuffer(buffer, flags, CL_BUFFER_CREATE_TYPE_REGION, &reg, &error_);
      bufptr = &subBuf;
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateSubBuffer() failed");
    }

    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), bufptr);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    size_t gws[1] = {64};
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

    _wrapper->clFinish(cmdQueues_[_deviceId]);
    if (useSubBuf_) _wrapper->clReleaseMemObject(subBuf);
    _wrapper->clReleaseMemObject(buffer);
  }

  timer.Stop();
  std::stringstream stream;

  static const char* Message[] = {" create+destroy time [uncached] ",
                                  " create+destroy time [cached  ] "};
  static const char* Type[] = {"DEV", "AHP", "UHP"};

  stream << Type[test_ / 2];
  stream << Message[test_ % 2];
  stream << " per allocation (ms) ";
  stream << bufSize / 1024 << " KB";
  if (useSubBuf_) stream << " subbuf ";
  testDescString = stream.str();
  _perfInfo = static_cast<float>(timer.GetElapsedTime() * 1000 / iter);

  delete[] values;
}

unsigned int OCLPerfMemCreate::close(void) { return OCLTestImp::close(); }
