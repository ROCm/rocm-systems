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

#include "OCLPerfDevMemWriteSpeed.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/opencl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_SIZES 1
static const unsigned int Sizes[NUM_SIZES] = {256 * 1024 * 1024};

const static char* strKernel =

    "__kernel void write_kernel(__global uint16 *dst, ulong size1, uint "
    "threads\n"
    "                          )\n"
    "{\n"
    "    uint16 pval = (uint16)(0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab,\
 0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab, 0xabababab);\n"
    "    int idx = get_global_id(0);\n"
    "    __global uint16 *dstEnd = dst + size1;\n"
    "    dst = &dst[idx];"
    "    do\n"
    "        {\n"
    "            *dst = pval;\n"
    "            dst += threads;\n"
    "        }\n"
    "    while (dst < dstEnd);\n"
    "}\n";

OCLPerfDevMemWriteSpeed::OCLPerfDevMemWriteSpeed() { _numSubTests = 1; }

OCLPerfDevMemWriteSpeed::~OCLPerfDevMemWriteSpeed() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfDevMemWriteSpeed::open(unsigned int test, char* units, double& conversion,
                                   unsigned int deviceId) {
  error_ = CL_SUCCESS;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = 0;
  kernel_ = 0;
  skip_ = false;
  dstBuffer_ = 0;
  nBytes = Sizes[0];
  cl_ulong loopCnt = nBytes / (16 * sizeof(cl_uint));
  cl_uint maxCUs;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_MAX_COMPUTE_UNITS,
                                     sizeof(cl_uint), &maxCUs, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  wgs = 64;
  const static cl_uint wavesPerCU = 8;
  nWorkItems = maxCUs * wavesPerCU * wgs;
  inputData = 0xabababab;
  nIter = 1000;

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

  kernel_ = _wrapper->clCreateKernel(program_, "write_kernel", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  dstBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, nBytes, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer(dstBuffer) failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &dstBuffer_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_ulong), (void*)&loopCnt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint), (void*)&nWorkItems);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
}

void OCLPerfDevMemWriteSpeed::run(void) {
  if (skip_) {
    return;
  }

  CPerfCounter timer;

  size_t gws[1] = {nWorkItems};
  size_t lws[1] = {wgs};

  // warm up
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  cl_uint* memResult;
  memResult = (cl_uint*)malloc(nBytes);
  if (0 == memResult) {
    CHECK_RESULT_NO_RETURN(0, "malloc failed!\n");
    return;
  }

  memset(memResult, 0, nBytes);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], dstBuffer_, CL_FALSE, 0, nBytes,
                                         memResult, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueReadBuffer dstBuffer_ failed!");
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  for (unsigned int i = 0; i < nBytes / sizeof(cl_uint); i++) {
    if (((cl_uint*)memResult)[i] != inputData) {
      CHECK_RESULT_NO_RETURN(0, "Data validation failed for warm up run!\n");
      free(memResult);
      return;
    }
  }

  free(memResult);

  timer.Reset();
  timer.Start();
  double sec2 = 0;
  cl_event* events = new cl_event[nIter];
  for (unsigned int i = 0; i < nIter; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                              NULL, &events[i]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();
  for (unsigned int i = 0; i < nIter; i++) {
    cl_ulong startTime = 0, endTime = 0;
    error_ = _wrapper->clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                                               sizeof(cl_ulong), &startTime, 0);
    CHECK_RESULT(error_, "clGetEventProfilingInfo failed");
    error_ = _wrapper->clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                                               sizeof(cl_ulong), &endTime, 0);
    CHECK_RESULT(error_, "clGetEventProfilingInfo failed");

    _wrapper->clReleaseEvent(events[i]);
    sec2 += endTime - startTime;
  }
  double sec = timer.GetElapsedTime();
  delete[] events;

  // write speed in GB/s
  double perf = ((double)nBytes * nIter * (double)(1e-09)) / sec;
  double perf2 = ((double)nBytes * nIter) / sec2;
  _perfInfo = (float)perf2;
  float perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes) i:%4d Wall time Perf: %.2f (GB/s)", nBytes, nIter,
           perfInfo);
  testDescString = buf;
}

unsigned int OCLPerfDevMemWriteSpeed::close(void) {
  if (!skip_) {
    if (dstBuffer_) {
      error_ = _wrapper->clReleaseMemObject(dstBuffer_);
      CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(srcBuffer_) failed");
    }
  }

  return OCLTestImp::close();
}
