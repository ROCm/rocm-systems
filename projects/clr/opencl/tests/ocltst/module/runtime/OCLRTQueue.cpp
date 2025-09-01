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

#include "OCLRTQueue.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

static const size_t Iterations = 0x100;
static const size_t IterationDivider = 2;
static const size_t MaxBuffers = IterationDivider;

#if EMU_ENV
static const size_t BufSize = 0x800;
#else
static const size_t BufSize = 0x800000;
#endif
const static char* strKernel =
    "__kernel void factorial(__global uint* out)                        \n"
    "{                                                                  \n"
    "   uint id = get_global_id(0);                                     \n"
    "   uint factorial = 1;                                             \n"
    "   for (uint i = 1; i < (id / 0x400); ++i)                         \n"
    "   {                                                               \n"
    "       factorial *= i;                                             \n"
    "   }                                                               \n"
    "    out[id] = factorial;                                           \n"
    "}                                                                  \n";

OCLRTQueue::OCLRTQueue() : rtQueue_(NULL), rtQueue1_(NULL), kernel2_(NULL) {
#ifndef CL_VERSION_2_0
  _numSubTests = 0;
  testID_ = 0;
  failed_ = false;
#else
  _numSubTests = 2;
  testID_ = 0;
  failed_ = false;
#endif
}

OCLRTQueue::~OCLRTQueue() {}

void OCLRTQueue::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
#ifdef CL_VERSION_2_0
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;
  size_t param_size = 0;
  char* strVersion = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strVersion = new char[param_size];
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, param_size, strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strVersion[7] < '2') {
    failed_ = true;
    return;
  }
  cl_uint rtQueues;
#define CL_DEVICE_MAX_REAL_TIME_COMPUTE_QUEUES_AMD 0x404D
#define CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_AMD 0x404E
#define CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_GRANULARITY_AMD 0x403A
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_MAX_REAL_TIME_COMPUTE_QUEUES_AMD,
                                sizeof(rtQueues), &rtQueues, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (rtQueues < 2) {
    failed_ = true;
    return;
  }

  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_AMD,
                                     sizeof(rtCUs_), &rtCUs_, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_MAX_COMPUTE_UNITS,
                                     sizeof(maxCUs_), &maxCUs_, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId],
                                     CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_GRANULARITY_AMD,
                                     sizeof(rtCUsGranularity_), &rtCUsGranularity_, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

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
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLRTQueue::run(void) {
#ifdef CL_VERSION_2_0
  if (failed_) {
    return;
  }

  void* values;
  CPerfCounter timer;
  cl_mem mapBuffer = buffers()[MaxBuffers];

  values = _wrapper->clEnqueueMapBuffer(cmdQueues_[_deviceId], mapBuffer, true,
                                        (CL_MAP_READ | CL_MAP_WRITE), 0, BufSize * sizeof(cl_uint),
                                        0, NULL, NULL, &error_);

  cl_mem buffer = buffers()[0];
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  // SubTest: 1
  size_t gws[1] = {BufSize};
  size_t x;

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  timer.Reset();
  timer.Start();
  for (x = 0; x < 1; x++) {
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  timer.Stop();

  double sec = timer.GetElapsedTime();
  // Buffer read bandwidth in GB/s
  double perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

  printf("\n Generic Queue(CUs: %d) Time:               %.3fs\n", maxCUs_, sec);

  // SubTest: 2
  bool test_rtq1 = true;
  if (testID_ == 0) {
    cu_ = rtCUs_ >> 1;
  } else {
    cu_ = rtCUs_;
    test_rtq1 = false;
  }
  if (cu_ == 0) {
    cu_ = rtCUs_;
    test_rtq1 = false;
  }

  if (cu_ < rtCUsGranularity_) {
    printf("The num of CUs is less than granularity, skipping...\n");
    return;
  }

  // Create a real time queue
#define CL_QUEUE_REAL_TIME_COMPUTE_UNITS_AMD 0x404f
  const cl_queue_properties cprops[] = {CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(0),
                                        CL_QUEUE_REAL_TIME_COMPUTE_UNITS_AMD, cu_, 0};
  rtQueue_ =
      _wrapper->clCreateCommandQueueWithProperties(context_, devices_[_deviceId], cprops, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(rtQueue_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  _wrapper->clFinish(rtQueue_);

  timer.Reset();
  timer.Start();
  for (x = 0; x < 1; x++) {
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    error_ = _wrapper->clEnqueueNDRangeKernel(rtQueue_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFinish(rtQueue_);

  timer.Stop();

  sec = timer.GetElapsedTime();
  // Buffer read bandwidth in GB/s
  perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

  printf(" RT Queue0 (CUs: %2d) Time:                  %.3fs\n", cu_, sec);

  rtQueue1_ = nullptr;
  if (test_rtq1) {
#define CL_QUEUE_MEDIUM_PRIORITY_AMD 0x4050
    const cl_queue_properties cprops2[] = {CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(0),
                                           CL_QUEUE_MEDIUM_PRIORITY_AMD, cu_, 0};
    rtQueue1_ = _wrapper->clCreateCommandQueueWithProperties(context_, devices_[_deviceId], cprops2,
                                                             &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");
  }
  if (rtQueue1_) {
    error_ =
        _wrapper->clEnqueueNDRangeKernel(rtQueue1_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    _wrapper->clFinish(rtQueue1_);

    timer.Reset();
    timer.Start();
    for (x = 0; x < 1; x++) {
      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

      error_ =
          _wrapper->clEnqueueNDRangeKernel(rtQueue1_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);

      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
    _wrapper->clFinish(rtQueue1_);

    timer.Stop();

    sec = timer.GetElapsedTime();
    // Buffer read bandwidth in GB/s
    perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

    printf(" RT Queue1 (CUs: %2d) Time:                  %.3fs\n", cu_, sec);
  } else {
    if (testID_ == 0) {
      printf(" RT Queue1 test was skipped. Not enough CUs - %2d)", cu_);
    }
  }

  // SubTest: 3
  timer.Reset();
  timer.Start();
  for (x = 0; x < 1; x++) {
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  timer.Stop();

  sec = timer.GetElapsedTime();
  // Buffer read bandwidth in GB/s
  perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

  printf(" Generic Queue(CUs: %d) Time:               %.3fs\n", maxCUs_ - rtCUs_, sec);

  // SubTest: 4
  for (x = 0; x < Iterations / 10; x++) {
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFlush(cmdQueues_[_deviceId]);
  timer.Reset();
  timer.Start();
  for (x = 0; x < 1; x++) {
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    error_ = _wrapper->clEnqueueNDRangeKernel(rtQueue_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFinish(rtQueue_);

  timer.Stop();
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  sec = timer.GetElapsedTime();
  // Buffer read bandwidth in GB/s
  perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

  printf(" Async RT0(CUs: %d) + Generic(CUs: %d) Time: %.3fs\n", cu_, maxCUs_ - rtCUs_, sec);

  // SubTest: 5
  if (rtQueue1_) {
    for (x = 0; x < Iterations / 10; x++) {
      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL,
                                                0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
    _wrapper->clFlush(cmdQueues_[_deviceId]);
    timer.Reset();
    timer.Start();
    for (x = 0; x < 1; x++) {
      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

      error_ =
          _wrapper->clEnqueueNDRangeKernel(rtQueue1_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);

      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
    _wrapper->clFinish(rtQueue1_);

    timer.Stop();
    _wrapper->clFinish(cmdQueues_[_deviceId]);

    sec = timer.GetElapsedTime();
    // Buffer read bandwidth in GB/s
    perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

    printf(" Async RT1(CUs: %d) + Generic(CUs: %d) Time: %.3fs\n", cu_, maxCUs_ - rtCUs_, sec);
  } else {
    if (testID_ == 0) {
      printf(" RT Queue1 test was skipped. Not enough CUs - %2d)", cu_);
    }
  }
  // SubTest: 6
  if (rtQueue1_) {
    for (x = 0; x < Iterations / 10; x++) {
      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL,
                                                0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
    _wrapper->clFlush(cmdQueues_[_deviceId]);
    timer.Reset();
    timer.Start();
    for (x = 0; x < 1; x++) {
      error_ =
          _wrapper->clEnqueueNDRangeKernel(rtQueue_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
    _wrapper->clFlush(rtQueue_);
    for (x = 0; x < 1; x++) {
      error_ =
          _wrapper->clEnqueueNDRangeKernel(rtQueue1_, kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }

    _wrapper->clFlush(rtQueue1_);
    _wrapper->clFinish(rtQueue_);
    _wrapper->clFinish(rtQueue1_);
    timer.Stop();
    _wrapper->clFlush(cmdQueues_[_deviceId]);

    sec = timer.GetElapsedTime();
    // Buffer read bandwidth in GB/s
    perf = ((double)BufSize * sizeof(cl_uint) * x * (double)(1e-09)) / sec;

    printf(" Async RT0(CUs: %d) + RT1(CUs: %d) + Generic(CUs: %d) Time: %.3fs\n", cu_, cu_,
           maxCUs_ - rtCUs_, sec);
    error_ =
        _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], mapBuffer, values, 0, NULL, NULL);
    _wrapper->clFinish(cmdQueues_[_deviceId]);
  } else {
    if (testID_ == 0) {
      printf(" RT Queue1 test was skipped. Not enough CUs - %2d)", cu_);
    }
  }
#endif
}

unsigned int OCLRTQueue::close(void) {
#ifdef CL_VERSION_2_0
  if (NULL != rtQueue_) {
    _wrapper->clReleaseCommandQueue(rtQueue_);
  }
  if (NULL != rtQueue1_) {
    _wrapper->clReleaseCommandQueue(rtQueue1_);
  }
  if (NULL != kernel2_) {
    _wrapper->clReleaseKernel(kernel2_);
  }

  return OCLTestImp::close();
#else
  return CL_SUCCESS;
#endif
}
