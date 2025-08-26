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

#include "OCLPerfDeviceEnqueueSier.h"

#include <Timer.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define KERNEL_CODE(...) #__VA_ARGS__

typedef struct {
  unsigned int threads;
} testStruct;

static unsigned int sizeList[] = {
    81, 243, 729, 2187, 6561, 19683, 59049,
};

const static char* strKernel = {KERNEL_CODE(
    \n __kernel void parentKernel(__global uint* buf, int width, int offsetx, int offsety) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  queue_t q = get_default_queue();

  int one_third = get_global_size(0) / 3;
  int two_thirds = 2 * one_third;

  if (x >= one_third && x < two_thirds && y >= one_third && y < two_thirds) {
    int idx = get_global_id(0);
    if (idx < 0) {
      buf[idx] = 0;
    }
  } else {
    if (one_third > 1 && x % one_third == 0 && y % one_third == 0) {
      const size_t grid[2] = {one_third, one_third};
      enqueue_kernel(q, 0, ndrange_2D(grid), ^{
        parentKernel(buf, width, x + offsetx, y + offsety);
      });
    }
  }
}
    \n)};

OCLPerfDeviceEnqueueSier::OCLPerfDeviceEnqueueSier() {
  _numSubTests = sizeof(sizeList) / sizeof(unsigned int);
  deviceQueue_ = NULL;
  failed_ = false;
  skip_ = false;
}

OCLPerfDeviceEnqueueSier::~OCLPerfDeviceEnqueueSier() {}

void OCLPerfDeviceEnqueueSier::open(unsigned int test, char* units, double& conversion,
                                    unsigned int deviceId) {
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

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
  delete strVersion;

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], "-cl-std=CL2.0", NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "parentKernel", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_ALLOC_HOST_PTR, 2048, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

  queueSize = 512 * 1024;

  image_size = sizeList[testID_];

#if defined(CL_VERSION_2_0)
  const cl_queue_properties cprops[] = {
      CL_QUEUE_PROPERTIES,
      static_cast<cl_queue_properties>(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                       CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_ON_DEVICE),
      CL_QUEUE_SIZE, queueSize, 0};
  deviceQueue_ =
      _wrapper->clCreateCommandQueueWithProperties(context_, devices_[deviceId], cprops, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");
#else
  skip_ = true;
  testDescString = "DeviceEnqueue NOT supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfDeviceEnqueueSier::run(void) {
  CPerfCounter timer;
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

  if (failed_) {
    return;
  }

  if (skip_) {
    return;
  }

  cl_mem buffer = buffers()[0];

  size_t gws[1] = {1};
  size_t lws[1] = {0};

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  int width = image_size, offsetx = 0, offsety = 0;
  error_ |= _wrapper->clSetKernelArg(kernel_, 1, sizeof(int), (void*)&width);
  error_ |= _wrapper->clSetKernelArg(kernel_, 2, sizeof(int), (void*)&offsetx);
  error_ |= _wrapper->clSetKernelArg(kernel_, 3, sizeof(int), (void*)&offsety);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, 0, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  _wrapper->clFinish(cmdQueues_[_deviceId]);

  size_t global_work_size[2] = {image_size, image_size};

  // Try to normalize the amount of work per test
  unsigned int repeats = 100;
  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < repeats; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL,
                                              global_work_size, 0, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

    _wrapper->clFinish(cmdQueues_[_deviceId]);
  }
  timer.Stop();

  double sec = timer.GetElapsedTime();

  unsigned int numOfKernels = (int)pow(8.0, log(image_size) / log(3) - 1);
  _perfInfo = (float)(numOfKernels * repeats) / (float)(sec * 1000000.);
  char buf[256];
  SNPRINTF(buf, sizeof(buf), "image_size = %5d, queue size %3dKB (Mdisp/s)", image_size,
           queueSize / 1024);
  testDescString = buf;
}

unsigned int OCLPerfDeviceEnqueueSier::close(void) {
  // FIXME: Re-enable CPU test once bug 10143 is fixed.
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return 0;
  }

  if (deviceQueue_) {
    error_ = _wrapper->clReleaseCommandQueue(deviceQueue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }

  return OCLTestImp::close();
}
