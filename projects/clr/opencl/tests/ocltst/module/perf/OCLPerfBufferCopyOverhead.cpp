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

#include "OCLPerfBufferCopyOverhead.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <complex>

#include "CL/opencl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

typedef struct {
  unsigned int iterations;
  int flushEvery;
} testStruct;

static testStruct testList[] = {
    {1, -1},       {1, -1},        {10, 1},         {10, -1},     {100, 1},    {100, 10},
    {100, -1},     {1000, 1},      {1000, 10},      {1000, 100},  {1000, -1},  {10000, 1},
    {10000, 10},   {10000, 100},   {10000, 1000},   {10000, -1},  {100000, 1}, {100000, 10},
    {100000, 100}, {100000, 1000}, {100000, 10000}, {100000, -1},
};

OCLPerfBufferCopyOverhead::OCLPerfBufferCopyOverhead() {
  _numSubTests = 2 * 2 * sizeof(testList) / sizeof(testStruct);
}

OCLPerfBufferCopyOverhead::~OCLPerfBufferCopyOverhead() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfBufferCopyOverhead::open(unsigned int test, char* units, double& conversion,
                                     unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test % (sizeof(testList) / sizeof(testStruct));

  context_ = 0;
  cmd_queue_ = 0;
  srcBuffer_ = 0;
  dstBuffer_ = 0;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
    platform = platforms[_platformIndex];
    char pbuf[100];
    error_ = _wrapper->clGetPlatformInfo(platforms[_platformIndex], CL_PLATFORM_VENDOR,
                                         sizeof(pbuf), pbuf, NULL);
    num_devices = 0;
    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    // Runtime returns an error when no GPU devices are present instead of just
    // returning 0 devices
    // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
    // Choose platform with GPU devices
    delete platforms;
  }

  bufSize_ = 4;

  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  cl_mem_flags flags = CL_MEM_READ_ONLY;
  sleep = ((test / (sizeof(testList) / sizeof(testStruct))) % 2) > 0;
  if (test >= ((sizeof(testList) / sizeof(testStruct)) * 2)) {
    srcHost = true;
    flags |= CL_MEM_ALLOC_HOST_PTR;
  } else {
    srcHost = false;
  }
  srcBuffer_ = _wrapper->clCreateBuffer(context_, flags, bufSize_, NULL, &error_);
  CHECK_RESULT(srcBuffer_ == 0, "clCreateBuffer(srcBuffer) failed");

  flags = CL_MEM_WRITE_ONLY;
  if (!srcHost) {
    flags |= CL_MEM_ALLOC_HOST_PTR;
  }
  dstBuffer_ = _wrapper->clCreateBuffer(context_, flags, bufSize_, NULL, &error_);
  CHECK_RESULT(dstBuffer_ == 0, "clCreateBuffer(dstBuffer) failed");
}

void OCLPerfBufferCopyOverhead::run(void) {
  CPerfCounter timer;
  cl_event event;
  cl_int eventStatus;
  unsigned int iter = testList[_openTest].iterations;

  // Warm up
  error_ = _wrapper->clEnqueueCopyBuffer(cmd_queue_, srcBuffer_, dstBuffer_, 0, 0, bufSize_, 0,
                                         NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < iter; i++) {
    error_ = _wrapper->clEnqueueCopyBuffer(cmd_queue_, srcBuffer_, dstBuffer_, 0, 0, bufSize_, 0,
                                           NULL, &event);

    CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");
    if ((testList[_openTest].flushEvery > 0) && (((i + 1) % testList[_openTest].flushEvery) == 0)) {
      if (sleep) {
        _wrapper->clFinish(cmd_queue_);
      } else {
        _wrapper->clFlush(cmd_queue_);
        error_ = _wrapper->clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int),
                                          &eventStatus, NULL);
        while (eventStatus > 0) {
          error_ = _wrapper->clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                            sizeof(cl_int), &eventStatus, NULL);
        }
      }
    }
    if (i != (iter - 1)) {
      _wrapper->clReleaseEvent(event);
    }
  }
  if (sleep) {
    _wrapper->clFinish(cmd_queue_);
  } else {
    _wrapper->clFlush(cmd_queue_);
    error_ = _wrapper->clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int),
                                      &eventStatus, NULL);
    while (eventStatus > 0) {
      error_ = _wrapper->clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int),
                                        &eventStatus, NULL);
    }
  }
  _wrapper->clReleaseEvent(event);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Buffer copy time in us
  double perf = sec * 1000. * 1000. / iter;

  const char* strSrc = NULL;
  const char* strDst = NULL;
  const char* strWait = NULL;
  if (srcHost) {
    strSrc = "host";
    strDst = "dev";
  } else {
    strSrc = "dev";
    strDst = "host";
  }
  if (sleep) {
    strWait = "sleep";
  } else {
    strWait = "spin";
  }
  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " %5s, s:%4s d:%4s i:%6d (us) ", strWait, strSrc, strDst, iter);
  testDescString = buf;
}

unsigned int OCLPerfBufferCopyOverhead::close(void) {
  if (srcBuffer_) {
    error_ = _wrapper->clReleaseMemObject(srcBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(srcBuffer_) failed");
  }
  if (dstBuffer_) {
    error_ = _wrapper->clReleaseMemObject(dstBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(dstBuffer_) failed");
  }
  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  return _crcword;
}
