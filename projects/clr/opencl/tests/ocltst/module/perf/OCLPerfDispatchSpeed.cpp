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

#include "OCLPerfDispatchSpeed.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define CHAR_BUF_SIZE 512

typedef struct {
  unsigned int iterations;
  int flushEvery;
} testStruct;

testStruct testList[] = {
    {1, -1},       {1, -1},        {10, 1},         {10, -1},     {100, 1},    {100, 10},
    {100, -1},     {1000, 1},      {1000, 10},      {1000, 100},  {1000, -1},  {10000, 1},
    {10000, 10},   {10000, 100},   {10000, 1000},   {10000, -1},  {100000, 1}, {100000, 10},
    {100000, 100}, {100000, 1000}, {100000, 10000}, {100000, -1},
};

unsigned int mapTestList[] = {1, 1, 10, 100, 1000, 10000, 100000};

void OCLPerfDispatchSpeed::genShader(void) {
  shader_.clear();
  shader_ +=
      "__kernel void _dispatchSpeed(__global float *outBuf)\n"
      "{\n"
      "    int i = (int) get_global_id(0);\n"
      "    if (i < 0)\n"
      "        outBuf[i] = 0.0f;\n"
      "}\n";
}

OCLPerfDispatchSpeed::OCLPerfDispatchSpeed() {
  testListSize = sizeof(testList) / sizeof(testStruct);
  _numSubTests = 2 * 2 * testListSize;
}

OCLPerfDispatchSpeed::~OCLPerfDispatchSpeed() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfDispatchSpeed::open(unsigned int test, char* units, double& conversion,
                                unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test % testListSize;

  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  outBuffer_ = 0;
  sleep = false;
  doWarmup = false;

  if ((test / testListSize) % 2) {
    doWarmup = true;
  }
  if (test >= (testListSize * 2)) {
    sleep = true;
  }

  bufSize_ = 64 * sizeof(cl_float);

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
#if 0
        // Get last for default
        platform = platforms[numPlatforms-1];
        for (unsigned i = 0; i < numPlatforms; ++i) {
#endif
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
    // if (num_devices > 0)
    //{
    //    platform = platforms[_platformIndex];
    //    break;
    //}
#if 0
        }
#endif
    delete platforms;
  } else {
    CHECK_RESULT(numPlatforms == 0, "No platforms available!");
  }

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

  outBuffer_ = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  genShader();
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &device, "", NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "_dispatchSpeed", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);
}

void OCLPerfDispatchSpeed::run(void) {
  int global = bufSize_ / sizeof(cl_float);
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  CPerfCounter timer;
  cl_event event;
  cl_int eventStatus;

  if (doWarmup) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, &event);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
    _wrapper->clFinish(cmd_queue_);
  }

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < testList[_openTest].iterations; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, &event);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
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
    if (i != (testList[_openTest].iterations - 1)) {
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

  // microseconds per launch
  double perf = (1000000.f * sec / testList[_openTest].iterations);
  const char* waitType;
  const char* extraChar;
  const char* n;
  const char* warmup;
  if (sleep) {
    waitType = "sleep";
    extraChar = "";
    n = "";
  } else {
    waitType = "spin";
    n = "n";
    extraChar = " ";
  }
  if (doWarmup) {
    warmup = "warmup";
  } else {
    warmup = "";
  }

  _perfInfo = (float)perf;
  char buf[256];
  if (testList[_openTest].flushEvery > 0) {
    SNPRINTF(buf, sizeof(buf), " %7d dispatches %s%sing every %5d %6s (us/disp)",
             testList[_openTest].iterations, waitType, n, testList[_openTest].flushEvery, warmup);
  } else {
    SNPRINTF(buf, sizeof(buf), " %7d dispatches (%s%s)              %6s (us/disp)",
             testList[_openTest].iterations, waitType, extraChar, warmup);
  }
  testDescString = buf;
}

unsigned int OCLPerfDispatchSpeed::close(void) {
  if (outBuffer_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
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

OCLPerfMapDispatchSpeed::OCLPerfMapDispatchSpeed() {
  testListSize = sizeof(mapTestList) / sizeof(unsigned int);
  _numSubTests = 2 * testListSize;
}

void OCLPerfMapDispatchSpeed::run(void) {
  cl_mem outBuffer;
  outBuffer = _wrapper->clCreateBuffer(context_, CL_MEM_ALLOC_HOST_PTR, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer);

  int global = bufSize_ / sizeof(cl_float);
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  CPerfCounter timer;

  if (doWarmup) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
    _wrapper->clFinish(cmd_queue_);
  }

  timer.Reset();
  timer.Start();
  void* mem;
  for (unsigned int i = 0; i < mapTestList[_openTest]; i++) {
    mem =
        _wrapper->clEnqueueMapBuffer(cmd_queue_, outBuffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                     0, bufSize_, 0, NULL, NULL, &error_);

    CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
    error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, outBuffer, mem, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueUnmapBuffer failed");
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }
  _wrapper->clFinish(cmd_queue_);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // microseconds per launch
  double perf = (1000000.f * sec / mapTestList[_openTest]);
  const char* warmup;
  if (doWarmup) {
    warmup = "warmup";
  } else {
    warmup = "";
  }

  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " %7d maps and dispatches %6s (us/disp)", mapTestList[_openTest],
           warmup);
  testDescString = buf;

  _wrapper->clReleaseMemObject(outBuffer);
}
