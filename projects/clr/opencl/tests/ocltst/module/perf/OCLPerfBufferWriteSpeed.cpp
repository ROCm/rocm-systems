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

#include "OCLPerfBufferWriteSpeed.h"

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

#define NUM_SIZES 8
// 256KB, 1 MB, 4MB, 16 MB
static const unsigned int Sizes[NUM_SIZES] = {1024,   32 * 1024, 64 * 1024, 128 * 1024,
                                              262144, 1048576,   4194304,   16777216};

static cl_uint blockedSubtests;

static const unsigned int Iterations[2] = {1, OCLPerfBufferWriteSpeed::NUM_ITER};

#define NUM_OFFSETS 1
static const unsigned int offsets[NUM_OFFSETS] = {0};
#define NUM_SUBTESTS (3 + NUM_OFFSETS)
extern const char* blkStr[2];

OCLPerfBufferWriteSpeed::OCLPerfBufferWriteSpeed() {
  _numSubTests = NUM_SIZES * NUM_SUBTESTS * 2;
  blockedSubtests = _numSubTests;
  _numSubTests += NUM_SIZES * NUM_SUBTESTS;
}

OCLPerfBufferWriteSpeed::~OCLPerfBufferWriteSpeed() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfBufferWriteSpeed::open(unsigned int test, char* units, double& conversion,
                                   unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;

  context_ = 0;
  cmd_queue_ = 0;
  outBuffer_ = 0;
  persistent = false;
  allocHostPtr = false;
  useHostPtr = false;
  hostMem = NULL;
  alignedMem = NULL;
  alignment = 4096;
  isAMD = false;

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
    if (num_devices > 0) {
      if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
        isAMD = true;
      }
      // platform = platforms[_platformIndex];
      // break;
    }
#if 0
        }
#endif
    delete platforms;
  }
  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  char getVersion[128];
  error_ = _wrapper->clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(getVersion),
                                       getVersion, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
  platformVersion[0] = getVersion[7];
  platformVersion[1] = getVersion[8];
  platformVersion[2] = getVersion[9];
  platformVersion[3] = '\0';
  bufSize_ = Sizes[_openTest % NUM_SIZES];

  if (((_openTest / NUM_SIZES) % NUM_SUBTESTS) > 2) {
    useHostPtr = true;
    offset = offsets[((_openTest / NUM_SIZES) % NUM_SUBTESTS) - 3];
  } else if ((((_openTest / NUM_SIZES) % NUM_SUBTESTS) == 2) && isAMD) {
    persistent = true;
  } else if (((_openTest / NUM_SIZES) % NUM_SUBTESTS) == 1) {
    allocHostPtr = true;
  }

  if (_openTest < blockedSubtests) {
    numIter = Iterations[_openTest / (NUM_SIZES * NUM_SUBTESTS)];
  } else {
    numIter = 4 * OCLPerfBufferWriteSpeed::NUM_ITER / ((_openTest % NUM_SIZES) + 1);
  }

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
  if (persistent) {
    flags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
  } else if (allocHostPtr) {
    flags |= CL_MEM_ALLOC_HOST_PTR;
  } else if (useHostPtr) {
    flags |= CL_MEM_USE_HOST_PTR;
    hostMem = (char*)malloc(bufSize_ + alignment - 1 + offset);
    CHECK_RESULT(hostMem == 0, "malloc(hostMem) failed");
    alignedMem = (char*)((((intptr_t)hostMem + alignment - 1) & ~(alignment - 1)) + offset);
  }
  outBuffer_ = _wrapper->clCreateBuffer(context_, flags, bufSize_, alignedMem, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  // Force memory to be on GPU if possible
  {
    cl_mem memBuffer = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
    CHECK_RESULT(memBuffer == 0, "clCreateBuffer(memBuffer) failed");

    _wrapper->clEnqueueCopyBuffer(cmd_queue_, outBuffer_, memBuffer, 0, 0, bufSize_, 0, NULL, NULL);
    _wrapper->clFinish(cmd_queue_);

    _wrapper->clReleaseMemObject(memBuffer);
  }
}

void OCLPerfBufferWriteSpeed::run(void) {
  CPerfCounter timer;
  char* mem = new char[bufSize_];
  cl_bool blocking = (_openTest < blockedSubtests) ? CL_TRUE : CL_FALSE;

  // Warm up
  error_ = _wrapper->clEnqueueWriteBuffer(cmd_queue_, outBuffer_, CL_TRUE, 0, bufSize_, mem, 0,
                                          NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueReadBuffer failed");

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < numIter; i++) {
    error_ = _wrapper->clEnqueueWriteBuffer(cmd_queue_, outBuffer_, blocking, 0, bufSize_, mem, 0,
                                            NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueReadBuffer failed");
  }
  if (blocking != CL_TRUE) {
    _wrapper->clFinish(cmd_queue_);
  }
  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Buffer write bandwidth in GB/s
  double perf = ((double)bufSize_ * numIter * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  char str[256];
  if (persistent) {
    SNPRINTF(str, sizeof(str), "PERSISTENT (GB/s)");
  } else if (allocHostPtr) {
    SNPRINTF(str, sizeof(str), "ALLOC_HOST_PTR (GB/s)");
  } else if (useHostPtr) {
    SNPRINTF(str, sizeof(str), "off: %4d USE_HOST_PTR (GB/s)", offset);
  } else {
    SNPRINTF(str, sizeof(str), "(GB/s)");
  }
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes) %3s i: %4d %29s ", bufSize_, blkStr[blocking], numIter,
           str);
  testDescString = buf;

  delete mem;
}

unsigned int OCLPerfBufferWriteSpeed::close(void) {
  if (outBuffer_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }
  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }
  if (hostMem) {
    free(hostMem);
  }

  return _crcword;
}

void OCLPerfBufferWriteRectSpeed::run(void) {
  CPerfCounter timer;
  char* mem = new char[bufSize_];
  size_t width = static_cast<size_t>(sqrt(static_cast<float>(bufSize_)));
  size_t bufOrigin[3] = {0, 0, 0};
  size_t hostOrigin[3] = {0, 0, 0};
  size_t region[3] = {width, width, 1};
  cl_bool blocking = (_openTest < blockedSubtests) ? CL_TRUE : CL_FALSE;

  // Skip for 1.0 platforms
  if ((platformVersion[0] == '1') && (platformVersion[2] == '0')) {
    char buf[256];
    SNPRINTF(buf, sizeof(buf), " SKIPPED ");
    testDescString = buf;
    return;
  }
  // Warm up
  error_ =
      _wrapper->clEnqueueWriteBufferRect(cmd_queue_, outBuffer_, CL_TRUE, bufOrigin, hostOrigin,
                                         region, width, 0, width, 0, mem, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueReadBufferRect failed");

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < numIter; i++) {
    error_ =
        _wrapper->clEnqueueWriteBufferRect(cmd_queue_, outBuffer_, blocking, bufOrigin, hostOrigin,
                                           region, width, 0, width, 0, mem, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueReadBufferRect failed");
  }
  if (blocking != CL_TRUE) {
    _wrapper->clFinish(cmd_queue_);
  }
  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Buffer write bandwidth in GB/s
  double perf = ((double)bufSize_ * numIter * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  char str[256];
  if (persistent) {
    SNPRINTF(str, sizeof(str), "PERSISTENT (GB/s)");
  } else if (allocHostPtr) {
    SNPRINTF(str, sizeof(str), "ALLOC_HOST_PTR (GB/s)");
  } else if (useHostPtr) {
    SNPRINTF(str, sizeof(str), "off: %4d USE_HOST_PTR (GB/s)", offset);
  } else {
    SNPRINTF(str, sizeof(str), "(GB/s)");
  }
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes) %3s i: %4d %29s ", bufSize_, blkStr[blocking], numIter,
           str);
  testDescString = buf;

  delete mem;
}
