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

#include "OCLPerfCPUMemSpeed.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>

#include "CL/opencl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_SIZES 4
// 256KB, 1 MB, 4MB, 16 MB
static const unsigned int Sizes[NUM_SIZES] = {262144, 1048576, 4194304, 16777216};

#define ITER_COUNT 2
static const unsigned int Iterations[2] = {1, OCLPerfCPUMemSpeed::NUM_ITER};
#define NUM_OFFSETS 1
static const unsigned int offsets[NUM_OFFSETS] = {0};
#define NUM_SUBTESTS (3 + NUM_OFFSETS)
OCLPerfCPUMemSpeed::OCLPerfCPUMemSpeed() {
  _numSubTests = NUM_SIZES * NUM_SUBTESTS * ITER_COUNT * 3;
}

OCLPerfCPUMemSpeed::~OCLPerfCPUMemSpeed() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfCPUMemSpeed::open(unsigned int test, char* units, double& conversion,
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
  testMemset = false;
  isAMD = false;
  gpuSrc = false;

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
    if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
      isAMD = true;
    }

    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    CHECK_RESULT(num_devices == 0, "No devices found, cannot proceed");
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
  }
  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  bufSize_ = Sizes[_openTest % NUM_SIZES];
  if (((_openTest / NUM_SIZES) % NUM_SUBTESTS) > 2) {
    useHostPtr = true;
    offset = offsets[((_openTest / NUM_SIZES) % NUM_SUBTESTS) - 3];
  } else if ((((_openTest / NUM_SIZES) % NUM_SUBTESTS) == 2) && isAMD) {
    persistent = true;
  } else if (((_openTest / NUM_SIZES) % NUM_SUBTESTS) == 1) {
    allocHostPtr = true;
  }

  numIter = Iterations[(_openTest / (NUM_SIZES * NUM_SUBTESTS)) % 2];
  if (_openTest >= (NUM_SIZES * NUM_SUBTESTS * ITER_COUNT * 2))
    testMemset = true;
  else if (_openTest >= (NUM_SIZES * NUM_SUBTESTS * ITER_COUNT)) {
    gpuSrc = true;
    numIter = std::min(numIter, 10u);
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

  cl_mem_flags flags;
  if (gpuSrc) {
    flags = CL_MEM_WRITE_ONLY;
    mapFlags = CL_MAP_READ;
  } else {
    flags = CL_MEM_READ_ONLY;
    mapFlags = CL_MAP_WRITE;
  }
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

    _wrapper->clEnqueueCopyBuffer(cmd_queue_, memBuffer, outBuffer_, 0, 0, bufSize_, 0, NULL, NULL);
    _wrapper->clFinish(cmd_queue_);

    _wrapper->clReleaseMemObject(memBuffer);
  }
}

void OCLPerfCPUMemSpeed::run(void) {
  CPerfCounter timer;

  void* mem;
  // Warm up
  mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, outBuffer_, CL_TRUE, mapFlags, 0, bufSize_, 0,
                                     NULL, NULL, &error_);

  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, outBuffer_, mem, 0, NULL, NULL);
  CHECK_RESULT(error_, "clEnqueueUnmapBuffer failed");
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, outBuffer_, CL_TRUE, mapFlags, 0, bufSize_, 0,
                                     NULL, NULL, &error_);

  char* cpumem = new char[bufSize_];

  timer.Reset();
  timer.Start();
  if (testMemset) {
    for (unsigned int i = 0; i < numIter; i++) {
      memset(mem, 0, bufSize_);
    }
  } else {
    if (gpuSrc) {
      for (unsigned int i = 0; i < numIter; i++) {
        memcpy((void*)cpumem, mem, bufSize_);
      }
    } else {
      for (unsigned int i = 0; i < numIter; i++) {
        memcpy(mem, (void*)cpumem, bufSize_);
      }
    }
  }

  timer.Stop();

  delete[] cpumem;

  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, outBuffer_, mem, 0, NULL, NULL);
  CHECK_RESULT(error_, "clEnqueueUnmapBuffer failed");
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  double sec = timer.GetElapsedTime();

  // Map read bandwidth in GB/s
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
  const char* str2 = NULL;
  if (testMemset)
    str2 = "memset to dev";
  else {
    if (gpuSrc)
      str2 = "memcpy from dev";
    else
      str2 = "memcpy to dev";
  }

  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes) %15s i: %4d %29s ", bufSize_, str2, numIter, str);
  testDescString = buf;
}

unsigned int OCLPerfCPUMemSpeed::close(void) {
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
