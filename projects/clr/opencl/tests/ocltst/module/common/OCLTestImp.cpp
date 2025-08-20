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

#include "OCLTestImp.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <cstdio>
#include <cstring>

/////////////////////////////////////////////////////////////////////////////

static unsigned int crcinit(unsigned int crc);
static int initializeSeed(void);

/////////////////////////////////////////////////////////////////////////////

OCLutil::Lock OCLTestImp::openDeviceLock;
OCLutil::Lock OCLTestImp::compileLock;

OCLTestImp::OCLTestImp()
    : _wrapper(0),
      _seed(0),
      error_(0),
      type_(0),
      deviceCount_(0),
      devices_(0),
      platform_(0),
      context_(0),
      program_(0),
      kernel_(0) {
  unsigned int i;
  for (i = 0; i < 256; i++) {
    _crctab[i] = crcinit(i << 24);
  }
  _perfInfo = 0;

  _wrapper = 0;
  _iterationCnt = 0;

  _seed = initializeSeed();

  _errorMsg = "";
  _errorFlag = false;
  type_ = CL_DEVICE_TYPE_GPU;
}

OCLTestImp::~OCLTestImp() {}
void OCLTestImp::useCPU() { type_ = CL_DEVICE_TYPE_CPU; }
void OCLTestImp::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  devices_ = 0;
  context_ = 0;
  program_ = 0;
  kernel_ = 0;
  deviceCount_ = 0;

  open(test, units, conversion, deviceId, getPlatformIndex());
}
void OCLTestImp::open(unsigned int test, char* units, double& conversion, unsigned int deviceId,
                      unsigned int platformIndex) {
  BaseTestImp::open();
  devices_ = 0;
  deviceCount_ = 0;
  context_ = 0;
  program_ = 0;
  kernel_ = 0;
  _deviceId = deviceId;
  _platformIndex = platformIndex;

  cl_uint numPlatforms = 0;
  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetPlatformIDs failed");
  CHECK_RESULT((numPlatforms == 0), "No platform found");

  cl_platform_id* platforms = new cl_platform_id[numPlatforms];
  error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");

  cl_platform_id platform = 0;
#if 0
  for(unsigned int i = 0; i < numPlatforms; ++i)
  {
    char buff[200];
    error_ = _wrapper->clGetPlatformInfo(platforms[i],CL_PLATFORM_VENDOR, sizeof(buff), buff, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
    if(strcmp(buff, "Advanced Micro Devices, Inc.") == 0)
    {
      platform = platforms[i];
      break;
    }
  }
#endif
  platform = platforms[_platformIndex];

  delete[] platforms;

  CHECK_RESULT((platform == 0), "AMD Platform not found");

  error_ = _wrapper->clGetDeviceIDs(platform, type_, 0, NULL, &deviceCount_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs() failed");

  devices_ = new cl_device_id[deviceCount_];
  error_ = _wrapper->clGetDeviceIDs(platform, type_, deviceCount_, devices_, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs() failed");

  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
  context_ = _wrapper->clCreateContext(props, deviceCount_, devices_, NULL, 0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateContext failed");

  cl_command_queue cmdQueue;
  for (unsigned int i = 0; i < deviceCount_; ++i) {
#ifndef CL_VERSION_2_0
    cmdQueue =
        _wrapper->clCreateCommandQueue(context_, devices_[i], CL_QUEUE_PROFILING_ENABLE, &error_);
#else
    cl_queue_properties prop[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cmdQueue = _wrapper->clCreateCommandQueueWithProperties(context_, devices_[i], prop, &error_);
#endif
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed");
    cmdQueues_.push_back(cmdQueue);
  }
  platform_ = platform;
}

unsigned int OCLTestImp::close() {
  for (unsigned int i = 0; i < buffers().size(); ++i) {
    error_ = _wrapper->clReleaseMemObject(buffers()[i]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseMemObject() failed");
  }
  buffers_.clear();

  if (kernel_ != 0) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseKernel() failed");
  }

  if (program_ != 0) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseProgram() failed");
  }

  for (unsigned int i = 0; i < cmdQueues_.size(); ++i) {
    error_ = _wrapper->clReleaseCommandQueue(cmdQueues_[i]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseCommandQueue() failed");
  }
  cmdQueues_.clear();

  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseContext() failed");
  }

  if (devices_) {
    delete[] devices_;
  }

  return BaseTestImp::close();
}

int OCLTestImp::genBitRand(int n) {
  int rslt;
  if (n <= 0 || n > 32) {
    assert(0);
    rslt = 0;
  } else if (n < 32) {
    _seed = _seed * 1103515245 + 12345;
    /*
     * return the most-significant n bits; they are the random ones (see
     * Knuth, Vol 2)
     */
    rslt = (_seed & 0x7fffffff) >> (31 - n);
  } else {
    rslt = (genBitRand(16) << 16) | genBitRand(16);
  }

  return rslt;
}

int OCLTestImp::genIntRand(int a, int b) {
  int r;
  int sign = 1;
  int mySmall;
  int delta;
  int bits = 0;
  int rslt;
  if (a > b) {
    mySmall = b;
    delta = a - b;
  } else {
    mySmall = a;
    delta = b - a;
  }
  if (delta == 0) {
    rslt = a;
    return (rslt);
  } else if (delta < 0) {
    sign = -1;
    delta = -delta;
  }
  delta &= 0x7fffffff;
  for (r = delta; r > 0; r >>= 1) {
    bits++;
  }
  do {
    r = genBitRand(bits);
  } while (r > delta);

  rslt = mySmall + r * sign;

  return (rslt);
}

void OCLTestImp::setOCLWrapper(OCLWrapper* wrapper) { _wrapper = wrapper; }

/////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32

#include <windows.h>

static int initializeSeed(void) {
  __int64 val;
  QueryPerformanceCounter((LARGE_INTEGER*)&val);
  return (int)val;
}

#endif  // _WIN32

/////////////////////////////////////////////////////////////////////////////

#ifdef __linux__

#include <sys/time.h>

static int initializeSeed(void) {
  struct timeval t;
  gettimeofday(&t, 0);
  return (int)t.tv_usec;
}

#endif  // __linux__

/////////////////////////////////////////////////////////////////////////////
//
// Same CRC32 as used by ogtst
//
static const unsigned int CRCMASK = 0x04c11db7;

static unsigned int crcinit(unsigned int crc) {
  int i;
  unsigned int ans = crc;

  for (i = 0; i < 8; i++) {
    if (ans & 0x80000000) {
      ans = (ans << 1) ^ CRCMASK;
    } else {
      ans <<= 1;
    }
  }
  return (ans);
}
