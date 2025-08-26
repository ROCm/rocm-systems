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

#include "OCLPerfBufferCopySpeed.h"

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
// 4KB, 8KB, 64KB, 256KB, 1 MB, 4MB, 16 MB, 16MB+10
static const unsigned int Sizes[NUM_SIZES] = {4096,    8192,    65536,    262144,
                                              1048576, 4194304, 16777216, 16777216 + 10};

static const unsigned int Iterations[2] = {1, OCLPerfBufferCopySpeed::NUM_ITER};

#define BUF_TYPES 4
//  16 ways to combine 4 different buffer types
#define NUM_SUBTESTS (BUF_TYPES * BUF_TYPES)

OCLPerfBufferCopySpeed::OCLPerfBufferCopySpeed() { _numSubTests = NUM_SIZES * NUM_SUBTESTS * 2; }

OCLPerfBufferCopySpeed::~OCLPerfBufferCopySpeed() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfBufferCopySpeed::setData(void* ptr, unsigned int size, unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  value = 0;
  for (unsigned int i = 0; i < size >> 2; i++) {
    ptr2[i] = value;
    value++;
  }
}

void OCLPerfBufferCopySpeed::checkData(void* ptr, unsigned int size, unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  value = 0;
  for (unsigned int i = 0; i < size >> 2; i++) {
    if (ptr2[i] != value) {
      printf("Data validation failed at %d!  Got 0x%08x 0x%08x 0x%08x 0x%08x\n", i, ptr2[i],
             ptr2[i + 1], ptr2[i + 2], ptr2[i + 3]);
      printf("Expected 0x%08x 0x%08x 0x%08x 0x%08x\n", value, value, value, value);
      CHECK_RESULT(true, "Data validation failed!");
      break;
    }
    value++;
  }
}

void OCLPerfBufferCopySpeed::open(unsigned int test, char* units, double& conversion,
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
  srcBuffer_ = 0;
  dstBuffer_ = 0;
  persistent[0] = false;
  persistent[1] = false;
  allocHostPtr[0] = false;
  allocHostPtr[1] = false;
  useHostPtr[0] = false;
  useHostPtr[1] = false;
  memptr[0] = NULL;
  memptr[1] = NULL;
  alignedmemptr[0] = NULL;
  alignedmemptr[1] = NULL;
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

  char getVersion[128];
  error_ = _wrapper->clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(getVersion),
                                       getVersion, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
  platformVersion[0] = getVersion[7];
  platformVersion[1] = getVersion[8];
  platformVersion[2] = getVersion[9];
  platformVersion[3] = '\0';
  bufSize_ = Sizes[_openTest % NUM_SIZES];
  unsigned int srcTest = (_openTest / NUM_SIZES) % BUF_TYPES;
  unsigned int dstTest = (_openTest / (NUM_SIZES * BUF_TYPES)) % BUF_TYPES;
  if (srcTest == 3) {
    useHostPtr[0] = true;
  } else if ((srcTest == 2) && isAMD) {
    persistent[0] = true;
  } else if (srcTest == 1) {
    allocHostPtr[0] = true;
  }
  if ((dstTest == 1) && isAMD) {
    persistent[1] = true;
  } else if (dstTest == 2) {
    allocHostPtr[1] = true;
  } else if (dstTest == 3) {
    useHostPtr[1] = true;
  }

  numIter = Iterations[_openTest / (NUM_SIZES * NUM_SUBTESTS)];

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
  if (persistent[0]) {
    flags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
  } else if (allocHostPtr[0]) {
    flags |= CL_MEM_ALLOC_HOST_PTR;
  } else if (useHostPtr[0]) {
    flags |= CL_MEM_USE_HOST_PTR;
    memptr[0] = malloc(bufSize_ + 4096);
    alignedmemptr[0] = (void*)(((size_t)memptr[0] + 4095) & ~4095);
  }
  srcBuffer_ = _wrapper->clCreateBuffer(context_, flags, bufSize_, alignedmemptr[0], &error_);
  CHECK_RESULT(srcBuffer_ == 0, "clCreateBuffer(srcBuffer) failed");
  void* mem;
  mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, srcBuffer_, CL_TRUE, CL_MAP_WRITE, 0, bufSize_, 0,
                                     NULL, NULL, &error_);
  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  setData(mem, bufSize_, 0x600df00d);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, srcBuffer_, mem, 0, NULL, NULL);

  flags = CL_MEM_WRITE_ONLY;
  if (persistent[1]) {
    flags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
  } else if (allocHostPtr[1]) {
    flags |= CL_MEM_ALLOC_HOST_PTR;
  } else if (useHostPtr[1]) {
    flags |= CL_MEM_USE_HOST_PTR;
    memptr[1] = malloc(bufSize_ + 4096);
    alignedmemptr[1] = (void*)(((size_t)memptr[1] + 4095) & ~4095);
  }
  dstBuffer_ = _wrapper->clCreateBuffer(context_, flags, bufSize_, alignedmemptr[1], &error_);
  CHECK_RESULT(dstBuffer_ == 0, "clCreateBuffer(dstBuffer) failed");

  // Force persistent memory to be on GPU
  if (persistent[0]) {
    cl_mem memBuffer = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
    CHECK_RESULT(memBuffer == 0, "clCreateBuffer(memBuffer) failed");

    _wrapper->clEnqueueCopyBuffer(cmd_queue_, memBuffer, dstBuffer_, 0, 0, bufSize_, 0, NULL, NULL);
    _wrapper->clFinish(cmd_queue_);

    _wrapper->clReleaseMemObject(memBuffer);
  }
  if (persistent[1]) {
    cl_mem memBuffer = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
    CHECK_RESULT(memBuffer == 0, "clCreateBuffer(memBuffer) failed");

    _wrapper->clEnqueueCopyBuffer(cmd_queue_, srcBuffer_, memBuffer, 0, 0, bufSize_, 0, NULL, NULL);
    _wrapper->clFinish(cmd_queue_);

    _wrapper->clReleaseMemObject(memBuffer);
  }
}

void OCLPerfBufferCopySpeed::run(void) {
  CPerfCounter timer;

  // Warm up
  error_ = _wrapper->clEnqueueCopyBuffer(cmd_queue_, srcBuffer_, dstBuffer_, 0, 0, bufSize_, 0,
                                         NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < numIter; i++) {
    error_ = _wrapper->clEnqueueCopyBuffer(cmd_queue_, srcBuffer_, dstBuffer_, 0, 0, bufSize_, 0,
                                           NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");
  }
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Buffer copy bandwidth in GB/s
  double perf = ((double)bufSize_ * numIter * (double)(1e-09)) / sec;

  void* mem;
  mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, dstBuffer_, CL_TRUE, CL_MAP_READ, 0, bufSize_, 0,
                                     NULL, NULL, &error_);
  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  checkData(mem, bufSize_, 0x600df00d);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, dstBuffer_, mem, 0, NULL, NULL);

  const char* strSrc = NULL;
  const char* strDst = NULL;
  if (persistent[0])
    strSrc = "per";
  else if (allocHostPtr[0])
    strSrc = "AHP";
  else if (useHostPtr[0])
    strSrc = "UHP";
  else
    strSrc = "dev";
  if (persistent[1])
    strDst = "per";
  else if (allocHostPtr[1])
    strDst = "AHP";
  else if (useHostPtr[1])
    strDst = "UHP";
  else
    strDst = "dev";
  // Double results when src and dst are both on device
  if ((persistent[0] || (!allocHostPtr[0] && !useHostPtr[0])) &&
      (persistent[1] || (!allocHostPtr[1] && !useHostPtr[1])))
    perf *= 2.0;
  // Double results when src and dst are both in sysmem
  if ((allocHostPtr[0] || useHostPtr[0]) && (allocHostPtr[1] || useHostPtr[1])) perf *= 2.0;
  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes) s:%s d:%s i:%4d (GB/s) ", bufSize_, strSrc, strDst,
           numIter);
  testDescString = buf;
}

unsigned int OCLPerfBufferCopySpeed::close(void) {
  if (srcBuffer_) {
    error_ = _wrapper->clReleaseMemObject(srcBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(srcBuffer_) failed");
  }
  if (dstBuffer_) {
    error_ = _wrapper->clReleaseMemObject(dstBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(dstBuffer_) failed");
  }
  if (memptr[0]) {
    free(memptr[0]);
  }
  if (memptr[1]) {
    free(memptr[1]);
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

void OCLPerfBufferCopyRectSpeed::run(void) {
  CPerfCounter timer;
  size_t width = static_cast<size_t>(sqrt(static_cast<float>(bufSize_)));
  size_t srcOrigin[3] = {0, 0, 0};
  size_t dstOrigin[3] = {0, 0, 0};
  size_t region[3] = {width, width, 1};
  // Clamp iteration count for non-local writes to shorten test runtime
  unsigned int testNumIter = numIter;

  if (allocHostPtr[1]) {
    testNumIter = (numIter < 100 ? numIter : 100);
  }

  // Skip for 1.0 platforms
  if ((platformVersion[0] == '1') && (platformVersion[2] == '0')) {
    char buf[256];
    SNPRINTF(buf, sizeof(buf), " SKIPPED ");
    testDescString = buf;
    return;
  }
  // Warm up
  error_ = _wrapper->clEnqueueCopyBufferRect(cmd_queue_, srcBuffer_, dstBuffer_, srcOrigin,
                                             dstOrigin, region, width, 0, width, 0, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueCopyBufferRect failed");
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < testNumIter; i++) {
    error_ =
        _wrapper->clEnqueueCopyBufferRect(cmd_queue_, srcBuffer_, dstBuffer_, srcOrigin, dstOrigin,
                                          region, width, 0, width, 0, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueCopyBufferRect failed");
  }
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Buffer copy bandwidth in GB/s
  double perf = ((double)bufSize_ * testNumIter * (double)(1e-09)) / sec;

  const char* strSrc = NULL;
  const char* strDst = NULL;
  if (persistent[0])
    strSrc = "per";
  else if (allocHostPtr[0])
    strSrc = "AHP";
  else if (useHostPtr[0])
    strSrc = "UHP";
  else
    strSrc = "dev";
  if (persistent[1])
    strDst = "per";
  else if (allocHostPtr[1])
    strDst = "AHP";
  else if (useHostPtr[1])
    strDst = "UHP";
  else
    strDst = "dev";
  // Double results when src and dst are both on device
  if ((persistent[0] || (!allocHostPtr[0] && !useHostPtr[0])) &&
      (persistent[1] || (!allocHostPtr[1] && !useHostPtr[1])))
    perf *= 2.0;
  // Double results when src and dst are both in sysmem
  if ((allocHostPtr[0] || useHostPtr[0]) && (allocHostPtr[1] || useHostPtr[1])) perf *= 2.0;
  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes) s:%s d:%s i:%4d (GB/s) ", bufSize_, strSrc, strDst,
           testNumIter);
  testDescString = buf;
}
