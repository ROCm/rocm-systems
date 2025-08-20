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

#include "OCLPerfSVMMap.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <sstream>
#include <string>

#include "CL/cl.h"
#include "CL/cl_ext.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_SIZES 5
static size_t sizeList[] = {
    0x040000, 0x080000, 0x100000, 0x200000, 0x400000,
};

#define NUM_FLAGS 4
static const cl_map_flags Flags[NUM_FLAGS] = {CL_MAP_READ, CL_MAP_WRITE, CL_MAP_READ | CL_MAP_WRITE,
                                              CL_MAP_WRITE_INVALIDATE_REGION};

OCLPerfSVMMap::OCLPerfSVMMap() {
  _numSubTests = NUM_SIZES * NUM_FLAGS;
  failed_ = false;
  skip_ = false;
}

OCLPerfSVMMap::~OCLPerfSVMMap() {}

void OCLPerfSVMMap::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
#if defined(CL_VERSION_2_0)
  _deviceId = deviceId;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  testFlag_ = test / NUM_SIZES;
  testSize_ = test % NUM_SIZES;

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  cl_device_svm_capabilities caps;
  error_ = clGetDeviceInfo(devices_[deviceId], CL_DEVICE_SVM_CAPABILITIES,
                           sizeof(cl_device_svm_capabilities), &caps, NULL);
  // check if CL_DEVICE_SVM_COARSE_GRAIN_BUFFER is set. Skip the test if not.
  if (!(caps & 0x1)) {
    skip_ = true;
    testDescString = "SVM NOT supported. Test Skipped.";
    return;
  }

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }
#else
  skip_ = true;
  testDescString = "SVM NOT supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfSVMMap::run(void) {
  if (skip_) {
    return;
  }

  if (failed_) {
    return;
  }
#if defined(CL_VERSION_2_0)
  void* buffer;
  CPerfCounter timer;
  void* hostPtr = NULL;

  const size_t bufSize = sizeList[testSize_] * sizeof(cl_int4);
  const cl_map_flags flag = Flags[testFlag_];
  const size_t iter = 100;

  timer.Reset();

  buffer = clSVMAlloc(context_, CL_MEM_READ_WRITE, bufSize, 0);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSVMAlloc() failed");

  for (size_t i = 0; i < iter; ++i) {
    timer.Start();

    error_ = clEnqueueSVMMap(cmdQueues_[_deviceId], CL_FALSE, flag, buffer, bufSize, 0, 0, 0);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueSVMMap() failed");

    error_ = clEnqueueSVMUnmap(cmdQueues_[_deviceId], buffer, 0, 0, 0);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueSVMUnmap() failed");

    _wrapper->clFinish(cmdQueues_[_deviceId]);

    timer.Stop();
  }

  clSVMFree(context_, (void*)buffer);

  char pFlags[4];
  pFlags[0] = (testFlag_ == 0 || testFlag_ == 2) ? 'R' : '_';  // CL_MAP_READ
  pFlags[1] = (testFlag_ == 1 || testFlag_ == 2) ? 'W' : '_';  // CL_MAP_WRITE
  pFlags[2] = (testFlag_ == 3) ? 'I' : '_';                    // CL_MAP_WRITE_INVALIDATE_REGION

  char buf[256];
  SNPRINTF(buf, sizeof(buf), "Map + Unmap (GB/s) for %6d KB, flags=%3s", (int)bufSize / 1024,
           pFlags);

  testDescString = buf;
  double sec = timer.GetElapsedTime();
  _perfInfo = static_cast<float>((bufSize * iter * (double)(1e-09)) / sec);
#endif
}

unsigned int OCLPerfSVMMap::close(void) { return OCLTestImp::close(); }
