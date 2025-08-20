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

#include "OCLPerfSVMMemFill.h"

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

#define NUM_MODES 3
#define NUM_CG_FLAGS 2
#define NUM_FG_FLAGS 3

static size_t typeSizeList[] = {
    1,  // sizeof(cl_uchar)
    2,   4, 8, 16, 32, 64,
    128,  // sizeof(cl_ulong16)
};

static unsigned int eleNumList[] = {
    0x0020000, 0x0080000, 0x0200000, 0x0800000, 0x2000000,
};

#if defined(CL_VERSION_2_0)
static const cl_svm_mem_flags CGFlags[NUM_CG_FLAGS] = {
    CL_MEM_READ_WRITE,
    CL_MEM_WRITE_ONLY,
};
static const cl_svm_mem_flags FGFlags[NUM_FG_FLAGS] = {
    0,
    CL_MEM_SVM_FINE_GRAIN_BUFFER,
    CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
};
#endif

OCLPerfSVMMemFill::OCLPerfSVMMemFill() {
  num_typeSize_ = sizeof(typeSizeList) / sizeof(size_t);
  num_elements_ = sizeof(eleNumList) / sizeof(unsigned int);
  _numSubTests = num_elements_ * num_typeSize_ * (NUM_FG_FLAGS * NUM_CG_FLAGS + 1);
  failed_ = false;
  skip_ = false;
}

OCLPerfSVMMemFill::~OCLPerfSVMMemFill() {}

void OCLPerfSVMMemFill::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

#if defined(CL_VERSION_2_0)
  FGSystem_ = (test >= (num_elements_ * num_typeSize_ * NUM_FG_FLAGS * NUM_CG_FLAGS));
  testFGFlag_ = (test / (num_elements_ * num_typeSize_ * NUM_CG_FLAGS)) % NUM_FG_FLAGS;
  testCGFlag_ = (test / (num_elements_ * num_typeSize_)) % NUM_CG_FLAGS;
  testTypeSize_ = typeSizeList[(test / num_elements_) % num_typeSize_];
  testNumEle_ = eleNumList[test % num_elements_];

  cl_device_svm_capabilities caps;
  error_ = clGetDeviceInfo(devices_[deviceId], CL_DEVICE_SVM_CAPABILITIES,
                           sizeof(cl_device_svm_capabilities), &caps, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if ((caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) == 0) {
    skip_ = true;  // Should never happen as OCL 2.0 devices are required to
                   // support coarse grain SVM
    testDescString = "Coarse Grain Buffer  NOT supported. Test Skipped.";
    return;
  } else if (testFGFlag_ > 0 && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0) {
    skip_ = true;  // No support for fine grain buffer SVM
    testDescString = "Fine Grain Buffer NOT supported. Test Skipped.";
    return;
  } else if (FGSystem_ && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) == 0) {
    skip_ = true;  // No support for fine grain system SVM
    testDescString = "Fine Grain System NOT supported. Test Skipped.";
    return;
  } else if (testFGFlag_ == 2 && ((caps & CL_DEVICE_SVM_ATOMICS) == 0)) {
    skip_ = true;  // No support for SVM Atomic
    testDescString = "SVM Atomic        NOT supported. Test Skipped.";
    return;
  }

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }
  return;
#else
  skip_ = true;
  testDescString = "SVM NOT supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfSVMMemFill::run(void) {
  if (skip_) {
    return;
  }

  if (failed_) {
    return;
  }
#if defined(CL_VERSION_2_0)
  cl_uint* buffer = NULL;
  CPerfCounter timer;
  size_t iter = 100, bufSize = testNumEle_ * 4;

  cl_mem_flags flags = CGFlags[testCGFlag_] | FGFlags[testFGFlag_];

  void* data = malloc(bufSize);

  timer.Reset();

  if (!FGSystem_) {
    buffer = (cl_uint*)clSVMAlloc(context_, flags, bufSize, (cl_uint)testTypeSize_);
    CHECK_RESULT(buffer == 0, "Allocation failed");
  } else {  // FGSystem_ = true
    buffer = (cl_uint*)malloc(bufSize);
    CHECK_RESULT(buffer == 0, "Allocation failed");
  }

  timer.Start();
  for (size_t i = 0; i < iter; ++i) {
    error_ = clEnqueueSVMMemFill(cmdQueues_[_deviceId], buffer, data, testTypeSize_, bufSize, 0,
                                 NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueSVMMemFill() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();

  if (!FGSystem_) {
    clSVMFree(context_, (void*)buffer);
  } else {
    free(buffer);
  }

  char pFlags[5];
  pFlags[0] = (testCGFlag_ == 0 || testCGFlag_ == 2) ? 'R' : '_';  // CL_MEM_READ_ONLY
  pFlags[1] = (testCGFlag_ == 0 || testCGFlag_ == 1) ? 'W' : '_';  // CL_MEM_WRITE_ONLY
  pFlags[2] = (testFGFlag_ == 1 || testFGFlag_ == 2) ? 'F' : '_';  // CL_MEM_SVM_FINE_GRAIN_BUFFER
  pFlags[3] = (testFGFlag_ == 2) ? 'A' : '_';                      // CL_MEM_SVM_ATOMICS

  char buf[256];

  if (!FGSystem_ && (testFGFlag_ == 0)) {
    SNPRINTF(buf, sizeof(buf),
             "Coarse Grain Buffer SVMMemFill (GB/s) for %6d KB, typeSize:%3d, "
             "flags=%4s",
             (int)bufSize / 1024, (int)testTypeSize_, pFlags);
  } else if (!FGSystem_ && (testFGFlag_ > 0)) {
    SNPRINTF(buf, sizeof(buf),
             "Fine Grain Buffer   SVMMemFill (GB/s) for %6d KB, typeSize:%3d, "
             "flags=%4s",
             (int)bufSize / 1024, (int)testTypeSize_, pFlags);
  } else if (FGSystem_) {
    SNPRINTF(buf, sizeof(buf),
             "Fine Grain System   SVMMemFill (GB/s) for %6d KB, typeSize:%3d, "
             "flags=%4s",
             (int)bufSize / 1024, (int)testTypeSize_, pFlags);
  }

  testDescString = buf;
  double sec = timer.GetElapsedTime();
  _perfInfo = static_cast<float>((bufSize * iter * (double)(1e-09)) / sec);
#endif
}

unsigned int OCLPerfSVMMemFill::close(void) { return OCLTestImp::close(); }
