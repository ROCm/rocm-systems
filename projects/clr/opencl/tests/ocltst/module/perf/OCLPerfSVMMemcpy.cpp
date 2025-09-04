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

#include "OCLPerfSVMMemcpy.h"

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
#define NUM_SRC_FLAGS 2
#define NUM_DST_FLAGS 2
#define NUM_FG_FLAGS 3

static size_t sizeList[NUM_SIZES] = {
    0x040000, 0x080000, 0x100000, 0x200000, 0x400000,
};

#if defined(CL_VERSION_2_0)
static const cl_svm_mem_flags srcFlagList[NUM_SRC_FLAGS] = {CL_MEM_READ_WRITE, CL_MEM_READ_ONLY};
static const cl_svm_mem_flags dstFlagList[NUM_DST_FLAGS] = {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
static const cl_svm_mem_flags FGFlags[NUM_FG_FLAGS] = {
    0,
    CL_MEM_SVM_FINE_GRAIN_BUFFER,
    CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
};
#endif

OCLPerfSVMMemcpy::OCLPerfSVMMemcpy() {
  _numSubTests = (NUM_SRC_FLAGS * NUM_DST_FLAGS * NUM_FG_FLAGS + 1) * NUM_SIZES;
  failed_ = false;
  skip_ = false;
}

OCLPerfSVMMemcpy::~OCLPerfSVMMemcpy() {}

void OCLPerfSVMMemcpy::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

#if defined(CL_VERSION_2_0)
  FGSystem_ = (test >= (NUM_SIZES * NUM_SRC_FLAGS * NUM_DST_FLAGS * NUM_FG_FLAGS));
  testFGFlag_ = (test / (NUM_SIZES * NUM_DST_FLAGS * NUM_SRC_FLAGS)) % (NUM_FG_FLAGS);
  testSrcFlag_ = (test / (NUM_SIZES * NUM_DST_FLAGS)) % (NUM_SRC_FLAGS);
  testDstFlag_ = (test / NUM_SIZES) % (NUM_DST_FLAGS);
  testSize_ = test % NUM_SIZES;

  cl_device_svm_capabilities caps;
  error_ = clGetDeviceInfo(devices_[deviceId], CL_DEVICE_SVM_CAPABILITIES,
                           sizeof(cl_device_svm_capabilities), &caps, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if ((caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) == 0) {
    skip_ = true;  // Should never happen as OCL 2.0 devices are required to
                   // support coarse grain SVM
    testDescString = "Coarse Grain Buffer  NOT supported. Test Skipped.";
    return;
  } else if ((testFGFlag_ > 0) && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0) {
    skip_ = true;  // No support for fine grain buffer SVM
    testDescString = "Fine Grain Buffer NOT supported. Test Skipped.";
    return;
  } else if (FGSystem_ && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) == 0) {
    skip_ = true;  // No support for fine grain system SVM
    testDescString = "Fine Grain System NOT supported. Test Skipped.";
    return;
  } else if ((testFGFlag_ == 2) && ((caps & CL_DEVICE_SVM_ATOMICS) == 0)) {
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

void OCLPerfSVMMemcpy::run(void) {
  if (skip_) {
    return;
  }

  if (failed_) {
    return;
  }
#if defined(CL_VERSION_2_0)
  cl_uint *src = NULL, *dst = NULL;
  CPerfCounter timer;

  size_t bufSize = sizeList[testSize_] * sizeof(cl_int4);
  size_t iter = 100;

  cl_mem_flags srcFlags = srcFlagList[testSrcFlag_] | FGFlags[testFGFlag_];
  cl_mem_flags dstFlags = dstFlagList[testDstFlag_] | FGFlags[testFGFlag_];

  size_t gws[1] = {bufSize / sizeof(cl_int4)};
  size_t lws[1] = {64};

  if (!FGSystem_) {
    src = (cl_uint*)clSVMAlloc(context_, srcFlags, bufSize, 0);
    CHECK_RESULT(src == 0, "Allocation failed");
    dst = (cl_uint*)clSVMAlloc(context_, dstFlags, bufSize, 0);
    CHECK_RESULT(dst == 0, "Allocation failed");
  } else {  // FGSystem_ == true
    src = (cl_uint*)malloc(bufSize);
    dst = (cl_uint*)malloc(bufSize);
  }

  timer.Reset();
  timer.Start();
  for (size_t i = 0; i < iter; ++i) {
    clEnqueueSVMMemcpy(cmdQueues_[_deviceId], false, dst, src, bufSize, 0, NULL, NULL);
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();

  if (!FGSystem_) {
    clSVMFree(context_, (void*)src);
    clSVMFree(context_, (void*)dst);
  } else {  // FGSystem_ = true
    free(src);
    free(dst);
  }

  char pSrcFlags[5];
  pSrcFlags[0] = (testSrcFlag_ == 0 || testSrcFlag_ == 1) ? 'R' : '_';  // CL_MEM_READ_ONLY
  pSrcFlags[1] = (testSrcFlag_ == 0) ? 'W' : '_';                       // CL_MEM_WRITE_ONLY
  pSrcFlags[2] =
      (testFGFlag_ == 1 || testFGFlag_ == 2) ? 'F' : '_';  // CL_MEM_SVM_FINE_GRAIN_BUFFER
  pSrcFlags[3] = (testFGFlag_ == 2) ? 'A' : '_';           // CL_MEM_SVM_ATOMICS
  pSrcFlags[4] = '\0';

  char pDstFlags[5];
  pDstFlags[0] = (testDstFlag_ == 0) ? 'R' : '_';
  pDstFlags[1] = (testDstFlag_ == 0 || testDstFlag_ == 1) ? 'W' : '_';
  pDstFlags[2] = (testFGFlag_ == 1 || testFGFlag_ == 2) ? 'F' : '_';
  pDstFlags[3] = (testFGFlag_ == 2) ? 'A' : '_';
  pSrcFlags[4] = '\0';

  char buf[256];

  if (FGSystem_) {
    SNPRINTF(buf, sizeof(buf), "Fine Grain System   SVMMemcpy (GB/s) for %6d KB, from:%4s to:%4s",
             (int)bufSize / 1024, pSrcFlags, pDstFlags);
  } else if (testFGFlag_ == 0) {
    SNPRINTF(buf, sizeof(buf), "Coarse Grain Buffer SVMMemcpy (GB/s) for %6d KB, from:%4s to:%4s",
             (int)bufSize / 1024, pSrcFlags, pDstFlags);
  } else {
    SNPRINTF(buf, sizeof(buf), "Fine Grain Buffer   SVMMemcpy (GB/s) for %6d KB, from:%4s to:%4s",
             (int)bufSize / 1024, pSrcFlags, pDstFlags);
  }

  testDescString = buf;
  double sec = timer.GetElapsedTime();
  _perfInfo = static_cast<float>((bufSize * iter * (double)(1e-09)) / sec);
#endif
}

unsigned int OCLPerfSVMMemcpy::close(void) { return OCLTestImp::close(); }
