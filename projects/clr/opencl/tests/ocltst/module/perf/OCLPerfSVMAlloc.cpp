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

#include "OCLPerfSVMAlloc.h"

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
#define NUM_CG_FLAGS 3
#define NUM_FG_FLAGS 3

static size_t sizeList[NUM_SIZES] = {
    0x040000, 0x080000, 0x100000, 0x200000, 0x400000,
};

#if defined(CL_VERSION_2_0)
static const cl_svm_mem_flags CGFlags[NUM_CG_FLAGS] = {
    CL_MEM_READ_WRITE,
    CL_MEM_WRITE_ONLY,
    CL_MEM_READ_ONLY,
};
static const cl_svm_mem_flags FGFlags[NUM_FG_FLAGS] = {
    0,
    CL_MEM_SVM_FINE_GRAIN_BUFFER,
    CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
};
#endif

static const char* strKernel =
    "__kernel void dummy(__global uint* out)    \n"
    "{                                          \n"
    "   uint id = get_global_id(0);             \n"
    "   uint value = 1;                         \n"
    "   if ((int)get_local_id(0) < 0)           \n"
    "       out[id] = value;                    \n"
    "}                                          \n";

OCLPerfSVMAlloc::OCLPerfSVMAlloc() {
  _numSubTests = NUM_CG_FLAGS * NUM_FG_FLAGS * NUM_SIZES + NUM_SIZES;
  failed_ = false;
  skip_ = false;
}

OCLPerfSVMAlloc::~OCLPerfSVMAlloc() {}

void OCLPerfSVMAlloc::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

#if defined(CL_VERSION_2_0)
  FGSystem_ = (test >= (NUM_CG_FLAGS * NUM_FG_FLAGS * NUM_SIZES));
  testFGFlag_ = (test / (NUM_SIZES * NUM_CG_FLAGS)) % NUM_FG_FLAGS;
  testCGFlag_ = (test / NUM_SIZES) % NUM_CG_FLAGS;
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
  } else if (testFGFlag_ > 0 && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0) {
    skip_ = true;  // No support for fine grain buffer SVM
    testDescString = "Fine Grain Buffer NOT supported. Test Skipped.";
    return;
  } else if (FGSystem_ && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) == 0) {
    skip_ = true;  // No support for fine grain system SVM
    testDescString = "Fine Grain System NOT supported. Test Skipped.";
    return;
  } else if (testFGFlag_ == 2 && (caps & CL_DEVICE_SVM_ATOMICS) == 0) {
    skip_ = true;  // No support for fine grain system SVM
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
  kernel_ = _wrapper->clCreateKernel(program_, "dummy", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");
  return;
#else
  skip_ = true;
  testDescString = "SVM NOT supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfSVMAlloc::run(void) {
  if (skip_) {
    return;
  }

  if (failed_) {
    return;
  }
#if defined(CL_VERSION_2_0)
  cl_uint* buffer = NULL;
  CPerfCounter timer;
  void* hostPtr = NULL;

  size_t bufSize = sizeList[testSize_] * sizeof(cl_int4);
  size_t iter = 100;

  cl_mem_flags flags = CGFlags[testCGFlag_] | FGFlags[testFGFlag_];

  timer.Reset();
  timer.Start();

  size_t gws[1] = {bufSize / sizeof(cl_int4)};
  size_t lws[1] = {64};

  for (size_t i = 0; i < iter; ++i) {
    if (!FGSystem_) {
      buffer = (cl_uint*)clSVMAlloc(context_, flags, bufSize, 0);
    } else {
      buffer = (cl_uint*)malloc(bufSize);
    }
    CHECK_RESULT(buffer == 0, "Allocation failed");

    error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, 0, buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

    _wrapper->clFinish(cmdQueues_[_deviceId]);

    if (!FGSystem_) {
      clSVMFree(context_, (void*)buffer);
    } else {
      free(buffer);
    }
  }

  timer.Stop();

  CPerfCounter timer2;
  timer2.Reset();
  size_t numN = 100;

  if (!FGSystem_) {
    buffer = (cl_uint*)clSVMAlloc(context_, flags, bufSize, 0);
  } else {
    buffer = (cl_uint*)malloc(bufSize);
  }
  CHECK_RESULT(buffer == 0, "Allocation failed");

  timer2.Start();
  for (size_t i = 0; i < numN; ++i) {
    error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, 0, buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer2.Stop();

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
    SNPRINTF(buf, sizeof(buf), "Coarse Grain Buffer Alloc + Free (GB/s) for %6d KB, flags=%4s",
             (int)bufSize / 1024, pFlags);
  } else if (!FGSystem_ && (testFGFlag_ > 0)) {
    SNPRINTF(buf, sizeof(buf), "Fine Grain Buffer   Alloc + Free (GB/s) for %6d KB, flags=%4s",
             (int)bufSize / 1024, pFlags);
  } else if (FGSystem_) {
    SNPRINTF(buf, sizeof(buf), "Fine Grain System   Alloc + Free (GB/s) for %6d KB, flags=N/A ",
             (int)bufSize / 1024);
  }

  testDescString = buf;
  double sec1 = timer.GetElapsedTime();
  double sec2 = timer2.GetElapsedTime();
  _perfInfo = static_cast<float>((bufSize * (double)(1e-09)) / (sec1 / iter - sec2 / numN));
#endif
}

unsigned int OCLPerfSVMAlloc::close(void) { return OCLTestImp::close(); }
