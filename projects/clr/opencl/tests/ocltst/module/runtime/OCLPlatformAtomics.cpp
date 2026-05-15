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

#include "OCLPlatformAtomics.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static char* strKernel =
    "__kernel void test_atomic_kernel(volatile __global atomic_int *pSync, "
    "volatile __global atomic_int *ptr, int numIterations)\n"
    "{                                                                         "
    "                                                 \n"
    "   while(atomic_load_explicit(pSync,  memory_order_acquire, "
    "memory_scope_all_svm_devices) == 0);                           \n"
    "   for (int i = 0; i < numIterations; i++) {                              "
    "                                                 \n"
    "        atomic_fetch_add_explicit(ptr, 1, memory_order_acq_rel, "
    "memory_scope_all_svm_devices);                             \n"
    "   }                                                                      "
    "                                                 \n"
    "}                                                                         "
    "                                                 \n";

OCLPlatformAtomics::OCLPlatformAtomics() {
  _numSubTests = 1;
  failed_ = false;
  svmCaps_ = 0;
}

OCLPlatformAtomics::~OCLPlatformAtomics() {}

void OCLPlatformAtomics::open(unsigned int test, char* units, double& conversion,
                              unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  size_t param_size = 0;
  char* strVersion = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strVersion = new char[param_size];
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, param_size, strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strVersion[7] < '2') {
    failed_ = true;
    return;
  }
  delete strVersion;

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], "-cl-std=CL2.0", NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "test_atomic_kernel", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");
}

static int AtomicLoad(volatile cl_int* object) {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
  return InterlockedExchangeAdd((volatile long*)object, 0);
#elif defined(__GNUC__)
  return __sync_add_and_fetch(object, 0);
#else
  printf("Atomic load not supported, aborting...");
  return 0;
#endif
}

static int AtomicIncrement(volatile cl_int* object) {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
  return _InterlockedIncrement((volatile long*)object);
#elif defined(__GNUC__)
  return __sync_fetch_and_add(object, 1);
#endif
  printf("Atomic increment not supported, aborting...");
  return 0;
}

void OCLPlatformAtomics::run(void) {
  if (failed_) return;

#ifdef CL_VERSION_2_0
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_SVM_CAPABILITIES,
                                     sizeof(svmCaps_), &svmCaps_, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetDeviceInfo()  failed");

  if (!(svmCaps_ & CL_DEVICE_SVM_ATOMICS)) {
    printf("SVM atomics not supported, skipping test...\n");
    return;
  }

  volatile cl_int* pSyncBuf = (volatile cl_int*)_wrapper->clSVMAlloc(
      context_, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 0);
  CHECK_RESULT(!pSyncBuf, "clSVMAlloc() failed");
  *pSyncBuf = 0;

  volatile cl_int* pAtomicBuf = (volatile cl_int*)_wrapper->clSVMAlloc(
      context_, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 0);
  CHECK_RESULT(!pAtomicBuf, "clSVMAlloc() failed");
  *pAtomicBuf = 0;

  error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, 0, (const void*)pSyncBuf);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArgSVMPointer() failed");

  error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, 1, (const void*)pAtomicBuf);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArgSVMPointer() failed");

  cl_int numIterations = 0x100000;
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_int), &numIterations);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t globalWorkSize[1] = {1};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, globalWorkSize,
                                            NULL, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  clFlush(cmdQueues_[_deviceId]);

  AtomicIncrement(pSyncBuf);

  // wait until we see some activity from a device (try to run host side
  // simultaneously).
  while (AtomicLoad(pAtomicBuf /*, memory_order_relaxed*/) == 0);

  for (int i = 0; i < numIterations; i++) {
    AtomicIncrement(pAtomicBuf);
  }

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "clFinish() failed");

  int expected = numIterations * 2;
  CHECK_RESULT(*pAtomicBuf != expected, "Expected: 0x%x, found: 0x%x", expected, *pAtomicBuf);

  _wrapper->clSVMFree(context_, (void*)pSyncBuf);
  _wrapper->clSVMFree(context_, (void*)pAtomicBuf);
#endif
}

unsigned int OCLPlatformAtomics::close(void) { return OCLTestImp::close(); }
