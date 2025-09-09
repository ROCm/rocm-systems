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

#include "OCLPerfSVMKernelArguments.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <sstream>
#include <string>

#include "CL/cl.h"
#include "CL/cl_ext.h"

static const size_t BufSize = 0x1000;
static const size_t Iterations = 0x10000;
static const size_t TotalQueues = 4;
static const size_t TotalBufs = 4;
static const size_t TotalArgs = 4;

#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

static const char* Arguments[TotalArgs] = {
    "__global uint* out",
    "__global uint* out, __global uint* buf0, __global uint* buf1, __global "
    "uint* buf2, __global uint* buf3",
    "__global uint* out, __global uint* buf0, __global uint* buf1, __global "
    "uint* buf2, __global uint* buf3, \n"
    "__global uint* buf4, __global uint* buf5, __global uint* buf6, __global "
    "uint* buf7, __global uint* buf8",
    "__global uint* out, __global uint* buf0, __global uint* buf1, __global "
    "uint* buf2, __global uint* buf3,\n"
    "__global uint* buf4, __global uint* buf5, __global uint* buf6, __global "
    "uint* buf7, __global uint* buf8,\n"
    "__global uint* buf9, __global uint* buf10, __global uint* buf11, __global "
    "uint* buf12, __global uint* buf13,\n"
    "__global uint* buf14, __global uint* buf15, __global uint* buf16, "
    "__global uint* buf17, __global uint* buf18"};

static const char* strKernel =
    "__kernel void dummy(%s)                    \n"
    "{                                          \n"
    "   uint id = get_global_id(0);             \n"
    "   uint value = 1;                         \n"
    "   out[id] = value;                        \n"
    "}                                          \n";

OCLPerfSVMKernelArguments::OCLPerfSVMKernelArguments() {
  _numSubTests = TotalQueues * TotalArgs;  // * TotalBufs;
  failed_ = false;
  skip_ = false;
}

OCLPerfSVMKernelArguments::~OCLPerfSVMKernelArguments() {}

void OCLPerfSVMKernelArguments::open(unsigned int test, char* units, double& conversion,
                                     unsigned int deviceId) {
#if defined(CL_VERSION_2_0)
  // cl_mem  buffer;
  _deviceId = deviceId;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  test_ = test;
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

  size_t numArguments = (test_ / TotalQueues) % TotalArgs;
  char* program = new char[4096];
  SNPRINTF(program, sizeof(char) * 4096, strKernel, Arguments[numArguments]);
  program_ =
      _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&program, NULL, &error_);
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

  delete[] program;

  static const size_t NumBuffs[TotalBufs] = {0x20, 0x100, 0x800, 0x2000};

  size_t bufSize = BufSize * sizeof(cl_int);

  numBufs_ = (unsigned int)NumBuffs[test_ / (TotalQueues * TotalArgs)];
  inOutBuffer = (void**)malloc(sizeof(void*) * numBufs_);

  for (size_t b = 0; b < numBufs_; ++b) {
    inOutBuffer[b] = clSVMAlloc(context_, CL_MEM_READ_WRITE, bufSize, 0);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSVMAlloc() failed");
  }
#else
  skip_ = true;
  testDescString = "SVM NOT supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfSVMKernelArguments::run(void) {
  if (skip_) {
    return;
  }

  if (failed_) {
    return;
  }
#if defined(CL_VERSION_2_0)
  CPerfCounter timer;
  static const size_t Queues[] = {1, 2, 4, 8};
  size_t numQueues = Queues[test_ % TotalQueues];
  cl_uint numArguments;
  _wrapper->clGetKernelInfo(kernel_, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &numArguments, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetKernelInfo() failed");

  size_t iter = Iterations / numQueues / numBufs_;
  iter = (iter == 0) ? 1 : iter;

  std::vector<cl_command_queue> cmdQueues(numQueues);
  for (size_t q = 0; q < numQueues; ++q) {
    cl_command_queue cmdQueue =
        _wrapper->clCreateCommandQueue(context_, devices_[_deviceId], 0, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed");
    cmdQueues[q] = cmdQueue;
  }
  // Warm-up
  for (size_t b = 0; b < (numBufs_ / numArguments); ++b) {
    for (size_t q = 0; q < numQueues; ++q) {
      for (cl_uint a = 0; a < numArguments; ++a) {
        void* buffer = inOutBuffer[(b * numArguments + a) % numBufs_];
        error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, a, buffer);
        CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArgSVMPointer() failed");
      }

      size_t gws[1] = {256};
      size_t lws[1] = {256};
      error_ =
          _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
  }
  for (size_t q = 0; q < numQueues; ++q) {
    _wrapper->clFinish(cmdQueues[q]);
  }

  size_t disp = 0;
  timer.Reset();
  timer.Start();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t b = 0; b < numBufs_; ++b) {
      for (size_t q = 0; q < numQueues; ++q) {
        for (cl_uint a = 0; a < numArguments; ++a) {
          void* buffer = inOutBuffer[(b * numArguments + a) % numBufs_];
          error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, a, buffer);
          CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArgSVMPointer() failed");
        }

        size_t gws[1] = {256};
        size_t lws[1] = {256};
        error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 0, NULL,
                                                  NULL);
        CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
        disp++;
      }
    }
  }
  for (size_t q = 0; q < numQueues; ++q) {
    _wrapper->clFinish(cmdQueues[q]);
  }
  timer.Stop();

  for (size_t q = 0; q < numQueues; ++q) {
    error_ = _wrapper->clReleaseCommandQueue(cmdQueues[q]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseCommandQueue() failed");
  }

  std::stringstream stream;
  stream << "Setup time (us) for " << numQueues << " queues, ";
  stream.flags(std::ios::right | std::ios::showbase);
  stream.width(2);
  stream << numArguments;
  stream << " arguments, ";
  stream.flags(std::ios::right | std::ios::showbase);
  stream.width(4);
  stream << numBufs_ << " buffers";
  testDescString = stream.str();
  _perfInfo = static_cast<float>(timer.GetElapsedTime() * 1000000 / disp);
#endif
}

unsigned int OCLPerfSVMKernelArguments::close(void) {
#if defined(CL_VERSION_2_0)
  for (size_t b = 0; b < numBufs_; ++b) {
    _wrapper->clSVMFree(context_, inOutBuffer[b]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clSVMFree() failed");
  }
#endif
  return OCLTestImp::close();
}
