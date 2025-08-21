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

#include "OCLDeviceAtomic.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#if EMU_ENV
static const cl_uint TotalElements = 8 * 32 * 256;
#else
static const cl_uint TotalElements = 256 * 1024 * 1024;
#endif
static const cl_uint ArraySize = 256;
static cl_uint hostArray[ArraySize];

#define KERNEL_CODE(...) #__VA_ARGS__

const static char* strKernel[] = {
    KERNEL_CODE(
    \n __kernel void atomic_test1(__global uint* res) {
      __global atomic_uint* inc = (__global atomic_uint*)res;
      atomic_fetch_add_explicit(inc, 1, memory_order_acq_rel, memory_scope_device);
    }
    \n __kernel void atomic_test2(__global uint* res) {
      __global atomic_uint* inc = (__global atomic_uint*)res;
      atomic_fetch_add_explicit(inc, 1, memory_order_acq_rel, memory_scope_device);
    }
    \n),
#if EMU_ENV
    KERNEL_CODE(
    \n __kernel void atomic_test1(__global uint* res) {
      for (uint i = 0; i < 8 * 32; ++i) {
        for (uint j = 0; j < 256; ++j) {
          __global atomic_uint* inc = (__global atomic_uint*)&res[j];
          uint val = atomic_load_explicit(inc, memory_order_acquire, memory_scope_device);
          if (0 != val) {
            res[1] = get_global_id(0);
            res[2] = i;
            return;
          }
        }
      }
    }
    \n __kernel void atomic_test2(__global uint* res) {
      if (get_global_id(0) == 8 * 20 * 100) {
        __global atomic_uint* inc = (__global atomic_uint*)res;
        // atomic_fetch_add_explicit(inc, 1, memory_order_acq_rel,
        // memory_scope_device);
        atomic_store_explicit(inc, get_global_id(0), memory_order_release, memory_scope_device);
      }
    }
    \n)
#else
    KERNEL_CODE(
    \n __kernel void atomic_test1(__global uint* res) {
      for (uint i = 0; i < 256 * 1024; ++i) {
        for (uint j = 0; j < 256; ++j) {
          __global atomic_uint* inc = (__global atomic_uint*)&res[j];
          uint val = atomic_load_explicit(inc, memory_order_acquire, memory_scope_device);
          if (0 != val) {
            res[1] = get_global_id(0);
            res[2] = i;
            return;
          }
        }
      }
    }
    \n __kernel void atomic_test2(__global uint* res) {
      if (get_global_id(0) == 64 * 1000 * 1000) {
        __global atomic_uint* inc = (__global atomic_uint*)res;
        // atomic_fetch_add_explicit(inc, 1, memory_order_acq_rel,
        // memory_scope_device);
        atomic_store_explicit(inc, get_global_id(0), memory_order_release, memory_scope_device);
      }
    }
    \n)
#endif
};

OCLDeviceAtomic::OCLDeviceAtomic() : hostQueue_(NULL), failed_(false), kernel2_(NULL) {
  _numSubTests = 2;
}

OCLDeviceAtomic::~OCLDeviceAtomic() {}

void OCLDeviceAtomic::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;
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

  char dbuffer[1024] = {0};
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel[test], NULL, &error_);
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

  kernel_ = _wrapper->clCreateKernel(program_, "atomic_test1", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  kernel2_ = _wrapper->clCreateKernel(program_, "atomic_test2", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  memset(hostArray, 0, sizeof(hostArray));
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR, sizeof(hostArray), &hostArray,
                                    &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

#if defined(CL_VERSION_2_0)
  const cl_queue_properties cprops[] = {CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(0),
                                        0};
  hostQueue_ =
      _wrapper->clCreateCommandQueueWithProperties(context_, devices_[deviceId], cprops, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLDeviceAtomic::run(void) {
  if (failed_) return;
  cl_mem buffer = buffers()[0];

  size_t gws[1] = {TotalElements};
  size_t gws2[1] = {1};
  size_t gws3[1] = {TotalElements};

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  if (testID_ == 0) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  } else {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws2, NULL,
                                              0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }

  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  if (testID_ == 0) {
    error_ =
        _wrapper->clEnqueueNDRangeKernel(hostQueue_, kernel2_, 1, NULL, gws, NULL, 0, NULL, NULL);
  } else {
    error_ =
        _wrapper->clEnqueueNDRangeKernel(hostQueue_, kernel2_, 1, NULL, gws3, NULL, 0, NULL, NULL);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  _wrapper->clFlush(cmdQueues_[_deviceId]);
  _wrapper->clFlush(hostQueue_);

  _wrapper->clFinish(cmdQueues_[_deviceId]);
  _wrapper->clFinish(hostQueue_);

  error_ = _wrapper->clEnqueueReadBuffer(hostQueue_, buffer, CL_TRUE, 0, sizeof(hostArray),
                                         hostArray, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");

  if (testID_ == 0) {
    if (hostArray[0] != 2 * TotalElements) {
      printf("Counter: %d, expected: %d\n", hostArray[0], 2 * TotalElements);
      CHECK_RESULT(true, "Incorrect result for device atomic inc!\n");
    }
  } else {
    printf("Value: %d, thread: %d, iter: %d\n", hostArray[0], hostArray[1], hostArray[2]);
    if (hostArray[0] == 0) {
      CHECK_RESULT(true, "Incorrect result for device atomic inc!\n");
    }
  }
}

unsigned int OCLDeviceAtomic::close(void) {
  if (NULL != hostQueue_) {
    _wrapper->clReleaseCommandQueue(hostQueue_);
  }
  if (NULL != kernel2_) {
    _wrapper->clReleaseKernel(kernel2_);
  }
  return OCLTestImp::close();
}
