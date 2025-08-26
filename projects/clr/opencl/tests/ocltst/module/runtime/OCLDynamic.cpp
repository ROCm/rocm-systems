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

#include "OCLDynamic.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#if EMU_ENV
static const cl_uint TotalElements = 1;
#else
static const cl_uint TotalElements = 128;
#endif  // EMU_ENV
static cl_uint hostArray[TotalElements];

#define KERNEL_CODE(...) #__VA_ARGS__

const static char* strKernel[] = {
    KERNEL_CODE(
    \n void block_fn(int tid, int mul, __global uint* res) { res[tid] = mul * 7 - 21; }

        __kernel void dynamic(__global uint* res) {
          int multiplier = 3;
          int tid = get_global_id(0);

          void (^kernelBlock)(void) = ^{
            block_fn(tid, multiplier, res);
          };

          res[tid] = -1;
          queue_t def_q = get_default_queue();
          ndrange_t ndrange = ndrange_1D(1);
          int enq_res;
          do {
            enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, kernelBlock);
            if (enq_res != 0 /*CL_SUCCESS*/) {
              res[tid] = -2;
            }
          } while (enq_res != 0);
        }
    \n),
    KERNEL_CODE(
    \n void block_fn(int tid, int mul, __global uint* res) { res[tid] = mul * 7 - 21; }

        __kernel void dynamic(__global uint* res, queue_t def_q) {
          int multiplier = 3;
          int tid = get_global_id(0);

          void (^kernelBlock)(void) = ^{
            block_fn(tid, multiplier, res);
          };

          res[tid] = -1;
          ndrange_t ndrange = ndrange_1D(1);
          // if (tid == 0) {
          int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);
          if (enq_res != 0 /*CL_SUCCESS*/) {
            res[tid] = -2;
            return;
          }
          //}
        }
    \n)};

OCLDynamic::OCLDynamic() {
  _numSubTests = 2;
  deviceQueue_ = NULL;
  failed_ = false;
}

OCLDynamic::~OCLDynamic() {}

void OCLDynamic::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  // FIXME: Re-enable CPU test once bug 10143 is fixed.
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

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

  kernel_ = _wrapper->clCreateKernel(program_, "dynamic", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  memset(hostArray, 0xee, sizeof(hostArray));
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                                    sizeof(hostArray), &hostArray, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
#if EMU_ENV
  cl_uint queueSize = 1;
#else
  cl_uint queueSize = (test == 0) ? 1 : 257 * 1024;
#endif  // EMU_ENV
#if defined(CL_VERSION_2_0)
  const cl_queue_properties cprops[] = {
      CL_QUEUE_PROPERTIES,
      static_cast<cl_queue_properties>(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                       CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_ON_DEVICE),
      CL_QUEUE_SIZE, queueSize, 0};
  deviceQueue_ =
      _wrapper->clCreateCommandQueueWithProperties(context_, devices_[deviceId], cprops, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLDynamic::run(void) {
  // FIXME: Re-enable CPU test once bug 10143 is fixed.
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

  if (failed_) return;
  cl_mem buffer = buffers()[0];

  size_t gws[1] = {TotalElements};
  size_t lws[1] = {16};

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  if (testID_ == 1) {
    error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_command_queue), &deviceQueue_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
  }

  size_t offset = 0;
  size_t region = TotalElements * sizeof(cl_uint);

  cl_uint* host = reinterpret_cast<cl_uint*>(_wrapper->clEnqueueMapBuffer(
      cmdQueues_[_deviceId], buffer, CL_TRUE, (CL_MAP_READ | CL_MAP_WRITE), offset, region, 0, NULL,
      NULL, &error_));
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1,
#if EMU_ENV
                                            NULL, gws, NULL, 0, NULL, NULL);
#else
                                            NULL, gws, lws, 0, NULL, NULL);
#endif  // EMU_ENV
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  _wrapper->clFinish(cmdQueues_[_deviceId]);

  for (unsigned int i = 0; i < TotalElements; ++i) {
    if (host[i] != 0) {
      printf("Bad value: a[%d] = %d\n", i, hostArray[i]);
      CHECK_RESULT(true, "Incorrect result for dependency!\n");
    }
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], buffer, host, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueUnmapBuffer() failed");

  _wrapper->clFinish(cmdQueues_[_deviceId]);
}

unsigned int OCLDynamic::close(void) {
  // FIXME: Re-enable CPU test once bug 10143 is fixed.
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return 0;
  }

  if (NULL != deviceQueue_) {
    _wrapper->clReleaseCommandQueue(deviceQueue_);
  }
  return OCLTestImp::close();
}
