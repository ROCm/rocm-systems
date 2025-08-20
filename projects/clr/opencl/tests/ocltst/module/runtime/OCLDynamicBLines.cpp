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

#include "OCLDynamicBLines.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static cl_int nLines = 2048;
const static cl_int blockDim = 64;
#define MAX_TESSELLATION 64

#define KERNEL_CODE(...) #__VA_ARGS__

const static char* strKernel[] =
{
    KERNEL_CODE(
    \n
        \x23 define MAX_TESSELLATION 64
    \n
        struct BezierLine
        {
            float2 CP[3];
            ulong vertexPos;
            int nVertices;
            int reserved;
        };
    \n
        __kernel
        void computeBezierLinePositions(int lidx, __global struct BezierLine* bLines,
            int nTessPoints, __global char* buf)
        {
            int idx = get_global_id(0);
            if (idx < nTessPoints) {
                float u = (float)idx / (float)(nTessPoints-1);
                float omu = 1.0f - u;

                float B3u[3];

                B3u[0] = omu * omu;
                B3u[1] = 2.0f * u * omu;
                B3u[2] = u * u;

                float2 position = {0, 0};

                for (int i = 0; i < 3; i++) {
                    position = position + B3u[i] * bLines[lidx].CP[i];
                }

                ((__global float2*)(bLines[lidx].vertexPos))[idx] = position;
            }
        }
    \n
        __kernel
        void computeBezierLines(__global struct BezierLine* bLines, int nLines, __global char* buf)
        {
            int lidx = get_global_id(0);

            if (lidx < nLines) {
                float curvature = length(bLines[lidx].CP[1] - 0.5f * (bLines[lidx].CP[0] + bLines[lidx].CP[2])) /
                    length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
                int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESSELLATION);

                if (bLines[lidx].vertexPos == 0) {
                    bLines[lidx].nVertices = nTessPoints;
                    uint value = atomic_add((__global volatile uint*)buf,
                        nTessPoints * sizeof(float2));
                    bLines[lidx].vertexPos = (ulong)(&buf[value]);
                }

                queue_t def_q = get_default_queue();
                ndrange_t ndrange = ndrange_1D(bLines[lidx].nVertices, 64);

                int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange,
                    ^{ computeBezierLinePositions(lidx, bLines, bLines[lidx].nVertices, buf); });
            }
        }
    \n
        __kernel
        void computeBezierLines2(__global struct BezierLine* bLines, int nLines, __global char* buf)
        {
            int lidx = get_global_id(0);

            if (lidx < nLines) {
                float curvature = length(bLines[lidx].CP[1] - 0.5f * (bLines[lidx].CP[0] + bLines[lidx].CP[2])) /
                    length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
                int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESSELLATION);

                if (bLines[lidx].vertexPos == 0) {
                    bLines[lidx].nVertices = nTessPoints;
                    uint value = atomic_add((__global volatile uint*)buf,
                        nTessPoints * sizeof(float2));
                    bLines[lidx].vertexPos = (ulong)(&buf[value]);
                }
            }
        }
    \n
    )
};

OCLDynamicBLines::OCLDynamicBLines() {
  _numSubTests = 1;
  deviceQueue_ = NULL;
  failed_ = false;
  bLines_ = NULL;
  hostArray_ = NULL;
  kernel2_ = NULL;
  kernel3_ = NULL;
}

OCLDynamicBLines::~OCLDynamicBLines() {}

void OCLDynamicBLines::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
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

  kernel_ = _wrapper->clCreateKernel(program_, "computeBezierLines", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  kernel2_ = _wrapper->clCreateKernel(program_, "computeBezierLines2", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  kernel3_ = _wrapper->clCreateKernel(program_, "computeBezierLinePositions", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  bLines_ = new BezierLine[nLines];

  cl_float2 last = {0, 0};
  for (int i = 0; i < nLines; i++) {
    bLines_[i].CP[0] = last;

    for (int j = 1; j < 3; j++) {
      bLines_[i].CP[j].s[0] = (float)rand() / (float)RAND_MAX;
      bLines_[i].CP[j].s[1] = (float)rand() / (float)RAND_MAX;
    }

    last = bLines_[i].CP[2];
    bLines_[i].vertexPos = 0;
    bLines_[i].nVertices = 0;
    bLines_[i].reserved = 0;
  }

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_USE_HOST_PTR, sizeof(BezierLine) * nLines,
                                    bLines_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

  hostArray_ = new cl_float2[nLines * (MAX_TESSELLATION + 1)];
  ((unsigned int*)hostArray_)[0] = sizeof(cl_float2);
  buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_USE_HOST_PTR,
                               sizeof(cl_float2) * nLines * MAX_TESSELLATION, hostArray_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

  cl_uint queueSize = 256 * 1024;
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

void OCLDynamicBLines::run(void) {
  CPerfCounter timer;
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

  if (failed_) return;

  cl_mem buffer = buffers()[0];
  cl_mem alloc = buffers()[1];

  size_t gws[1] = {nLines};
  size_t lws[1] = {blockDim};

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  error_ |= _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_int), &nLines);
  error_ |= _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_mem), &alloc);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  _wrapper->clFinish(cmdQueues_[_deviceId]);

  for (int i = 0; i < nLines; i++) {
    bLines_[i].vertexPos = 0;
    bLines_[i].nVertices = 0;
    bLines_[i].reserved = 0;
  }
  ((unsigned int*)hostArray_)[0] = sizeof(cl_float2);

  timer.Reset();
  timer.Start();
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();
  double sec = timer.GetElapsedTime();

  for (int i = 0; i < nLines; i++) {
    bLines_[i].vertexPos = 0;
    bLines_[i].nVertices = 0;
    bLines_[i].reserved = 0;
  }
  unsigned int allocSize = ((unsigned int*)hostArray_)[0];
  ((unsigned int*)hostArray_)[0] = sizeof(cl_float2);

  //
  // Host emulation
  //
  timer.Reset();
  timer.Start();
  // Step 1. Fill the jobs
  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), &buffer);
  error_ |= _wrapper->clSetKernelArg(kernel2_, 1, sizeof(cl_int), &nLines);
  error_ |= _wrapper->clSetKernelArg(kernel2_, 2, sizeof(cl_mem), &alloc);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel2_, 1, NULL, gws, lws, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  _wrapper->clFinish(cmdQueues_[_deviceId]);

  // Step 2. Run all jobs
  for (int lidx = 0; lidx < nLines; lidx++) {
    // Readback the new dimension.
    error_ = _wrapper->clSetKernelArg(kernel3_, 0, sizeof(cl_int), &lidx);
    error_ |= _wrapper->clSetKernelArg(kernel3_, 1, sizeof(cl_mem), &buffer);
    error_ |= _wrapper->clSetKernelArg(kernel3_, 2, sizeof(cl_int), &bLines_[lidx].nVertices);
    error_ |= _wrapper->clSetKernelArg(kernel3_, 3, sizeof(cl_mem), &alloc);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    size_t gwsL[1] = {static_cast<size_t>(bLines_[lidx].nVertices)};
    size_t lwsL[1] = {blockDim};

    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel3_, 1, NULL, gws, lws, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }

  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();
  double sec2 = timer.GetElapsedTime();

  if (memcmp(&allocSize, hostArray_, sizeof(cl_uint)) != 0) {
    CHECK_RESULT(true, "Validaiton failed!");
  }

  if (sec >= sec2) {
    _perfInfo = (float)(sec2 - sec);
    CHECK_RESULT(true, "Device enqueue is slower than emulation (sec)");
    return;
  }

  _perfInfo = (float)(((sec2 - sec) / sec) * 100);
  testDescString = "Device enqueue is (%%) faster";
}

unsigned int OCLDynamicBLines::close(void) {
  // FIXME: Re-enable CPU test once bug 10143 is fixed.
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return 0;
  }

  delete[] bLines_;
  delete[] hostArray_;

  if (NULL != deviceQueue_) {
    _wrapper->clReleaseCommandQueue(deviceQueue_);
  }
  if (NULL != kernel2_) {
    _wrapper->clReleaseKernel(kernel2_);
  }
  if (NULL != kernel3_) {
    _wrapper->clReleaseKernel(kernel3_);
  }
  return OCLTestImp::close();
}
