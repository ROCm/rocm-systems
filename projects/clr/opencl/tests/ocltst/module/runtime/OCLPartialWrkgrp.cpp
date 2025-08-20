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

#include "OCLPartialWrkgrp.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

static const size_t BufSize = 0x1000;

const static char* strKernel =
    "__kernel void fillX(__global int4* out)                                \n"
    "{                                                                      \n"
    "   int id = get_global_id(0);                                          \n"
    "   out[id].x = id;                                                     \n"
    "}                                                                      \n"
    "                                                                       \n"
    "__kernel void fillXY(__global int4* out)                               \n"
    "{                                                                      \n"
    "   int id = get_global_id(0) + get_global_id(1) * get_global_size(0);  \n"
    "   out[id].x = get_global_id(0);                                       \n"
    "   out[id].y = get_global_id(1);                                       \n"
    "}                                                                      \n"
    "                                                                       \n"
    "__kernel void fillXYZ(__global int4* out)                              \n"
    "{                                                                      \n"
    "   int id = get_global_id(0) + get_global_id(1) * get_global_size(0) + \n"
    "       get_global_id(2) * get_global_size(0) * get_global_size(1);     \n"
    "   out[id].x = get_global_id(0);                                       \n"
    "   out[id].y = get_global_id(1);                                       \n"
    "   out[id].z = get_global_id(2);                                       \n"
    "}                                                                      \n";

OCLPartialWrkgrp::OCLPartialWrkgrp() {
  _numSubTests = 3;
  isOCL2_ = true;
}

OCLPartialWrkgrp::~OCLPartialWrkgrp() {}

void OCLPartialWrkgrp::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  _openTest = test;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  char version[128];
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_VERSION, sizeof(version), version, NULL);

  if (_openTest != 0 && strstr(version, "OpenCL 2.0") == NULL) {
    isOCL2_ = false;
    return;
  }

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  switch (_openTest) {
    case 0:
      error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
      break;
    case 1:
      error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId],
                                        "-cl-uniform-work-group-size -cl-std=CL2.0", NULL, NULL);
      break;
    case 2:
      error_ =
          _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], "-cl-std=CL2.0", NULL, NULL);
      break;
    default:
      CHECK_RESULT(false, "Invalid test number > _numSubTests");
      return;
  }

  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "fillX", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, BufSize * sizeof(cl_int4), NULL,
                                    &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPartialWrkgrp::run(void) {
  if (!isOCL2_) return;
  unsigned int* values;
  cl_mem buffer = buffers()[0];
  values = reinterpret_cast<unsigned int*>(new cl_int4[BufSize]);

  //
  // Check unaligned workgroup in X dimension
  //

  // Clear destination buffer
  memset(values, 0, BufSize * sizeof(cl_int4));
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                          BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t gws[1] = {BufSize - 1};
  size_t lws[1] = {256};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                            NULL, NULL);

  switch (_openTest) {
    case 2:
      if (error_ != CL_SUCCESS) {
        CHECK_RESULT(false, "clEnqueueNDRangeKernel() failed");
        return;
      }
      error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                             BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");

      for (size_t x = 0; x < BufSize; ++x) {
        if (x == (BufSize - 1)) {
          CHECK_RESULT((values[4 * x] != 0), "Comparison failed!");
        } else {
          CHECK_RESULT((values[4 * x] != x), "Comparison failed!");
        }
      }
      break;
    case 1:
    case 0:
      CHECK_RESULT((error_ != CL_INVALID_WORK_GROUP_SIZE),
                   "clEnqueueNDRangeKernel(): "
                   "Expected to fail for non-uniform work group sizes!");
      break;
    default:
      CHECK_RESULT(false, "Invalid test number > _numSubTests");
      return;
  }

  error_ = _wrapper->clReleaseKernel(kernel_);
  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseKernel() failed");

  //
  // Check unaligned workgroup in X and Y dimensions
  //
  kernel_ = _wrapper->clCreateKernel(program_, "fillXY", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  // Clear destination buffer
  memset(values, 0, BufSize * sizeof(cl_int4));
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                          BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t gws2[2] = {0x3f, 0x3f};
  size_t lws2[2] = {16, 16};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, gws2, lws2, 0,
                                            NULL, NULL);

  switch (_openTest) {
    case 2:
      if (error_ != CL_SUCCESS) {
        CHECK_RESULT(false, "clEnqueueNDRangeKernel() failed");
        return;
      }
      error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                             BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");

      for (size_t y = 0; y < 0x40; ++y) {
        for (size_t x = 0; x < 0x3f; ++x) {
          size_t id = x + y * 0x3f;
          if (y == 0x3f) {
            CHECK_RESULT((values[4 * id] != 0), "Comparison failed!");
            CHECK_RESULT((values[4 * id + 1] != 0), "Comparison failed!");
          } else {
            CHECK_RESULT((values[4 * id] != x), "Comparison failed!");
            CHECK_RESULT((values[4 * id + 1] != y), "Comparison failed!");
          }
        }
      }
      break;
    case 1:
    case 0:
      CHECK_RESULT((error_ != CL_INVALID_WORK_GROUP_SIZE),
                   "clEnqueueNDRangeKernel(): "
                   "Expected to fail for non-uniform work group sizes!");
      break;
    default:
      CHECK_RESULT(false, "Invalid test number > _numSubTests");
      return;
  }

  error_ = _wrapper->clReleaseKernel(kernel_);
  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseKernel() failed");

  //
  // Check unaligned workgroup in X, Y and Z dimensions
  //
  kernel_ = _wrapper->clCreateKernel(program_, "fillXYZ", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  // Clear destination buffer
  memset(values, 0, BufSize * sizeof(cl_int4));
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                          BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t gws3[3] = {0xf, 0x10, 0xf};
  size_t lws3[3] = {4, 4, 4};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 3, NULL, gws3, lws3, 0,
                                            NULL, NULL);
  switch (_openTest) {
    case 2:
      if (error_ != CL_SUCCESS) {
        CHECK_RESULT(false, "clEnqueueNDRangeKernel() failed");
        return;
      }
      error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                             BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");

      for (size_t z = 0; z < 0x10; ++z) {
        for (size_t y = 0; y < 0x10; ++y) {
          for (size_t x = 0; x < 0xf; ++x) {
            size_t id = x + y * 0xf + z * 0xf0;
            if (z == 0xf) {
              CHECK_RESULT((values[4 * id] != 0), "Comparison failed!");
              CHECK_RESULT((values[4 * id + 1] != 0), "Comparison failed!");
              CHECK_RESULT((values[4 * id + 2] != 0), "Comparison failed!");
            } else {
              CHECK_RESULT((values[4 * id] != x), "Comparison failed!");
              CHECK_RESULT((values[4 * id + 1] != y), "Comparison failed!");
              CHECK_RESULT((values[4 * id + 2] != z), "Comparison failed!");
            }
          }
        }
      }
      break;
    case 1:
    case 0:
      CHECK_RESULT((error_ != CL_INVALID_WORK_GROUP_SIZE),
                   "clEnqueueNDRangeKernel(): "
                   "Expected fail for non-uniform work group sizes!");
      break;
    default:
      CHECK_RESULT(false, "Invalid test number > _numSubTests");
      return;
  }

  delete[] values;
}

unsigned int OCLPartialWrkgrp::close(void) { return OCLTestImp::close(); }
