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

#include "OCLUnalignedCopy.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "CL/cl_ext.h"

static const int BufSize = 64;

OCLUnalignedCopy::OCLUnalignedCopy() {
  _numSubTests = 1;
  failed_ = false;
}

OCLUnalignedCopy::~OCLUnalignedCopy() {}

void OCLUnalignedCopy::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  _deviceId = deviceId;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }
  cl_mem buffer;
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, BufSize * sizeof(cl_int4), NULL,
                                    &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, BufSize * sizeof(cl_int4), NULL,
                                    &error_);
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLUnalignedCopy::run(void) {
  if (failed_) {
    return;
  }

  char* values = new char[BufSize];
  char* results = new char[BufSize];

  for (int i = 0; i < BufSize; ++i) {
    values[i] = i;
  }

  static const char TestCnt = 7;
  char sizes[TestCnt][3] = {
      {5, 7, 13}, {5, 7, 12}, {4, 9, 12}, {4, 9, 15}, {27, 16, 15}, {27, 16, 13}, {32, 16, 13},
  };

  for (int i = 0; i < TestCnt; ++i) {
    error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers_[0], CL_FALSE, 0,
                                            BufSize, values, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

    cl_uint pattern = 0;
    error_ = /*_wrapper->*/ clEnqueueFillBuffer(cmdQueues_[_deviceId], buffers_[1], &pattern,
                                                sizeof(pattern), 0, BufSize, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueFillBuffer() failed");

    error_ = _wrapper->clEnqueueCopyBuffer(cmdQueues_[_deviceId], buffers_[0], buffers_[1],
                                           sizes[i][0], sizes[i][1], sizes[i][2], 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");

    error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[1], CL_TRUE, 0, BufSize,
                                           results, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");

    for (int j = 0; j < sizes[i][1]; ++j) {
      CHECK_RESULT(results[j] != 0, "Comparison failed");
    }
    for (int j = sizes[i][1], k = 0; j < (sizes[i][1] + sizes[i][2]); ++j, ++k) {
      CHECK_RESULT(results[j] != sizes[i][0] + k, "Comparison failed");
    }
    for (int j = (sizes[i][1] + sizes[i][2]); j < BufSize; ++j) {
      CHECK_RESULT(results[j] != 0, "Comparison failed");
    }
  }

  delete[] values;
  delete[] results;
}

unsigned int OCLUnalignedCopy::close(void) { return OCLTestImp::close(); }
