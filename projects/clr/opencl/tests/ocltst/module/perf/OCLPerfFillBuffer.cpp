/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLPerfFillBuffer.h"

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

static size_t typeSizeList[] = {
    1,  // sizeof(cl_uchar)
    2,   4, 8, 16, 32, 64,
    128,  // sizeof(cl_ulong16)
};

static unsigned int eleNumList[] = {
    0x0020000, 0x0080000, 0x0200000, 0x0800000, 0x2000000,
};

OCLPerfFillBuffer::OCLPerfFillBuffer() {
  num_typeSize_ = sizeof(typeSizeList) / sizeof(size_t);
  num_elements_ = sizeof(eleNumList) / sizeof(unsigned int);
  _numSubTests = num_elements_ * num_typeSize_;
  failed_ = false;
  skip_ = false;
}

OCLPerfFillBuffer::~OCLPerfFillBuffer() {}

void OCLPerfFillBuffer::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  testTypeSize_ = typeSizeList[(test / num_elements_) % num_typeSize_];
  testNumEle_ = eleNumList[test % num_elements_];

  bufSize_ = testNumEle_ * 4;

  buffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, bufSize_, 0, &error_);
  CHECK_RESULT(buffer_ == 0, "clCreateBuffer(buffer_) failed");

  return;
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfFillBuffer::run(void) {
  CPerfCounter timer;
  size_t iter = 100;

  void* data = malloc(testTypeSize_);

  timer.Reset();
  timer.Start();
  for (size_t i = 0; i < iter; ++i) {
    error_ = clEnqueueFillBuffer(cmdQueues_[_deviceId], buffer_, data, testTypeSize_, 0, bufSize_,
                                 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueFillBuffer() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();

  char buf[256];

  SNPRINTF(buf, sizeof(buf), "FillBuffer (GB/s) for %6d KB, typeSize:%3d", (int)bufSize_ / 1024,
           (int)testTypeSize_);

  testDescString = buf;
  double sec = timer.GetElapsedTime();
  _perfInfo = static_cast<float>((bufSize_ * iter * (double)(1e-09)) / sec);
}

unsigned int OCLPerfFillBuffer::close(void) {
  if (buffer_) {
    error_ = _wrapper->clReleaseMemObject(buffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(buffer) failed");
  }
  return OCLTestImp::close();
}
