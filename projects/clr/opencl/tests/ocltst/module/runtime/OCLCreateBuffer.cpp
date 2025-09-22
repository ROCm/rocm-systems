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

#include "OCLCreateBuffer.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <sstream>
#ifdef __linux__
#include <unistd.h>
#endif

#include "CL/cl.h"

const static size_t MaxSubTests = 1;

OCLCreateBuffer::OCLCreateBuffer() {
  _numSubTests = MaxSubTests;
  failed_ = false;
  maxSize_ = 0;
}

OCLCreateBuffer::~OCLCreateBuffer() {}

void OCLCreateBuffer::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;

  size_t size;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                            &maxSize_, &size);
//! Workaround out of range issue in Windows 32bit apps
#if defined(_WIN32) && !defined(_WIN64)
  static const size_t MaxSizeLimit = 512 * 1024 * 1024;
  if (maxSize_ > MaxSizeLimit) {
    maxSize_ = MaxSizeLimit;
  }
#endif
#if EMU_ENV
  maxSize_ = 1000;
#endif  // EMU_ENV
  cl_mem buf = NULL;

  // Make sure to use a size that's multiple of 8 (64bit).
  maxSize_ &= 0xFFFFFFFFFFFFFFF8;

  buf = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, maxSize_, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");

  buffers_.push_back(buf);
}

void OCLCreateBuffer::run(void) {
  CPerfCounter timer;

  cl_ulong pattern = PATTERN_20_64BIT;
  timer.Reset();
  timer.Start();
  error_ = /*_wrapper->*/ clEnqueueFillBuffer(cmdQueues_[_deviceId], buffers_[0], &pattern,
                                              sizeof(pattern), 0, maxSize_, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueFillBuffer() failed");
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  size_t maxSteps = maxSize_;
#ifdef __linux__
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (maxSteps > (size_t)(pages * page_size / 2)) {
    maxSteps = (size_t)pages * page_size / 2;
  }
#endif
  void* resultBuf = NULL;

  // Reduce the buffer for the step transfers ahead of the allocation,
  // since huge buffers may cause paging and very low performance
  maxSteps /= 16;
  while ((resultBuf = malloc(maxSteps)) == NULL) {
    maxSteps /= 2;
    continue;
  }

  checkResult(maxSteps, resultBuf, PATTERN_20_64BIT);

  memset(resultBuf, PATTERN_2A_08BIT, maxSteps);

  writeBuffer(maxSteps, resultBuf);

  memset(resultBuf, 0x00, maxSteps);
  checkResult(maxSteps, resultBuf, PATTERN_2A_64BIT);

  free(resultBuf);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  _perfInfo = (float)sec * 1000.f;
  std::stringstream str;
  str << "Max single alloc (size of ";
  str << maxSize_;
  str << " bytes) ";

  testDescString = str.str();
  str << "Max single read/write (size of ";
  str << maxSize_;
  str << " bytes) create time (ms):";

  testDescString = str.str();
}

void OCLCreateBuffer::checkResult(size_t maxSteps, void* resultBuf, cl_ulong pattern) {
  size_t startPoint = 0;
  while ((startPoint) < maxSize_) {
    cl_event ee;
    size_t readSize = maxSteps;
    if ((startPoint + maxSteps) > maxSize_) {
      readSize = maxSize_ - startPoint;
    }
    error_ = /*wrapper->*/ clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_FALSE,
                                               startPoint, readSize, resultBuf, 0, NULL, &ee);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
    _wrapper->clFinish(cmdQueues_[_deviceId]);

    size_t err_cnt = 0;
    size_t chk_cnt = readSize / sizeof(cl_ulong);

    cl_ulong* cc = reinterpret_cast<cl_ulong*>(resultBuf);

    for (size_t i = 0; i < chk_cnt; i++) {
      if (cc[i] != pattern) {
        err_cnt++;
      }
    }
    if (err_cnt != 0) {
      error_ = -1;
      CHECK_RESULT((error_ != CL_SUCCESS), "checkResult() failed");
      break;
    }
    startPoint += maxSteps;
  }
}

void OCLCreateBuffer::writeBuffer(size_t maxSteps, void* dataBuf) {
  size_t startPoint = 0;
  while ((startPoint) < maxSize_) {
    cl_event ee;
    size_t writeSize = maxSteps;
    if ((startPoint + maxSteps) > maxSize_) {
      writeSize = maxSize_ - startPoint;
    }
    error_ = /*wrapper->*/ clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers_[0], CL_FALSE,
                                                startPoint, writeSize, dataBuf, 0, NULL, &ee);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
    _wrapper->clFinish(cmdQueues_[_deviceId]);
    startPoint += maxSteps;
  }
}

unsigned int OCLCreateBuffer::close(void) { return OCLTestImp::close(); }
