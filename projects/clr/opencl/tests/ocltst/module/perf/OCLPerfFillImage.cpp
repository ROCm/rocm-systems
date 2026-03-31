/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLPerfFillImage.h"

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

static unsigned int sizeList[] = {
    256, 512, 1024, 2048, 4096, 8192,
};

OCLPerfFillImage::OCLPerfFillImage() {
  num_sizes_ = sizeof(sizeList) / sizeof(unsigned int);
  _numSubTests = num_sizes_;
  failed_ = false;
  skip_ = false;
}

OCLPerfFillImage::~OCLPerfFillImage() {}

void OCLPerfFillImage::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  bufSize_ = sizeList[test % num_sizes_];

  cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};
  buffer_ = _wrapper->clCreateImage2D(context_, CL_MEM_WRITE_ONLY, &format, bufSize_, bufSize_, 0,
                                      NULL, &error_);
  CHECK_RESULT(buffer_ == 0, "clCreateImage2D(imageBuffer_) failed");

  return;
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfFillImage::run(void) {
  CPerfCounter timer;
  size_t iter = 100;

  cl_uint4 fillColor = {1, 1, 1, 1};
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {bufSize_, bufSize_, 1};

  timer.Reset();
  timer.Start();
  for (size_t i = 0; i < iter; ++i) {
    error_ = clEnqueueFillImage(cmdQueues_[_deviceId], buffer_, (const void*)&fillColor, origin,
                                region, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueFillImage() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();

  char buf[256];

  SNPRINTF(buf, sizeof(buf), "FillImage (GB/s) for %4dx%4d ", (int)bufSize_, (int)bufSize_);

  testDescString = buf;
  double sec = timer.GetElapsedTime();
  _perfInfo = static_cast<float>((bufSize_ * bufSize_ * 4 * iter * (double)(1e-09)) / sec);
}

unsigned int OCLPerfFillImage::close(void) {
  if (buffer_) {
    error_ = _wrapper->clReleaseMemObject(buffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(buffer) failed");
  }
  return OCLTestImp::close();
}
