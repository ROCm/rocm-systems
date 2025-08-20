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

#include "OCLPerfImageCreate.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/opencl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_SIZES 4
static const unsigned int Sizes[NUM_SIZES] = {256, 512, 1024, 2048};

#if defined(CL_VERSION_2_0)
#define NUM_FORMATS 3
static const cl_image_format formats[NUM_FORMATS] = {
    {CL_RGBA, CL_UNSIGNED_INT8}, {CL_sRGBA, CL_UNORM_INT8}, {CL_DEPTH, CL_UNORM_INT16}};
static const char* textFormats[NUM_FORMATS] = {
    "CL_RGBA , CL_UNSIGNED_INT8", "CL_sRGBA, CL_UNORM_INT8   ", "CL_DEPTH, CL_UNORM_INT16  "};
static const unsigned int formatSize[NUM_FORMATS] = {sizeof(CL_UNSIGNED_INT8),
                                                     sizeof(CL_UNORM_INT8), sizeof(CL_UNORM_INT16)};
#else
#define NUM_FORMATS 1
static const cl_image_format formats[NUM_FORMATS] = {{CL_RGBA, CL_UNSIGNED_INT8}};
static const char* textFormats[NUM_FORMATS] = {"CL_RGBA, CL_UNSIGNED_INT8"};
static const unsigned int formatSize[NUM_FORMATS] = {sizeof(CL_UNSIGNED_INT8)};
#endif

OCLPerfImageCreate::OCLPerfImageCreate() { _numSubTests = NUM_SIZES * NUM_FORMATS; }

OCLPerfImageCreate::~OCLPerfImageCreate() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfImageCreate::setData(void* ptr, unsigned int size, unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  for (unsigned int i = 0; i < size >> 2; i++) {
    ptr2[i] = value;
    value++;
  }
}

void OCLPerfImageCreate::open(unsigned int test, char* units, double& conversion,
                              unsigned int deviceId) {
  error_ = CL_SUCCESS;
  testId_ = test;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = 0;
  kernel_ = 0;
  cmd_queue_ = 0;
  outBuffer_ = 0;
  skip_ = false;

  // check device version
  size_t param_size = 0;
  char* strVersion = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strVersion = new char[param_size];
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, param_size, strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strVersion[7] < '2') {
    skip_ = true;
    testDescString = "sRGBA Image not supported for < 2.0 devices. Test Skipped.";
    delete strVersion;
    return;
  }
  delete strVersion;
  size_t size;
  cl_bool imageSupport_ = false;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_IMAGE_SUPPORT,
                                     sizeof(imageSupport_), &imageSupport_, &size);
  if (!imageSupport_) {
    printf("\n%s\n", "Image not supported, skipping this test!");
    skip_ = true;
    return;
  }
  bufSize_ = Sizes[test % NUM_SIZES];
  bufnum_ = (test / NUM_SIZES) % NUM_FORMATS;
  memSize = bufSize_ * bufSize_ * formatSize[bufnum_];
  numIter = 100;

  outBuffer_ = (cl_mem*)malloc(numIter * sizeof(cl_mem));
  memptr = new char[memSize];

  cmd_queue_ = cmdQueues_[_deviceId];
}

void OCLPerfImageCreate::run(void) {
  if (skip_) {
    return;
  }

  CPerfCounter timer;

  cl_image_desc imageInfo;

  memset(&imageInfo, 0x0, sizeof(cl_image_desc));

  imageInfo.image_type = CL_MEM_OBJECT_IMAGE2D;
  imageInfo.image_width = bufSize_;
  imageInfo.image_height = bufSize_;
  imageInfo.image_depth = 1;
  imageInfo.image_array_size = 1;
  imageInfo.image_row_pitch = bufSize_ * formatSize[bufnum_];
  imageInfo.image_slice_pitch = imageInfo.image_row_pitch * (bufSize_);

  setData(memptr, memSize, 0xdeadbeef);

  char* dstmem = new char[memSize];
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {1, 1, 1};

  timer.Reset();
  timer.Start();

  for (unsigned int i = 0; i < numIter; ++i) {
    outBuffer_[i] = clCreateImage(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                  &formats[bufnum_], &imageInfo, memptr, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "Error clCreateImage()");

    error_ = _wrapper->clEnqueueReadImage(cmd_queue_, outBuffer_[i], CL_TRUE, origin, region, 0, 0,
                                          dstmem, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueReadImage failed");
    _wrapper->clFinish(cmd_queue_);
  }

  timer.Stop();

  delete dstmem;

  double sec = timer.GetElapsedTime();

  // Image create in GB/s
  double perf = ((double)memSize * numIter * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  char buf[256];
  unsigned int fmt_num = (testId_ / NUM_SIZES) % NUM_FORMATS;
  SNPRINTF(buf, sizeof(buf), " (%4dx%4d) fmt:%s(%1d) i: %4d (GB/s) ", bufSize_, bufSize_,
           textFormats[fmt_num], formatSize[bufnum_], numIter);
  testDescString = buf;
}

unsigned int OCLPerfImageCreate::close(void) {
  if (skip_) {
    return CL_SUCCESS;
  }
  if (memptr) {
    delete memptr;
  }
  if (outBuffer_) {
    for (unsigned int i = 0; i < numIter; ++i) {
      if (outBuffer_[i]) {
        error_ = _wrapper->clReleaseMemObject(outBuffer_[i]);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_[i]) failed");
      }
    }
  }
  return OCLTestImp::close();
}
