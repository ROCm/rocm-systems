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

#include "OCLPerf3DImageWriteSpeed.h"

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

#define KERNEL_CODE(...) #__VA_ARGS__

#define NUM_SIZES 4
static const unsigned int Sizes[NUM_SIZES] = {64, 128, 256, 512};

#define NUM_FORMATS 1
static const cl_image_format formats[NUM_FORMATS] = {{CL_RGBA, CL_UNSIGNED_INT8}};
static const char* textFormats[NUM_FORMATS] = {"CL_RGBA , CL_UNSIGNED_INT8"};
static const unsigned int formatSize[NUM_FORMATS] = {sizeof(CL_UNSIGNED_INT8)};

const static char* strKernel = {KERNEL_CODE(
  \n __kernel void image_kernel(write_only image3d_t input) {
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);

  int4 coords = (int4)(x, y, z, 0);
  write_imageui(input, coords, (1, 1, 1, 1));
}
  \n)};

OCLPerf3DImageWriteSpeed::OCLPerf3DImageWriteSpeed() { _numSubTests = NUM_SIZES * NUM_FORMATS; }

OCLPerf3DImageWriteSpeed::~OCLPerf3DImageWriteSpeed() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerf3DImageWriteSpeed::open(unsigned int test, char* units, double& conversion,
                                    unsigned int deviceId) {
  error_ = CL_SUCCESS;
  testId_ = test;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = 0;
  kernel_ = 0;
  cmd_queue_ = 0;
  imageBuffer_ = 0;
  skip_ = false;

  char charbuf[1024];
  size_t retsize;
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  if (!strstr(charbuf, "cl_khr_3d_image_writes")) {
    skip_ = true;
    testDescString = "3D Write not supported. Test Skipped.";
    return;
  }

  bufSize_ = Sizes[test % NUM_SIZES];
  bufnum_ = (test / NUM_SIZES) % NUM_FORMATS;
  memSize_ = bufSize_ * bufSize_ * bufSize_ * formatSize[bufnum_];

  cmd_queue_ = cmdQueues_[_deviceId];

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
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

  kernel_ = _wrapper->clCreateKernel(program_, "image_kernel", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  imageBuffer_ = _wrapper->clCreateImage3D(context_, CL_MEM_WRITE_ONLY, &formats[bufnum_], bufSize_,
                                           bufSize_, bufSize_, 0, 0, NULL, &error_);
  CHECK_RESULT(imageBuffer_ == 0, "clCreateImage(imageBuffer_) failed");

  // set kernel arguments
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &imageBuffer_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
}

void OCLPerf3DImageWriteSpeed::run(void) {
  if (skip_) {
    return;
  }

  CPerfCounter timer;
  unsigned int fmt_num = (testId_ / NUM_SIZES) % NUM_FORMATS;

  size_t gws[3] = {bufSize_, bufSize_, bufSize_};
  size_t lws[3] = {8, 8, 4};

  // warm up
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 3, NULL, gws, lws, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  _wrapper->clFinish(cmd_queue_);

  // checkData
  char* bufptr = (char*)malloc(memSize_);

  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {bufSize_, bufSize_, bufSize_};
  size_t image_row_pitch = bufSize_ * formatSize[bufnum_];
  size_t image_slice_pitch = image_row_pitch * bufSize_;
  error_ = clEnqueueReadImage(cmd_queue_, imageBuffer_, true, origin, region, image_row_pitch,
                              image_slice_pitch, bufptr, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadImage() failed");

  for (size_t i = 0; i < bufSize_ * bufSize_ * bufSize_ * 4; ++i) {
    if (bufptr[i] != 1) {
      printf("(%4dx%4dx%4d) fmt:%s(%1u) checkData() fail, image_ptr[%u] = %d\n", bufSize_, bufSize_,
             bufSize_, textFormats[fmt_num], formatSize[bufnum_], (unsigned int)i, (int)bufptr[i]);
      CHECK_RESULT_NO_RETURN(0, "Data validation failed!\n");
      char buf[256];
      SNPRINTF(buf, sizeof(buf), " (%4dx%4dx%4d) fmt:%s(%1d) checkData() FAILED! ", bufSize_,
               bufSize_, bufSize_, textFormats[fmt_num], formatSize[bufnum_]);
      testDescString = buf;
      return;
    }
  }
  delete bufptr;

  // test begins
  unsigned int numIter = 5;

  timer.Reset();
  timer.Start();

  for (unsigned int i = 0; i < numIter; ++i) {
    error_ =
        _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 3, NULL, gws, lws, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    _wrapper->clFinish(cmd_queue_);
  }

  timer.Stop();

  double sec = timer.GetElapsedTime();

  // write_image speed in GB/s
  double perf = ((double)memSize_ * numIter * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%3dx%3dx%3d) fmt:%s(%1u) i: %2d (GB/s) ", bufSize_, bufSize_,
           bufSize_, textFormats[fmt_num], formatSize[bufnum_], numIter);
  testDescString = buf;
}

unsigned int OCLPerf3DImageWriteSpeed::close(void) {
  if (!skip_) {
    if (imageBuffer_) {
      error_ = _wrapper->clReleaseMemObject(imageBuffer_);
      CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(imageBuffer_) failed");
    }
  }
  return OCLTestImp::close();
}
