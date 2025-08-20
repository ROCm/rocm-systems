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

#include "OCLPerfImageReadWrite.h"

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
static const unsigned int Sizes[NUM_SIZES] = {256, 512, 1024, 2048};

#if defined(CL_VERSION_2_0)
#define NUM_FORMATS 2
static const cl_image_format formats[NUM_FORMATS] = {{CL_RGBA, CL_UNSIGNED_INT8},
                                                     {CL_sRGBA, CL_UNORM_INT8}};
static const char* textFormats[NUM_FORMATS] = {"CL_RGBA , CL_UNSIGNED_INT8",
                                               "CL_sRGBA, CL_UNORM_INT8   "};
static const unsigned int formatSize[NUM_FORMATS] = {sizeof(CL_UNSIGNED_INT8),
                                                     sizeof(CL_UNORM_INT8)};
#else
#define NUM_FORMATS 1
static const cl_image_format formats[NUM_FORMATS] = {{CL_RGBA, CL_UNSIGNED_INT8}};
static const char* textFormats[NUM_FORMATS] = {"CL_RGBA , CL_UNSIGNED_INT8"};
static const unsigned int formatSize[NUM_FORMATS] = {sizeof(CL_UNSIGNED_INT8)};
#endif

const static char* strKernel = {KERNEL_CODE(
  \n __constant sampler_t s_nearest = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;
  \n __kernel void image_kernel(read_write image2d_t image, uint zero) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int offset = y * get_image_width(image) + x;

  int2 coords = (int2)(x, y);
  uint4 tmp = read_imageui(image, s_nearest, coords);

  write_imageui(image, coords, 1 + tmp * zero);
}
  \n)};

OCLPerfImageReadWrite::OCLPerfImageReadWrite() { _numSubTests = NUM_SIZES * NUM_FORMATS; }

OCLPerfImageReadWrite::~OCLPerfImageReadWrite() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfImageReadWrite::setData(void* ptr, unsigned int size, unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  for (unsigned int i = 0; i < size >> 2; i++) {
    ptr2[i] = value;
    value++;
  }
}

void OCLPerfImageReadWrite::open(unsigned int test, char* units, double& conversion,
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

  // check device version
  size_t param_size = 0;
  char* strVersion = 0;
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_OPENCL_C_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strVersion = new char[param_size];
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_OPENCL_C_VERSION, param_size,
                                     strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strVersion[9] < '2') {
    skip_ = true;
    testDescString =
        "Image read_write qualifier not supported in OpenCL C < 2.0. Test "
        "Skipped.";
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

  memptr = new char[memSize];

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

  // create image
  setData(memptr, memSize, 0x0);
  imageBuffer_ =
      _wrapper->clCreateImage2D(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                &formats[bufnum_], bufSize_, bufSize_, 0, memptr, &error_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clCreateImage2D() failed");

  const unsigned int zero = 0;

  // set kernel arguments
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &imageBuffer_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(unsigned int), &zero);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
}

void OCLPerfImageReadWrite::run(void) {
  if (skip_) {
    return;
  }

  CPerfCounter timer;

  size_t gws[2] = {bufSize_, bufSize_};
  size_t lws[2] = {8, 8};

  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 2, NULL, gws, lws, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  _wrapper->clFinish(cmd_queue_);

  timer.Reset();
  timer.Start();

  for (unsigned int i = 0; i < numIter; ++i) {
    error_ =
        _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 2, NULL, gws, lws, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    _wrapper->clFinish(cmd_queue_);
  }

  timer.Stop();

  double sec = timer.GetElapsedTime();

  // speed in GB/s
  double perf = ((double)memSize * numIter * (double)(1e-09)) * 2 / sec;

  _perfInfo = (float)perf;
  char buf[256];
  unsigned int fmt_num = (testId_ / NUM_SIZES) % NUM_FORMATS;
  SNPRINTF(buf, sizeof(buf), " (%4dx%4d) fmt:%s(%1d) i: %4d (GB/s) ", bufSize_, bufSize_,
           textFormats[fmt_num], formatSize[bufnum_], numIter);
  testDescString = buf;
}

unsigned int OCLPerfImageReadWrite::close(void) {
  if (!skip_) {
    if (memptr) {
      delete memptr;
    }
    if (imageBuffer_) {
      error_ = _wrapper->clReleaseMemObject(imageBuffer_);
      CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(imageBuffer_) failed");
    }
  }
  return OCLTestImp::close();
}
