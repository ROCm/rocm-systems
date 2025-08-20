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

#include "OCLPerfImageCopyCorners.h"

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

#define NUM_SIZES 2
static const unsigned int Sizes0[NUM_SIZES] = {512, 16384};
static const unsigned int Sizes1[NUM_SIZES] = {16384, 512};

#define NUM_FORMATS 3
static const cl_image_format formats[NUM_FORMATS] = {
    {CL_RGBA, CL_UNSIGNED_INT8}, {CL_R, CL_UNSIGNED_INT32}, {CL_RGBA, CL_UNSIGNED_INT32}};
static const char* textFormats[NUM_FORMATS] = {"R8G8B8A8", "R32", "R32G32B32A32"};
static const unsigned int formatSize[NUM_FORMATS] = {4 * sizeof(cl_uchar), 1 * sizeof(cl_uint),
                                                     4 * sizeof(cl_uint)};

static const unsigned int Iterations[2] = {1, OCLPerfImageCopyCorners::NUM_ITER};

#define NUM_SUBTESTS 3
OCLPerfImageCopyCorners::OCLPerfImageCopyCorners() {
  _numSubTests = NUM_SIZES * NUM_SUBTESTS * NUM_FORMATS * 2;
}

OCLPerfImageCopyCorners::~OCLPerfImageCopyCorners() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfImageCopyCorners::setData(void* ptr, unsigned int pitch, unsigned int size) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  unsigned int value = 0;
  for (unsigned int i = 0; i < size >> 2; i++) {
    ptr2[i] = value;
    value++;
  }
}

void OCLPerfImageCopyCorners::checkData(void* ptr, unsigned int pitch, unsigned int size) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  unsigned int value = 0;
  for (unsigned int i = 0; i < size >> 2; i++) {
    if (ptr2[i] != value) {
      printf("Data validation failed at %d!  Got 0x%08x 0x%08x 0x%08x 0x%08x\n", i, ptr2[i],
             ptr2[i + 1], ptr2[i + 2], ptr2[i + 3]);
      printf("Expected 0x%08x 0x%08x 0x%08x 0x%08x\n", value, value, value, value);
      CHECK_RESULT(true, "Data validation failed!");
      break;
    }
    value++;
  }
}

void OCLPerfImageCopyCorners::open(unsigned int test, char* units, double& conversion,
                                   unsigned int deviceId) {
  cl_uint typeOfDevice = type_;
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  size_t queryOut = 0;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;

  context_ = 0;
  cmd_queue_ = 0;
  srcBuffer_ = 0;
  dstBuffer_ = 0;
  srcImage_ = false;
  dstImage_ = false;
  skip_ = false;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
#if 0
        // Get last for default
        platform = platforms[numPlatforms-1];
        for (unsigned i = 0; i < numPlatforms; ++i) {
#endif
    platform = platforms[_platformIndex];
    char pbuf[100];
    error_ = _wrapper->clGetPlatformInfo(platforms[_platformIndex], CL_PLATFORM_VENDOR,
                                         sizeof(pbuf), pbuf, NULL);
    num_devices = 0;
    /* Get the number of requested devices */
    error_ =
        _wrapper->clGetDeviceIDs(platforms[_platformIndex], typeOfDevice, 0, NULL, &num_devices);
    // Runtime returns an error when no GPU devices are present instead of just
    // returning 0 devices
    // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
    // Choose platform with GPU devices
    // if (num_devices > 0)
    //{
    //    platform = platforms[_platformIndex];
    //    break;
    //}
#if 0
        }
#endif
    delete platforms;
  }

  bufnum_ = (_openTest / (NUM_SIZES * NUM_SUBTESTS)) % NUM_FORMATS;

  if ((((_openTest / NUM_SIZES) % NUM_SUBTESTS) + 1) & 1) {
    srcImage_ = true;
  }
  if ((((_openTest / NUM_SIZES) % NUM_SUBTESTS) + 1) & 2) {
    dstImage_ = true;
  }

  numIter = Iterations[_openTest / (NUM_SIZES * NUM_SUBTESTS * NUM_FORMATS)];

  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, typeOfDevice, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];
  size_t size;
  cl_bool imageSupport_ = false;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport_),
                                     &imageSupport_, &size);
  if (!imageSupport_) {
    printf("\n%s\n", "Image not supported, skipping this test!");
    skip_ = true;
    return;
  }

  if (_openTest % NUM_SIZES) {
    error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t),
                                       &queryOut, NULL);
    bufSizeW_ = (cl_uint)queryOut;
    bufSizeH_ = Sizes1[_openTest % NUM_SIZES];
  } else {
    error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t),
                                       &queryOut, NULL);
    bufSizeW_ = Sizes0[_openTest % NUM_SIZES];
    bufSizeH_ = (cl_uint)queryOut;
  }

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  cl_mem_flags flags = CL_MEM_WRITE_ONLY;
  void* mem;
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {bufSizeW_, bufSizeH_, 1};
  size_t image_row_pitch;
  size_t image_slice_pitch;
  unsigned int memSize;
  if (dstImage_) {
    dstBuffer_ = _wrapper->clCreateImage2D(context_, flags, &formats[bufnum_], bufSizeW_, bufSizeH_,
                                           0, NULL, &error_);
    CHECK_RESULT(dstBuffer_ == 0, "clCreateImage(dstBuffer) failed");
    mem = _wrapper->clEnqueueMapImage(cmd_queue_, dstBuffer_, CL_TRUE, CL_MAP_WRITE, origin, region,
                                      &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &error_);
    CHECK_RESULT(error_, "clEnqueueMapImage failed");
    memSize = (unsigned int)image_row_pitch * bufSizeH_;
  } else {
    dstBuffer_ = _wrapper->clCreateBuffer(
        context_, flags, bufSizeW_ * bufSizeH_ * formatSize[bufnum_], NULL, &error_);
    CHECK_RESULT(dstBuffer_ == 0, "clCreateBuffer(dstBuffer) failed");
    mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, dstBuffer_, CL_TRUE, CL_MAP_WRITE, 0,
                                       bufSizeW_ * bufSizeH_ * formatSize[bufnum_], 0, NULL, NULL,
                                       &error_);
    CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
    memSize = (unsigned int)bufSizeW_ * bufSizeH_ * formatSize[bufnum_];
    image_row_pitch = 0;
  }
  setData(mem, (unsigned int)image_row_pitch, memSize);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, dstBuffer_, mem, 0, NULL, NULL);

  flags = CL_MEM_READ_ONLY;
  if (srcImage_) {
    srcBuffer_ = _wrapper->clCreateImage2D(context_, flags, &formats[bufnum_], bufSizeW_, bufSizeH_,
                                           0, NULL, &error_);
    CHECK_RESULT(srcBuffer_ == 0, "clCreateImage(srcBuffer) failed");
    mem = _wrapper->clEnqueueMapImage(cmd_queue_, srcBuffer_, CL_TRUE, CL_MAP_WRITE, origin, region,
                                      &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &error_);
    CHECK_RESULT(error_, "clEnqueueMapImage failed");
    memSize = (unsigned int)image_row_pitch * bufSizeH_;
  } else {
    srcBuffer_ = _wrapper->clCreateBuffer(
        context_, flags, bufSizeW_ * bufSizeH_ * formatSize[bufnum_], NULL, &error_);
    CHECK_RESULT(srcBuffer_ == 0, "clCreateBuffer(srcBuffer) failed");
    mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, srcBuffer_, CL_TRUE, CL_MAP_WRITE, 0,
                                       bufSizeW_ * bufSizeH_ * formatSize[bufnum_], 0, NULL, NULL,
                                       &error_);
    CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
    memSize = (unsigned int)bufSizeW_ * bufSizeH_ * formatSize[bufnum_];
    image_row_pitch = 0;
  }
  setData(mem, (unsigned int)image_row_pitch, memSize);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, srcBuffer_, mem, 0, NULL, NULL);
}

void OCLPerfImageCopyCorners::run(void) {
  if (skip_) {
    return;
  }
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {bufSizeW_, bufSizeH_, 1};

  // Warm up
  if (srcImage_ == false) {
    error_ = _wrapper->clEnqueueCopyBufferToImage(cmd_queue_, srcBuffer_, dstBuffer_, 0, origin,
                                                  region, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueCopyBufferToImage failed");
  } else if (dstImage_ == false) {
    error_ = _wrapper->clEnqueueCopyImageToBuffer(cmd_queue_, srcBuffer_, dstBuffer_, origin,
                                                  region, 0, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueCopyImageToBuffer failed");
  } else {
    error_ = _wrapper->clEnqueueCopyImage(cmd_queue_, srcBuffer_, dstBuffer_, origin, origin,
                                          region, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueCopyImage failed");
  }
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  CPerfCounter timer;
  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < numIter; i++) {
    if (srcImage_ == false) {
      error_ = _wrapper->clEnqueueCopyBufferToImage(cmd_queue_, srcBuffer_, dstBuffer_, 0, origin,
                                                    region, 0, NULL, NULL);
      CHECK_RESULT(error_, "clEnqueueCopyBufferToImage failed");
    } else if (dstImage_ == false) {
      error_ = _wrapper->clEnqueueCopyImageToBuffer(cmd_queue_, srcBuffer_, dstBuffer_, origin,
                                                    region, 0, 0, NULL, NULL);
      CHECK_RESULT(error_, "clEnqueueCopyImageToBuffer failed");
    } else {
      error_ = _wrapper->clEnqueueCopyImage(cmd_queue_, srcBuffer_, dstBuffer_, origin, origin,
                                            region, 0, NULL, NULL);
      CHECK_RESULT(error_, "clEnqueueCopyImage failed");
    }
  }
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Image copy bandwidth in GB/s
  double perf =
      ((double)bufSizeW_ * bufSizeH_ * formatSize[bufnum_] * 2 * numIter * (double)(1e-09)) / sec;

  const char* strSrc = NULL;
  const char* strDst = NULL;
  if (srcImage_)
    strSrc = "img";
  else
    strSrc = "buf";
  if (dstImage_)
    strDst = "img";
  else
    strDst = "buf";
  void* mem;
  size_t image_row_pitch;
  size_t image_slice_pitch;
  unsigned int memSize;
  if (dstImage_) {
    mem = _wrapper->clEnqueueMapImage(cmd_queue_, dstBuffer_, CL_TRUE, CL_MAP_READ, origin, region,
                                      &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &error_);
    CHECK_RESULT(error_, "clEnqueueMapImage failed");
    memSize = (unsigned int)image_row_pitch * bufSizeH_;
  } else {
    mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, dstBuffer_, CL_TRUE, CL_MAP_READ, 0,
                                       bufSizeW_ * bufSizeH_ * formatSize[bufnum_], 0, NULL, NULL,
                                       &error_);
    CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
    memSize = (unsigned int)bufSizeW_ * bufSizeH_ * formatSize[bufnum_];
    image_row_pitch = 0;
  }
  checkData(mem, (unsigned int)image_row_pitch, memSize);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, dstBuffer_, mem, 0, NULL, NULL);
  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%4dx%4d) fmt:%s src:%s dst:%s i: %4d (GB/s) ", bufSizeW_, bufSizeH_,
           textFormats[bufnum_], strSrc, strDst, numIter);
  testDescString = buf;
}

unsigned int OCLPerfImageCopyCorners::close(void) {
  if (skip_) {
    return CL_SUCCESS;
  }
  _wrapper->clFinish(cmd_queue_);

  if (srcBuffer_) {
    error_ = _wrapper->clReleaseMemObject(srcBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(srcBuffer_) failed");
  }
  if (dstBuffer_) {
    error_ = _wrapper->clReleaseMemObject(dstBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(dstBuffer_) failed");
  }
  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  return _crcword;
}
