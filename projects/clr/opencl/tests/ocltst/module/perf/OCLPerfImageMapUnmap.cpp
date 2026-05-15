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

#include "OCLPerfImageMapUnmap.h"

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

#define NUM_SIZES 1
static const unsigned int Sizes0[2] = {0xc0, 0x18a};

#define NUM_FORMATS 1
static const cl_image_format formats[NUM_FORMATS] = {{CL_R, CL_SNORM_INT16}};
static const char* textFormats[NUM_FORMATS] = {"R16"};
static const unsigned int formatSize[NUM_FORMATS] = {2 * sizeof(cl_uchar)};

static const unsigned int Iterations[2] = {1, OCLPerfImageMapUnmap::NUM_ITER};

#define NUM_SUBTESTS 1
OCLPerfImageMapUnmap::OCLPerfImageMapUnmap() {
  _numSubTests = NUM_SIZES * NUM_SUBTESTS * NUM_FORMATS * 1;
}

OCLPerfImageMapUnmap::~OCLPerfImageMapUnmap() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfImageMapUnmap::setData(void* ptr, unsigned int pitch, unsigned int size,
                                   unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  value = 0;
  for (unsigned int i = 0; i < size >> 2; i++) {
    ptr2[i] = value;
    value++;
  }
}

void OCLPerfImageMapUnmap::checkData(void* ptr, unsigned int pitch, unsigned int size,
                                     unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  value = 0;
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

void OCLPerfImageMapUnmap::open(unsigned int test, char* units, double& conversion,
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
  skip_ = false;
  context_ = 0;
  cmd_queue_ = 0;
  srcBuffer_ = 0;
  dstBuffer_ = 0;
  srcImage_ = false;
  dstImage_ = false;

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

  srcImage_ = true;

  dstImage_ = false;

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

  bufSizeW_ = Sizes0[0];
  bufSizeH_ = Sizes0[1];

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  cl_mem_flags flags = CL_MEM_WRITE_ONLY;
  cl_mem_flags flags2 = CL_MEM_WRITE_ONLY;
  void* mem;
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {bufSizeW_, bufSizeH_, 1};
  size_t image_row_pitch;
  size_t image_slice_pitch;
  cl_image_desc imageInfo;

  memset(&imageInfo, 0x0, sizeof(cl_image_desc));

  imageInfo.image_type = CL_MEM_OBJECT_IMAGE2D;
  imageInfo.image_width = bufSizeW_;
  imageInfo.image_height = bufSizeH_;
  imageInfo.image_depth = 1;
  imageInfo.image_array_size = 1;
  imageInfo.image_row_pitch = bufSizeW_ * formatSize[bufnum_];
  imageInfo.image_slice_pitch = imageInfo.image_row_pitch * (bufSizeH_);

  void* host_ptr = malloc(imageInfo.image_row_pitch * imageInfo.image_height);

  unsigned int memSize;
  if (dstImage_) {
    dstBuffer_ = _wrapper->clCreateImage2D(context_, flags, &formats[bufnum_], bufSizeW_, bufSizeH_,
                                           0, host_ptr, &error_);
    CHECK_RESULT(dstBuffer_ == 0, "clCreateImage(dstBuffer) failed");
    mem = _wrapper->clEnqueueMapImage(cmd_queue_, dstBuffer_, CL_TRUE, CL_MAP_WRITE, origin, region,
                                      &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &error_);
    CHECK_RESULT(error_, "clEnqueueMapImage failed");
    memSize = (unsigned int)image_row_pitch * bufSizeH_;
  } else {
    dstBuffer_ = _wrapper->clCreateBuffer(
        context_, flags2, bufSizeW_ * bufSizeH_ * formatSize[bufnum_], NULL, &error_);
    CHECK_RESULT(dstBuffer_ == 0, "clCreateBuffer(dstBuffer) failed");
    mem = _wrapper->clEnqueueMapBuffer(cmd_queue_, dstBuffer_, CL_TRUE, CL_MAP_WRITE, 0,
                                       bufSizeW_ * bufSizeH_ * formatSize[bufnum_], 0, NULL, NULL,
                                       &error_);
    CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
    memSize = (unsigned int)bufSizeW_ * bufSizeH_ * formatSize[bufnum_];
    image_row_pitch = 0;
  }
  setData(mem, (unsigned int)image_row_pitch, memSize, 0xdeadbeef);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, dstBuffer_, mem, 0, NULL, NULL);

  flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
  if (srcImage_) {
    srcBuffer_ =
        _wrapper->clCreateImage(context_, flags, &formats[bufnum_], &imageInfo, host_ptr, &error_);
    CHECK_RESULT(srcBuffer_ == 0, "clCreateImage(srcBuffer) failed");
    mem = _wrapper->clEnqueueMapImage(cmd_queue_, srcBuffer_, CL_TRUE, CL_MAP_WRITE, origin, region,
                                      &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &error_);
    CHECK_RESULT(error_, "clEnqueueMapImage failed");
    memSize = (unsigned int)image_row_pitch * bufSizeH_;
    error_ = _wrapper->clFinish(cmd_queue_);
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
  setData(mem, (unsigned int)image_row_pitch, memSize, 0x600df00d);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, srcBuffer_, mem, 0, NULL, NULL);
  error_ = _wrapper->clFinish(cmd_queue_);
}

void OCLPerfImageMapUnmap::run(void) {
  if (skip_) {
    return;
  }
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {bufSizeW_, bufSizeH_, 1};

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
  checkData(mem, (unsigned int)image_row_pitch, memSize, 0x600df00d);
  _wrapper->clEnqueueUnmapMemObject(cmd_queue_, dstBuffer_, mem, 0, NULL, NULL);
  _perfInfo = 0;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%4dx%4d) fmt:%s src:%s dst:%s i: %4d (GB/s) ", bufSizeW_, bufSizeH_,
           textFormats[bufnum_], strSrc, strDst, numIter);
  testDescString = buf;
}

unsigned int OCLPerfImageMapUnmap::close(void) {
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
