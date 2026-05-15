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

#include "OCLReadWriteImage.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <sstream>
#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#include "CL/cl.h"

const static size_t imageSize = 4;
const static size_t MaxSubTests = 4;

static const char* rgba8888_kernel_read =
    "\n"
    "__kernel void read_rgba8888(read_only image2d_t srcimg, __global uchar4 "
    "*dst, sampler_t sampler)\n"
    "{\n"
    "    int    tid_x = get_global_id(0);\n"
    "    int    tid_y = get_global_id(1);\n"
    "    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
    "    float4 color;\n"
    "\n"
    "    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y)) * 255.0f;\n"
    "    dst[indx] = convert_uchar4_rte(color);\n"
    "\n"
    "}\n";

static const char* rgba8888_kernel_write =
    "\n"
    "__kernel void write_rgba8888(__global unsigned char *src, write_only "
    "image2d_t dstimg)\n"
    "{\n"
    "    int            tid_x = get_global_id(0);\n"
    "    int            tid_y = get_global_id(1);\n"
    "    int            indx = tid_y * get_image_width(dstimg) + tid_x;\n"
    "    float4         color;\n"
    "\n"
    "    indx *= 4;\n"
    "    color = (float4)((float)src[indx+0], (float)src[indx+1], "
    "(float)src[indx+2], (float)src[indx+3]);\n"
    "    color /= (float4)(255.0f, 255.0f, 255.0f, 255.0f);\n"
    "    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
    "\n"
    "}\n";

OCLReadWriteImage::OCLReadWriteImage() {
  _numSubTests = MaxSubTests;
  done_ = false;
  imageWidth = imageSize;
  imageHeight = imageSize;
  imageDepth = imageSize;
}

OCLReadWriteImage::~OCLReadWriteImage() {}

bool OCLReadWriteImage::verifyImageData(unsigned char* inputImageData, unsigned char* output,
                                        size_t width, size_t height) {
  for (unsigned int i = 0; i < 4 * width * height; i++) {
    if (output[i] != inputImageData[i]) {
      printf(
          "Verification failed at byte %u in the output image => %x != %x "
          "[reference]\n",
          i, output[i], inputImageData[i]);
      return false;
    }
  }
  return true;
}
void OCLReadWriteImage::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;

  cl_bool imageSupport;
  size_t size;
  for (size_t i = 0; i < deviceCount_; ++i) {
    _wrapper->clGetDeviceInfo(devices_[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport),
                              &imageSupport, &size);
    if (!imageSupport) {
      testDescString = "Image not supported, skipping this test! ";
      done_ = true;
      return;
    }
  }

  if (test == 1) {
    program_ =
        _wrapper->clCreateProgramWithSource(context_, 1, &rgba8888_kernel_read, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

    error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], NULL, NULL, NULL);
    if (error_ != CL_SUCCESS) {
      char programLog[1024];
      _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                      programLog, 0);
      printf("\n%s\n", programLog);
      fflush(stdout);
    }
    CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

    kernel_ = _wrapper->clCreateKernel(program_, "read_rgba8888", &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");
  } else if ((test == 2) || (test == 3)) {
    program_ =
        _wrapper->clCreateProgramWithSource(context_, 1, &rgba8888_kernel_write, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

    error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], NULL, NULL, NULL);
    if (error_ != CL_SUCCESS) {
      char programLog[1024];
      _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                      programLog, 0);
      printf("\n%s\n", programLog);
      fflush(stdout);
    }
    CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

    kernel_ = _wrapper->clCreateKernel(program_, "write_rgba8888", &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");
  }

  cl_mem memory;
  cl_image_format imgageFormat;
  imgageFormat.image_channel_order = CL_RGBA;
  imgageFormat.image_channel_data_type = CL_UNORM_INT8;
  bufferSize = imageWidth * imageHeight * 4 * sizeof(unsigned char);

  memory = _wrapper->clCreateImage2D(context_, CL_MEM_READ_WRITE, &imgageFormat, imageWidth,
                                     imageHeight, 0, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateImage() failed");

  buffers_.push_back(memory);

  if ((test == 1) || (test == 2) || (test == 3)) {
    memory = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, bufferSize, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(memory);
  }
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLReadWriteImage::run(void) {
  if (done_) {
    return;
  }

  const unsigned int inputImageData[imageSize][imageSize] = {
      {0xc0752fac, 0x67c3fb43, 0xf215d309, 0xd8465724},
      {0xc13a8c58, 0xae5727e6, 0x19a55158, 0x9409484d},
      {0xc5f3d073, 0xc0af4ffe, 0xb1d86352, 0x93931df3},
      {0xc120a78e, 0x207fb909, 0x97f4ca1f, 0x72cbfea3}};

  unsigned char* outputPtr = (unsigned char*)malloc(bufferSize);

  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {imageWidth, imageHeight, 1};
  bool validation;
  size_t threads[2];

  switch (testID_) {
    case 0:  // ImageWrite (w/ sDMA) and ImageRead (w/ sDMA)
      error_ = _wrapper->clEnqueueWriteImage(cmdQueues_[_deviceId], buffers_[0], true, origin,
                                             region, 0, 0, inputImageData, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteImage() failed");

      error_ = _wrapper->clEnqueueReadImage(cmdQueues_[_deviceId], buffers_[0], true, origin,
                                            region, 0, 0, outputPtr, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadImage() failed");

      validation =
          verifyImageData((unsigned char*)&inputImageData, outputPtr, imageWidth, imageHeight);
      if (validation) {
        printf("ImageWrite (w/ sDMA)   -> ImageRead (w/ sDMA)   passed!\n");
      } else {
        CHECK_RESULT(true, "ImageWrite (w/ sDMA) -> ImageRead (w/ sDMA) failed!\n");
      }
      break;
    case 1:  // ImageWrite (w/ sDMA) and ImageRead (w/ kernel)
      error_ = _wrapper->clEnqueueWriteImage(cmdQueues_[_deviceId], buffers_[0], true, origin,
                                             region, 0, 0, inputImageData, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteImage() failed");

      cl_sampler sampler;
      sampler = _wrapper->clCreateSampler(context_, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE,
                                          CL_FILTER_NEAREST, &error_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateSampler failed");

      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof buffers_[0], &buffers_[0]);
      error_ |= clSetKernelArg(kernel_, 1, sizeof buffers_[1], &buffers_[1]);
      error_ |= clSetKernelArg(kernel_, 2, sizeof sampler, &sampler);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed\n");

      threads[0] = (unsigned int)imageWidth;
      threads[1] = (unsigned int)imageHeight;

      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, threads,
                                                NULL, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

      error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[1], CL_TRUE, 0,
                                             bufferSize, outputPtr, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");

      validation =
          verifyImageData((unsigned char*)&inputImageData, outputPtr, imageWidth, imageHeight);
      if (validation) {
        printf("ImageWrite (w/ sDMA)   -> ImageRead (w/ kernel) passed!\n");
      } else {
        CHECK_RESULT(true, "ImageWrite (w/ sDMA) -> ImageRead (w/ kernel) failed!\n");
      }

      break;
    case 2:  // ImageWrite (w/ kernel) and ImageRead (w/ sDMA)
      error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers_[1], CL_TRUE, 0,
                                              bufferSize, inputImageData, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof buffers_[1], &buffers_[1]);
      error_ |= clSetKernelArg(kernel_, 1, sizeof buffers_[0], &buffers_[0]);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed\n");

      threads[0] = (unsigned int)imageWidth;
      threads[1] = (unsigned int)imageHeight;

      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, threads,
                                                NULL, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

      error_ = _wrapper->clEnqueueReadImage(cmdQueues_[_deviceId], buffers_[0], true, origin,
                                            region, 0, 0, outputPtr, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadImage() failed");

      validation =
          verifyImageData((unsigned char*)&inputImageData, outputPtr, imageWidth, imageHeight);
      if (validation) {
        printf("ImageWrite (w/ kernel) -> ImageRead (w/ sDMA)   passed!\n");
      } else {
        CHECK_RESULT(true, "ImageWrite (w/ kernel) -> ImageRead (w/ sDMA) failed!\n");
      }
      break;
    case 3:  // ImageWrite (w/ kernel) and ImageRead (w/ kernel)
      error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers_[1], CL_TRUE, 0,
                                              bufferSize, inputImageData, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof buffers_[1], &buffers_[1]);
      error_ |= clSetKernelArg(kernel_, 1, sizeof buffers_[0], &buffers_[0]);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed\n");

      threads[0] = (unsigned int)imageWidth;
      threads[1] = (unsigned int)imageHeight;

      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, threads,
                                                NULL, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

      // recreate the program_ to use the read kernel
      program_ =
          _wrapper->clCreateProgramWithSource(context_, 1, &rgba8888_kernel_read, NULL, &error_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

      error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], NULL, NULL, NULL);
      if (error_ != CL_SUCCESS) {
        char programLog[1024];
        _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                        programLog, 0);
        printf("\n%s\n", programLog);
        fflush(stdout);
      }
      CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

      kernel_ = _wrapper->clCreateKernel(program_, "read_rgba8888", &error_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

      sampler = _wrapper->clCreateSampler(context_, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE,
                                          CL_FILTER_NEAREST, &error_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateSampler failed");

      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof buffers_[0], &buffers_[0]);
      error_ |= clSetKernelArg(kernel_, 1, sizeof buffers_[1], &buffers_[1]);
      error_ |= clSetKernelArg(kernel_, 2, sizeof sampler, &sampler);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed\n");

      threads[0] = (unsigned int)imageWidth;
      threads[1] = (unsigned int)imageHeight;

      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, threads,
                                                NULL, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

      error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[1], CL_TRUE, 0,
                                             bufferSize, outputPtr, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");

      validation =
          verifyImageData((unsigned char*)&inputImageData, outputPtr, imageWidth, imageHeight);
      if (validation) {
        printf("ImageWrite (w/ kernel) -> ImageRead (w/ kernel) passed!\n");
      } else {
        CHECK_RESULT(true, "ImageWrite (w/ kernel) -> ImageRead (w/ kernel) failed!\n");
      }

      break;
  }

  free(outputPtr);
}

unsigned int OCLReadWriteImage::close(void) { return OCLTestImp::close(); }
