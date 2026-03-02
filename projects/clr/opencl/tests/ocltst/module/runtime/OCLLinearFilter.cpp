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

#include "OCLLinearFilter.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static size_t ImageSize = 4;

const static char* strKernel =
    "const sampler_t g_Sampler =    CLK_FILTER_LINEAR |                 \n"
    "                               CLK_ADDRESS_CLAMP_TO_EDGE |         \n"
    "                               CLK_NORMALIZED_COORDS_FALSE;        \n"
    "                                                                   \n"
    "__kernel void linear3D(__read_only image3d_t img3D, __global float4* "
    "f4Tata) \n"
    "{                                                                  \n"
    "   float4 f4Index = { 2.25f, 1.75f, 0.5f, 0.0f };                  \n"
    "	// copy interpolated data in result buffer                      \n"
    "	f4Tata[0] = read_imagef(img3D, g_Sampler, f4Index);             \n"
    "}                                                                  \n"
    "                                                                   \n"
    "__kernel void linear2D(__read_only image2d_t img2D, __global float4* "
    "f4Tata) \n"
    "{                                                                  \n"
    "   float2 f2Index = { 2.25f, 1.75f };                              \n"
    "	// copy interpolated data in result buffer                      \n"
    "	f4Tata[0] = read_imagef(img2D, g_Sampler, f2Index);             \n"
    "}                                                                  \n"
    "                                                                   \n";

OCLLinearFilter::OCLLinearFilter() {
  done_ = false;
  _numSubTests = 2;
}

OCLLinearFilter::~OCLLinearFilter() {}

void OCLLinearFilter::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

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

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
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

  const char* kernels[2] = {"linear3D", "linear2D"};
  kernel_ = _wrapper->clCreateKernel(program_, kernels[test], &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem memory;
  size_t offset[3] = {0, 0, 0};
  cl_image_format imageFormat = {CL_RGBA, CL_FLOAT};

  if (test == 0) {
    float data[ImageSize][ImageSize][ImageSize][4];
    float index = 0.f;
    size_t region[3] = {ImageSize, ImageSize, ImageSize};
    for (size_t z = 0; z < ImageSize; ++z) {
      for (size_t y = 0; y < ImageSize; ++y) {
        for (size_t x = 0; x < ImageSize; ++x) {
          data[z][y][x][0] = (float)x;
          data[z][y][x][1] = (float)y;
          data[z][y][x][2] = (float)z;
          data[z][y][x][3] = 1.0f;
        }
      }
    }
    memory = _wrapper->clCreateImage3D(context_, CL_MEM_READ_ONLY, &imageFormat, ImageSize,
                                       ImageSize, ImageSize, 0, 0, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateImage() failed");

    error_ = _wrapper->clEnqueueWriteImage(cmdQueues_[_deviceId], memory, true, offset, region, 0,
                                           0, data, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteImage() failed");
  } else {
    float data[4][ImageSize][ImageSize];
    size_t region[3] = {ImageSize, ImageSize, 1};
    for (size_t y = 0; y < ImageSize; ++y) {
      for (size_t x = 0; x < ImageSize; ++x) {
        data[y][x][0] = (float)x;
        data[y][x][1] = (float)y;
        data[y][x][2] = data[y][x][3] = 1.0f;
      }
    }

    memory = _wrapper->clCreateImage2D(context_, CL_MEM_READ_ONLY, &imageFormat, ImageSize,
                                       ImageSize, 0, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateImage() failed");
    error_ = _wrapper->clEnqueueWriteImage(cmdQueues_[_deviceId], memory, true, offset, region, 0,
                                           0, data, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteImage() failed");
  }
  buffers_.push_back(memory);

  memory =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, 4 * sizeof(cl_float), NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(memory);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLLinearFilter::run(void) {
  if (done_) {
    return;
  }

  cl_float values[4] = {0.f, 0.f, 0.f, 0.f};
  cl_float ref[2] = {1.75f, 1.25f};
  cl_mem image = buffers()[0];
  cl_mem buffer = buffers()[1];

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &image);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t gws[1] = {0x1};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, true, 0,
                                         4 * sizeof(cl_float), values, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
  for (cl_uint i = 0; i < 2; ++i) {
    if (values[i] != ref[i]) {
      printf("%.2f != %.2f [ref]", values[i], ref[i]);
      CHECK_RESULT(true, " - Incorrect result for linear filtering!\n");
    }
  }
}

unsigned int OCLLinearFilter::close(void) { return OCLTestImp::close(); }
