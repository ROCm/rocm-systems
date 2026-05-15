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

#include "OCLCreateImage.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <sstream>
#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#include "CL/cl.h"

const static size_t ImageSize = 4;
const static size_t MaxSubTests = 5;

const static char* strKernel =
    "const sampler_t g_Sampler =    CLK_FILTER_LINEAR |                 \n"
    "                               CLK_ADDRESS_CLAMP_TO_EDGE |         \n"
    "                               CLK_NORMALIZED_COORDS_FALSE;        \n"
    "                                                                   \n"
    "__kernel void linear3D(__read_only image3d_t img3D, __global float4* "
    "f4Tata) \n"
    "{                                                                  \n"
    "   float4 f4Index = { 2.25f, 1.75f, 0.5f, 0.0f };                  \n"
    "   // copy interpolated data in result buffer                      \n"
    "   f4Tata[0] = read_imagef(img3D, g_Sampler, f4Index);             \n"
    "}                                                                  \n"
    "                                                                   \n"
    "__kernel void linear2D(__read_only image2d_t img2D, __global float4* "
    "f4Tata) \n"
    "{                                                                  \n"
    "   float2 f2Index = { 2.25f, 1.75f };                              \n"
    "   // copy interpolated data in result buffer                      \n"
    "   f4Tata[0] = read_imagef(img2D, g_Sampler, f2Index);             \n"
    "}                                                                  \n"
    "                                                                   \n"
    "__kernel void linear1DArray(__read_only image1d_array_t img1DA, __global "
    "float4* f4Tata) \n"
    "{                                                                  \n"
    "   float2 f2Index = { 2.25f, 0 };                                  \n"
    "   // copy interpolated data in result buffer                      \n"
    "   f4Tata[0] = read_imagef(img1DA, g_Sampler, f2Index);             \n"
    "}                                                                  \n"
    "                                                                   \n"
    "__kernel void linear2DArray(__read_only image2d_array_t img2DA, __global "
    "float4* f4Tata) \n"
    "{                                                                  \n"
    "   float4 f4Index = { 2.25f, 1.75f, 0.0f, 0.0f };                  \n"
    "   // copy interpolated data in result buffer                      \n"
    "   f4Tata[0] = read_imagef(img2DA, g_Sampler, f4Index);            \n"
    "}                                                                  \n"
    "                                                                   \n"
    "__kernel void point1DBuffer(__read_only image1d_buffer_t img1DB, __global "
    "float4* f4Tata) \n"
    "{                                                                  \n"
    "   int index = 2;                                                  \n"
    "   // copy interpolated data in result buffer                      \n"
    "   f4Tata[0] = read_imagef(img1DB, index);                         \n"
    "}                                                                  \n"
    "                                                                   \n";

OCLCreateImage::OCLCreateImage() {
  _numSubTests = MaxSubTests;
  done_ = false;
  ImageSizeX = ImageSize;
  ImageSizeY = ImageSize;
  ImageSizeZ = ImageSize;
}

OCLCreateImage::~OCLCreateImage() {}

void OCLCreateImage::open(unsigned int test, char* units, double& conversion,
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

  cl_ulong max2DWidth;
  cl_ulong max2DHeight;

  cl_ulong max3DWidth;
  cl_ulong max3DHeight;
  cl_ulong max3DDepth;

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                            &maxSize_, &size);

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(cl_ulong),
                            &max2DWidth, &size);

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(cl_ulong),
                            &max2DHeight, &size);

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(cl_ulong),
                            &max3DWidth, &size);

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(cl_ulong),
                            &max3DHeight, &size);

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(cl_ulong),
                            &max3DDepth, &size);

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

  const char* kernels[] = {"linear3D", "linear2D", "linear2DArray", "linear1DArray",
                           "point1DBuffer"};
  unsigned int dimensions[] = {3, 2, 3, 2, 1};
  kernel_ = _wrapper->clCreateKernel(program_, kernels[test], &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem memory;
  cl_mem buf = NULL;
  cl_image_desc desc;
  size_t offset[3] = {0, 0, 0};
  cl_image_format imageFormat = {CL_RGBA, CL_FLOAT};

  desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  desc.image_array_size = 0;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.buffer = (cl_mem)NULL;

  if (test == 0) {
    desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    if (is64BitApp()) {
      ImageSizeX = max3DWidth;
      ImageSizeY = maxSize_ / (ImageSizeX * 16);
      if (ImageSizeY > (max3DHeight)) {
        ImageSizeY = max3DHeight;
      }
      ImageSizeZ = maxSize_ / (ImageSizeX * ImageSizeY * 16);
#if EMU_ENV
      ImageSizeX = ImageSizeY = ImageSizeZ = 4;
#endif  // EMU_ENV
    } else {
      ImageSizeX = 4;
      ImageSizeY = 4;
      ImageSizeZ = 4;
    }
    desc.image_width = ImageSizeX;
    desc.image_height = ImageSizeY;
    desc.image_depth = ImageSizeZ;
  }
  if (test == 1) {
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    if (is64BitApp()) {
      ImageSizeX = max2DWidth - 0x10;
      ImageSizeY = maxSize_ / (ImageSizeX * 16 * 2);
      if (ImageSizeY >= max2DHeight) {
        ImageSizeY = max2DHeight - 0x1000;
      }
#ifdef __linux__
      // On linux, if the size of total system memory is less than 4GB,
      // then, we can allocate much smaller image.
      // TODO, need to find the root cause
      struct sysinfo myinfo;
      unsigned long total_bytes;

      sysinfo(&myinfo);
      total_bytes = myinfo.mem_unit * myinfo.totalram;
      if ((total_bytes / (1024 * 1024)) <= 4096) {
        ImageSizeY /= 2;
      }
#endif
#if EMU_ENV
      ImageSizeX = ImageSizeY = 4;
#endif  // EMU_ENV
    } else {
      ImageSizeX = 4;
      ImageSizeY = 4;
    }
    ImageSizeZ = 0;
    desc.image_width = ImageSizeX;
    desc.image_height = ImageSizeY;
    desc.image_depth = 0;
  } else if (test == 2) {
    desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    ImageSizeX = ImageSize;
    ImageSizeY = ImageSize;
    ImageSizeZ = ImageSize;
    desc.image_width = ImageSizeX;
    desc.image_height = ImageSizeY;
    desc.image_depth = 0;
    desc.image_array_size = ImageSize;
  } else if (test == 3) {
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
    ImageSizeX = ImageSize;
    ImageSizeY = ImageSize;
    ImageSizeZ = 0;
    desc.image_width = ImageSize;
    desc.image_height = ImageSize;
    desc.image_depth = 0;
    desc.image_array_size = ImageSize;
  } else if (test == 4) {
    ImageSizeX = ImageSize;
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    buf = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, ImageSizeX * 4 * sizeof(cl_float),
                                   NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    ImageSizeY = 0;
    ImageSizeZ = 0;
    desc.image_width = ImageSizeX;
    desc.image_height = 0;
    desc.image_depth = 0;
    desc.buffer = buf;
  }

  memory = _wrapper->clCreateImage(context_, CL_MEM_READ_ONLY, &imageFormat, &desc, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateImage() failed");

  float fillColor[4] = {1.f, 1.f, 1.f, 1.f};

  if (dimensions[test] == 1) {
    float data[4][ImageSize];
    size_t region[3] = {ImageSize, 1, 1};

    error_ = _wrapper->clEnqueueFillImage(cmdQueues_[_deviceId], memory, fillColor, offset, region,
                                          0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueFillImage() failed");
    error_ = _wrapper->clEnqueueReadImage(cmdQueues_[_deviceId], memory, true, offset, region, 0, 0,
                                          data, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadImage() failed");

    for (size_t x = 0; x < ImageSize; ++x) {
      if (0 != memcmp(&data[x], fillColor, sizeof(fillColor))) {
        CHECK_RESULT(true, "Fill image validation failed");
      }
      data[x][0] = (float)x;
      data[x][1] = data[x][2] = data[x][3] = 1.0f;
    }
    error_ = _wrapper->clEnqueueWriteImage(cmdQueues_[_deviceId], memory, true, offset, region, 0,
                                           0, data, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteImage() failed");
  } else if (dimensions[test] == 2) {
    size_t region[3] = {ImageSizeX, ImageSizeY, 1};

    error_ = _wrapper->clEnqueueFillImage(cmdQueues_[_deviceId], memory, fillColor, offset, region,
                                          0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueFillImage() failed");

    float* data;
    size_t ActualImageSizeY = ImageSizeY;
    size_t maxImageSize = maxSize_;
#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (maxImageSize > ((size_t)pages * page_size)) {
      maxImageSize = ((size_t)pages * page_size);
    }
#endif
    while ((((ImageSizeX * ActualImageSizeY * sizeof(float) * 4) / (1024 * 1024)) >=
            (size_t)4 * 1024) ||
           ((ImageSizeX * ActualImageSizeY * sizeof(float) * 4) >= (maxImageSize / 2))) {
      if (ActualImageSizeY == 1) {
        break;
      }
      ActualImageSizeY /= 2;
    }
    while ((data = (float*)malloc(ImageSizeX * ActualImageSizeY * sizeof(float) * 4)) == NULL) {
      if (ActualImageSizeY == 1) {
        break;
      }
      ActualImageSizeY /= 2;
    }
    if (data == NULL) {
      CHECK_RESULT(true, "malloc() failed");
    }

    size_t remainSizeY = ImageSizeY;
    while (remainSizeY > 0) {
      ActualImageSizeY = (remainSizeY > ActualImageSizeY) ? ActualImageSizeY : remainSizeY;
      size_t tmpRange[3] = {ImageSizeX, ActualImageSizeY, 1};
      error_ = _wrapper->clEnqueueReadImage(cmdQueues_[_deviceId], memory, true, offset, tmpRange,
                                            0, 0, data, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadImage() failed");

      for (size_t y = 0; y < ActualImageSizeY; ++y) {
        for (size_t x = 0; x < ImageSizeX; ++x) {
          size_t offsetData = (y * ImageSizeX + x) * 4;
          if (0 != memcmp(&data[offsetData], fillColor, sizeof(fillColor))) {
            CHECK_RESULT(true, "Fill image validation failed");
          }
          data[offsetData + 0] = (float)x;
          data[offsetData + 1] = (float)y;
          data[offsetData + 2] = data[offsetData + 3] = 1.0f;
        }
      }
      error_ = _wrapper->clEnqueueWriteImage(cmdQueues_[_deviceId], memory, true, offset, tmpRange,
                                             0, 0, data, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteImage() failed");
      remainSizeY -= ActualImageSizeY;
      offset[1] += ActualImageSizeY;
    }
    free(data);
  } else if (dimensions[test] == 3) {
    float* data;

    float index = 0.f;
    size_t region[3] = {ImageSizeX, ImageSizeY, ImageSizeZ};
    error_ = _wrapper->clEnqueueFillImage(cmdQueues_[_deviceId], memory, fillColor, offset, region,
                                          0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueFillImage() failed");

    size_t ActualImageSizeZ = ImageSizeZ;
    size_t maxImageSize = maxSize_;
#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (maxImageSize > ((size_t)pages * page_size)) {
      maxImageSize = ((size_t)pages * page_size);
    }
#endif
    while (
        (((ImageSizeX * ImageSizeY * ActualImageSizeZ * sizeof(float) * 4) / (1024 * 1024)) >=
         (size_t)4 * 1024) ||
        ((ImageSizeX * ImageSizeY * ActualImageSizeZ * sizeof(float) * 4) >= (maxImageSize / 2))) {
      if (ActualImageSizeZ == 1) {
        break;
      }
      ActualImageSizeZ /= 2;
    }
    while ((data = (float*)malloc(ImageSizeX * ImageSizeY * ActualImageSizeZ * sizeof(float) *
                                  4)) == NULL) {
      if (ActualImageSizeZ == 1) {
        break;
      }
      ActualImageSizeZ -= 1;
    }
    if (data == NULL) {
      CHECK_RESULT(true, "malloc() failed");
    }

    size_t remainSizeZ = ImageSizeZ;
    while (remainSizeZ > 0) {
      ActualImageSizeZ = (remainSizeZ > ActualImageSizeZ) ? ActualImageSizeZ : remainSizeZ;
      size_t tmpRange[3] = {ImageSizeX, ImageSizeY, ActualImageSizeZ};
      error_ = _wrapper->clEnqueueReadImage(cmdQueues_[_deviceId], memory, true, offset, tmpRange,
                                            0, 0, data, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadImage() failed");

      for (size_t z = 0; z < ActualImageSizeZ; ++z) {
        for (size_t y = 0; y < ImageSizeY; ++y) {
          for (size_t x = 0; x < ImageSizeX; ++x) {
            size_t offset = (((z * ImageSizeY) + y) * ImageSizeX + x) * 4;
            if (0 != memcmp(&data[offset], fillColor, sizeof(fillColor))) {
              CHECK_RESULT(true, "Fill image validation failed");
            }
            data[offset + 0] = (float)x;
            data[offset + 1] = (float)y;
            data[offset + 2] = (float)z;
            data[offset + 3] = 1.0f;
          }
        }
      }
      error_ = _wrapper->clEnqueueWriteImage(cmdQueues_[_deviceId], memory, true, offset, tmpRange,
                                             0, 0, data, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteImage() failed");
      remainSizeZ -= ActualImageSizeZ;
      offset[2] += ActualImageSizeZ;
    }
    free(data);
  }

  buffers_.push_back(memory);

  memory =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, 4 * sizeof(cl_float), NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(memory);
  if (buf != NULL) {
    buffers_.push_back(buf);
  }
  size_t imageSizebyte = (ImageSizeY != 0) ? ImageSizeY * ImageSizeX : ImageSizeX;
  imageSizebyte *= (ImageSizeZ != 0) ? ImageSizeZ : 1;
  imageSizebyte *= 16;  //  16 bytes per pixel, imageFormat = {CL_RGBA,CL_FLOAT}
  char strImgSize[200];
  if (imageSizebyte >= 1024 * 1024) {
    sprintf(strImgSize, "%5ld MB", (long)(imageSizebyte / (1024 * 1024)));
  } else {
    sprintf(strImgSize, "%6ld Bytes", (long)imageSizebyte);
  }
  std::stringstream str;
  str << " (";
  str << ImageSizeX;
  str << ", ";
  str << ImageSizeY;
  str << ",  ";
  str << ImageSizeZ;
  str << ") ";
  str << strImgSize;

  testDescString = str.str();
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLCreateImage::run(void) {
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
  if (testID_ == 4) {
    ref[0] = 2.0f;
  }
  for (cl_uint i = 0; i < static_cast<cl_uint>((testID_ >= 3) ? 1 : 2); ++i) {
    if (values[i] != ref[i]) {
      printf("%.2f != %.2f [ref]", values[i], ref[i]);
      CHECK_RESULT(true, " - Incorrect result for linear filtering!\n");
    }
  }
}

unsigned int OCLCreateImage::close(void) { return OCLTestImp::close(); }
