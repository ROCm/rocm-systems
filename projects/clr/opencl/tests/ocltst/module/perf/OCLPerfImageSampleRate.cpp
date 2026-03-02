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

#include "OCLPerfImageSampleRate.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_TYPES 6
static const cl_image_format formats[NUM_TYPES] = {
    {CL_R, CL_UNSIGNED_INT8}, {CL_RG, CL_UNSIGNED_INT8}, {CL_RGBA, CL_UNSIGNED_INT8},
    {CL_R, CL_FLOAT},         {CL_RGBA, CL_HALF_FLOAT},  {CL_RGBA, CL_FLOAT}};
static const char* types[NUM_TYPES] = {"R8",   "R8G8",          "R8G8B8A8",
                                       "R32F", "R16G16B16A16F", "R32G32B32A32F"};
static const unsigned int typeSizes[NUM_TYPES] = {1, 2, 4, 4, 8, 16};

#define NUM_SIZES 12
static const unsigned int sizes[NUM_SIZES] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

#define NUM_BUFS 6
#define MAX_BUFS (1 << (NUM_BUFS - 1))

OCLPerfImageSampleRate::OCLPerfImageSampleRate() {
  _numSubTests = NUM_TYPES * NUM_SIZES * NUM_BUFS;
}

OCLPerfImageSampleRate::~OCLPerfImageSampleRate() {}

void OCLPerfImageSampleRate::setKernel(void) {
  shader_.clear();
  shader_ +=
      "kernel void sampleRate(global float4* outBuffer, unsigned int "
      "inBufSize, unsigned int writeIt,\n";
  char buf[256];
  for (unsigned int i = 0; i < numBufs_; i++) {
    SNPRINTF(buf, sizeof(buf), "read_only image2d_t inBuffer%d", i);
    shader_ += buf;
    if (i < (numBufs_ - 1)) {
      shader_ += ",";
    }
    shader_ += "\n";
  }
  shader_ += ")\n";
  shader_ +=
      "{\n"
      "    uint gid = get_global_id(0);\n"
      "    uint inputIdx = gid % inBufSize;\n"
      "    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
      "    float4 tmp = (float4)0.0f;\n";

  for (unsigned int i = 0; i < numBufs_; i++) {
    SNPRINTF(buf, sizeof(buf),
             "    tmp += read_imagef(inBuffer%d, sampler, (int2)( gid %% "
             "inBufSize, (gid / inBufSize) %% inBufSize));\n",
             i);
    shader_ += buf;
  }
  shader_ +=
      "    if (writeIt*(unsigned int)tmp.x) outBuffer[gid] = tmp;\n"
      "}\n";
  // printf("Shader -> %s\n", shader_.c_str());
}

void OCLPerfImageSampleRate::setData(cl_mem buffer, unsigned int val) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width_, width_, 1};
  size_t image_row_pitch;
  size_t image_slice_pitch;
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapImage(
      cmd_queue_, buffer, true, CL_MAP_WRITE, origin, region, &image_row_pitch, &image_slice_pitch,
      0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < width_ * width_; i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

void OCLPerfImageSampleRate::checkData(cl_mem buffer) {
#if 0
    float* data = (float *)_wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_READ, 0, outBufSize_, 0, NULL, NULL, &error_);
    for (unsigned int i = 0; i < outBufSize_/sizeof(float); i++)
    {
        if (data[i] != (float)numBufs_) {
            printf("Data validation failed at %d! Got %f, expected %f\n", i, data[i], (float)numBufs_);
            break;
        }
    }
    error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfImageSampleRate::open(unsigned int test, char* units, double& conversion,
                                  unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;
  skip_ = false;
  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  inBuffer_ = 0;
  outBuffer_ = 0;

  // We compute a square domain
  width_ = sizes[test % NUM_SIZES];
  numBufs_ = (1 << ((test / NUM_SIZES) % NUM_BUFS));
  typeIdx_ = (test / (NUM_SIZES * NUM_BUFS)) % NUM_TYPES;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
    platform = platforms[_platformIndex];
    num_devices = 0;
    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    delete platforms;
  }
  /*
   * If we could find a platform, use it.
   */
  CHECK_RESULT(platform == 0, "Couldn't find platform with GPU devices, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
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
  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  inBuffer_ = (cl_mem*)malloc(sizeof(cl_mem) * numBufs_);
  memset(inBuffer_, 0, sizeof(cl_mem) * numBufs_);
  for (unsigned int i = 0; i < numBufs_; i++) {
    inBuffer_[i] = _wrapper->clCreateImage2D(context_, CL_MEM_READ_ONLY, &formats[typeIdx_], width_,
                                             width_, 0, NULL, &error_);
    CHECK_RESULT(inBuffer_[i] == 0, "clCreateImage2D(inBuffer) failed");
  }

  outBufSize_ = sizes[NUM_SIZES - 1] * sizes[NUM_SIZES - 1] * sizeof(cl_float4);
  outBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, outBufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  setKernel();
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  const char* buildOps = NULL;
  error_ = _wrapper->clBuildProgram(program_, 1, &device, buildOps, NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "sampleRate", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(outBuffer) failed");
  unsigned int sizeDW = width_;
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(unsigned int), (void*)&sizeDW);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(sizeDW) failed");
  unsigned int writeIt = 0;
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(unsigned int), (void*)&writeIt);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(writeIt) failed");
  for (unsigned int i = 0; i < numBufs_; i++) {
    error_ = _wrapper->clSetKernelArg(kernel_, i + 3, sizeof(cl_mem), (void*)&inBuffer_[i]);
    CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(inBuffer) failed");
    // setData(inBuffer_[i], 0x3f800000);
  }
  // setData(outBuffer_, 0xdeadbeef);
}

void OCLPerfImageSampleRate::run(void) {
  if (skip_) {
    return;
  }
  int global = outBufSize_ / typeSizes[typeIdx_];
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};
  unsigned int maxIter = MAX_ITERATIONS * (MAX_BUFS / numBufs_);

  CPerfCounter timer;

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < maxIter; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
  }

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue_);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // checkData(outBuffer_);
  // Compute GB/s
  double perf = ((double)outBufSize_ * numBufs_ * (double)maxIter * (double)(1e-09)) / sec;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), "Domain %dx%d,  %13s, %2d images,%4dx%4d (GB/s)", sizes[NUM_SIZES - 1],
           sizes[NUM_SIZES - 1], types[typeIdx_], numBufs_, width_, width_);

  _perfInfo = (float)perf;
  testDescString = buf;
}

unsigned int OCLPerfImageSampleRate::close(void) {
  if (skip_) {
    return CL_SUCCESS;
  }
  _wrapper->clFinish(cmd_queue_);

  if (inBuffer_) {
    for (unsigned int i = 0; i < numBufs_; i++) {
      if (inBuffer_[i]) {
        error_ = _wrapper->clReleaseMemObject(inBuffer_[i]);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(inBuffer_) failed");
      }
    }
    free(inBuffer_);
  }
  if (outBuffer_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
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
