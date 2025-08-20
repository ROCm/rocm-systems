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

#include "OCLPerfMatrixTranspose.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

static const unsigned int NUM_BLOCK_SIZES = 2;
static const unsigned int blockSizes[NUM_BLOCK_SIZES] = {8, 16};
static const unsigned int NUM_MATRIX_DIMS = 2;
static const unsigned int matrixDims[NUM_MATRIX_DIMS] = {1024, 1920};
static const char* matrixtranspose_kernel =
    "kernel void matrixTranspose(global uint *restrict inBuf, global uint "
    "*restrict outBuf, local uint *localBuf, uint blockSize, uint width, uint "
    "height)\n"
    "{\n"
    "    uint globalIdx = get_global_id(0);\n"
    "    uint globalIdy = get_global_id(1);\n"

    "    uint localIdx = get_local_id(0);\n"
    "    uint localIdy = get_local_id(1);\n"

    "    /* copy from input to local memory */\n"
    "    /* Note that we transpose the x and y coordinates when storing */\n"
    "    localBuf[localIdx*blockSize + localIdy] = inBuf[globalIdy*width + "
    "globalIdx];\n"

    "    /* wait until the whole block is filled */\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"

    "    uint groupIdx = get_group_id(0);\n"
    "    uint groupIdy = get_group_id(1);\n"

    "    /* calculate the corresponding target location for transpose  by "
    "inverting x and y values*/\n"
    "    /* Here we don't swap localIdx and localIdy, this is to get larger "
    "bursts when threads write to memory. */\n"
    "    /* To make this work, we've swapped the coordinates when we write to "
    "local memory. */\n"
    "    uint targetGlobalIdx = groupIdy*blockSize + localIdx;\n"
    "    uint targetGlobalIdy = groupIdx*blockSize + localIdy;\n"

    "    /* calculate the corresponding raster indices of source and target "
    "*/\n"
    "    uint targetIndex  = targetGlobalIdy*height     + targetGlobalIdx;\n"
    "    uint sourceIndex  = localIdy       * blockSize + localIdx;\n"

    "    outBuf[targetIndex] = localBuf[sourceIndex];\n"
    "}\n";

OCLPerfMatrixTranspose::OCLPerfMatrixTranspose() {
  _numSubTests = NUM_BLOCK_SIZES * NUM_MATRIX_DIMS;
}

OCLPerfMatrixTranspose::~OCLPerfMatrixTranspose() {}

void OCLPerfMatrixTranspose::setData(cl_mem buffer) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < height_; i++) {
    for (unsigned int j = 0; j < width_; j++) {
      *(data + i * width_ + j) = i * width_ + j;
    }
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

void OCLPerfMatrixTranspose::fillData(cl_mem buffer, unsigned int val) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < width_ * height_; i++) {
    data[i] = val;
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

void OCLPerfMatrixTranspose::checkData(cl_mem buffer) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_READ, 0, bufSize_, 0, NULL, NULL, &error_);
  bool err = false;
  for (unsigned int i = 0; (i < width_) && !err; i++) {
    for (unsigned int j = 0; (j < height_) && !err; j++) {
      if (*(data + i * height_ + j) != (j * width_ + i)) {
        printf("Data mismatch at (%d, %d)!  Got %d, expected %d\n", j, i, *(data + i * height_ + j),
               j * width_ + i);
        err = true;
        break;
      }
    }
    break;
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfMatrixTranspose::open(unsigned int test, char* units, double& conversion,
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

  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  inBuffer_ = 0;
  outBuffer_ = 0;

  blockSize_ = blockSizes[_openTest % NUM_BLOCK_SIZES];
  width_ = matrixDims[_openTest / NUM_BLOCK_SIZES];
  height_ = width_;
  // We compute a square domain
  bufSize_ = width_ * height_ * sizeof(cl_uint);

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
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    // Runtime returns an error when no GPU devices are present instead of just
    // returning 0 devices
    // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
    // Choose platform with GPU devices
    if (num_devices > 0) {
      // platform = platforms[_platformIndex];
      // break;
    }
#if 0
        }
#endif
    delete platforms;
  }
  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find platform with GPU devices, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  inBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, bufSize_, NULL, &error_);
  CHECK_RESULT(inBuffer_ == 0, "clCreateBuffer(inBuffer) failed");
  setData(inBuffer_);

  outBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");
  fillData(outBuffer_, 0xdeadbeef);

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&matrixtranspose_kernel,
                                                 NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  char* buildOps = NULL;
  error_ = _wrapper->clBuildProgram(program_, 1, &device, buildOps, NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "matrixTranspose", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&inBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint) * blockSize_ * blockSize_, NULL);
  error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_uint), (void*)&blockSize_);
  error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_uint), (void*)&width_);
  error_ = _wrapper->clSetKernelArg(kernel_, 5, sizeof(cl_uint), (void*)&height_);
}

void OCLPerfMatrixTranspose::run(void) {
  size_t global_work_size[2] = {width_, height_};
  size_t local_work_size[2] = {blockSize_, blockSize_};

  CPerfCounter timer;

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < MAX_ITERATIONS; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 2, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
  }

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue_);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  checkData(outBuffer_);
  // Compute GB/s
  double perf = ((double)bufSize_ * (double)MAX_ITERATIONS * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  testDescString = "";
  char str[64];
  sprintf(str, "(%d,%d) matrix with (%2d,%2d) block size %fms (GB/s) ", width_, height_, blockSize_,
          blockSize_, (sec / (double)MAX_ITERATIONS) * 1000.);
  testDescString += str;
}

unsigned int OCLPerfMatrixTranspose::close(void) {
  _wrapper->clFinish(cmd_queue_);

  if (inBuffer_) {
    error_ = _wrapper->clReleaseMemObject(inBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(inBuffer_) failed");
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
