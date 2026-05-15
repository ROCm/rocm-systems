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

#include "OCLPerfSampleRate.h"

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

#define NUM_TYPES 3
static const char* types[NUM_TYPES] = {"float", "float2", "float4"};
static const unsigned int typeSizes[NUM_TYPES] = {4, 8, 16};

#define NUM_SIZES 12
static const unsigned int sizes[NUM_SIZES] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

#define NUM_BUFS 6
#define MAX_BUFS (1 << (NUM_BUFS - 1))

OCLPerfSampleRate::OCLPerfSampleRate() {
  _numSubTests = NUM_TYPES * NUM_SIZES * NUM_BUFS;
  skip_ = false;
}

OCLPerfSampleRate::~OCLPerfSampleRate() {}

void OCLPerfSampleRate::setKernel(void) {
  shader_.clear();
  shader_ +=
      "kernel void sampleRate(global DATATYPE* outBuffer, unsigned int "
      "inBufSize, unsigned int writeIt,\n";
  char buf[256];
  for (unsigned int i = 0; i < numBufs_; i++) {
    SNPRINTF(buf, sizeof(buf), "global DATATYPE* inBuffer%d", i);
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
      "    DATATYPE tmp = (DATATYPE)0.0f;\n";

  for (unsigned int i = 0; i < numBufs_; i++) {
    SNPRINTF(buf, sizeof(buf), "    tmp += inBuffer%d[inputIdx];\n", i);
    shader_ += buf;
  }
  if (typeSizes[typeIdx_] > 4) {
    shader_ +=
        "    if (writeIt*(unsigned int)tmp.x) outBuffer[gid] = tmp;\n"
        "}\n";
  } else {
    shader_ +=
        "    if (writeIt*(unsigned int)tmp) outBuffer[gid] = tmp;\n"
        "}\n";
  }
  // printf("Shader -> %s\n", shader_.c_str());
}

void OCLPerfSampleRate::setData(cl_mem buffer, unsigned int val) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  if (data == NULL) {
    if ((error_ == CL_MEM_OBJECT_ALLOCATION_FAILURE) || (error_ == CL_OUT_OF_RESOURCES) ||
        (error_ == CL_OUT_OF_HOST_MEMORY)) {
      printf("WARNING: Not enough memory, skipped\n");
      error_ = CL_SUCCESS;
      skip_ = true;
    } else {
      CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueMapBuffer failed");
    }
    return;
  }
  for (unsigned int i = 0; i < bufSize_ / sizeof(unsigned int); i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

void OCLPerfSampleRate::checkData(cl_mem buffer) {
  float* data = (float*)_wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_READ, 0,
                                                     outBufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < outBufSize_ / sizeof(float); i++) {
    if (data[i] != (float)numBufs_) {
      printf("Data validation failed at %d! Got %f, expected %f\n", i, data[i], (float)numBufs_);
      break;
    }
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfSampleRate::open(unsigned int test, char* units, double& conversion,
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

  // We compute a square domain
  width_ = sizes[test % NUM_SIZES];
  typeIdx_ = (test / NUM_SIZES) % NUM_TYPES;
  bufSize_ = width_ * width_ * typeSizes[typeIdx_];
  numBufs_ = (1 << (test / (NUM_SIZES * NUM_TYPES)));

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
    inBuffer_[i] = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, bufSize_, NULL, &error_);
    CHECK_RESULT(inBuffer_[i] == 0, "clCreateBuffer(inBuffer) failed");
  }

  outBufSize_ = sizes[NUM_SIZES - 1] * sizes[NUM_SIZES - 1] * typeSizes[NUM_TYPES - 1];
  outBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, outBufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  setKernel();
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  const char* buildOps = NULL;
  SNPRINTF(charbuf, sizeof(charbuf), "-D DATATYPE=%s", types[typeIdx_]);
  buildOps = charbuf;
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
  unsigned int sizeDW = width_ * width_;
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(unsigned int), (void*)&sizeDW);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(sizeDW) failed");
  unsigned int writeIt = 0;
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(unsigned int), (void*)&writeIt);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(writeIt) failed");
  for (unsigned int i = 0; i < numBufs_; i++) {
    error_ = _wrapper->clSetKernelArg(kernel_, i + 3, sizeof(cl_mem), (void*)&inBuffer_[i]);
    CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(inBuffer) failed");
    setData(inBuffer_[i], 0x3f800000);
    if (skip_) return;
  }
  setData(outBuffer_, 0xdeadbeef);
}

void OCLPerfSampleRate::run(void) {
  int global = outBufSize_ / typeSizes[typeIdx_];
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};
  unsigned int maxIter = MAX_ITERATIONS * (MAX_BUFS / numBufs_);

  if (skip_) return;

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
  SNPRINTF(buf, sizeof(buf), "Domain %dx%d, %2d bufs, %6s, %4dx%4d (GB/s)", sizes[NUM_SIZES - 1],
           sizes[NUM_SIZES - 1], numBufs_, types[typeIdx_], width_, width_);

  _perfInfo = (float)perf;
  testDescString = buf;
}

unsigned int OCLPerfSampleRate::close(void) {
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
