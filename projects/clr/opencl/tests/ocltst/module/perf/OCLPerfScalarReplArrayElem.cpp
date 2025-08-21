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

#include "OCLPerfScalarReplArrayElem.h"

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

#define NUM_SIZES 1
static const unsigned int Sizes[NUM_SIZES] = {16777216};  // 16

static void genKernelSource(const char* vtypeName, unsigned arrayLen, unsigned loopCount,
                            char* source) {
  sprintf(source,
          "%s foo(uint lid, __local %s *localLocal)\n"
          "{\n"
          "    %s val0 = 0.0f;\n"
          "    %s val1 = 0.0f;\n"
          "    for (int i = 0; i < %d; ++i) {\n"
          "      val0 += localLocal[lid];\n"
          "      lid += 16;\n"
          "    }\n"
          "    %s val = val0+val1;\n"
          "    return val;\n"
          "}\n"
          "__kernel __attribute__((reqd_work_group_size(64,1,1)))"
          "  void _ldsReadSpeed(__global %s *outBuf)\n"
          "{\n"
          "    uint gid = (int) get_global_id(0);\n"
          "    uint lid = (int) get_local_id(0);\n"
          "    __local %s localLocal[%d];\n"
          "    outBuf[gid] = foo(lid, localLocal);\n"
          "}\n",
          vtypeName, vtypeName, vtypeName, vtypeName, loopCount, vtypeName, vtypeName, vtypeName,
          arrayLen);
}

typedef struct {
  const char* name;
  unsigned nBytes;
} ExplicitType;

static const ExplicitType tyChar = {"char", 1};
static const ExplicitType tyShort = {"short", 2};
static const ExplicitType tyInt = {"int", 4};
static const ExplicitType tyLong = {"long", 8};
static const ExplicitType tyFloat = {"float", 4};
static const ExplicitType tyDouble = {"double", 8};

typedef struct {
  ExplicitType elemType;
  unsigned nElems;
  const char* name;
  unsigned getSize() const { return elemType.nBytes * nElems; }
} VectorType;

static const VectorType vecTypes[] = {
    {tyChar, 8, "char8"},     {tyShort, 4, "short4"},   {tyInt, 2, "int2"},
    {tyFloat, 2, "float2"},   {tyLong, 1, "long"},

    {tyChar, 16, "char16"},   {tyShort, 8, "short8"},   {tyInt, 4, "int4"},
    {tyFloat, 4, "float4"},   {tyLong, 2, "long2"},

    {tyShort, 16, "short16"}, {tyInt, 8, "int8"},       {tyFloat, 8, "float8"},
    {tyLong, 4, "long4"},

    {tyInt, 16, "int16"},     {tyFloat, 16, "float16"}, {tyLong, 8, "long8"},

    {tyLong, 16, "long16"}};
static const unsigned ldsBytes = 4 * 4096;
static const unsigned nVecTypes = sizeof(vecTypes) / sizeof(VectorType);

void OCLPerfScalarReplArrayElem::genShader(unsigned int idx) {
  VectorType vecType = vecTypes[idx];
  ExplicitType elemType = vecType.elemType;
  unsigned vecSize = vecType.nElems;
  unsigned arrayLen = ldsBytes / vecType.getSize();
  unsigned loopCount = arrayLen / 16;
  char source[7192];
  genKernelSource(vecType.name, arrayLen, loopCount, source);
  shader_ = std::string(source);
  numReads_ = loopCount;
  itemWidth_ = vecType.getSize();
}

OCLPerfScalarReplArrayElem::OCLPerfScalarReplArrayElem() { _numSubTests = NUM_SIZES * nVecTypes; }

OCLPerfScalarReplArrayElem::~OCLPerfScalarReplArrayElem() {}

void OCLPerfScalarReplArrayElem::setData(cl_mem buffer, float val) {
  float* data = (float*)_wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_WRITE, 0,
                                                     bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < (bufSize_ >> 2); i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

void OCLPerfScalarReplArrayElem::checkData(cl_mem buffer) {
  float* data = (float*)_wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_READ, 0,
                                                     bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < (bufSize_ >> 2); i++) {
    if (data[i] != (float)numReads_) {
      printf("Data validation failed at index %d!\n", i);
      printf("Expected %d %d %d %d\nGot %d %d %d %d\n", numReads_, numReads_, numReads_, numReads_,
             (unsigned int)data[i], (unsigned int)data[i + 1], (unsigned int)data[i + 2],
             (unsigned int)data[i + 3]);
      CHECK_RESULT_NO_RETURN(0, "Data validation failed!\n");
      break;
    }
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfScalarReplArrayElem::open(unsigned int test, char* units, double& conversion,
                                      unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;

  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  outBuffer_ = 0;
  _openTest = test;

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

  width_ = Sizes[test % NUM_SIZES];
  shaderIdx_ = test / NUM_SIZES;
  bufSize_ = width_;

  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  outBuffer_ = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  genShader(shaderIdx_);
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &device, "", NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "_ldsReadSpeed", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);

  // setData(outBuffer_, 1.2345678f);
}

void OCLPerfScalarReplArrayElem::run(void) {
  int global = bufSize_ / itemWidth_;
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  CPerfCounter timer;

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < NUM_ITER; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }
  _wrapper->clFinish(cmd_queue_);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Constant bandwidth in GB/s
  double perf = ((double)global * numReads_ * itemWidth_ * NUM_ITER * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " %10s %8d threads, %4d reads (GB/s)", vecTypes[shaderIdx_].name,
           global, numReads_);
  testDescString = buf;
  // checkData(outBuffer_);
}

unsigned int OCLPerfScalarReplArrayElem::close(void) {
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
