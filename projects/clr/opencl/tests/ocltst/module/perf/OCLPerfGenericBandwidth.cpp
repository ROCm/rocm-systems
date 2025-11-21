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

#include "OCLPerfGenericBandwidth.h"

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

#define NUM_SIZES 4
// 256KB, 1 MB, 4MB, 16 MB
static const unsigned int Sizes[NUM_SIZES] = {262144, 1048576, 4194304, 16777216};

void OCLPerfGenericBandwidth::genShader(unsigned int idx) {
  shader_.clear();
  if (idx == 0) {
    shader_ +=
        "__kernel __attribute__((reqd_work_group_size(64,1,1))) void "
        "_genericReadSpeed(global float *outBuf, global float *inBuf, local "
        "float *inLocal, float c, char useLocal)\n"
        "{\n"
        "    int gid = (int) get_global_id(0);\n"
        "    int lid = (int) get_local_id(0);\n"
        "    float val0 = 0.0f;\n"
        "    float val1 = 0.0f;\n"
        "    float *localLocal;\n"
        "    int hacklid = gid % 64;\n"
        "    if (useLocal)\n"
        "        localLocal = inLocal;\n"
        "    else\n"
        "        localLocal = inBuf;\n"
        "    for (int i = 0; i < (768/64); i++) {\n"
        "        localLocal[hacklid + i*64] = lid;\n"
        "    }\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "#pragma nounroll\n"
        "    for (uint i = 0; i < 32;i++)\n"
        "    {\n"
        "        val0 += localLocal[lid+0];\n"
        "        val1 += localLocal[lid+64];\n"
        "        val0 += localLocal[lid+128];\n"
        "        val1 += localLocal[lid+192];\n"
        "        val0 += localLocal[lid+256];\n"
        "        val1 += localLocal[lid+320];\n"
        "        val0 += localLocal[lid+384];\n"
        "        val1 += localLocal[lid+448];\n"
        "        lid += 1;\n"
        "    }\n"
        "    val0 += val1;\n"
        "    val1 = min(val0,1.0f);\n"
        "    if ((lid + val1) < 0){\n"
        "        outBuf[gid] = val0;\n"
        "    }\n"
        "}\n";
    dataSizeBytes_ = 768 * 4;
  } else {
    shader_ +=
        "__kernel __attribute__((reqd_work_group_size(64,1,1))) void "
        "_genericReadSpeed(global float *outBuf, global float *inBuf, local "
        "float *inLocal, float c, char useLocal)\n"
        "{\n"
        "    uint gid = (uint) get_global_id(0);\n"
        "    int lid = (int) get_local_id(0);\n"
        "    float val0 = 0.0f;\n"
        "    float val1 = 0.0f;\n"
        "    float *localLocal;\n"
        "    uint hacklid = gid % 64;\n"
        "    if (useLocal)\n"
        "        localLocal = inLocal;\n"
        "    else\n"
        "        localLocal = inBuf;\n"
        "    for (int i = 0; i < (256/64); i++) {\n"
        "        localLocal[hacklid + i*64] = lid;\n"
        "    }\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    #pragma nounroll\n"
        "    for (uint i = 0; i < 32;i++)\n"
        "    {\n"
        "        val0 += localLocal[8*i+0];\n"
        "        val1 += localLocal[8*i+1];\n"
        "        val0 += localLocal[8*i+2];\n"
        "        val1 += localLocal[8*i+3];\n"
        "        val0 += localLocal[8*i+4];\n"
        "        val1 += localLocal[8*i+5];\n"
        "        val0 += localLocal[8*i+6];\n"
        "        val1 += localLocal[8*i+7];\n"
        "    }\n"
        "    val0 += val1;\n"
        "    val1 = min(val0,1.0f);\n"
        "    if ((lid + val1) < 0){\n"
        "        outBuf[gid] = val0;\n"
        "    }\n"
        "}\n";
    dataSizeBytes_ = 256 * 4;
  }
}

OCLPerfGenericBandwidth::OCLPerfGenericBandwidth() { _numSubTests = NUM_SIZES * 4; }

OCLPerfGenericBandwidth::~OCLPerfGenericBandwidth() {}

void OCLPerfGenericBandwidth::setData(cl_mem buffer, float val) {
  float* data = (float*)_wrapper->clEnqueueMapBuffer(
      cmdQueues_[_deviceId], buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < (bufSize_ >> 2); i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmdQueues_[_deviceId]);
}

void OCLPerfGenericBandwidth::checkData(cl_mem buffer) {
  float* data = (float*)_wrapper->clEnqueueMapBuffer(
      cmdQueues_[_deviceId], buffer, true, CL_MAP_READ, 0, bufSize_, 0, NULL, NULL, &error_);
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
  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmdQueues_[_deviceId]);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfGenericBandwidth::open(unsigned int test, char* units, double& conversion,
                                   unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  _crcword = 0;
  conversion = 1.0f;

  failed = false;
  kernel_ = 0;
  inBuffer_ = 0;
  outBuffer_ = 0;
  useLDS_ = ((test / NUM_SIZES) % 2) == 0 ? 1 : 0;

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
    failed = true;
    return;
  }
  delete strVersion;

  numReads_ = 32;
  width_ = Sizes[test % NUM_SIZES];
  shaderIdx_ = test / (NUM_SIZES * 2);

  bufSize_ = width_;

  inBuffer_ = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
  CHECK_RESULT(inBuffer_ == 0, "clCreateBuffer(inBuffer) failed");

  outBuffer_ = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  genShader(shaderIdx_);
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], "-cl-std=CL2.0", NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "_genericReadSpeed", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  float foo = 0;
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&inBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 2, 1024 * sizeof(cl_float), (void*)NULL);
  error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_float), (void*)&foo);
  error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_char), (void*)&useLDS_);

  setData(outBuffer_, 1.2345678f);
}

void OCLPerfGenericBandwidth::run(void) {
  if (failed) return;
  int global = bufSize_ / sizeof(cl_float);
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  CPerfCounter timer;

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < NUM_ITER; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  char buf[256];
  const char* buf2;
  if (useLDS_)
    buf2 = "LDS";
  else
    buf2 = "global";
  const char* buf3;
  if (shaderIdx_ == 0) {
    buf3 = "reads";
    numReads_ *= 8;
  } else {
    buf3 = "broadcast";
    numReads_ *= 8;
  }
  // LDS bandwidth in GB/s
  // We have one extra write per LDS location to initialize LDS
  double perf = ((double)global * (numReads_ * sizeof(cl_float) + dataSizeBytes_ / 64) * NUM_ITER *
                 (double)(1e-09)) /
                sec;

  _perfInfo = (float)perf;
  SNPRINTF(buf, sizeof(buf), " %6s %9s %8d threads, %3d reads (GB/s) ", buf2, buf3, global,
           numReads_);
  testDescString = buf;
  // checkData(outBuffer_);
}

unsigned int OCLPerfGenericBandwidth::close(void) {
  if (inBuffer_) {
    error_ = _wrapper->clReleaseMemObject(inBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(inBuffer_) failed");
  }
  if (outBuffer_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }

  return OCLTestImp::close();
}
