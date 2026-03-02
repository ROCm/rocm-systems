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

#include "OCLPerfLDSReadSpeed.h"

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

void OCLPerfLDSReadSpeed::genShader(unsigned int idx) {
  shader_.clear();
  if (idx == 0) {
    shader_ +=
        "__kernel __attribute__((reqd_work_group_size(64,1,1))) void "
        "_ldsReadSpeed(__global float *outBuf, float c)\n"
        "{\n"
        "    uint gid = (int) get_global_id(0);\n"
        "    uint lid = (int) get_local_id(0);\n"
        "    __local float localLocal[2048];\n"
        "    float val1 = c;\n"
        "    float val2 = c;\n"
        "    float val3 = c;\n"
        "    float val4 = c;\n"
        "    uint hacklid = gid % 64;\n"
        "    for (int i = 0; i < (2048/64); i++) {\n"
        "        localLocal[hacklid + i*64] = lid;\n"
        "    }\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    val1 += localLocal[lid+0];\n"
        "    val2 += localLocal[lid+64];\n"
        "    val3 += localLocal[lid+128];\n"
        "    val4 += localLocal[lid+192];\n"
        "    val1 += localLocal[lid+256];\n"
        "    val2 += localLocal[lid+320];\n"
        "    val3 += localLocal[lid+384];\n"
        "    val4 += localLocal[lid+448];\n"
        "    val1 += localLocal[lid+512];\n"
        "    val2 += localLocal[lid+576];\n"
        "    val3 += localLocal[lid+640];\n"
        "    val4 += localLocal[lid+704];\n"
        "    val1 += localLocal[lid+768];\n"
        "    val2 += localLocal[lid+832];\n"
        "    val3 += localLocal[lid+896];\n"
        "    val4 += localLocal[lid+960];\n"
        "    val1 += localLocal[lid+1024];\n"
        "    val2 += localLocal[lid+1088];\n"
        "    val3 += localLocal[lid+1152];\n"
        "    val4 += localLocal[lid+1216];\n"
        "    val1 += localLocal[lid+1280];\n"
        "    val2 += localLocal[lid+1344];\n"
        "    val3 += localLocal[lid+1408];\n"
        "    val4 += localLocal[lid+1472];\n"
        "    val1 += localLocal[lid+1536];\n"
        "    val2 += localLocal[lid+1600];\n"
        "    val3 += localLocal[lid+1664];\n"
        "    val4 += localLocal[lid+1728];\n"
        "    val1 += localLocal[lid+1792];\n"
        "    val2 += localLocal[lid+1856];\n"
        "    val3 += localLocal[lid+1920];\n"
        "    val4 += localLocal[lid+1984];\n"
        "    outBuf[gid] = val1+val2+val3+val4;\n"
        "}\n";
    ldsSizeBytes_ = 2048 * 4;
  } else if (idx == 1) {
    shader_ +=
        "__kernel __attribute__((reqd_work_group_size(64,1,1))) void "
        "_ldsReadSpeed(__global float *outBuf, float c)\n"
        "{\n"
        "    uint gid = (uint) get_global_id(0);\n"
        "    int lid = (int) get_local_id(0);\n"
        "    __local float localLocal[768];\n"
        "    float val0 = 0.0f;\n"
        "    float val1 = 0.0f;\n"
        "    uint hacklid = gid % 64;\n"
        "    for (int i = 0; i < (768/64); i++) {\n"
        "        localLocal[hacklid + i*64] = lid;\n"
        "    }\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "#pragma nounroll\n"
        "for (uint i = 0; i < 32;i++)\n"
        "{\n"
        "    val0 += localLocal[lid+0];\n"
        "    val1 += localLocal[lid+64];\n"
        "    val0 += localLocal[lid+128];\n"
        "    val1 += localLocal[lid+192];\n"
        "    val0 += localLocal[lid+256];\n"
        "    val1 += localLocal[lid+320];\n"
        "    val0 += localLocal[lid+384];\n"
        "    val1 += localLocal[lid+448];\n"
        "    lid += 1;\n"
        "}\n"
        "val0 += val1;\n"
        "val1 = min(val0,1.0f);\n"
        "if ((lid + val1) < 0){\n"
        "    outBuf[gid] = val0;\n"
        "}\n"
        "}\n";
    ldsSizeBytes_ = 768 * 4;
  } else {
    shader_ +=
        "__kernel __attribute__((reqd_work_group_size(64,1,1))) void "
        "_ldsReadSpeed(__global float *outBuf, float c)\n"
        "{\n"
        "    uint gid = (uint) get_global_id(0);\n"
        "    int lid = (int) get_local_id(0);\n"
        "    __local float localLocal[256];\n"
        "    float val0 = 0.0f;\n"
        "    float val1 = 0.0f;\n"
        "    uint hacklid = gid % 64;\n"
        "    for (int i = 0; i < (256/64); i++) {\n"
        "        localLocal[hacklid + i*64] = lid;\n"
        "    }\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "#pragma nounroll\n"
        "for (uint i = 0; i < 32;i++)\n"
        "{\n"
        "    val0 += localLocal[8*i+0];\n"
        "    val1 += localLocal[8*i+1];\n"
        "    val0 += localLocal[8*i+2];\n"
        "    val1 += localLocal[8*i+3];\n"
        "    val0 += localLocal[8*i+4];\n"
        "    val1 += localLocal[8*i+5];\n"
        "    val0 += localLocal[8*i+6];\n"
        "    val1 += localLocal[8*i+7];\n"
        "}\n"
        "val0 += val1;\n"
        "val1 = min(val0,1.0f);\n"
        "if ((lid + val1) < 0){\n"
        "    outBuf[gid] = val0;\n"
        "}\n"
        "}\n";
    ldsSizeBytes_ = 256 * 4;
  }
}

OCLPerfLDSReadSpeed::OCLPerfLDSReadSpeed() { _numSubTests = NUM_SIZES * 3; }

OCLPerfLDSReadSpeed::~OCLPerfLDSReadSpeed() {}

void OCLPerfLDSReadSpeed::setData(cl_mem buffer, float val) {
  float* data = (float*)_wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_WRITE, 0,
                                                     bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < (bufSize_ >> 2); i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_);
}

void OCLPerfLDSReadSpeed::checkData(cl_mem buffer) {
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
  _wrapper->clFinish(cmd_queue_);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfLDSReadSpeed::open(unsigned int test, char* units, double& conversion,
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

  numReads_ = 32;
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

  float foo = 0;
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_float), (void*)&foo);

  setData(outBuffer_, 1.2345678f);
}

void OCLPerfLDSReadSpeed::run(void) {
  int global = bufSize_ / sizeof(cl_float);
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

  char buf[256];
  const char* buf2;
  if (shaderIdx_ == 0) {
    buf2 = " def kernel";
  } else if (shaderIdx_ == 1) {
    buf2 = "SI friendly";
    numReads_ *= 8;
  } else {
    buf2 = "  broadcast";
    numReads_ *= 8;
  }
  // LDS bandwidth in GB/s
  // We have one extra write per LDS location to initialize LDS
  double perf = ((double)global * (numReads_ * sizeof(cl_float) + ldsSizeBytes_ / 64) * NUM_ITER *
                 (double)(1e-09)) /
                sec;

  _perfInfo = (float)perf;
  SNPRINTF(buf, sizeof(buf), " %s %8d threads, %3d reads (GB/s) ", buf2, global, numReads_);
  testDescString = buf;
  // checkData(outBuffer_);
}

unsigned int OCLPerfLDSReadSpeed::close(void) {
  _wrapper->clFinish(cmd_queue_);

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
