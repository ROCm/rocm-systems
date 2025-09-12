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

#include "OCLPerfProgramGlobalWrite.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

static const unsigned int NUM_SIZES = 4;
static const unsigned int NUM_READ_MODES = 6;
// Limit to 32 reads for now
static const unsigned int MAX_READ_MODES = 4;

static const unsigned int NumReads[NUM_READ_MODES] = {1, 4, 16, 32, 64, 128};
// 256KB, 1 MB, 4MB, 16 MB
static const unsigned int Sizes[NUM_SIZES] = {262144, 1048576, 4194304, 16777216};
static const unsigned int MaxTypes = 6;
static unsigned int NumTypes = MaxTypes;
static const char* types[MaxTypes] = {"char", "short", "int", "long", "float", "double"};
static unsigned int StartType = 0;
static const unsigned int NumVecWidths = 3;  // 5; char8 global scope does not work; bug opened
static const char* vecWidths[NumVecWidths] = {"", "2", "4"};        //, "8", "16"};
static const unsigned int vecWidths_int[NumVecWidths] = {1, 2, 4};  //, 8, 16};
static const unsigned int TypeSize[MaxTypes] = {sizeof(cl_char),  sizeof(cl_short),
                                                sizeof(cl_int),   sizeof(cl_long),
                                                sizeof(cl_float), sizeof(cl_double)};
#define CHAR_BUF_SIZE 512

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif
void OCLPerfProgramGlobalWrite::genShader(unsigned int type, unsigned int vecWidth,
                                          unsigned int numReads, unsigned int bufSize) {
  char buf[CHAR_BUF_SIZE];

  shader_.clear();
  shader_ +=
      "#ifdef USE_ARENA\n"
      "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n"
      "#endif\n";
  shader_ +=
      "#ifdef USE_AMD_DOUBLES\n"
      "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
      "#endif\n";
  shader_ +=
      "#ifdef USE_KHR_DOUBLES\n"
      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
      "#endif\n";
  SNPRINTF(buf, CHAR_BUF_SIZE, "__global %s%s gp[%d];\n", types[type], vecWidths[vecWidth],
           bufSize);
  shader_.append(buf);
  SNPRINTF(buf, CHAR_BUF_SIZE,
           "__kernel void __attribute__((reqd_work_group_size(64,1,1))) "
           "_WriteSpeed(constant uint * restrict constBuf)\n");
  shader_.append(buf);
  shader_ +=
      "{\n"
      "    uint i = (uint) get_global_id(0);\n";
  if (numReads == 1) {
    SNPRINTF(buf, CHAR_BUF_SIZE, "    %s%s temp = 0;\n", types[type], vecWidths[vecWidth]);
    shader_.append(buf);
    shader_ += "    const unsigned int Max = constBuf[0];\n";
    shader_ +=
        "    *(gp + i % Max) = 0;\n"
        "}\n";
  } else {
    SNPRINTF(buf, CHAR_BUF_SIZE, "    %s%s temp0 = 0;\n", types[type], vecWidths[vecWidth]);
    shader_.append(buf);
    SNPRINTF(buf, CHAR_BUF_SIZE, "    %s%s temp1 = 0;\n", types[type], vecWidths[vecWidth]);
    shader_.append(buf);
    SNPRINTF(buf, CHAR_BUF_SIZE, "    %s%s temp2 = 0;\n", types[type], vecWidths[vecWidth]);
    shader_.append(buf);
    SNPRINTF(buf, CHAR_BUF_SIZE, "    %s%s temp3 = 0;\n", types[type], vecWidths[vecWidth]);
    shader_.append(buf);
    shader_ +=
        "    const unsigned int Max = constBuf[0];\n"
        "    unsigned int idx0 = (i % Max) + constBuf[1];\n"
        "    unsigned int idx1 = (i % Max) + constBuf[2];\n"
        "    unsigned int idx2 = (i % Max) + constBuf[3];\n"
        "    unsigned int idx3 = (i % Max) + constBuf[4];\n";

    for (unsigned int i = 0; i < (numReads >> 2); i++) {
      shader_ += "    *(gp + idx0) = idx0;\n";
      shader_ += "    *(gp + idx1) = idx1;\n";
      shader_ += "    *(gp + idx2) = idx2;\n";
      shader_ += "    *(gp + idx3) = idx3;\n";
      shader_ += "    idx0 += constBuf[5];\n";
      shader_ += "    idx1 += constBuf[5];\n";
      shader_ += "    idx2 += constBuf[5];\n";
      shader_ += "    idx3 += constBuf[5];\n";
    }
    shader_ += "}\n";
  }
  SNPRINTF(buf, CHAR_BUF_SIZE, "__kernel void __dummyRead(global %s%s *in)\n", types[type],
           vecWidths[vecWidth]);
  shader_.append(buf);
  shader_ +=
      "{\n"
      "    uint i = (uint) get_global_id(0);\n";
  SNPRINTF(buf, CHAR_BUF_SIZE, "    in[i] = gp[i];\n");
  shader_.append(buf);
  shader_ += "}\n";
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

OCLPerfProgramGlobalWrite::OCLPerfProgramGlobalWrite() {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  context_ = 0;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
    // Get last for default
    platform = platforms[numPlatforms - 1];
    for (unsigned i = 0; i < numPlatforms; ++i) {
      char pbuf[100];
      error_ =
          _wrapper->clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
      num_devices = 0;
      /* Get the number of requested devices */
      error_ = _wrapper->clGetDeviceIDs(platforms[i], type_, 0, NULL, &num_devices);
      // Runtime returns an error when no GPU devices are present instead of
      // just returning 0 devices
      // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
      // Choose platform with GPU devices
      if (num_devices > 0) {
        platform = platforms[i];
        break;
      }
    }
    delete platforms;
  }

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

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  char* p = strstr(charbuf, "cl_khr_byte_addressable_store");
  char* p2 = strstr(charbuf, "cl_khr_fp64");

  NumTypes = MaxTypes;
  if (!p) {
    // No arena ops
    NumTypes -= 2;
    StartType = 2;
  }
  if (!p2) {
    // Doubles not supported
    NumTypes--;
  }
  _numSubTests = NumTypes * NumVecWidths * NUM_SIZES * MAX_READ_MODES;
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  skip_ = false;
}

OCLPerfProgramGlobalWrite::~OCLPerfProgramGlobalWrite() {}

void OCLPerfProgramGlobalWrite::open(unsigned int test, char* units, double& conversion,
                                     unsigned int deviceId) {
  error_ = CL_SUCCESS;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = 0;
  kernel_ = 0;
  cmd_queue_ = 0;
  outBuffer_ = 0;
  constBuffer_ = 0;

#if defined(CL_VERSION_2_0)
  cl_device_id device;
  numReads_ = NumReads[test % MAX_READ_MODES];
  width_ = Sizes[(test / MAX_READ_MODES) % NUM_SIZES];
  vecSizeIdx_ = (test / (MAX_READ_MODES * NUM_SIZES)) % NumVecWidths;
  typeIdx_ = (test / (MAX_READ_MODES * NUM_SIZES * NumVecWidths)) % NumTypes + StartType;

  bufSize_ = width_;

  cmd_queue_ = cmdQueues_[_deviceId];

  device = devices_[_deviceId];

  outBuffer_ = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  constBuffer_ = _wrapper->clCreateBuffer(context_, 0, 16 * 2, NULL, &error_);
  CHECK_RESULT(constBuffer_ == 0, "clCreateBuffer(constBuffer) failed");

  genShader(typeIdx_, vecSizeIdx_, numReads_, bufSize_ / (TypeSize[typeIdx_] * (1 << vecSizeIdx_)));
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  std::string args;
  args.clear();
  if (typeIdx_ < 2) {
    args += "-D USE_ARENA ";
  }
  args += "-cl-std=CL2.0";
  error_ = _wrapper->clBuildProgram(program_, 1, &device, args.c_str(), NULL, NULL);
  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "_WriteSpeed", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&constBuffer_);

  unsigned int* cBuf = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, constBuffer_, true, CL_MAP_WRITE, 0, 16 * 2, 0, NULL, NULL, &error_);
  // Force all wavefronts to fetch the same data.  We are looking for peak speed
  // here.
  cBuf[0] = 64;
  // These values are chosen to assure there is no data reuse within a clause.
  // If caching is not working, then the uncached numbers will be low.
  cBuf[1] = 0;
  cBuf[2] = 64;
  cBuf[3] = 128;
  cBuf[4] = 192;
  cBuf[5] = 0;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, constBuffer_, cBuf, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_);
#else
  skip_ = true;
  testDescString = "Program scope globals not supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

void OCLPerfProgramGlobalWrite::run(void) {
  if (skip_) {
    return;
  }
#if defined(CL_VERSION_2_0)
  int global = bufSize_ / (TypeSize[typeIdx_] * (1 << vecSizeIdx_));
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

  // Program scope global write bandwidth in GB/s
  double perf = ((double)bufSize_ * numReads_ * NUM_ITER * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  char buf[256];
  char buf2[256];
  SNPRINTF(buf, sizeof(buf), "%s%s", types[typeIdx_], vecWidths[vecSizeIdx_]);
  SNPRINTF(buf2, sizeof(buf2), " %-8s (%8d) %2d reads: (GB/s) ", buf, width_, numReads_);
  testDescString = buf2;
#endif
}

unsigned int OCLPerfProgramGlobalWrite::close(void) {
#if defined(CL_VERSION_2_0)
  if (cmd_queue_) _wrapper->clFinish(cmd_queue_);

  if (outBuffer_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }
  if (constBuffer_) {
    error_ = _wrapper->clReleaseMemObject(constBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(constBuffer_) failed");
  }
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
  }
#endif
  return OCLTestImp::close();
}
