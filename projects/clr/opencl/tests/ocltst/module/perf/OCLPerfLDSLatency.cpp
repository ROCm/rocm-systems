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

#include "OCLPerfLDSLatency.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

static const unsigned int NUM_SIZES = 5;
// 2k up to 64MB
static const unsigned int Sizes[NUM_SIZES] = {2048, 4096, 8192, 16384, 32768};

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif
void OCLPerfLDSLatency::genShader() {
  shader_.clear();

  // DO NOT PUBLISH
  // Adopted from SiSoft Sandra 2013's memory latency test
  shader_ +=
      "__kernel\n"
      //"__attribute__((work_group_size_hint(1, 1, 1)))\n"
      "void MemWalker(\n"
      "    global uint * restrict input,\n"
      "    global uint * restrict output,\n"
      "    const uint uCount,  const uint uSize,\n"
      "    const uint uOffset, const int bMem, const uint repeats)\n"
      "{\n"
      "    uint o = uOffset;\n"
      "    uint lid = get_local_id(0);\n"
      "    uint x = lid*o;\n"
      "    local uint lclData[8192];\n"
      "\n"
      "    {\n"
      "        uint i = uCount;\n"
      "        while (i--) {\n"
      "            uint oldX = x;\n"
      "            x = input[x];\n"
      "            lclData[oldX] = x;\n"
      "        }\n"
      "    }\n"
      "\n"
      "    x = lid*uOffset;\n"
      "    for (uint loop = 0; loop < repeats; loop++) {\n"
      "        uint i = uCount;\n"
      "        while (i--) {\n"
      "            x = lclData[x] + o;\n"
      "        }\n"
      "    }\n"
      "\n"
      "    output[0] = x;\n"
      "}\n";

  // printf("shader:\n%s\n", shader_.c_str());
  shader_ += "\n\n";
  shader_ +=
      "__kernel\n"
      //"__attribute__((work_group_size_hint(1, 1, 1)))\n"
      "void Overhead(\n"
      "    global uint * restrict input,\n"
      "    global uint * restrict output,\n"
      "    const uint uCount,  const uint uSize,\n"
      "    const uint uOffset, const int bMem, const uint repeats)\n"
      "{\n"
      "    local uint lclData[8192];\n"
      "#ifdef USE_FLOAT\n"
      "    {\n"
      "        uint x = 0;\n"
      "        uint i = uCount;\n"
      "        while (i--) {\n"
      "            uint oldX = x;\n"
      "            x = input[x] /* + o*/;\n"
      "            lclData[oldX] = x;\n"
      "        }\n"
      "    }\n"
      "    float x = (float)input[0];\n"
      "    for (uint loop = 0; loop < repeats; loop++) {\n"
      "        uint i = uCount;\n"
      "        x = (float)uOffset*x;\n"
      "        while (i--) {\n"
      "            x += (float)i;\n"
      "        }\n"
      "    }\n"
      "    output[0] = (uint)x + uOffset*lclData[8191];\n"
      "#else\n"
      "    {\n"
      "        uint x = 0;\n"
      "        uint i = uCount;\n"
      "        while (i--) {\n"
      "            uint oldX = x;\n"
      "            x = input[x] /* + o*/;\n"
      "            lclData[oldX] = x;\n"
      "        }\n"
      "    }\n"
      "    uint x = input[0];\n"
      "    for (uint loop = 0; loop < repeats; loop++) {\n"
      "        uint i = uCount;\n"
      "        x = x*uOffset;\n"
      "        while (i--) {\n"
      "            x += i;\n"
      "        }\n"
      "    }\n"
      "    output[0] = x + uOffset*lclData[8191];\n"
      "#endif\n"
      "}\n";
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

OCLPerfLDSLatency::OCLPerfLDSLatency() {
  _numSubTests = NUM_SIZES * 2;
  maxSize_ = Sizes[NUM_SIZES - 1] * 2048;
}

OCLPerfLDSLatency::~OCLPerfLDSLatency() {}

void OCLPerfLDSLatency::setData(cl_mem buffer, unsigned int val) {
  void* ptr = _wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_WRITE, 0, width_, 0,
                                           NULL, NULL, &error_);
  unsigned int* data = (unsigned int*)ptr;
  for (unsigned int i = 0; i < bufSizeDW_; i++) {
    data[(i * (1024 + 17)) % bufSizeDW_] = ((i + 1) * (1024 + 17)) % bufSizeDW_;
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, ptr, 0, NULL, NULL);
  clFinish(cmd_queue_);
}

void OCLPerfLDSLatency::checkData(cl_mem buffer) {
  void* ptr = _wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_READ, 0,
                                           sizeof(cl_uint), 0, NULL, NULL, &error_);

  unsigned int* data = (unsigned int*)ptr;
  if (data[0] != 0) {
    printf("OutData= 0x%08x\n", data[0]);
    CHECK_RESULT_NO_RETURN(data[0] != 0, "Data validation failed!\n");
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, ptr, 0, NULL, NULL);
}

void OCLPerfLDSLatency::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  moreThreads = false;

  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  inBuffer_ = 0;
  outBuffer_ = 0;
  _errorFlag = false;  // Reset error code so a single error doesn't prevent
                       // other subtests from running
  _errorMsg = "";
  isAMD_ = false;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");

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
      if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
        isAMD_ = true;
      }
    }

    delete platforms;
  }

  width_ = Sizes[test % NUM_SIZES];

  bufSizeDW_ = width_ / sizeof(cl_uint);
  moreThreads = ((test / NUM_SIZES) % 2) ? true : false;

  CHECK_RESULT(platform == 0, "Couldn't find OpenCL platform, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "Failed to allocate devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  device = devices[0];

  free(devices);
  devices = NULL;
  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  cl_uint flags;
  flags = 0;
  inBuffer_ = _wrapper->clCreateBuffer(context_, flags, width_, NULL, &error_);
  CHECK_RESULT(inBuffer_ == 0, "clCreateBuffer(inBuffer) failed");

  outBuffer_ = _wrapper->clCreateBuffer(context_, 0, 1 * sizeof(cl_uint), NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  genShader();
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  std::string args;
  args.clear();
  if (isAMD_) args += " -D USE_FLOAT";

  error_ = _wrapper->clBuildProgram(program_, 1, &device, args.c_str(), NULL, NULL);
  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "MemWalker", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel(MemWalker) failed");

  kernel2_ = _wrapper->clCreateKernel(program_, "Overhead", &error_);
  CHECK_RESULT(kernel2_ == 0, "clCreateKernel(Overhead) failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&inBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint), (void*)&bufSizeDW_);
  error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_uint), (void*)&bufSizeDW_);
  unsigned int zero = 0;
  error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_uint), (void*)&zero);
  int bMem = 1;
  error_ = _wrapper->clSetKernelArg(kernel_, 5, sizeof(cl_int), (void*)&bMem);
  // Limit the repeats, large buffers will have more samples, but the test runs
  // for a long time
  repeats_ = std::max((maxSize_ >> 4) / bufSizeDW_, 1u);
  error_ = _wrapper->clSetKernelArg(kernel_, 6, sizeof(cl_uint), (void*)&repeats_);

  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), (void*)&inBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 1, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 2, sizeof(cl_uint), (void*)&bufSizeDW_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 3, sizeof(cl_uint), (void*)&bufSizeDW_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 4, sizeof(cl_uint), (void*)&zero);
  error_ = _wrapper->clSetKernelArg(kernel2_, 5, sizeof(cl_int), (void*)&bMem);
  error_ = _wrapper->clSetKernelArg(kernel2_, 6, sizeof(cl_uint), (void*)&repeats_);

  setData(inBuffer_, (int)1.0f);
}

void OCLPerfLDSLatency::run(void) {
  int global = 1;
  int local = 1;

  if (moreThreads) {
    if (isAMD_) {
      global *= 64;
      local *= 64;
    } else {
      global *= 32;
      local *= 32;
    }
  }
  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  // Warm-up
  unsigned int warmup = 128;
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint), (void*)&warmup);
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint), (void*)&bufSizeDW_);
  _wrapper->clFinish(cmd_queue_);

  // Restore input buffer when finished as it may have been modified by RW test
  setData(inBuffer_, (int)1.0f);

  CPerfCounter timer, timer2;

  timer.Reset();
  timer.Start();

  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");

  _wrapper->clFinish(cmd_queue_);

  timer.Stop();

  checkData(outBuffer_);

  timer2.Reset();
  timer2.Start();

  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel2_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");

  _wrapper->clFinish(cmd_queue_);

  timer2.Stop();
  double sec = timer.GetElapsedTime() - timer2.GetElapsedTime();

  // Read latency in ns
  double perf = sec * (double)(1e09) / ((double)bufSizeDW_ * (double)repeats_);

  _perfInfo = (float)perf;
  char buf[256];
  char buf2[32];
  buf2[0] = '\0';
  SNPRINTF(buf, sizeof(buf), "%10s %2d threads, %8d reads, %5d repeats (ns)", buf2, global,
           bufSizeDW_, repeats_);
  testDescString = buf;
}

unsigned int OCLPerfLDSLatency::close(void) {
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
  if (kernel2_) {
    error_ = _wrapper->clReleaseKernel(kernel2_);
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
