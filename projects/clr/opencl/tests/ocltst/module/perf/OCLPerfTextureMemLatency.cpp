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

#include "OCLPerfTextureMemLatency.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

static const unsigned int NUM_SIZES = 13;
// 2k up to 64MB
static const cl_uint2 Dims[NUM_SIZES] = {{{32, 16}},    {{32, 32}},   {{64, 32}},    {{64, 64}},
                                         {{128, 64}},   {{128, 128}}, {{256, 128}},  {{256, 256}},
                                         {{512, 256}},  {{512, 512}}, {{1024, 512}}, {{1024, 1024}},
                                         {{2048, 1024}}};
// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif
void OCLPerfTextureMemLatency::genShader() {
  shader_.clear();

  // Adopted from SiSoft Sandra 2013's memory latency test
  shader_ +=
      "constant sampler_t insample = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
      "__kernel\n"
      "__attribute__((work_group_size_hint(1, 1, 1)))\n"
      "void MemWalker(\n"
      "    read_only image2d_t input,\n"
      "    __global uint * restrict output,\n"
      "    const uint uCount,  const uint uSize,\n"
      "    const uint4 uOffset, const int bMem, const uint repeats)\n"
      "{\n"
      "    uint4 o = uOffset;\n"
      "    uint lid = get_local_id(0);\n"
      "    uint4 x = lid*o;\n"
      "\n"
      "    for (uint loop = 0; (loop < repeats); loop++) {\n"
      "        uint i = uCount;\n"
      "        int2 nx = (int2)(0,0);\n"
      "        nx = (int2)((x.y << 8) | x.x, (x.w << 8) | x.z);\n"
      "        while (i--) {\n"
      "            x = read_imageui(input, insample, nx);\n"
      "            x.x += o.x;\n"
      "            x.z += o.z;\n"
      "            nx = (int2)((x.y << 8) | x.x, (x.w << 8) | x.z);\n"
      "        }\n"
      "    }\n"
      "\n"
      "    output[0] = x.x + x.y;\n"
      "}\n";

  // printf("shader:\n%s\n", shader_.c_str());
  shader_ += "\n\n";
  shader_ +=
      "__kernel\n"
      "__attribute__((work_group_size_hint(1, 1, 1)))\n"
      "void Overhead(\n"
      "    read_only image2d_t input,\n"
      "    __global uint * restrict output,\n"
      "    const uint uCount,  const uint uSize,\n"
      "    const uint4 uOffset, const int bMem, const uint repeats)\n"
      "{\n"
      "    uint4 o = uOffset;\n"
      "    uint lid = get_local_id(0);\n"
      "    uint4 x = lid*o;\n"
      "    x += o;\n"
      "    int2 nx;\n"
      "    for (uint loop = 0; loop < repeats; loop++) {\n"
      "        uint i = uCount;\n"
      "        nx = (int2)(0,0);\n"
      "        nx = (int2)((x.y << 8) | x.x, (x.w << 8) | x.z);\n"
      "        while (i--) {\n"
      "            x.x = nx.x  + o.x;\n"
      "            x.z = nx.y  + o.y;\n"
      "            nx = (int2)((x.y << 8) | x.x, (x.w << 8) | x.z);\n"
      "        }\n"
      "    }\n"
      "    output[0] = nx.x | nx.y;\n"
      "}\n";
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

OCLPerfTextureMemLatency::OCLPerfTextureMemLatency() {
  _numSubTests = NUM_SIZES;
  maxSize_ = Dims[NUM_SIZES - 1].s[0] * Dims[NUM_SIZES - 1].s[1];
}

OCLPerfTextureMemLatency::~OCLPerfTextureMemLatency() {}

void OCLPerfTextureMemLatency::setData(cl_mem buffer, unsigned int val) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width_, height_, 1};

  void* ptr =
      _wrapper->clEnqueueMapImage(cmd_queue_, buffer, true, CL_MAP_WRITE, origin, region,
                                  &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &error_);
  CHECK_RESULT(error_, "clEnqueueMapImage failed.");
  unsigned int* data = (unsigned int*)ptr;
  unsigned int nextOffset = 0;
  for (unsigned int i = 0; i < bufSizeDW_; i++) {
    unsigned int offset = ((1024 + 17) * (i + 1)) % bufSizeDW_;
    unsigned int x, y;
    x = offset % width_;
    y = offset / width_;
    unsigned int newx, newy;
    newx = nextOffset % width_;
    newy = nextOffset / width_;
    data[newy * image_row_pitch / sizeof(unsigned int) + newx] = (y << 16) | (x & 0xffff);
    nextOffset = offset;
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, ptr, 0, NULL, NULL);
  clFinish(cmd_queue_);
}

void OCLPerfTextureMemLatency::checkData(cl_mem buffer) {
  void* ptr = _wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, true, CL_MAP_READ, 0,
                                           sizeof(cl_uint), 0, NULL, NULL, &error_);

  unsigned int* data = (unsigned int*)ptr;
  if (data[0] != 0) {
    printf("OutData= 0x%08x\n", data[0]);
    CHECK_RESULT_NO_RETURN(data[0] != 0, "Data validation failed!\n");
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, ptr, 0, NULL, NULL);
  clFinish(cmd_queue_);
}

void OCLPerfTextureMemLatency::open(unsigned int test, char* units, double& conversion,
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
  inBuffer_ = 0;
  outBuffer_ = 0;
  _errorFlag = false;  // Reset error code so a single error doesn't prevent
                       // other subtests from running
  _errorMsg = "";

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
      }
    }

    delete platforms;
  }

  width_ = Dims[test % NUM_SIZES].s[0];
  height_ = Dims[test % NUM_SIZES].s[1];

  bufSizeDW_ = width_ * height_;

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

  device = devices[0];

  free(devices);

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};
  inBuffer_ = _wrapper->clCreateImage2D(context_, CL_MEM_READ_ONLY, &format, width_, height_, 0,
                                        NULL, &error_);
  CHECK_RESULT(inBuffer_ == 0, "clCreateImage(inBuffer) failed");

  outBuffer_ = _wrapper->clCreateBuffer(context_, 0, sizeof(cl_uint), NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  genShader();
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  std::string args;
  args.clear();

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
  CHECK_RESULT(kernel_ == 0, "clCreateKernel(Overhead) failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&inBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint), (void*)&bufSizeDW_);
  error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_uint), (void*)&bufSizeDW_);
  cl_uint4 zero;
  zero.s[0] = 0;
  zero.s[1] = 0;
  zero.s[2] = 0;
  zero.s[3] = 0;
  error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_uint4), (void*)&zero);
  int bMem = 1;
  error_ = _wrapper->clSetKernelArg(kernel_, 5, sizeof(cl_int), (void*)&bMem);
  repeats_ = std::max((maxSize_ >> 2) / bufSizeDW_, 1u);
  error_ = _wrapper->clSetKernelArg(kernel_, 6, sizeof(cl_uint), (void*)&repeats_);

  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), (void*)&inBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 1, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 2, sizeof(cl_uint), (void*)&bufSizeDW_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 3, sizeof(cl_uint), (void*)&bufSizeDW_);
  error_ = _wrapper->clSetKernelArg(kernel2_, 4, sizeof(cl_uint4), (void*)&zero);
  error_ = _wrapper->clSetKernelArg(kernel2_, 5, sizeof(cl_int), (void*)&bMem);
  error_ = _wrapper->clSetKernelArg(kernel2_, 6, sizeof(cl_uint), (void*)&repeats_);

  setData(inBuffer_, (int)1.0f);
}

void OCLPerfTextureMemLatency::run(void) {
  int global = 1;
  int local = 1;

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
  SNPRINTF(buf, sizeof(buf), "%8d reads, %5d repeats (ns)", bufSizeDW_, repeats_);
  testDescString = buf;
}

unsigned int OCLPerfTextureMemLatency::close(void) {
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
