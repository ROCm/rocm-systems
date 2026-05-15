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

#include "OCLPerfDeviceConcurrency.h"

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

typedef struct {
  double x;
  double y;
  double width;
} coordRec;

static coordRec coords[] = {
    {0.0, 0.0, 0.00001},  // All black
};

static unsigned int numCoords = sizeof(coords) / sizeof(coordRec);

static const char* float_mandel_vec =
    "__kernel void mandelbrot(__global uint *out, uint width, float xPos, "
    "float yPos, float xStep, float yStep, uint maxIter)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    int i = tid % (width/4);\n"
    "    int j = tid / (width/4);\n"
    "    int4 veci = (int4)(4*i, 4*i+1, 4*i+2, 4*i+3);\n"
    "    int4 vecj = (int4)(j, j, j, j);\n"
    "    float4 x0;\n"
    "    x0.s0 = (float)(xPos + xStep*veci.s0);\n"
    "    x0.s1 = (float)(xPos + xStep*veci.s1);\n"
    "    x0.s2 = (float)(xPos + xStep*veci.s2);\n"
    "    x0.s3 = (float)(xPos + xStep*veci.s3);\n"
    "    float4 y0;\n"
    "    y0.s0 = (float)(yPos + yStep*vecj.s0);\n"
    "    y0.s1 = (float)(yPos + yStep*vecj.s1);\n"
    "    y0.s2 = (float)(yPos + yStep*vecj.s2);\n"
    "    y0.s3 = (float)(yPos + yStep*vecj.s3);\n"
    "\n"
    "    float4 x = x0;\n"
    "    float4 y = y0;\n"
    "\n"
    "    uint iter = 0;\n"
    "    float4 tmp;\n"
    "    int4 stay;\n"
    "    int4 ccount = 0;\n"
    "    float4 savx = x;\n"
    "    float4 savy = y;\n"
    "    stay = (x*x+y*y) <= (float4)(4.0f, 4.0f, 4.0f, 4.0f);\n"
    "    for (iter = 0; (stay.s0 | stay.s1 | stay.s2 | stay.s3) && (iter < "
    "maxIter); iter+=16)\n"
    "    {\n"
    "        x = savx;\n"
    "        y = savy;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = x*x + x0 - y*y;\n"
    "        y = 2.0f * x * y + y0;\n"
    "        x = tmp*tmp + x0 - y*y;\n"
    "        y = 2.0f * tmp * y + y0;\n"
    "\n"
    "        stay = (x*x+y*y) <= (float4)(4.0f, 4.0f, 4.0f, 4.0f);\n"
    "        savx = (stay ? x : savx);\n"
    "        savy = (stay ? y : savy);\n"
    "        ccount -= stay*16;\n"
    "    }\n"
    "    // Handle remainder\n"
    "    if (!(stay.s0 & stay.s1 & stay.s2 & stay.s3))\n"
    "    {\n"
    "        iter = 16;\n"
    "        do\n"
    "        {\n"
    "            x = savx;\n"
    "            y = savy;\n"
    "            // More efficient to use scalar ops here: Why?\n"
    "            stay.s0 = ((x.s0*x.s0+y.s0*y.s0) <= 4.0f) && (ccount.s0 < "
    "maxIter);\n"
    "            stay.s1 = ((x.s1*x.s1+y.s1*y.s1) <= 4.0f) && (ccount.s1 < "
    "maxIter);\n"
    "            stay.s2 = ((x.s2*x.s2+y.s2*y.s2) <= 4.0f) && (ccount.s2 < "
    "maxIter);\n"
    "            stay.s3 = ((x.s3*x.s3+y.s3*y.s3) <= 4.0f) && (ccount.s3 < "
    "maxIter);\n"
    "            tmp = x;\n"
    "            x = x*x + x0 - y*y;\n"
    "            y = 2.0f*tmp*y + y0;\n"
    "            ccount += stay;\n"
    "            iter--;\n"
    "            savx.s0 = (stay.s0 ? x.s0 : savx.s0);\n"
    "            savx.s1 = (stay.s1 ? x.s1 : savx.s1);\n"
    "            savx.s2 = (stay.s2 ? x.s2 : savx.s2);\n"
    "            savx.s3 = (stay.s3 ? x.s3 : savx.s3);\n"
    "            savy.s0 = (stay.s0 ? y.s0 : savy.s0);\n"
    "            savy.s1 = (stay.s1 ? y.s1 : savy.s1);\n"
    "            savy.s2 = (stay.s2 ? y.s2 : savy.s2);\n"
    "            savy.s3 = (stay.s3 ? y.s3 : savy.s3);\n"
    "        } while ((stay.s0 | stay.s1 | stay.s2 | stay.s3) && iter);\n"
    "    }\n"
    "    __global uint4 *vecOut = (__global uint4 *)out;\n"
    "    vecOut[tid] = convert_uint4(ccount);\n"
    "}\n";

OCLPerfDeviceConcurrency::OCLPerfDeviceConcurrency() {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
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
    if (num_devices > MAX_DEVICES) {
      num_devices = MAX_DEVICES;
    }
    delete platforms;
  }
  _numSubTests = num_devices;
}

OCLPerfDeviceConcurrency::~OCLPerfDeviceConcurrency() {}

void OCLPerfDeviceConcurrency::setData(cl_mem buffer, unsigned int idx, unsigned int val) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_[idx], buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < width_; i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_[idx], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_[idx]);
}

void OCLPerfDeviceConcurrency::checkData(cl_mem buffer, unsigned int idx) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_[idx], buffer, true, CL_MAP_READ, 0, bufSize_, 0, NULL, NULL, &error_);
  totalIters = 0;
  for (unsigned int i = 0; i < width_; i++) {
    totalIters += data[i];
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_[idx], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_[idx]);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfDeviceConcurrency::open(unsigned int test, char* units, double& conversion,
                                    unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  num_devices = 0;
  cl_device_id* devices = NULL;
  unsigned int i;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;

  context_ = 0;

  for (i = 0; i < MAX_DEVICES; i++) {
    cmd_queue_[i] = 0;
    program_[i] = 0;
    kernel_[i] = 0;
    outBuffer_[i] = 0;
  }

  // Maximum iteration count
  // NOTE: Some kernels are unrolled 16 times, so make sure maxIter is divisible
  // by 16 NOTE: Can increase to get better peak performance numbers, but be
  // sure not to TDR slow ASICs! NOTE:. for warmup run we use maxIter = 256 and
  // then for the actual run we use maxIter = 8388608 * (engine_clock / 1000).
  maxIter = 256;

  // NOTE: Width needs to be divisible by 4 because the float_mandel_vec kernel
  // processes 4 pixels at once NOTE: Can increase to get better peak
  // performance numbers, but be sure not to TDR slow ASICs!
  width_ = 256;

  // We compute a square domain
  bufSize_ = width_ * sizeof(cl_uint);

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
    if (num_devices > MAX_DEVICES) {
      num_devices = MAX_DEVICES;
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

  /* Get the requested devices */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  context_ = _wrapper->clCreateContext(NULL, num_devices, devices, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cur_devices = _openTest + 1;

  for (i = 0; i < cur_devices; i++) {
    cmd_queue_[i] = _wrapper->clCreateCommandQueue(context_, devices[i], 0, NULL);
    CHECK_RESULT(cmd_queue_[i] == 0, "clCreateCommandQueue failed");
    outBuffer_[i] = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
    CHECK_RESULT(outBuffer_[i] == 0, "clCreateBuffer(outBuffer) failed");
  }

  const char* tmp;
  tmp = float_mandel_vec;

  for (i = 0; i < cur_devices; i++) {
    program_[i] =
        _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
    CHECK_RESULT(program_[i] == 0, "clCreateProgramWithSource failed");

    error_ = _wrapper->clBuildProgram(program_[i], 1, &devices[i], "", NULL, NULL);

    if (error_ != CL_SUCCESS) {
      cl_int intError;
      char log[16384];
      intError = _wrapper->clGetProgramBuildInfo(program_[i], devices[i], CL_PROGRAM_BUILD_LOG,
                                                 16384 * sizeof(char), log, NULL);
      printf("Build error on device %d -> %s\n", i, log);

      CHECK_RESULT(0, "clBuildProgram failed");
    }
  }

  for (i = 0; i < cur_devices; i++) {
    kernel_[i] = _wrapper->clCreateKernel(program_[i], "mandelbrot", &error_);
    CHECK_RESULT(kernel_[i] == 0, "clCreateKernel failed");
  }

  coordIdx = _openTest % numCoords;
  float xStep = (float)(coords[coordIdx].width / (double)width_);
  float yStep = (float)(-coords[coordIdx].width / (double)width_);
  float xPos = (float)(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
  float yPos = (float)(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clSetKernelArg(kernel_[i], 0, sizeof(cl_mem), (void*)&outBuffer_[i]);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 1, sizeof(cl_uint), (void*)&width_);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 2, sizeof(cl_float), (void*)&xPos);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 3, sizeof(cl_float), (void*)&yPos);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 4, sizeof(cl_float), (void*)&xStep);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 5, sizeof(cl_float), (void*)&yStep);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 6, sizeof(cl_uint), (void*)&maxIter);
  }

  for (i = 0; i < cur_devices; i++) {
    setData(outBuffer_[i], i, 0xdeadbeef);
  }

  cl_uint clkFrequency = 0;
  error_ = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clkFrequency),
                           &clkFrequency, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  assert(clkFrequency > 0);
  maxIter = (unsigned int)(8388608 * ((float)clkFrequency / 1000));
  maxIter = (maxIter + 15) & ~15;
}

void OCLPerfDeviceConcurrency::run(void) {
  int global = width_ >> 2;
  // We handle 4 pixels per thread
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};
  unsigned int i;

  // Warmup
  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_[i], kernel_[i], 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }

  for (i = 0; i < cur_devices; i++) {
    _wrapper->clFlush(cmd_queue_[i]);
  }

  for (i = 0; i < cur_devices; i++) {
    _wrapper->clFinish(cmd_queue_[i]);
  }

  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clSetKernelArg(kernel_[i], 6, sizeof(cl_uint), (void*)&maxIter);
  }

  CPerfCounter timer;

  timer.Reset();
  timer.Start();

  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_[i], kernel_[i], 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }

  for (i = 0; i < cur_devices; i++) {
    _wrapper->clFlush(cmd_queue_[i]);
  }

  for (i = 0; i < cur_devices; i++) {
    _wrapper->clFinish(cmd_queue_[i]);
  }

  timer.Stop();
  double sec = timer.GetElapsedTime();

  unsigned long long expected = (unsigned long long)width_ * (unsigned long long)maxIter;

  for (i = 0; i < cur_devices; i++) {
    checkData(outBuffer_[i], i);
    CHECK_RESULT(totalIters != expected, "Incorrect iteration count detected!");
  }

  _perfInfo = (float)sec;
  char buf[128];
  SNPRINTF(buf, sizeof(buf), "time for %2d devices (s) (%2d queues) ", cur_devices, cur_devices);
  testDescString = buf;
}

unsigned int OCLPerfDeviceConcurrency::close(void) {
  unsigned int i;

  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }

  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clReleaseKernel(kernel_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel(kernel_) failed");
  }

  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clReleaseProgram(program_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram(program_) failed");
  }

  for (i = 0; i < cur_devices; i++) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }

  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  return _crcword;
}
