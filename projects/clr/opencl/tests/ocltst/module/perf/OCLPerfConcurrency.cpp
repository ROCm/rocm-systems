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

#include "OCLPerfConcurrency.h"

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

OCLPerfConcurrency::OCLPerfConcurrency() { _numSubTests = 10 * numCoords; }

OCLPerfConcurrency::~OCLPerfConcurrency() {}

void OCLPerfConcurrency::setData(cl_mem buffer, unsigned int val) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_[0], buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < width_; i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_[0], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_[0]);
}

void OCLPerfConcurrency::checkData(cl_mem buffer) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_[0], buffer, true, CL_MAP_READ, 0, bufSize_, 0, NULL, NULL, &error_);
  totalIters = 0;
  for (unsigned int i = 0; i < width_; i++) {
    totalIters += data[i];
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_[0], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_[0]);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfConcurrency::open(unsigned int test, char* units, double& conversion,
                              unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  unsigned int i;

  if (type_ != CL_DEVICE_TYPE_GPU) {
    char msg[256];
    SNPRINTF(msg, sizeof(msg), "No GPU devices present. Exiting!\t");
    testDescString = msg;
    return;
  }

  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;

  context_ = 0;

  for (i = 0; i < MAX_ASYNC_QUEUES; i++) {
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
#if 0
        // Get last for default
        platform = platforms[numPlatforms-1];
        for (i = 0; i < numPlatforms; ++i) {
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

  cl_uint numAsyncQueues;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD,
                                     sizeof(numAsyncQueues), &numAsyncQueues, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  CHECK_RESULT(numAsyncQueues > MAX_ASYNC_QUEUES, "numAsyncQueues is too large for this test");

  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &numCUs,
                                     &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  switch (_openTest) {
    case 0:
      num_cmd_queues = num_programs = num_kernels = num_outbuffers = 1;
      break;

    case 1:
      num_cmd_queues = 1;
      num_programs = 1;
      num_kernels = 1;
      num_outbuffers = 2;
      break;

    case 2:
      num_cmd_queues = 1;
      num_programs = 2;
      num_kernels = 2;
      num_outbuffers = 2;
      break;

    case 3:
      num_cmd_queues = num_programs = num_kernels = num_outbuffers = 2;
      break;

    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
      num_cmd_queues = num_programs = num_kernels = num_outbuffers = numAsyncQueues % 8;
      break;

    default:
      break;
  }

  for (i = 0; i < num_cmd_queues; i++) {
    cmd_queue_[i] = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
    CHECK_RESULT(cmd_queue_[i] == 0, "clCreateCommandQueue failed");
  }

  for (i = 0; i < num_outbuffers; i++) {
    outBuffer_[i] = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
    CHECK_RESULT(outBuffer_[i] == 0, "clCreateBuffer(outBuffer) failed");
  }

  const char* tmp;
  tmp = float_mandel_vec;

  for (i = 0; i < num_programs; i++) {
    program_[i] =
        _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
    CHECK_RESULT(program_[i] == 0, "clCreateProgramWithSource failed");

    error_ = _wrapper->clBuildProgram(program_[i], 1, &device, "", NULL, NULL);

    if (error_ != CL_SUCCESS) {
      cl_int intError;
      char log[16384];
      intError = _wrapper->clGetProgramBuildInfo(program_[i], device, CL_PROGRAM_BUILD_LOG,
                                                 16384 * sizeof(char), log, NULL);
      printf("Build error -> %s\n", log);

      CHECK_RESULT(0, "clBuildProgram failed");
    }
  }

  for (i = 0; i < num_kernels; i++) {
    kernel_[i] = _wrapper->clCreateKernel(program_[i], "mandelbrot", &error_);
    CHECK_RESULT(kernel_[i] == 0, "clCreateKernel failed");
  }

  coordIdx = _openTest % numCoords;
  float xStep = (float)(coords[coordIdx].width / (double)width_);
  float yStep = (float)(-coords[coordIdx].width / (double)width_);
  float xPos = (float)(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
  float yPos = (float)(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

  for (i = 0; i < num_kernels; i++) {
    error_ = _wrapper->clSetKernelArg(kernel_[i], 0, sizeof(cl_mem), (void*)&outBuffer_[i]);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 1, sizeof(cl_uint), (void*)&width_);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 2, sizeof(cl_float), (void*)&xPos);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 3, sizeof(cl_float), (void*)&yPos);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 4, sizeof(cl_float), (void*)&xStep);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 5, sizeof(cl_float), (void*)&yStep);
    error_ = _wrapper->clSetKernelArg(kernel_[i], 6, sizeof(cl_uint), (void*)&maxIter);
  }

  for (i = 0; i < num_outbuffers; i++) {
    setData(outBuffer_[i], 0xdeadbeef);
  }

  unsigned int clkFrequency = 0;
  error_ = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clkFrequency),
                           &clkFrequency, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  assert(clkFrequency > 0);
  maxIter = (unsigned int)(((8388608 * ((float)clkFrequency / 1000)) * numCUs) / 128);
  maxIter = (maxIter + 15) & ~15;
}

void OCLPerfConcurrency::run(void) {
  // Test runs only on GPU
  if (type_ != CL_DEVICE_TYPE_GPU) return;

  int global = width_ >> 2;
  // We handle 4 pixels per thread
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};
  unsigned int i;

  // Warmup
  for (i = 0; i < num_kernels; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_[i % num_cmd_queues], kernel_[i], 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }

  for (i = 0; i < num_cmd_queues; i++) {
    _wrapper->clFlush(cmd_queue_[i]);
  }

  for (i = 0; i < num_cmd_queues; i++) {
    _wrapper->clFinish(cmd_queue_[i]);
  }

  for (i = 0; i < num_kernels; i++) {
    error_ = _wrapper->clSetKernelArg(kernel_[i], 6, sizeof(cl_uint), (void*)&maxIter);
  }

  CPerfCounter timer;

  timer.Reset();
  timer.Start();

  for (i = 0; i < num_kernels; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_[i % num_cmd_queues], kernel_[i], 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }

  if (_openTest == 1) {
    error_ = _wrapper->clSetKernelArg(kernel_[0], 0, sizeof(cl_mem), (void*)&outBuffer_[1]);
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_[0], kernel_[0], 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }

  for (i = 0; i < num_cmd_queues; i++) {
    _wrapper->clFlush(cmd_queue_[i]);
  }

  for (i = 0; i < num_cmd_queues; i++) {
    _wrapper->clFinish(cmd_queue_[i]);
  }

  timer.Stop();
  double sec = timer.GetElapsedTime();

  unsigned long long expected = (unsigned long long)width_ * (unsigned long long)maxIter;

  for (i = 0; i < num_outbuffers; i++) {
    checkData(outBuffer_[i]);
    CHECK_RESULT(totalIters != expected, "Incorrect iteration count detected!");
  }

  _perfInfo = (float)sec;
  if (_openTest == 0)
    testDescString = "time for 1 kernel  (s)               ";
  else if (_openTest == 1)
    testDescString = "time for 2 kernels (s) (same kernel) ";
  else if (_openTest == 2)
    testDescString = "time for 2 kernels (s) (diff kernels)";
  else {
    char buf[128];
    SNPRINTF(buf, sizeof(buf), "time for %d kernels (s) (   %d queues) ", num_kernels,
             num_cmd_queues);
    testDescString = buf;
  }
}

unsigned int OCLPerfConcurrency::close(void) {
  unsigned int i;

  // Test runs only on GPU
  if (type_ != CL_DEVICE_TYPE_GPU) return 0;

  _wrapper->clFinish(cmd_queue_[0]);

  for (i = 0; i < num_outbuffers; i++) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }

  for (i = 0; i < num_kernels; i++) {
    error_ = _wrapper->clReleaseKernel(kernel_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel(kernel_) failed");
  }

  for (i = 0; i < num_programs; i++) {
    error_ = _wrapper->clReleaseProgram(program_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram(program_) failed");
  }

  for (i = 0; i < num_cmd_queues; i++) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_[i]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }

  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  return _crcword;
}
