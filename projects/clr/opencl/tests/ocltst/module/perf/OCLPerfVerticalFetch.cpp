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

#include "OCLPerfVerticalFetch.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <sstream>

#include "CL/opencl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_SIZES 1
#define WIDTH 4952
#define HEIGHT 3288
unsigned int Sizes[NUM_SIZES] = {WIDTH * HEIGHT * 4};

#define KERNEL_CODE(...) #__VA_ARGS__
const static char* strKernel = KERNEL_CODE(
\n __kernel void ResizeVerticalFilter(
    const __global uint* inputImage, const unsigned int inputColumns, const unsigned int inputRows,
    __local uint* inputImageCache, const int numCachedPixels, __global uint* dst) {
  const unsigned int startY = get_group_id(1) * get_local_size(1);
  float scale = 0.5f;
  const float support = 0.5f;
  const int cacheRangeStartY = max((int)((startY + 0.5f) / 1.0f + support + 0.5f), (int)(0));
  const int cacheRangeEndY = min((int)(cacheRangeStartY + numCachedPixels), (int)inputRows);
  const unsigned int x = get_global_id(0);
  event_t e = async_work_group_strided_copy(inputImageCache,
                                            inputImage + cacheRangeStartY * inputColumns + x,
                                            cacheRangeEndY - cacheRangeStartY, inputColumns, 0);
  wait_group_events(1, &e);

  if (get_local_id(1) == 0) {
    //    uint sum = 0;
    //    for (unsigned int chunk = 0; chunk < numCachedPixels; chunk++) {
    //      sum += inputImageCache[chunk];
    //    }
    atomic_add(dst, inputImageCache[0]);
  }
}
\n);

OCLPerfVerticalFetch::OCLPerfVerticalFetch() {
  ptr_ = nullptr;
  _numSubTests = 6;
}

OCLPerfVerticalFetch::~OCLPerfVerticalFetch() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfVerticalFetch::open(unsigned int test, char* units, double& conversion,
                                unsigned int deviceId) {
  error_ = CL_SUCCESS;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = 0;
  kernel_ = 0;
  skip_ = false;
  dstBuffer_ = 0;
  cl_ulong loopCnt = nBytes / (16 * sizeof(cl_uint));
  cl_uint maxCUs;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_MAX_COMPUTE_UNITS,
                                     sizeof(cl_uint), &maxCUs, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  wgs = 64;
  const static cl_uint wavesPerCU = 8;
  nWorkItems = maxCUs * wavesPerCU * wgs;
  uint32_t memLoc = CL_MEM_USE_HOST_PTR;

  inputData = 0x1;
  switch (test) {
    case 0:
      nIter = 1;
      mem_type_ = "UHP";
      break;
    case 1:
      nIter = 100;
      mem_type_ = "UHP";
      break;
    case 2:
      nIter = 1;
      memLoc = CL_MEM_ALLOC_HOST_PTR;
      mem_type_ = "AHP";
      break;
    case 3:
      nIter = 100;
      memLoc = CL_MEM_ALLOC_HOST_PTR;
      mem_type_ = "AHP";
      break;
    case 4:
      nIter = 1;
      memLoc = 0;
      mem_type_ = "dev";
      break;
    case 5:
      nIter = 1000;
      memLoc = 0;
      mem_type_ = "dev";
      break;
  }

  std::string nameFile("dim.ini");
  std::fstream is(nameFile.c_str(), std::fstream::in | std::fstream::binary);
  std::string line;
  if (is.is_open()) {
    size_t posStart = 0;
    do {
      std::getline(is, line);
    } while (line.find_first_of('/', posStart) != std::string::npos);
    // Find global/local
    posStart = 0;
    size_t posEnd = 1;
    std::string dimS = line.substr(posStart, posEnd - posStart);
    dim = std::stoi(dimS.c_str(), nullptr, 10);
    posStart = posEnd;
    posEnd = line.find_first_of('[', posStart);
    for (cl_uint i = 0; i < dim; ++i) {
      posStart = posEnd + 1;
      posEnd = line.find_first_of(',', posStart);
      std::string global = line.substr(posStart, posEnd - posStart);
      gws[i] = std::stoi(global.c_str(), nullptr, 10);
    }
    posEnd = line.find_first_of('[', posStart);
    for (cl_uint i = 0; i < dim; ++i) {
      posStart = posEnd + 1;
      posEnd = line.find_first_of(',', posStart);
      std::string global = line.substr(posStart, posEnd - posStart);
      lws[i] = std::stoi(global.c_str(), nullptr, 10);
    }
    posEnd = line.find_first_of('[', posStart);
    posStart = posEnd + 1;
    posEnd = line.find_first_of(',', posStart);
    std::string global = line.substr(posStart, posEnd - posStart);
    numCachedPixels_ = std::stoi(global.c_str(), nullptr, 10);
    is.close();
  } else {
    dim = 2;
    gws[0] = WIDTH;
    gws[1] = 512;
    lws[0] = 1;
    lws[1] = 256;
    numCachedPixels_ = 1676;
  }
  cl_uint width = static_cast<cl_uint>(gws[0]);
  cl_uint height = numCachedPixels_ * static_cast<cl_uint>(gws[1] / lws[1]);
  if (gws[1] > 512) {
    gws[1] = 512;
  }
  Sizes[0] = width * height * sizeof(int);
  nBytes = Sizes[0];

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "ResizeVerticalFilter", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  if (memLoc == CL_MEM_USE_HOST_PTR) {
    ptr_ = malloc(nBytes);
  }
  srcBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY | memLoc, nBytes, ptr_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer(srcBuffer) failed");
  void* mem;
  mem = _wrapper->clEnqueueMapBuffer(cmdQueues_[_deviceId], srcBuffer_, CL_TRUE,
                                     CL_MAP_READ | CL_MAP_WRITE, 0, nBytes, 0, NULL, NULL, &error_);
  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  for (unsigned int i = 0; i < nBytes / sizeof(cl_uint); ++i) {
    reinterpret_cast<cl_uint*>(mem)[i] = inputData;
  }

  dstBuffer_ =
      _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(cl_uint), NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer(dstBuffer) failed");
  _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], srcBuffer_, mem, 0, NULL, NULL);
  mem = _wrapper->clEnqueueMapBuffer(cmdQueues_[_deviceId], dstBuffer_, CL_TRUE,
                                     CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_uint), 0, NULL, NULL,
                                     &error_);
  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  memset(mem, 0, sizeof(cl_uint));
  _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], dstBuffer_, mem, 0, NULL, NULL);

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &srcBuffer_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_uint), (void*)&width);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint), (void*)&height);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 3, numCachedPixels_ * sizeof(cl_uint), 0);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_uint), (void*)&numCachedPixels_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 5, sizeof(cl_mem), (void*)&dstBuffer_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
}

void OCLPerfVerticalFetch::run(void) {
  if (skip_) {
    return;
  }

  CPerfCounter timer;

  // warm up
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, dim, NULL, gws, lws, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  cl_uint* memResult;
  memResult = (cl_uint*)malloc(sizeof(cl_uint));
  if (0 == memResult) {
    CHECK_RESULT_NO_RETURN(0, "malloc failed!\n");
    return;
  }

  memset(memResult, 0, sizeof(cl_uint));
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], dstBuffer_, CL_FALSE, 0,
                                         sizeof(cl_uint), memResult, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueReadBuffer dstBuffer_ failed!");
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  if (memResult[0] != ((gws[0] * gws[1]) / (lws[0] * lws[1]))) {
    CHECK_RESULT_NO_RETURN(0, "Data validation failed for warm up run!\n");
    // free(memResult);
    // return;
  }

  free(memResult);

  timer.Reset();
  timer.Start();
  double sec2 = 0;
  cl_event* events = new cl_event[nIter];
  for (unsigned int i = 0; i < nIter; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, dim, NULL, gws, lws,
                                              0, NULL, &events[i]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    _wrapper->clFinish(cmdQueues_[_deviceId]);
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();
  for (unsigned int i = 0; i < nIter; i++) {
    cl_ulong startTime = 0, endTime = 0;
    error_ = _wrapper->clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                                               sizeof(cl_ulong), &startTime, 0);
    CHECK_RESULT(error_, "clGetEventProfilingInfo failed");
    error_ = _wrapper->clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                                               sizeof(cl_ulong), &endTime, 0);
    CHECK_RESULT(error_, "clGetEventProfilingInfo failed");

    _wrapper->clReleaseEvent(events[i]);
    sec2 += endTime - startTime;
  }
  double sec = timer.GetElapsedTime();
  delete[] events;

  // read speed in GB/s
  double perf = ((double)nBytes * nIter * (double)(1e-09)) / sec;
  double perf2 = ((double)nBytes * nIter) / sec2;
  _perfInfo = (float)perf2;
  float perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes, %s) i:%4d Wall time Perf: %.2f (GB/s)", nBytes,
           mem_type_, nIter, perfInfo);
  testDescString = buf;
}

unsigned int OCLPerfVerticalFetch::close(void) {
  if (!skip_) {
    if (srcBuffer_) {
      error_ = _wrapper->clReleaseMemObject(srcBuffer_);
      CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(srcBuffer_) failed");
    }

    if (dstBuffer_) {
      error_ = _wrapper->clReleaseMemObject(dstBuffer_);
      CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(srcBuffer_) failed");
    }
  }
  if (ptr_ != nullptr) {
    free(ptr_);
    ptr_ = nullptr;
  }

  return OCLTestImp::close();
}
