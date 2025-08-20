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

#include "OCLPerfDoubleDMASeq.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <cmath>
#include <sstream>
#include <string>

#include "CL/cl.h"
#include "CL/cl_ext.h"

#ifdef _WIN32
const size_t blockX = 128;
const size_t blockY = 128;
const size_t blockZ = 256;
#else
const size_t blockX = 256;
const size_t blockY = 256;
const size_t blockZ = 512;
#endif

const size_t chunk = 16;
const size_t size_S = blockX * blockY * blockZ * sizeof(cl_float4);
const size_t size_s = blockX * blockY * chunk * sizeof(cl_float4);
static const int WindowWidth = 80;

const size_t MaxQueues = 3;

static const char* strKernel =
    "__kernel void dummy(__global float4* out)  \n"
    "{                                          \n"
    "   uint id = get_global_id(0);             \n"
    "   float4 value = (float4)(1.0f, 2.0f, 3.0f, 4.0f);  \n"
    "   uint factorial = 1;                     \n"
    "   for (uint i = 1; i < (id / 0x400); ++i)\n"
    "   {                                       \n"
    "       factorial *= i;                     \n"
    "   }                                       \n"
    "   out[id] = value * factorial;            \n"
    "}                                          \n";

OCLPerfDoubleDMASeq::OCLPerfDoubleDMASeq() {
  _numSubTests = MaxQueues * 2;
  failed_ = false;
}

OCLPerfDoubleDMASeq::~OCLPerfDoubleDMASeq() {}

void OCLPerfDoubleDMASeq::open(unsigned int test, char* units, double& conversion,
                               unsigned int deviceId) {
  _deviceId = deviceId;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  test_ = test;
  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }
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
  kernel_ = _wrapper->clCreateKernel(program_, "dummy", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  size_t bufSize = size_s;
  cl_mem buffer;
  test_ %= MaxQueues;
  events_ = ((test / MaxQueues) == 0) ? false : true;
  size_t numBufs = (test_ % MaxQueues) + 1;
  for (size_t b = 0; b < numBufs; ++b) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, bufSize, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size_S,
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfDoubleDMASeq::run(void) {
  if (failed_) {
    return;
  }
  CPerfCounter timer;
  const int numQueues = (test_ % MaxQueues) + 1;
  const int numBufs = numQueues;

  std::vector<cl_command_queue> cmdQueues(numQueues);
  int q;
  cl_command_queue_properties qProp = 0;
  for (q = 0; q < numQueues; ++q) {
    cl_command_queue cmdQueue =
        _wrapper->clCreateCommandQueue(context_, devices_[_deviceId], qProp, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed");
    cmdQueues[q] = cmdQueue;
  }
  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "Execution failed");

  float* Data_s = (float*)_wrapper->clEnqueueMapBuffer(cmdQueues[0], buffers_[numBufs], CL_TRUE,
                                                       CL_MAP_READ | CL_MAP_WRITE, 0, size_S, 0,
                                                       NULL, NULL, &error_);

  size_t gws[1] = {size_s / (4 * sizeof(float))};
  size_t lws[1] = {256};

  // Warm-up
  for (q = 0; q < numQueues; ++q) {
    error_ |= _wrapper->clEnqueueWriteBuffer(cmdQueues[q], buffers_[q], CL_FALSE, 0, size_s,
                                             (char*)Data_s, 0, NULL, NULL);
    error_ |= _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[q]);
    error_ |=
        _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 0, NULL, NULL);
    error_ |= _wrapper->clEnqueueReadBuffer(cmdQueues[q], buffers_[q], CL_FALSE, 0, size_s,
                                            (char*)Data_s, 0, NULL, NULL);
    error_ |= _wrapper->clFinish(cmdQueues[q]);
  }

  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "Execution failed");

  size_t s_done = 0;
  cl_event x[MaxQueues] = {0};

  /*----------  pass2:  copy Data_s to and from GPU Buffers ----------*/
  s_done = 0;
  timer.Reset();
  timer.Start();
  int idx = numBufs - 1;
  // Start from the last so read/write won't go to the same DMA when kernel is
  // executed
  q = numQueues - 1;
  size_t iter = 0;
  if (events_) {
    while (1) {
      error_ |= _wrapper->clEnqueueWriteBuffer(cmdQueues[q], buffers_[idx], CL_FALSE, 0, size_s,
                                               (char*)Data_s + s_done, 0, NULL, NULL);

      // Implicit flush of DMA engine on kernel start, because memory dependency
      error_ |= _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[idx]);
      int prevQ;
      if (q == 0) {
        prevQ = numQueues - 1;
      } else {
        prevQ = q - 1;
      }
      if ((x[prevQ] != NULL) && (numQueues != 1)) {
        error_ |= _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 1,
                                                   &x[prevQ], &x[q]);
        error_ |= _wrapper->clReleaseEvent(x[prevQ]);
        x[prevQ] = NULL;
      } else {
        error_ |= _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 0,
                                                   NULL, &x[q]);
        if (numQueues == 1) {
          error_ |= _wrapper->clReleaseEvent(x[q]);
          x[q] = NULL;
        }
      }
      error_ |= _wrapper->clFlush(cmdQueues[q]);

      // Change the queue
      error_ |= _wrapper->clEnqueueReadBuffer(cmdQueues[q], buffers_[idx], CL_FALSE, 0, size_s,
                                              (char*)Data_s + s_done, 0, NULL, NULL);

      if ((s_done += size_s) >= size_S) {
        break;
      }

      error_ |= _wrapper->clFlush(cmdQueues[q]);
      ++iter;
      ++idx %= numBufs;
      ++q %= numQueues;
    }
    for (q = 0; q < numQueues; ++q) {
      if (x[q] != NULL) {
        error_ |= _wrapper->clReleaseEvent(x[q]);
      }
    }
  } else {
    while (1) {
      error_ |= _wrapper->clEnqueueWriteBuffer(cmdQueues[q], buffers_[idx], CL_FALSE, 0, size_s,
                                               (char*)Data_s + s_done, 0, NULL, NULL);

      // Implicit flush of DMA engine on kernel start, because memory dependency
      error_ |= _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[idx]);
      error_ |=
          _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 0, NULL, NULL);

      // Change the queue
      error_ |= _wrapper->clEnqueueReadBuffer(cmdQueues[q], buffers_[idx], CL_FALSE, 0, size_s,
                                              (char*)Data_s + s_done, 0, NULL, NULL);

      if ((s_done += size_s) >= size_S) {
        break;
      }

      error_ |= _wrapper->clFlush(cmdQueues[q]);
      ++iter;
      ++idx %= numBufs;
      ++q %= numQueues;
    }
  }

  for (q = 0; q < numQueues; ++q) {
    error_ |= _wrapper->clFinish(cmdQueues[q]);
  }
  timer.Stop();

  error_ |=
      _wrapper->clEnqueueUnmapMemObject(cmdQueues[0], buffers_[numBufs], Data_s, 0, NULL, NULL);

  error_ |= _wrapper->clFinish(cmdQueues[0]);
  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "Execution failed");

  for (q = 0; q < numQueues; ++q) {
    error_ = _wrapper->clReleaseCommandQueue(cmdQueues[q]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseCommandQueue() failed");
  }

  double GBytes = (double)(2 * size_S) / (double)(1000 * 1000 * 1000);
  _perfInfo = static_cast<float>(GBytes / timer.GetElapsedTime());

  std::stringstream stream;
  stream << "Write/Kernel/Read operation ";

  stream << numQueues << " queues ";
  if (events_) {
    stream << " (use events) ";
  }
  stream << " [GB/s]";

  stream.flags(std::ios::right | std::ios::showbase);
  testDescString = stream.str();
}

unsigned int OCLPerfDoubleDMASeq::close(void) { return OCLTestImp::close(); }
