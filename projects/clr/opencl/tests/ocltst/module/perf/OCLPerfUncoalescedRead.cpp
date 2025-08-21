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

#include "OCLPerfUncoalescedRead.h"

#include <string.h>

#include <iomanip>
#include <sstream>

#include "Timer.h"

const char* OCLPerfUncoalescedRead::kernel_str =
    "#define NUM_READS 32\n\
    __kernel void read_uncoalescing(__global float *input,__global float *output)\n\
    {\n\
        float val = (float)(0.0f);\n\
        size_t gid = get_global_id(0);\n\
        val = val + input[gid * NUM_READS + 0];\n\
        val = val + input[gid * NUM_READS + 1];\n\
        val = val + input[gid * NUM_READS + 2];\n\
        val = val + input[gid * NUM_READS + 3];\n\
        val = val + input[gid * NUM_READS + 4];\n\
        val = val + input[gid * NUM_READS + 5];\n\
        val = val + input[gid * NUM_READS + 6];\n\
        val = val + input[gid * NUM_READS + 7];\n\
        val = val + input[gid * NUM_READS + 8];\n\
        val = val + input[gid * NUM_READS + 9];\n\
        val = val + input[gid * NUM_READS + 10];\n\
        val = val + input[gid * NUM_READS + 11];\n\
        val = val + input[gid * NUM_READS + 12];\n\
        val = val + input[gid * NUM_READS + 13];\n\
        val = val + input[gid * NUM_READS + 14];\n\
        val = val + input[gid * NUM_READS + 15];\n\
        val = val + input[gid * NUM_READS + 16];\n\
        val = val + input[gid * NUM_READS + 17];\n\
        val = val + input[gid * NUM_READS + 18];\n\
        val = val + input[gid * NUM_READS + 19];\n\
        val = val + input[gid * NUM_READS + 20];\n\
        val = val + input[gid * NUM_READS + 21];\n\
        val = val + input[gid * NUM_READS + 22];\n\
        val = val + input[gid * NUM_READS + 23];\n\
        val = val + input[gid * NUM_READS + 24];\n\
        val = val + input[gid * NUM_READS + 25];\n\
        val = val + input[gid * NUM_READS + 26];\n\
        val = val + input[gid * NUM_READS + 27];\n\
        val = val + input[gid * NUM_READS + 28];\n\
        val = val + input[gid * NUM_READS + 29];\n\
        val = val + input[gid * NUM_READS + 30];\n\
        val = val + input[gid * NUM_READS + 31];\n\
        output[gid] = val;\n\
    }\n";

OCLPerfUncoalescedRead::OCLPerfUncoalescedRead() { _numSubTests = 3; }

OCLPerfUncoalescedRead::~OCLPerfUncoalescedRead() {}

void OCLPerfUncoalescedRead::open(unsigned int test, char* units, double& conversion,
                                  unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "error_ opening test");
  silentFailure = false;
  _openTest = test;
  program_ = 0;
  kernel_ = 0;
  input_buff = NULL;

  if (test > 0) {
    size_t param_size = 0;
    char* strVersion = 0;
    error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_OPENCL_C_VERSION, 0, 0,
                                       &param_size);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
    strVersion = (char*)malloc(param_size);
    error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_OPENCL_C_VERSION, param_size,
                                       strVersion, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
    if (strVersion[9] < '2') {
      printf("\nOpenCL C 2.0 not supported\n");
      silentFailure = true;
    }
    free(strVersion);
    if (silentFailure) return;
  }

  cl_mem buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                           SIZE * NUM_READS * sizeof(cl_float), 0, &error_);
  buffers_.push_back(buffer);
  buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, SIZE * sizeof(cl_float), 0, &error_);
  buffers_.push_back(buffer);

  srand(0x8956);
  input_buff = (float*)malloc(SIZE * NUM_READS * sizeof(float));
  for (unsigned int i = 0; i < SIZE * NUM_READS; ++i) {
    input_buff[i] = (float)rand();
  }

  error_ =
      _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                     SIZE * NUM_READS * sizeof(cl_float), input_buff, 0, 0, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");

  float* buff = (float*)_wrapper->clEnqueueMapBuffer(cmdQueues_[_deviceId], buffers_[1], CL_TRUE,
                                                     CL_MAP_WRITE, 0, SIZE * sizeof(cl_float), 0, 0,
                                                     0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer failed");
  memset(buff, 0, SIZE * sizeof(cl_float));
  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], buffers_[1], buff, 0, 0, 0);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer failed");

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &kernel_str, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource failed");
  std::string compileOptions = "";
  if (test > 0) {
    compileOptions = "-cl-std=CL2.0";
  }

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], compileOptions.c_str(), NULL,
                                    NULL);

  if (error_ != CL_SUCCESS) {
    char log[400];
    _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 400, log,
                                    0);
    printf("\n\n%s\n\n", log);
  }

  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram failed");
  kernel_ = _wrapper->clCreateKernel(program_, "read_uncoalescing", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&buffers_[1]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
}

void OCLPerfUncoalescedRead::validate(void) {
  bool success = true;
  float* buff =
      (float*)_wrapper->clEnqueueMapBuffer(cmdQueues_[_deviceId], buffers_[1], CL_TRUE, CL_MAP_READ,
                                           0, SIZE * sizeof(cl_float), 0, 0, 0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer failed");
  for (unsigned int i = 0; i < SIZE; ++i) {
    volatile float val = 0;
    for (int j = 0; j < NUM_READS; ++j) {
      val += input_buff[i * NUM_READS + j];
    }
    if (val != buff[i]) {
      success = false;
      std::string errorMsg = "Invalid result.  Expected: ";
      errorMsg += std::to_string(val);
      errorMsg += " Actual result: ";
      errorMsg += std::to_string(buff[i]);
      CHECK_RESULT(true, errorMsg.c_str());
      break;
    }
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], buffers_[1], buff, 0, 0, 0);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer failed");
}

void OCLPerfUncoalescedRead::run(void) {
  if (silentFailure) {
    return;
  }
  CPerfCounter timer;

  // Warm up
  size_t workGroupSize = SIZE;
  for (int i = 0; i < 50; ++i) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                              &workGroupSize, NULL, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
    _wrapper->clFinish(cmdQueues_[_deviceId]);
  }

  cl_event eventArr[NUM_ITER];
  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < NUM_ITER; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                              &workGroupSize, NULL, 0, NULL, &eventArr[i]);

    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  }
  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_RESULT(error_, "clFinish failed");
  timer.Stop();
  double sec1 = timer.GetElapsedTime();
  double sec2 = 0;
  for (unsigned int i = 0; i < NUM_ITER; ++i) {
    cl_ulong startTime = 0, endTime = 0;
    error_ = _wrapper->clGetEventProfilingInfo(eventArr[i], CL_PROFILING_COMMAND_START,
                                               sizeof(cl_ulong), &startTime, 0);
    CHECK_RESULT(error_, "clGetEventProfilingInfo failed");
    error_ = _wrapper->clGetEventProfilingInfo(eventArr[i], CL_PROFILING_COMMAND_END,
                                               sizeof(cl_ulong), &endTime, 0);
    CHECK_RESULT(error_, "clGetEventProfilingInfo failed");
    sec2 += 1e-9 * (endTime - startTime);
    error_ = _wrapper->clReleaseEvent(eventArr[i]);
    CHECK_RESULT(error_, "clReleaseEvent failed");
  }

  validate();

  // Buffer copy bandwidth in GB/s
  double perf1 = ((double)SIZE * NUM_READS * NUM_ITER * sizeof(cl_float) * (double)(1e-09)) / sec1;
  double perf2 = ((double)SIZE * NUM_READS * NUM_ITER * sizeof(cl_float) * (double)(1e-09)) / sec2;
  _perfInfo = (float)perf2;

  std::ostringstream strStream;
  switch (_openTest) {
    case 0:
      strStream << "OCL1.2      ";
      break;
    case 1:
      strStream << "OCL2.0      ";
      break;
    case 2:
      strStream << "OCL2.0/flag ";
      break;
  }

  strStream << std::fixed << std::setprecision(2) << perf1 << " timer GB/s ";
  strStream << "time: " << std::setprecision(3) << sec1 << "s (profile GB/s)";
  testDescString = strStream.str();
  ;
}

unsigned int OCLPerfUncoalescedRead::close(void) {
  if (input_buff) {
    free(input_buff);
  }
  return OCLTestImp::close();
}
