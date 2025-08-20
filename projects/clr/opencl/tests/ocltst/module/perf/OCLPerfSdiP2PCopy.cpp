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

#include "OCLPerfSdiP2PCopy.h"

#include <string.h>

#include "Timer.h"

#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_SIZES 5
// 64KB, 256KB, 1 MB, 4MB, 16 MB
static const unsigned int Sizes[NUM_SIZES] = {65536, 262144, 1048576, 4194304, 16777216};

OCLPerfSdiP2PCopy::OCLPerfSdiP2PCopy() {
  // If there are two different gpus in the system,
  // we have to test each of them
  _numSubTests = 2 * NUM_SIZES;
}

OCLPerfSdiP2PCopy::~OCLPerfSdiP2PCopy() {}

void OCLPerfSdiP2PCopy::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  cl_uint numPlatforms = 0;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  _crcword = 0;
  conversion = 1.0f;
  _openTest = test % NUM_SIZES;
  bufSize_ = Sizes[_openTest];
  error_ = 0;
  srcBuff_ = 0;
  inputArr_ = 0;
  outputArr_ = 0;
  extPhysicalBuff_ = 0;
  silentFailure = false;
  busAddressableBuff_ = 0;
  devices_[0] = devices_[1] = 0;
  contexts_[0] = contexts_[1] = 0;
  cmd_queues_[0] = cmd_queues_[1] = 0;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(numPlatforms == 0, "clGetPlatformIDs failed");
  error_ = _wrapper->clGetPlatformIDs(1, &platform, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  error_ = _wrapper->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (num_devices != 2) {
    printf(
        "\nSilent Failure: Two GPUs are required to run OCLPerfSdiP2PCopy "
        "test\n");
    silentFailure = true;
    return;
  }
  error_ = _wrapper->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices_, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
  if (test >= NUM_SIZES) {
    cl_device_id temp = devices_[0];
    devices_[0] = devices_[1];
    devices_[1] = temp;
  }
  size_t param_size = 0;
  char* strExtensions = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_EXTENSIONS, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strExtensions = (char*)malloc(param_size);
  error_ =
      _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_EXTENSIONS, param_size, strExtensions, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strstr(strExtensions, "cl_amd_bus_addressable_memory") == 0) {
    printf(
        "\nSilent Failure: cl_amd_bus_addressable_memory extension is not "
        "enabled on GPU 0\n");
    silentFailure = true;
    free(strExtensions);
    return;
  }
  free(strExtensions);
  error_ = _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_EXTENSIONS, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strExtensions = (char*)malloc(param_size);
  error_ =
      _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_EXTENSIONS, param_size, strExtensions, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strstr(strExtensions, "cl_amd_bus_addressable_memory") == 0) {
    printf(
        "\nSilent Failure: cl_amd_bus_addressable_memory extension is not "
        "enabled on GPU 1\n");
    silentFailure = true;
    free(strExtensions);
    return;
  }
  free(strExtensions);
  deviceNames_ = " [";
  param_size = 0;
  char* strDeviceName = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_NAME, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strDeviceName = (char*)malloc(param_size);
  error_ = _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_NAME, param_size, strDeviceName, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  deviceNames_ = deviceNames_ + strDeviceName;
  free(strDeviceName);
  error_ = _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_NAME, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strDeviceName = (char*)malloc(param_size);
  error_ = _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_NAME, param_size, strDeviceName, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  deviceNames_ = deviceNames_ + "->";
  deviceNames_ = deviceNames_ + strDeviceName;
  free(strDeviceName);
  deviceNames_ = deviceNames_ + "]";
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

  contexts_[0] = _wrapper->clCreateContext(props, 1, &devices_[0], 0, 0, &error_);
  CHECK_RESULT(contexts_[0] == 0, "clCreateContext failed");
  contexts_[1] = _wrapper->clCreateContext(props, 1, &devices_[1], 0, 0, &error_);
  CHECK_RESULT(contexts_[1] == 0, "clCreateContext failed");
  cmd_queues_[0] = _wrapper->clCreateCommandQueue(contexts_[0], devices_[0], 0, NULL);
  CHECK_RESULT(cmd_queues_[0] == 0, "clCreateCommandQueue failed");
  cmd_queues_[1] = _wrapper->clCreateCommandQueue(contexts_[1], devices_[1], 0, NULL);
  CHECK_RESULT(cmd_queues_[1] == 0, "clCreateCommandQueue failed");
  busAddressableBuff_ =
      _wrapper->clCreateBuffer(contexts_[0], CL_MEM_BUS_ADDRESSABLE_AMD, bufSize_, 0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer failed");
  error_ = _wrapper->clEnqueueMakeBuffersResidentAMD(cmd_queues_[0], 1, &busAddressableBuff_, true,
                                                     &busAddr_, 0, 0, 0);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMakeBuffersResidentAMD failed");
  extPhysicalBuff_ = _wrapper->clCreateBuffer(contexts_[1], CL_MEM_EXTERNAL_PHYSICAL_AMD, bufSize_,
                                              &busAddr_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer failed");
  srcBuff_ = _wrapper->clCreateBuffer(contexts_[1], CL_MEM_READ_WRITE, bufSize_, 0, &error_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clCreateBuffer failed");
  inputArr_ = (cl_uint*)malloc(bufSize_);
  outputArr_ = (cl_uint*)malloc(bufSize_);
  for (unsigned int i = 0; i < (bufSize_ / sizeof(cl_uint)); ++i) {
    inputArr_[i] = i + 1;
    outputArr_[i] = 0;
  }
  error_ = _wrapper->clEnqueueWriteBuffer(cmd_queues_[1], srcBuff_, CL_TRUE, 0, bufSize_, inputArr_,
                                          0, 0, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteBuffer failed");
}

void OCLPerfSdiP2PCopy::run(void) {
  if (silentFailure) {
    return;
  }
  CPerfCounter timer;
  // Warm up
  error_ = _wrapper->clEnqueueCopyBuffer(cmd_queues_[1], srcBuff_, extPhysicalBuff_, 0, 0, bufSize_,
                                         0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");
  error_ = _wrapper->clFinish(cmd_queues_[1]);
  CHECK_RESULT(error_, "clFinish failed");
  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < NUM_ITER; i++) {
    error_ = _wrapper->clEnqueueCopyBuffer(cmd_queues_[1], srcBuff_, extPhysicalBuff_, 0, 0,
                                           bufSize_, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");
  }
  error_ = _wrapper->clFinish(cmd_queues_[1]);
  CHECK_RESULT(error_, "clFinish failed");
  timer.Stop();
  double sec = timer.GetElapsedTime();
  error_ = _wrapper->clEnqueueReadBuffer(cmd_queues_[0], busAddressableBuff_, CL_TRUE, 0, bufSize_,
                                         outputArr_, 0, 0, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteBuffer failed");
  CHECK_RESULT((memcmp(inputArr_, outputArr_, bufSize_) != 0), "copy failed");
  // Buffer copy bandwidth in GB/s
  double perf = ((double)bufSize_ * NUM_ITER * (double)(1e-09)) / sec;
  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%8d bytes) i:%4d (GB/s) %s", bufSize_, NUM_ITER,
           deviceNames_.c_str());
  testDescString = buf;
}

unsigned int OCLPerfSdiP2PCopy::close(void) {
  if (srcBuff_) {
    error_ = _wrapper->clReleaseMemObject(srcBuff_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject failed");
  }
  if (extPhysicalBuff_) {
    error_ = _wrapper->clReleaseMemObject(extPhysicalBuff_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject failed");
  }
  if (busAddressableBuff_) {
    error_ = _wrapper->clReleaseMemObject(busAddressableBuff_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject failed");
  }
  if (cmd_queues_[0]) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queues_[0]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (cmd_queues_[1]) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queues_[1]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (contexts_[0]) {
    error_ = _wrapper->clReleaseContext(contexts_[0]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }
  if (contexts_[1]) {
    error_ = _wrapper->clReleaseContext(contexts_[1]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }
  if (inputArr_) {
    free(inputArr_);
  }
  if (outputArr_) {
    free(outputArr_);
  }
  return _crcword;
}
