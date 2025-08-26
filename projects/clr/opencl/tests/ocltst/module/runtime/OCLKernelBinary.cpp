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

#include "OCLKernelBinary.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static char* strKernel12 =
    "typedef struct ST {                                \n"
    "  int i0;                                          \n"
    "  int i1;                                          \n"
    "} ST_t;                                            \n"
    "                                                   \n"
    "__constant ST_t STCArray[2] = {                    \n"
    "  { 1, 0 },                                        \n"
    "  { 2, 1 }                                         \n"
    "};                                                 \n"
    "                                                   \n"
    "__kernel void foo (__global int *p, int n)         \n"
    "{                                                  \n"
    "  int s = 0;                                       \n"
    "  int i;                                           \n"
    "  for (i=0; i < n; ++i) {                          \n"
    "    s += STCArray[i].i0 + STCArray[i].i1;          \n"
    "  }                                                \n"
    "  *p = s;                                          \n"
    "}                                                  \n";

const static char* strKernel20 =
    "typedef struct ST {                                \n"
    "  int i0;                                          \n"
    "  int i1;                                          \n"
    "} ST_t;                                            \n"
    "                                                   \n"
    "__constant ST_t STCArray[2] = {                    \n"
    "  { -1, 0 },                                       \n"
    "  { 3, -1 }                                        \n"
    "};                                                 \n"
    "                                                   \n"
    "__global int var = 1;                              \n"
    "                                                   \n"
    "__kernel void foo (__global int *p, int n)         \n"
    "{                                                  \n"
    "  int s = 0;                                       \n"
    "  int i;                                           \n"
    "  for (i=0; i < n; ++i) {                          \n"
    "    s += STCArray[i].i0 + STCArray[i].i1 + var++;  \n"
    "  }                                                \n"
    "  p[get_global_id(0)] = s;                         \n"
    "}                                                  \n";

OCLKernelBinary::OCLKernelBinary() { _numSubTests = 2; }

OCLKernelBinary::~OCLKernelBinary() {}

void OCLKernelBinary::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);

  char strVersion[128];
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_VERSION, sizeof(strVersion),
                                     strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  if (test == 1 && strVersion[7] < '2') {
    program_ = NULL;
    return;
  }

  const char *options, *options0;
  const char* strKernel;
  switch (test) {
    case 0:
      options = "";
      options0 = "-O0";
      strKernel = strKernel12;
      break;
    case 1:
      options = "-cl-std=CL2.0";
      options0 = "-cl-std=CL2.0 -O0";
      strKernel = strKernel20;
      break;
    default:
      assert(false);
      return;
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], options, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  size_t* sizes = new size_t[deviceCount_];
  CHECK_RESULT(((sizes != NULL) ? false : true), "malloc()");
  size_t* sizes1 = new size_t[deviceCount_];
  CHECK_RESULT(((sizes1 != NULL) ? false : true), "malloc()");
  size_t* sizes2 = new size_t[deviceCount_];
  CHECK_RESULT(((sizes2 != NULL) ? false : true), "malloc()");

  unsigned int programInfoDeviceIdIndex = 0;
  cl_device_id* programInfoDevices = new cl_device_id[deviceCount_];
  CHECK_RESULT(((programInfoDevices != NULL) ? false : true), "malloc()");
  // get an array of device Id's that relate to values order returned by
  // 'clGetProgramInfo'
  error_ = _wrapper->clGetProgramInfo(
      program_, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * deviceCount_, programInfoDevices, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetProgramInfo()");
  // map between the class devices_ array and the programInfoDeviceId array
  for (unsigned int i = 0; i < deviceCount_; i++) {
    if (devices_[deviceId] == programInfoDevices[i]) {
      programInfoDeviceIdIndex = i;
    }
  }
  delete[] programInfoDevices;

  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                                      sizeof(size_t) * deviceCount_, sizes, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetProgramInfo()");

  unsigned char** binaries = new unsigned char*[deviceCount_];
  CHECK_RESULT(((binaries != NULL) ? false : true), "malloc()");

  for (unsigned int i = 0; i < deviceCount_; i++) {
    if (sizes[i] > 0) {
      binaries[i] = new unsigned char[sizes[i]];
      CHECK_RESULT(((binaries[i] != NULL) ? false : true), "malloc()");
    } else {
      binaries[i] = NULL;
    }
  }

  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                                      sizeof(unsigned char*) * deviceCount_, binaries, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetProgramInfo()");

  error_ = _wrapper->clReleaseProgram(program_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clReleaseProgram()");

  const unsigned char* cBinary = binaries[programInfoDeviceIdIndex];
  cl_int status;
  program_ = _wrapper->clCreateProgramWithBinary(context_, 1, &devices_[deviceId],
                                                 &(sizes[programInfoDeviceIdIndex]), &cBinary,
                                                 &status, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithBinary()");

  for (unsigned int i = 0; i < deviceCount_; i++) {
    if (binaries[i] != NULL) delete[] binaries[i];
  }
  delete[] binaries;

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], options0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetProgramInfo()");

  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                                      sizeof(size_t) * deviceCount_, sizes1, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "1st clGetProgramInfo()");

  kernel_ = _wrapper->clCreateKernel(program_, "foo", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "1st clCreateKernel() failed");

  _wrapper->clReleaseKernel(kernel_);
  CHECK_RESULT((error_ != CL_SUCCESS), "1st clReleaseKernel() failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], options0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetProgramInfo()");

  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                                      sizeof(size_t) * deviceCount_, sizes2, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "2nd clGetProgramInfo()");

  kernel_ = _wrapper->clCreateKernel(program_, "foo", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "2nd clCreateKernel() failed");

  cl_mem buffer;
  buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, 2 * sizeof(cl_uint), NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

  delete[] sizes;
  delete[] sizes1;
  delete[] sizes2;
}

void OCLKernelBinary::run(void) {
  if (program_ == NULL) {
    return;
  }

  cl_mem buffer = buffers()[0];
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  cl_int num = 2;
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_int), &num);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t gws[1] = {2};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  cl_uint outputV[2] = {0};
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, true, 0,
                                         2 * sizeof(cl_uint), outputV, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
  if (outputV[0] != 4) {
    CHECK_RESULT(true, "Incorrect result of kernel execution!");
  }
}

unsigned int OCLKernelBinary::close(void) { return OCLTestImp::close(); }
