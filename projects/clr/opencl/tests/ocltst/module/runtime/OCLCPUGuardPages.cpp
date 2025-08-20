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

#include "OCLCPUGuardPages.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#ifdef _WIN32
#include <excpt.h>
#include <windows.h>  // for EXCEPTION_ACCESS_VIOLATION

int filter(unsigned int code, struct _EXCEPTION_POINTERS* ep) {
  printf("In filter\n");
  if (code == EXCEPTION_ACCESS_VIOLATION) {
    printf("caught AV as expected.");
    return EXCEPTION_EXECUTE_HANDLER;
  } else {
    printf("didn't catch AV, unexpected.");
    return EXCEPTION_CONTINUE_SEARCH;
  };
}

#else
#include <signal.h>

#include <csignal>
#include <cstdlib>
#include <iostream>

void segfault_sigaction(int signal, siginfo_t* si, void* arg) {
  printf("Caught segfault at address %p\n", si->si_addr);
  exit(0);
}

#endif

const static char* strKernel =
    "__kernel void simple_in_out_test( int in_offset, \n"
    "                                  int out_offset, \n"
    "                                  __global float4* in,          \n"
    "                                  __global float4* out) { \n"
    "unsigned int gid = get_global_id(0);\n"
    "out[gid + out_offset] = in[gid + in_offset] * -1.f;"
    "}";

testOCLCPUGuardPagesStruct testOCLCPUGuardPagesList[] = {
    {false, false, 1024, 0, 0}, {true, false, 1024, 0, 0},  {false, false, 1024, 0, 0},
    {true, true, 1024, 0, 0},   {false, false, 1024, 0, 0}, {true, true, 1024, 0, 0}};

OCLCPUGuardPages::OCLCPUGuardPages() {
  _numSubTests = sizeof(testOCLCPUGuardPagesList) / sizeof(testOCLCPUGuardPagesStruct);

  /*
      struct sigaction sa;

      memset(&sa, 0, sizeof(sa));
      sigemptyset(&sa.sa_mask);
      sa.sa_sigaction = segfault_sigaction;
      sa.sa_flags   = SA_SIGINFO;

      sigaction(SIGSEGV, &sa, NULL);
  */
}

OCLCPUGuardPages::~OCLCPUGuardPages() {}

void OCLCPUGuardPages::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  // Initialize the current test parameters.
  testValues = testOCLCPUGuardPagesList[test];

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

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

  kernel_ = _wrapper->clCreateKernel(program_, "simple_in_out_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  // Create input and output buffers for the test.
  cl_mem inBuffer, outBuffer;
  cl_float4* dummyIn = new cl_float4[testValues.items];
  for (int i = 0; i < testValues.items; i++) {
    dummyIn[i].s[0] = dummyIn[i].s[1] = dummyIn[i].s[2] = dummyIn[i].s[3] = i * 1.f;
  }
  inBuffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                      testValues.items * sizeof(cl_float4), NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], inBuffer, 1, 0,
                                          testValues.items * sizeof(cl_float4), dummyIn, 0, 0, 0);
  buffers_.push_back(inBuffer);

  outBuffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                       testValues.items * sizeof(cl_float4), NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(outBuffer);
  delete[] dummyIn;
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLCPUGuardPages::run(void) {
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_int), &testValues.in_offset);
  error_ |= _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_int), &testValues.out_offset);
  error_ |= _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_mem), &buffers()[0]);
  error_ |= _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_mem), &buffers()[1]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t globalThreads[1];
  globalThreads[0] = testValues.items;
  size_t localThreads[1] = {256};

#ifdef _WIN32
  //    LPTOP_LEVEL_EXCEPTION_FILTER pOriginalFilter =
  //    SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
  //    AddVectoredExceptionHandler(1,MyVectorExceptionFilter);

  try {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                              globalThreads, localThreads, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  } catch (...) {
    printf("exception caught in OCLTest...\n");
  }

#else
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, globalThreads,
                                            localThreads, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
#endif
}

unsigned int OCLCPUGuardPages::close(void) { return OCLTestImp::close(); }
