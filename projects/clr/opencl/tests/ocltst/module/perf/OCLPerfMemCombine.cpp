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

#include "OCLPerfMemCombine.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/opencl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

struct TestParams {
  const char* type;
  unsigned int numCombine;
  unsigned int assignSize;
};

TestParams testParams[]
    // char type causes shader compiler to crash. reenable once get a fix for
    // the shader compiler
    //= {{"char", 16}, {"short", 8}, {"int", 4}, {"long", 4}, {"float", 4}};
    //= {{"char", 16, 1}, {"short", 8, 2}, {"int", 4, 4}, {"long", 4, 8},
    = {{"short", 8, 2},  {"int", 4, 4},      {"long", 4, 8},   {"float", 4, 4},
       {"char4", 4, 4},  {"uchar16", 4, 16}, {"short2", 4, 4}, {"int2", 4, 8},
       {"uint4", 4, 16}, {"long2", 4, 16},   {"float2", 4, 8}};

const int numTests = sizeof(testParams) / sizeof(TestParams);

// Generate a kernel that does array loads and stores, which should be combined
// by MemCombine
void genCombineVLoadVStores(const char* type, int loopSize, int numCombine, char* ret) {
  sprintf(ret,
          "__kernel void combine_vload_vstores(__global %s"
          " * restrict src, __global %s *result) {\n",
          type, type);
  strcat(ret, "  int id = get_global_id(0);\n");
  strcat(ret, "  int gsize = get_global_size(0);\n");
  char buf[256];
  sprintf(buf, "  for (int i = 0; i < %d; i+=gsize) {\n", loopSize);
  strcat(ret, buf);
  sprintf(buf, "    int j = (i+id) * %d;\n", numCombine);
  strcat(ret, buf);
  for (int i = 0; i < numCombine; ++i) {
    sprintf(buf, "    result[j+%d] = src[j+%d];\n", i, i);
    strcat(ret, buf);
  }
  strcat(ret, "  }\n}\n");
}

void OCLPerfMemCombine::setData(cl_mem buffer, unsigned int bufSize, unsigned char val) {
  unsigned char* data = (unsigned char*)_wrapper->clEnqueueMapBuffer(
      cmdQueues_[0], buffer, true, CL_MAP_WRITE, 0, bufSize, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < bufSize; ++i) data[i] = val;

  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[0], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmdQueues_[0]);
}

void print1Darray(unsigned char* buffer, unsigned int bufSize) {
  for (unsigned int i = 0; i < bufSize; ++i) {
    if (i % 32 == 0) printf("\n");
    printf("%d ", buffer[i]);
  }
  printf("\n");
}

void OCLPerfMemCombine::checkData(cl_mem buffer, unsigned int bufSize, unsigned int limit,
                                  unsigned char defVal) {
  unsigned char* data = (unsigned char*)_wrapper->clEnqueueMapBuffer(
      cmdQueues_[0], buffer, true, CL_MAP_READ, 0, bufSize, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < bufSize; i++) {
    unsigned char expected;
    if (i < limit) {
      expected = 1U;
    } else {
      expected = defVal;
    }
    if (data[i] != expected) {
      printf("at index %d:\n", i);
      print1Darray(&data[i], 16);
      CHECK_RESULT(1, "incorrect output data detected!");
      break;
    }
  }

  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[0], buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmdQueues_[0]);
}

OCLPerfMemCombine::OCLPerfMemCombine() { _numSubTests = numTests; }

OCLPerfMemCombine::~OCLPerfMemCombine() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfMemCombine::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  _openTest = test;

  context_ = 0;
  kernel_ = NULL;
  program_ = NULL;

  OCLTestImp::open(test, units, conversion, deviceId);

  cl_mem inBuffer = _wrapper->clCreateBuffer(context_, 0, inSize_, NULL, &error_);
  CHECK_RESULT(inBuffer == 0, "clCreateBuffer(inBuffer) failed");
  buffers_.push_back(inBuffer);

  cl_mem outBuffer = _wrapper->clCreateBuffer(context_, 0, outSize_, NULL, &error_);
  CHECK_RESULT(outBuffer == 0, "clCreateBuffer(outBuffer) failed");
  buffers_.push_back(outBuffer);

  createKernel(testParams[test].type, testParams[test].numCombine);
  setData(inBuffer, inSize_, 1U);
  setData(outBuffer, outSize_, 0);
  dataRange_ = loopSize_ * numCombine_ * testParams[test].assignSize;
}

void OCLPerfMemCombine::createKernel(const char* type, int numCombine) {
  dataType_ = type;
  numCombine_ = numCombine;

  /////////////////////////////////////////////////////////////////
  // Load CL file, build CL program object, create CL kernel object
  /////////////////////////////////////////////////////////////////
  char source[1024];
  genCombineVLoadVStores(type, loopSize_, numCombine, source);
  size_t sourceSize[] = {strlen(source)};
  const char* src = &source[0];

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &src, sourceSize, &error_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clCreateProgramWithSource failed");

  /* create a cl program executable for all the devices specified */
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);
    return;
  }

  /* get a kernel object handle for a kernel with the given name */
  const char* kernelName = "combine_vload_vstores";
  kernel_ = _wrapper->clCreateKernel(program_, kernelName, &error_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clCreateProgramWithSource failed");

  /*** Set appropriate arguments to the kernel ***/
  /* the input array to the kernel */
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers()[0]);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");

  /* the output array to the kernel */
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&buffers()[1]);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
}

void OCLPerfMemCombine::run(void) {
  size_t globalThreads[1];
  size_t localThreads[1];

  globalThreads[0] = 64;
  localThreads[0] = 64;

  CPerfCounter timer;
  timer.Reset();
  timer.Start();

  for (unsigned int i = 0; i < NUM_ITER; ++i) {
    /*
     * Enqueue a kernel run call.
     */
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[0], kernel_, 1, NULL, globalThreads,
                                              localThreads, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }
  _wrapper->clFinish(cmdQueues_[0]);

  timer.Stop();
  double sec = timer.GetElapsedTime();
  char buf[256];
  SNPRINTF(buf, sizeof(buf), "%d %-8s (sec)", numCombine_, dataType_);
  testDescString = buf;
  _perfInfo = (float)sec;

  checkData(buffers()[1], outSize_, dataRange_, 0);
  return;
}

unsigned int OCLPerfMemCombine::close(void) { return OCLTestImp::close(); }
