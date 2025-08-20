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

#include "OCLGLBuffer.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

const static char* strKernel =
    "__kernel void glbuffer_test( __global uint4 *source, __global uint4 "
    "*glDest, __global uint4 *clDest)   \n"
    "{                                                                         "
    "                             \n"
    "    int  tid = get_global_id(0);                                          "
    "                             \n"
    "    clDest[ tid ] = source[ tid ] + (uint4)(1);                           "
    "                             \n"
    "    glDest[ tid ] = source[ tid ] + (uint4)(2);                           "
    "                             \n"
    "}                                                                         "
    "                             \n";

OCLGLBuffer::OCLGLBuffer() : inGLBuffer_(0), outGLBuffer_(0) { _numSubTests = 1; }

OCLGLBuffer::~OCLGLBuffer() {}

void OCLGLBuffer::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  // Build the kernel
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed (%d)", error_);

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed (%d)", error_);

  kernel_ = _wrapper->clCreateKernel(program_, "glbuffer_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed (%d)", error_);
}

void OCLGLBuffer::run(void) {
  if (_errorFlag) {
    return;
  }

  cl_mem buffer;
  cl_uint4 inData[c_numOfElements] = {{{0}}};
  cl_uint4 outDataCL[c_numOfElements] = {{{0}}};
  cl_uint4 outDataGL[c_numOfElements] = {{{0}}};

  // Initialize input data with random values
  for (unsigned int i = 0; i < c_numOfElements; i++) {
    for (unsigned int j = 0; j < sizeof(cl_uint4) / sizeof(cl_uint); j++) {
      inData[i].s[j] = (unsigned int)rand();
    }
  }

  // Generate and Bind in & out OpenGL buffers
  glGenBuffers(1, &inGLBuffer_);
  glGenBuffers(1, &outGLBuffer_);

  glBindBuffer(GL_ARRAY_BUFFER, inGLBuffer_);
  glBufferData(GL_ARRAY_BUFFER, c_numOfElements * sizeof(cl_uint4), inData, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer_);
  glBufferData(GL_ARRAY_BUFFER, c_numOfElements * sizeof(cl_uint4), outDataGL, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glFinish();

  // Create input buffer from GL input buffer
  buffer = _wrapper->clCreateFromGLBuffer(context_, CL_MEM_READ_ONLY, inGLBuffer_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create input GL buffer (%d)", error_);
  buffers_.push_back(buffer);

  // Create output buffer from GL output buffer
  buffer = _wrapper->clCreateFromGLBuffer(context_, CL_MEM_WRITE_ONLY, outGLBuffer_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create output GL buffer (%d)", error_);
  buffers_.push_back(buffer);

  // Create a CL output buffer
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, c_numOfElements * sizeof(cl_uint4),
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed (%d)", error_);
  buffers_.push_back(buffer);

  // Assign args and execute
  for (unsigned int i = 0; i < buffers_.size(); i++) {
    error_ = _wrapper->clSetKernelArg(kernel_, i, sizeof(cl_mem), &buffers()[i]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);
  }

  error_ =
      _wrapper->clEnqueueAcquireGLObjects(cmdQueues_[_deviceId], 2, &buffers()[0], 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);

  size_t gws[1] = {c_numOfElements};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed (%d)", error_);

  error_ =
      _wrapper->clEnqueueReleaseGLObjects(cmdQueues_[_deviceId], 2, &buffers()[0], 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReleaseGLObjects failed (%d)", error_);

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clFinish() failed (%d)", error_);

  // Get the results from both CL and GL buffers
  error_ =
      _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers()[2], CL_TRUE, 0,
                                    c_numOfElements * sizeof(cl_uint4), outDataCL, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "Unable to read output CL array! (%d)", error_);

  glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer_);
  void* glMem = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
  memcpy(outDataGL, glMem, c_numOfElements * sizeof(cl_uint4));
  glUnmapBuffer(GL_ARRAY_BUFFER);

  cl_uint4 expectedCL = {{0}};
  cl_uint4 expectedGL = {{0}};

  // Check output
  for (unsigned int i = 0; i < c_numOfElements; ++i) {
    // Calculate expected value in CL output buffer (input + 1)
    expectedCL = inData[i];
    expectedCL.s[0]++;
    expectedCL.s[1]++;
    expectedCL.s[2]++;
    expectedCL.s[3]++;

    // Calculate expected value in GL output buffer (input + 2)
    expectedGL = inData[i];
    expectedGL.s[0] += 2;
    expectedGL.s[1] += 2;
    expectedGL.s[2] += 2;
    expectedGL.s[3] += 2;

    // Compare expected output with actual data received
    for (unsigned int j = 0; j < sizeof(cl_uint4) / sizeof(cl_uint); j++) {
      CHECK_RESULT((outDataCL[i].s[j] != expectedCL.s[j]),
                   "Element %d in CL output buffer is incorrect!\n\t \
                         expected:{%d, %d, %d, %d} differs from actual:{%d, %d, %d, %d}",
                   i, expectedCL.s[0], expectedCL.s[1], expectedCL.s[2], expectedCL.s[3],
                   outDataCL[i].s[0], outDataCL[i].s[1], outDataCL[i].s[2], outDataCL[i].s[3]);
      CHECK_RESULT((outDataGL[i].s[j] != expectedGL.s[j]),
                   "Element %d in GL output buffer is incorrect!\n\t \
                         expected:{%d, %d, %d, %d} differs from actual:{%d, %d, %d, %d}",
                   i, expectedGL.s[0], expectedGL.s[1], expectedGL.s[2], expectedGL.s[3],
                   outDataGL[i].s[0], outDataGL[i].s[1], outDataGL[i].s[2], outDataGL[i].s[3]);
    }
  }
}

unsigned int OCLGLBuffer::close(void) {
  for (unsigned int i = 0; i < buffers().size(); ++i) {
    clReleaseMemObject(buffers()[i]);
  }
  buffers_.clear();

  // Delete GL in & out buffers
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDeleteBuffers(1, &inGLBuffer_);
  inGLBuffer_ = 0;
  glDeleteBuffers(1, &outGLBuffer_);
  outGLBuffer_ = 0;

  return OCLGLCommon::close();
}
