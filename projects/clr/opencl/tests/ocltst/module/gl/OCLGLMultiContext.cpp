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

#include "OCLGLMultiContext.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

const static char* strKernel =
    "__kernel void glmulticontext_test( __global uint4 *source, __global uint4 "
    "*dest)   \n"
    "{                                                                         "
    "         \n"
    "    int  tid = get_global_id(0);                                          "
    "         \n"
    "    dest[ tid ] = source[ tid ] + (uint4)(1);                             "
    "         \n"
    "}                                                                         "
    "         \n";

OCLGLMultiContext::OCLGLMultiContext() {
  memset(contextData_, 0, sizeof(contextData_));
  _numSubTests = 1;
}

OCLGLMultiContext::~OCLGLMultiContext() {}

void OCLGLMultiContext::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  cl_context_properties properties[7] = {0};
  for (unsigned int i = 0; i < c_glContextCount; i++) {
    createGLContext(contextData_[i].glContext);
    getCLContextPropertiesFromGLContext(contextData_[i].glContext, properties);

    // Create new CL context from GL context
    contextData_[i].clContext =
        _wrapper->clCreateContext(properties, 1, &devices_[_deviceId], NULL, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateContext() failed (%d)", error_);

    // Create command queue for new context
    contextData_[i].clCmdQueue =
        _wrapper->clCreateCommandQueue(contextData_[i].clContext, devices_[_deviceId], 0, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed (%d)", error_);

    // Build the kernel
    contextData_[i].clProgram = _wrapper->clCreateProgramWithSource(contextData_[i].clContext, 1,
                                                                    &strKernel, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed (%d)", error_);

    error_ = _wrapper->clBuildProgram(contextData_[i].clProgram, 1, &devices_[deviceId], NULL, NULL,
                                      NULL);
    if (error_ != CL_SUCCESS) {
      char programLog[1024];
      _wrapper->clGetProgramBuildInfo(contextData_[i].clProgram, devices_[deviceId],
                                      CL_PROGRAM_BUILD_LOG, 1024, programLog, 0);
      printf("\n%s\n", programLog);
      fflush(stdout);
    }
    CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed (%d)", error_);

    contextData_[i].clKernel =
        _wrapper->clCreateKernel(contextData_[i].clProgram, "glmulticontext_test", &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed (%d)", error_);
  }
}

void OCLGLMultiContext::run() {
  if (_errorFlag) {
    return;
  }

  cl_uint4 inOutData[c_numOfElements] = {{{0}}};
  cl_uint4 expectedData[c_numOfElements] = {{{0}}};

  // Initialize input data with random values
  for (unsigned int i = 0; i < c_numOfElements; i++) {
    for (unsigned int j = 0; j < sizeof(cl_uint4) / sizeof(cl_uint); j++) {
      inOutData[i].s[j] = (unsigned int)rand();
      expectedData[i].s[j] = inOutData[i].s[j] + c_glContextCount;
    }
  }

  for (unsigned int i = 0; i < c_glContextCount; i++) {
    makeCurrent(contextData_[i].glContext);

    // Generate and Bind in & out OpenGL buffers
    GLuint inGLBuffer = 0, outGLBuffer = 0;
    glGenBuffers(1, &inGLBuffer);
    glGenBuffers(1, &outGLBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, inGLBuffer);
    glBufferData(GL_ARRAY_BUFFER, c_numOfElements * sizeof(cl_uint4), inOutData, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer);
    glBufferData(GL_ARRAY_BUFFER, c_numOfElements * sizeof(cl_uint4), NULL, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glFinish();

    // Create input buffer from GL input buffer
    contextData_[i].inputBuffer = _wrapper->clCreateFromGLBuffer(
        contextData_[i].clContext, CL_MEM_READ_ONLY, inGLBuffer, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create input GL buffer (%d)", error_);

    // Create output buffer from GL output buffer
    contextData_[i].outputBuffer = _wrapper->clCreateFromGLBuffer(
        contextData_[i].clContext, CL_MEM_WRITE_ONLY, outGLBuffer, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create output GL buffer (%d)", error_);

    error_ = _wrapper->clSetKernelArg(contextData_[i].clKernel, 0, sizeof(cl_mem),
                                      &(contextData_[i].inputBuffer));
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);

    error_ = _wrapper->clSetKernelArg(contextData_[i].clKernel, 1, sizeof(cl_mem),
                                      &(contextData_[i].outputBuffer));
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);

    error_ = _wrapper->clEnqueueAcquireGLObjects(contextData_[i].clCmdQueue, 1,
                                                 &(contextData_[i].inputBuffer), 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);

    error_ = _wrapper->clEnqueueAcquireGLObjects(contextData_[i].clCmdQueue, 1,
                                                 &(contextData_[i].outputBuffer), 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);

    size_t gws[1] = {c_numOfElements};
    error_ = _wrapper->clEnqueueNDRangeKernel(contextData_[i].clCmdQueue, contextData_[i].clKernel,
                                              1, NULL, gws, NULL, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed (%d)", error_);

    error_ = _wrapper->clEnqueueReleaseGLObjects(contextData_[i].clCmdQueue, 1,
                                                 &(contextData_[i].inputBuffer), 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReleaseGLObjects failed (%d)", error_);

    error_ = _wrapper->clEnqueueReleaseGLObjects(contextData_[i].clCmdQueue, 1,
                                                 &(contextData_[i].outputBuffer), 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReleaseGLObjects failed (%d)", error_);

    error_ = _wrapper->clFinish(contextData_[i].clCmdQueue);
    CHECK_RESULT((error_ != CL_SUCCESS), "clFinish() failed (%d)", error_);

    glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer);
    void* glMem = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    memcpy(inOutData, glMem, c_numOfElements * sizeof(cl_uint4));
    glUnmapBuffer(GL_ARRAY_BUFFER);

    _wrapper->clReleaseMemObject(contextData_[i].inputBuffer);
    _wrapper->clReleaseMemObject(contextData_[i].outputBuffer);

    // Delete GL buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &inGLBuffer);
    inGLBuffer = 0;
    glDeleteBuffers(1, &outGLBuffer);
    outGLBuffer = 0;
  }

  // Compare expected output with actual data received
  for (unsigned int i = 0; i < c_numOfElements; i++) {
    for (unsigned int j = 0; j < sizeof(cl_uint4) / sizeof(cl_uint); j++) {
      CHECK_RESULT((inOutData[i].s[j] != expectedData[i].s[j]),
                   "Element %d is incorrect!\n\t \
                                                                       expected:{%d, %d, %d, %d} differs from actual:{%d, %d, %d, %d}",
                   i, expectedData[i].s[0], expectedData[i].s[1], expectedData[i].s[2],
                   expectedData[i].s[3], inOutData[i].s[0], inOutData[i].s[1], inOutData[i].s[2],
                   inOutData[i].s[3]);
    }
  }
}

unsigned int OCLGLMultiContext::close() {
  for (unsigned int i = 0; i < c_glContextCount; i++) {
    makeCurrent(contextData_[i].glContext);
    _wrapper->clReleaseKernel(contextData_[i].clKernel);
    _wrapper->clReleaseProgram(contextData_[i].clProgram);
    _wrapper->clReleaseCommandQueue(contextData_[i].clCmdQueue);
    _wrapper->clReleaseContext(contextData_[i].clContext);
    destroyGLContext(contextData_[i].glContext);
  }
  return OCLGLCommon::close();
}
