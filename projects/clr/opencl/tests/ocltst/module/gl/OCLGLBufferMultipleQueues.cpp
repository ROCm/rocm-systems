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

#include "OCLGLBufferMultipleQueues.h"

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
    "    glDest[ tid ] = source[ tid ] + (uint4)(2);                           "
    "                             \n"
    "    clDest[ tid ] = source[ tid ] + (uint4)(1);                           "
    "                             \n"
    "}                                                                         "
    "                             \n";

OCLGLBufferMultipleQueues::OCLGLBufferMultipleQueues() { _numSubTests = 1; }

OCLGLBufferMultipleQueues::~OCLGLBufferMultipleQueues() {}

void OCLGLBufferMultipleQueues::open(unsigned int test, char* units, double& conversion,
                                     unsigned int deviceId) {
  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  // Create multiple queues for the device (first add already created queue in
  // OCLGLCommon::open, then add a second queue)
  deviceCmdQueues_.resize(QUEUES_PER_DEVICE_COUNT);
  deviceCmdQueues_[0] = cmdQueues_[deviceId];
  for (int queueIndex = 1; queueIndex < QUEUES_PER_DEVICE_COUNT; queueIndex++) {
    cl_command_queue cmdQueue =
        _wrapper->clCreateCommandQueue(context_, devices_[deviceId], 0, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed");
    deviceCmdQueues_[queueIndex] = cmdQueue;
  }

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

void OCLGLBufferMultipleQueues::run(void) {
  if (_errorFlag) {
    return;
  }

  inputGLBufferPerQueue_.resize(QUEUES_PER_DEVICE_COUNT, NULL);
  outputGLBufferPerQueue_.resize(QUEUES_PER_DEVICE_COUNT, NULL);
  outputCLBufferPerQueue_.resize(QUEUES_PER_DEVICE_COUNT, NULL);

  std::vector<std::vector<cl_uint4> > inData(QUEUES_PER_DEVICE_COUNT);  // Input data per queue

  inGLBufferIDs_.resize(QUEUES_PER_DEVICE_COUNT, 0);
  outGLBufferIDs_.resize(QUEUES_PER_DEVICE_COUNT, 0);
  for (int queueIndex = 0; queueIndex < QUEUES_PER_DEVICE_COUNT; queueIndex++) {
    // Initialize input data with random values
    inData[queueIndex].resize(BUFFER_ELEMENTS_COUNT);
    for (int i = 0; i < BUFFER_ELEMENTS_COUNT; i++) {
      for (unsigned int j = 0; j < sizeof(cl_uint4) / sizeof(cl_uint); j++) {
        inData[queueIndex][i].s[j] = (unsigned int)rand();
      }
    }

    // Generate and Bind in & out OpenGL buffers
    glGenBuffers(1, &inGLBufferIDs_[queueIndex]);
    glGenBuffers(1, &outGLBufferIDs_[queueIndex]);

    glBindBuffer(GL_ARRAY_BUFFER, inGLBufferIDs_[queueIndex]);
    glBufferData(GL_ARRAY_BUFFER, BUFFER_ELEMENTS_COUNT * sizeof(cl_uint4), &inData[queueIndex][0],
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, outGLBufferIDs_[queueIndex]);
    glBufferData(GL_ARRAY_BUFFER, BUFFER_ELEMENTS_COUNT * sizeof(cl_uint4), NULL, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glFinish();

    // Create input buffer from GL input buffer
    inputGLBufferPerQueue_[queueIndex] = _wrapper->clCreateFromGLBuffer(
        context_, CL_MEM_READ_ONLY, inGLBufferIDs_[queueIndex], &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create input GL buffer (%d)", error_);

    // Create output buffer from GL output buffer
    outputGLBufferPerQueue_[queueIndex] = _wrapper->clCreateFromGLBuffer(
        context_, CL_MEM_WRITE_ONLY, outGLBufferIDs_[queueIndex], &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create output GL buffer (%d)", error_);

    // Create a CL output buffer
    outputCLBufferPerQueue_[queueIndex] = _wrapper->clCreateBuffer(
        context_, CL_MEM_WRITE_ONLY, BUFFER_ELEMENTS_COUNT * sizeof(cl_uint4), NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed (%d)", error_);
  }

  for (int queueIndex = 0; queueIndex < QUEUES_PER_DEVICE_COUNT; queueIndex++) {
    // Assign arguments to kernel according to queue index
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem),
                                      &inputGLBufferPerQueue_[queueIndex]);  // Input source
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);
    error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem),
                                      &outputGLBufferPerQueue_[queueIndex]);  // Output glDest
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);
    error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_mem),
                                      &outputCLBufferPerQueue_[queueIndex]);  // Output clDest
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);

    // Acquire input GL buffer
    error_ = _wrapper->clEnqueueAcquireGLObjects(
        deviceCmdQueues_[queueIndex], 1, &inputGLBufferPerQueue_[queueIndex], 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);

    // Acquire output GL buffer
    error_ = _wrapper->clEnqueueAcquireGLObjects(
        deviceCmdQueues_[queueIndex], 1, &outputGLBufferPerQueue_[queueIndex], 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);

    // Enqueue the kernel
    size_t gws[1] = {BUFFER_ELEMENTS_COUNT};
    error_ = _wrapper->clEnqueueNDRangeKernel(deviceCmdQueues_[queueIndex], kernel_, 1, NULL, gws,
                                              NULL, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed (%d)", error_);

    // Release input GL buffer
    error_ = _wrapper->clEnqueueReleaseGLObjects(
        deviceCmdQueues_[queueIndex], 1, &inputGLBufferPerQueue_[queueIndex], 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReleaseGLObjects failed (%d)", error_);

    // Release output GL buffer
    error_ = _wrapper->clEnqueueReleaseGLObjects(
        deviceCmdQueues_[queueIndex], 1, &outputGLBufferPerQueue_[queueIndex], 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReleaseGLObjects failed (%d)", error_);

    // Flush commands in order to trigger the operations
    error_ = _wrapper->clFlush(deviceCmdQueues_[queueIndex]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clFlush() failed (%d)", error_);
  }

  for (int queueIndex = 0; queueIndex < QUEUES_PER_DEVICE_COUNT; queueIndex++) {
    // Get the results from CL buffer (in a synchronous manner)
    cl_uint4 outDataCL[BUFFER_ELEMENTS_COUNT];
    error_ = _wrapper->clEnqueueReadBuffer(
        deviceCmdQueues_[queueIndex], outputCLBufferPerQueue_[queueIndex], CL_TRUE, 0,
        BUFFER_ELEMENTS_COUNT * sizeof(cl_uint4), outDataCL, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to read output CL array! (%d)", error_);

    cl_uint4 outDataGL[BUFFER_ELEMENTS_COUNT] = {{{0}}};
    glBindBuffer(GL_ARRAY_BUFFER, outGLBufferIDs_[queueIndex]);  // why again
    void* glMem = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    memcpy(outDataGL, glMem, BUFFER_ELEMENTS_COUNT * sizeof(cl_uint4));
    glUnmapBuffer(GL_ARRAY_BUFFER);

    cl_uint4 expectedCL = {{0}};
    cl_uint4 expectedGL = {{0}};

    // Check output
    for (int i = 0; i < BUFFER_ELEMENTS_COUNT; ++i) {
      // Calculate expected value in CL output buffer (input + 1)
      expectedCL = inData[queueIndex][i];
      expectedCL.s[0]++;
      expectedCL.s[1]++;
      expectedCL.s[2]++;
      expectedCL.s[3]++;

      // Calculate expected value in GL output buffer (input + 2)
      expectedGL = inData[queueIndex][i];
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
}

unsigned int OCLGLBufferMultipleQueues::close(void) {
  // Release cl buffers (must be done before releasing the associated GL
  // buffers)
  for (int bufferIndex = 0; bufferIndex < (int)inputGLBufferPerQueue_.size(); bufferIndex++) {
    error_ = _wrapper->clReleaseMemObject(inputGLBufferPerQueue_[bufferIndex]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseMemObject() failed");
  }

  for (int bufferIndex = 0; bufferIndex < (int)outputGLBufferPerQueue_.size(); bufferIndex++) {
    error_ = _wrapper->clReleaseMemObject(outputGLBufferPerQueue_[bufferIndex]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseMemObject() failed");
  }

  for (int bufferIndex = 0; bufferIndex < (int)outputCLBufferPerQueue_.size(); bufferIndex++) {
    error_ = _wrapper->clReleaseMemObject(outputCLBufferPerQueue_[bufferIndex]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseMemObject() failed");
  }

  // Delete GL in & out buffers
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  if (!inGLBufferIDs_.empty()) {
    glDeleteBuffers((int)inGLBufferIDs_.size(), &inGLBufferIDs_[0]);
  }

  if (!outGLBufferIDs_.empty()) {
    glDeleteBuffers((int)outGLBufferIDs_.size(), &outGLBufferIDs_[0]);
  }

  // Release queues created by open method, the first queue per device is
  // released by base class
  for (int queueIndex = 1; queueIndex < (int)deviceCmdQueues_.size(); queueIndex++) {
    error_ = _wrapper->clReleaseCommandQueue(deviceCmdQueues_[queueIndex]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseCommandQueue() failed");
  }
  deviceCmdQueues_.clear();

  return OCLGLCommon::close();
}
