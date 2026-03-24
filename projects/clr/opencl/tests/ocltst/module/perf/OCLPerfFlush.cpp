/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLPerfFlush.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <sstream>
#include <string>

#include "CL/cl.h"

static const cl_uint Iterations = 0x10000;
static const cl_uint IterationDivider = 2;
static const size_t MaxBuffers = IterationDivider;
static size_t BufSize = 0x1000;

const static char* strKernel =
    "__kernel void factorial(__global uint* out)                        \n"
    "{                                                                  \n"
    "   uint id = get_global_id(0);                                     \n"
    "   uint factorial = 1;                                             \n"
    "   for (uint i = 1; i < (id / 0x10000); ++i)                       \n"
    "   {                                                               \n"
    "       factorial *= i;                                             \n"
    "   }                                                               \n"
    "    out[id] = factorial;                                            \n"
    "}                                                                  \n";

unsigned int NumTests = 3;

OCLPerfFlush::OCLPerfFlush() {
  _numSubTests = NumTests;
  failed_ = false;
}

OCLPerfFlush::~OCLPerfFlush() {}

void OCLPerfFlush::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
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
  size_t maxWorkGroupSize = 1;
  cl_uint computePower = 1;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                     sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
  computePower *= static_cast<cl_uint>(maxWorkGroupSize);
  cl_uint maxComputeUnits = 1;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_MAX_COMPUTE_UNITS,
                                     sizeof(maxComputeUnits), &maxComputeUnits, NULL);
  computePower *= 32 * maxComputeUnits;
  BufSize =
      (BufSize < static_cast<size_t>(computePower)) ? static_cast<size_t>(computePower) : BufSize;
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
  kernel_ = _wrapper->clCreateKernel(program_, "factorial", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  for (size_t i = 0; i < MaxBuffers; ++i) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, BufSize * sizeof(cl_uint), NULL,
                                      &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }
}

void OCLPerfFlush::run(void) {
  if (failed_) {
    return;
  }
  for (size_t y = 0; y < IterationDivider; ++y) {
    cl_mem buffer = buffers()[y];

    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

    size_t gws[1] = {BufSize};
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  CPerfCounter timer;
  const char* descriptions[] = {"Single batch: ", "clFlush():    ", "clFinish():   "};

  timer.Reset();
  timer.Start();
  cl_uint x;
  for (x = 0; x < Iterations / IterationDivider; x++) {
    for (size_t y = 0; y < IterationDivider; ++y) {
      cl_mem buffer = buffers()[y];

      error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
      CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

      size_t gws[1] = {BufSize};
      error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL,
                                                0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
    }
    if (test_ == 1) {
      _wrapper->clFlush(cmdQueues_[_deviceId]);
    } else if (test_ == 2) {
      _wrapper->clFinish(cmdQueues_[_deviceId]);
    }
  }
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  timer.Stop();

  std::stringstream stream;
  stream << "Loop[" << std::hex << Iterations << "], " << descriptions[test_];
  stream << "(sec)";
  testDescString = stream.str();
  _perfInfo = static_cast<float>(timer.GetElapsedTime());
}

unsigned int OCLPerfFlush::close(void) { return OCLTestImp::close(); }
