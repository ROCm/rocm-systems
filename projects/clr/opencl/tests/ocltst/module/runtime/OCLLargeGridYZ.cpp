/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLLargeGridYZ.h"

#include <stdio.h>
#include <string.h>

#include <vector>

#include "CL/cl.h"

// gfx12 can't address more than 2^16 work-groups in Y or Z, so the runtime must reject
// such launches instead of silently skipping work.
static const size_t MaxGroupsYZ = 0x10000;
static const size_t BufSize = 100000;

const static char* strKernel =
    "__kernel void fillY(__global uint* out)                                \n"
    "{                                                                      \n"
    "   out[get_global_id(1)] = get_global_id(1);                           \n"
    "}                                                                      \n"
    "                                                                       \n"
    "__kernel void fillZ(__global uint* out)                                \n"
    "{                                                                      \n"
    "   out[get_global_id(2)] = get_global_id(2);                           \n"
    "}                                                                      \n";

OCLLargeGridYZ::OCLLargeGridYZ() {
  _numSubTests = 1;
  isGfx12_ = false;
}

OCLLargeGridYZ::~OCLLargeGridYZ() {}

void OCLLargeGridYZ::open(unsigned int test, char* units, double& conversion,
                          unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  char name[128] = {0};
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_NAME, sizeof(name), name, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_NAME failed");
  isGfx12_ = (strncmp(name, "gfx12", 5) == 0);
  if (!isGfx12_) {
    printf("Skipping test: the work-group count limit only applies to gfx12+\n");
    return;
  }

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource() failed");

  // Build without -cl-std=CL2.0 so work-groups must be uniform. That makes the runtime pick
  // local size 1 for a prime global size when the app passes no local size.
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  cl_mem buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, BufSize * sizeof(cl_uint),
                                           NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

bool OCLLargeGridYZ::enqueue(cl_kernel kernel, cl_uint dims, const size_t* gws, const size_t* lws,
                             cl_int expected) {
  cl_mem buffer = buffers()[0];
  error_ = _wrapper->clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
  if (error_ != CL_SUCCESS) {
    printf("clSetKernelArg() failed (%d)\n", error_);
    return false;
  }

  if (expected == CL_SUCCESS) {
    // Fill with a pattern no work-item writes, so skipped work-items are detected.
    std::vector<cl_uint> pattern(BufSize, 0xdeadbeef);
    error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                            BufSize * sizeof(cl_uint), pattern.data(), 0, NULL,
                                            NULL);
    if (error_ != CL_SUCCESS) {
      printf("clEnqueueWriteBuffer() failed (%d)\n", error_);
      return false;
    }
  }

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel, dims, NULL, gws, lws, 0,
                                            NULL, NULL);
  if (error_ != expected) {
    printf("clEnqueueNDRangeKernel() returned %d, expected %d\n", error_, expected);
    return false;
  }
  return true;
}

bool OCLLargeGridYZ::verify(size_t count) {
  std::vector<cl_uint> values(BufSize);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers()[0], CL_TRUE, 0,
                                         BufSize * sizeof(cl_uint), values.data(), 0, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    printf("clEnqueueReadBuffer() failed (%d)\n", error_);
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (values[i] != i) {
      printf("Mismatch at %zu: got 0x%x\n", i, values[i]);
      return false;
    }
  }
  return true;
}

void OCLLargeGridYZ::run(void) {
  if (!isGfx12_) return;

  cl_kernel fillY = _wrapper->clCreateKernel(program_, "fillY", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel(fillY) failed");
  cl_kernel fillZ = _wrapper->clCreateKernel(program_, "fillZ", &error_);
  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clCreateKernel(fillZ) failed");

  const size_t lws2[2] = {1, 1};
  const size_t lws3[3] = {1, 1, 1};

  // One work-group over the limit in Y and Z must be rejected.
  const size_t gwsY[2] = {1, MaxGroupsYZ + 1};
  CHECK_RESULT_NO_RETURN(!enqueue(fillY, 2, gwsY, lws2, CL_INVALID_GLOBAL_WORK_SIZE),
                         "Y over the work-group limit wasn't rejected");
  const size_t gwsZ[3] = {1, 1, MaxGroupsYZ + 1};
  CHECK_RESULT_NO_RETURN(!enqueue(fillZ, 3, gwsZ, lws3, CL_INVALID_GLOBAL_WORK_SIZE),
                         "Z over the work-group limit wasn't rejected");

  // 65537 is prime, so the runtime-selected local size is 1 and the launch must be rejected.
  CHECK_RESULT_NO_RETURN(!enqueue(fillY, 2, gwsY, NULL, CL_INVALID_GLOBAL_WORK_SIZE),
                         "Y over the limit with runtime local size wasn't rejected");

  // Exactly at the limit is valid and must write every row.
  const size_t gwsMax[2] = {1, MaxGroupsYZ};
  CHECK_RESULT_NO_RETURN(!enqueue(fillY, 2, gwsMax, lws2, CL_SUCCESS) || !verify(MaxGroupsYZ),
                         "Y at the work-group limit failed");

  // More work-items than the limit is fine when the work-group count stays within it.
  const size_t gwsBig[2] = {1, BufSize};
  const size_t lwsBig[2] = {1, 2};
  CHECK_RESULT_NO_RETURN(!enqueue(fillY, 2, gwsBig, lwsBig, CL_SUCCESS) || !verify(BufSize),
                         "Y with large local size failed");

  _wrapper->clReleaseKernel(fillZ);
  _wrapper->clReleaseKernel(fillY);
}

unsigned int OCLLargeGridYZ::close(void) { return OCLTestImp::close(); }
