/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLPerfCommandQueue.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <sstream>
#include <string>

#include "CL/cl.h"
#include "CL/cl_ext.h"

static const size_t BufSize = 0x1000;
static const size_t Iterations = 0x100;
static const size_t TotalQueues = 4;
static const size_t TotalBufs = 4;

OCLPerfCommandQueue::OCLPerfCommandQueue() {
  _numSubTests = TotalQueues * TotalBufs;
  failed_ = false;
}

OCLPerfCommandQueue::~OCLPerfCommandQueue() {}

void OCLPerfCommandQueue::open(unsigned int test, char* units, double& conversion,
                               unsigned int deviceId) {
  cl_mem buffer;
  _deviceId = deviceId;
  CPerfCounter timer;
  timer.Reset();
  timer.Start();

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  timer.Stop();
  if (test == 0) {
    printf("Runtime load/init time: %0.2f ms\n", static_cast<float>(timer.GetElapsedTime() * 1000));
  }
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
  static const size_t MemObjects[] = {1, 100, 1000, 5000};
  size_t numMems = MemObjects[test_ / TotalBufs];
  size_t bufSize = BufSize * sizeof(cl_int4);
  for (size_t b = 0; b < numMems; ++b) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bufSize, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfCommandQueue::run(void) {
  if (failed_) {
    return;
  }
  unsigned int* values;
  values = reinterpret_cast<unsigned int*>(new cl_int4[BufSize]);
  CPerfCounter timer;
  static const size_t Queues[] = {1, 2, 4, 8};
  size_t numQueues = Queues[test_ % TotalQueues];

  // Clear destination buffer
  memset(values, 0, BufSize * sizeof(cl_int4));

  size_t iter = Iterations / (numQueues * ((size_t)1 << (test_ / TotalBufs + 1)));
  std::vector<cl_command_queue> cmdQueues(numQueues);

  timer.Reset();
  timer.Start();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t q = 0; q < numQueues; ++q) {
      cl_command_queue cmdQueue =
          _wrapper->clCreateCommandQueue(context_, devices_[_deviceId], 0, &error_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed");
      cmdQueues[q] = cmdQueue;
    }
    timer.Stop();
    for (size_t q = 0; q < numQueues; ++q) {
      for (size_t b = 0; b < buffers_.size(); ++b) {
        error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues[q], buffers_[b], CL_TRUE, 0,
                                                sizeof(cl_int4), values, 0, NULL, NULL);
      }
    }
    timer.Start();
    for (size_t q = 0; q < numQueues; ++q) {
      error_ = _wrapper->clReleaseCommandQueue(cmdQueues[q]);
      CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseCommandQueue() failed");
    }
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
  }

  timer.Stop();

  std::stringstream stream;

  stream << "Create+destroy time for " << numQueues << " queues and " << buffers_.size()
         << " buffers";
  stream.precision(3);
  stream.width(5);
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream << "(ms)";
  testDescString = stream.str();
  _perfInfo = static_cast<float>(timer.GetElapsedTime() * 1000 / (iter * numQueues));
  delete[] values;
}

unsigned int OCLPerfCommandQueue::close(void) { return OCLTestImp::close(); }
