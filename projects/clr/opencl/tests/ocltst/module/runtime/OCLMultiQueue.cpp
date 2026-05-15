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

#include "OCLMultiQueue.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <sstream>
#include <string>

#include "CL/cl.h"

const static char* strKernel =
    "__kernel void                                     \n"
    "copyInc(__global uint* dst, __global uint* src)   \n"
    "{                                                 \n"
    "    uint index = get_global_id(0);                \n"
    "                                                  \n"
    "    dst[index] = src[index] + 1;                  \n"
    "}                                                 \n";

static bool useGPU = true;

static const cl_uint NumQueues = 8;  // must be power of 2
static cl_uint NumElements = 4096;
#if EMU_ENV
static const cl_uint NumRuns = 16;
#else
static const cl_uint NumRuns = 16384;
#endif

static const cl_uint ExecutionsPerQueue = 256;
std::stringstream lerror;

class MemTransfer {
 public:
  MemTransfer(OCLWrapper* wrapper, cl_context context, cl_command_queue queue, cl_uint numElements)
      : wrapper_(wrapper), context_(context), queue_(queue), numElements_(numElements), count_(0) {}

  ~MemTransfer() {
    wrapper_->clReleaseMemObject(dst_);
    wrapper_->clReleaseMemObject(src_);
  }

  bool create() {
    cl_int err;
    size_t size = numElements_ * sizeof(cl_uint);
    cl_uint* data = new cl_uint[numElements_];
    memset(data, 0, size);

    src_ = wrapper_->clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR, size, data, &err);
    if (src_ == NULL) {
      lerror << "clReleaseContext failed";
      delete[] data;
      return false;
    }
    dst_ = wrapper_->clCreateBuffer(context_, 0, size, NULL, &err);
    if (dst_ == NULL) {
      lerror << "clCreateBuffer() failed";
      delete[] data;
      return false;
    }

    delete[] data;
    return true;
  }

  bool run(cl_kernel kernel) {
    size_t global_work_size[1];
    size_t local_work_size[1];
    size_t size = numElements_ * sizeof(cl_uint);

    global_work_size[0] = (numElements_ + 63) / 64 * 64;
    local_work_size[0] = 64;

    if (CL_SUCCESS != wrapper_->clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&dst_)) {
      return false;
    }

    if (CL_SUCCESS != wrapper_->clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&src_)) {
      return false;
    }

    if (CL_SUCCESS !=
        wrapper_->clEnqueueNDRangeKernel(queue_, kernel, 1, NULL, (const size_t*)global_work_size,
                                         (const size_t*)local_work_size, 0, NULL, NULL)) {
      lerror << "clEnqueueNDRangeKernel() failed";
      return false;
    }

    // Copy dst into src
    if (CL_SUCCESS != wrapper_->clEnqueueCopyBuffer(queue_, dst_, src_, 0, 0, size, 0, 0, NULL)) {
      lerror << "clEnqueueCopyBuffer() failed";
      return false;
    }
    count_++;
    return true;
  }

  bool check() {
    size_t size = numElements_ * sizeof(cl_uint);
    cl_event event;
    void* ptr = wrapper_->clEnqueueMapBuffer(queue_, src_, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL,
                                             NULL, NULL);
    cl_uint* data = reinterpret_cast<cl_uint*>(ptr);

    for (cl_uint i = 0; i < numElements_; ++i) {
      if (data[i] != count_) {
        return false;
      }
    }
    wrapper_->clEnqueueUnmapMemObject(queue_, src_, ptr, 0, NULL, &event);
    wrapper_->clWaitForEvents(1, &event);
    wrapper_->clReleaseEvent(event);
    return true;
  }

  void flush() { wrapper_->clFlush(queue_); }

 private:
  OCLWrapper* wrapper_;
  cl_context context_;
  cl_command_queue queue_;
  cl_uint numElements_;
  cl_uint count_;
  cl_mem dst_;
  cl_mem src_;
};

MemTransfer* work[NumQueues];

bool test(cl_kernel, cl_uint, cl_uint);

OCLMultiQueue::OCLMultiQueue() {
  _numSubTests = 0;
  for (cl_uint i = 1; i <= NumQueues; i <<= 1, _numSubTests++);
  failed_ = false;
}

OCLMultiQueue::~OCLMultiQueue() {}

void OCLMultiQueue::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  test_ = test;
  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    testDescString = "GPU device is required for this test!\n";
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
  NumElements = (NumElements < static_cast<size_t>(computePower))
                    ? static_cast<size_t>(computePower)
                    : NumElements;
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
  kernel_ = _wrapper->clCreateKernel(program_, "copyInc", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");
}

void OCLMultiQueue::run(void) {
  if (failed_) {
    return;
  }

  // Run test
  cl_uint queues = 1 << test_;
  if (!test(kernel_, NumRuns / queues, queues)) {
    lerror << "We failed a test run!";
    CHECK_RESULT(true, lerror.str().c_str());
  }
}

unsigned int OCLMultiQueue::close(void) { return OCLTestImp::close(); }

bool OCLMultiQueue::test(cl_kernel kernel, cl_uint numRuns, cl_uint numQueues) {
  cl_command_queue cmd_queue[NumQueues];
  CPerfCounter timer;

  for (cl_uint i = 0; i < numQueues; ++i) {
    cmd_queue[i] = _wrapper->clCreateCommandQueue(context_, devices_[_deviceId], 0, &error_);
    if (cmd_queue[i] == (cl_command_queue)0) {
      _wrapper->clReleaseContext(context_);
      testDescString = "clCreateCommandQueue() failed";
      return false;
    }
    work[i] = new MemTransfer(_wrapper, context_, cmd_queue[i], NumElements);
    if (work[i] == NULL || !work[i]->create()) {
      testDescString = "Test creation failed";
      return false;
    }
  }

  timer.Reset();
  timer.Start();

  cl_uint dispatchCount = ExecutionsPerQueue / numQueues;
  for (cl_uint i = 0; i < numRuns; ++i) {
    for (cl_uint j = 0; j < numQueues; ++j) {
      if (!work[j]->run(kernel)) {
        testDescString = "Execution failed";
        return false;
      }
      // Every queue should have a dispatch after 256 executions,
      // but the time for dispatch on each queue
      // will be shifted on dispatchCount
      if (((i % dispatchCount) == 0) && (((i / dispatchCount) % numQueues) == j)) {
        work[j]->flush();
      }
    }
  }

  for (cl_uint i = 0; i < numQueues; ++i) {
    _wrapper->clFinish(cmd_queue[i]);
  }

  timer.Stop();

  for (cl_uint j = 0; j < numQueues; ++j) {
    if (!work[j]->check()) {
      testDescString = "Result Check fails!";
      return false;
    }
  }
  std::stringstream stream;

  stream << "Num Queues: " << numQueues << ", Executions Per Queue: ";
  stream.flags(std::ios::right | std::ios::showbase);
  stream.width(5);
  stream << numRuns;
  stream.precision(3);
  stream << ", Time: " << (float)(timer.GetElapsedTime()) << " seconds";

  for (cl_uint i = 0; i < numQueues; ++i) {
    delete work[i];
    _wrapper->clReleaseCommandQueue(cmd_queue[i]);
  }
  testDescString = stream.str();

  return true;
}
