/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLEmptyDeviceEnqueue.h"

#include <stdio.h>
#include <string.h>

#include <vector>

#include "CL/cl.h"

static const cl_uint TotalElements = 128;
static cl_uint hostArray[TotalElements];

#define KERNEL_CODE(...) #__VA_ARGS__

// Sub-test 0: enqueue in a runtime-never-taken branch (carries hidden args at -O0).
// Sub-test 1: plain kernel with out-of-line helper, no enqueue (faithful BurnIn-style).
// Sub-test 2: real device enqueue (regression guard, run with a device queue).
static const char* strKernel[] = {
    KERNEL_CODE(
    \n void child_fn(int tid, __global uint* res) { res[tid] = 0xC0FFEEu; }

        __kernel void empty_enqueue_deadbranch(__global uint* res, uint sentinel) {
          int tid = get_global_id(0);
          res[tid] = (uint)tid;
          if (sentinel == 0xDEADBEEFu) {
            void (^kernelBlock)(void) = ^{ child_fn(tid, res); };
            queue_t def_q = get_default_queue();
            ndrange_t ndrange = ndrange_1D(1);
            if (enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, kernelBlock) != 0) {
              res[tid] = 0xBADu;
            }
          }
        }
    \n),
    KERNEL_CODE(
    \n uint helper(uint x) { return x * 7u - 21u; }

        __kernel void empty_enqueue_plain(__global uint* res) {
          int tid = get_global_id(0);
          res[tid] = helper((uint)tid);
        }
    \n),
    KERNEL_CODE(
    \n void child_fn(int tid, __global uint* res) { res[tid] = 0u; }

        __kernel void real_enqueue(__global uint* res) {
          int tid = get_global_id(0);
          void (^kernelBlock)(void) = ^{ child_fn(tid, res); };
          res[tid] = 0xFFFFFFFFu;
          queue_t def_q = get_default_queue();
          ndrange_t ndrange = ndrange_1D(1);
          if (enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock) != 0) {
            res[tid] = 0xBADu;
          }
        }
    \n)};

static const char* kernelName[] = {"empty_enqueue_deadbranch", "empty_enqueue_plain",
                                   "real_enqueue"};
static const char* buildOpts[] = {"-cl-std=CL2.0 -cl-opt-disable",
                                  "-cl-std=CL2.0 -cl-opt-disable", "-cl-std=CL2.0"};

OCLEmptyDeviceEnqueue::OCLEmptyDeviceEnqueue() {
  _numSubTests = 3;
  deviceQueue_ = NULL;
  skip_ = false;
  testID_ = 0;
}

OCLEmptyDeviceEnqueue::~OCLEmptyDeviceEnqueue() {}

bool OCLEmptyDeviceEnqueue::hasDeviceEnqueueMetadata(unsigned int deviceId) {
  cl_uint numDevices = 0;
  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_NUM_DEVICES, sizeof(numDevices),
                                      &numDevices, NULL);
  if (error_ != CL_SUCCESS || numDevices == 0) return false;

  std::vector<size_t> sizes(numDevices, 0);
  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                                      numDevices * sizeof(size_t), sizes.data(), NULL);
  if (error_ != CL_SUCCESS) return false;

  std::vector<std::vector<unsigned char> > storage(numDevices);
  std::vector<unsigned char*> binaries(numDevices, NULL);
  for (cl_uint i = 0; i < numDevices; ++i) {
    storage[i].resize(sizes[i] ? sizes[i] : 1);
    binaries[i] = storage[i].data();
  }
  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                                      numDevices * sizeof(unsigned char*), binaries.data(), NULL);
  if (error_ != CL_SUCCESS) return false;

  std::vector<cl_device_id> progDevices(numDevices);
  error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_DEVICES,
                                      numDevices * sizeof(cl_device_id), progDevices.data(), NULL);
  if (error_ != CL_SUCCESS) return false;

  static const char* markers[] = {"hidden_default_queue", "hidden_completion_action"};
  const size_t numMarkers = sizeof(markers) / sizeof(markers[0]);
  for (cl_uint i = 0; i < numDevices; ++i) {
    if (progDevices[i] != devices_[deviceId]) continue;
    const unsigned char* buf = binaries[i];
    const size_t len = sizes[i];
    for (size_t m = 0; m < numMarkers; ++m) {
      const size_t mlen = strlen(markers[m]);
      if (len < mlen) continue;
      for (size_t off = 0; off + mlen <= len; ++off) {
        if (memcmp(buf + off, markers[m], mlen) == 0) return true;
      }
    }
  }
  return false;
}

void OCLEmptyDeviceEnqueue::open(unsigned int test, char* units, double& conversion,
                                 unsigned int deviceId) {
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;

  // Device-side enqueue requires OpenCL >= 2.0.
  size_t param_size = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo(CL_DEVICE_VERSION) failed");
  std::vector<char> strVersion(param_size);
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, param_size,
                                     strVersion.data(), 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo(CL_DEVICE_VERSION) failed");
  if (strVersion[7] < '2') {  // "OpenCL X.Y ..." -> index 7 is the major digit.
    skip_ = true;
    return;
  }

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel[test], NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource() failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], buildOpts[test], NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  // Strict precondition for the no-queue O0 sub-tests: the kernel must actually
  // carry the hidden device-enqueue ABI args, otherwise the test is meaningless.
  if (test == 0 || test == 1) {
    const bool hasMeta = hasDeviceEnqueueMetadata(deviceId);
    // Distinguish a failed binary query from genuinely-absent metadata so the
    // failure message is actionable. hasDeviceEnqueueMetadata leaves error_ set
    // to the last clGetProgramInfo status (CL_SUCCESS on the found/not-found
    // paths, the error code if a query failed).
    CHECK_RESULT((error_ != CL_SUCCESS),
                 "clGetProgramInfo() failed during device-enqueue metadata scan");
    CHECK_RESULT(!hasMeta,
                 "Kernel does not carry hidden device-enqueue ABI args; precondition not met");
  }

  kernel_ = _wrapper->clCreateKernel(program_, kernelName[test], &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  memset(hostArray, 0xee, sizeof(hostArray));
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                                    sizeof(hostArray), &hostArray, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

  // Only the regression sub-test creates a default on-device queue. Sub-tests 0
  // and 1 intentionally leave deviceQueue_ == NULL (no default device queue).
  if (test == 2) {
    const cl_queue_properties cprops[] = {
        CL_QUEUE_PROPERTIES,
        static_cast<cl_queue_properties>(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                         CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_ON_DEVICE),
        CL_QUEUE_SIZE, 256 * 1024, 0};
    deviceQueue_ = _wrapper->clCreateCommandQueueWithProperties(context_, devices_[deviceId],
                                                                cprops, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");
  }
}

void OCLEmptyDeviceEnqueue::run(void) {
  if (type_ == CL_DEVICE_TYPE_CPU) return;
  if (skip_) return;

  cl_mem buffer = buffers()[0];
  size_t gws[1] = {TotalElements};

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg(0) failed");

  if (testID_ == 0) {
    cl_uint sentinel = 0u;  // != 0xDEADBEEF -> the enqueue branch is never taken.
    error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_uint), &sentinel);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg(1) failed");
  }

  cl_uint* host = reinterpret_cast<cl_uint*>(_wrapper->clEnqueueMapBuffer(
      cmdQueues_[_deviceId], buffer, CL_TRUE, (CL_MAP_READ | CL_MAP_WRITE), 0,
      TotalElements * sizeof(cl_uint), 0, NULL, NULL, &error_));
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer() failed");

  // The key assertion: dispatch must succeed even with no default device queue
  // for sub-tests 0 and 1 (PR #7178). Local size NULL lets the runtime choose.
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clFinish() failed");

  for (cl_uint i = 0; i < TotalElements; ++i) {
    cl_uint expected;
    if (testID_ == 0) {
      expected = i;
    } else if (testID_ == 1) {
      expected = (cl_uint)(i * 7u - 21u);
    } else {
      expected = 0u;
    }
    if (host[i] != expected) {
      printf("Bad value: res[%u] = 0x%x (expected 0x%x)\n", i, host[i], expected);
      CHECK_RESULT(true, "Incorrect result");
    }
  }

  error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], buffer, host, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueUnmapMemObject() failed");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
}

unsigned int OCLEmptyDeviceEnqueue::close(void) {
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return 0;
  }
  if (NULL != deviceQueue_) {
    _wrapper->clReleaseCommandQueue(deviceQueue_);
    deviceQueue_ = NULL;
  }
  return OCLTestImp::close();
}
