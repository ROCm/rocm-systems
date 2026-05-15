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

#include "OCLPerfDeviceEnqueueEvent.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define KERNEL_CODE(...) #__VA_ARGS__

typedef struct {
  unsigned int threads;
} testStruct;

static testStruct testList[] = {
    {64}, {128}, {256}, {512}, {1024}, {2048}, {4096},
};

static unsigned int qsizeList[] = {
    16, 32, 64, 128, 256, 512,
};

static unsigned int levelList[] = {
    1,
    2,
    4,
    8,
};

const static char* strKernel = {KERNEL_CODE(
  \n __kernel void childKernel(__global uint* buf, uint level, clk_event_t wait_evt) {
  int idx = get_global_id(0);
  if (idx < 0) {
    buf[idx] = 0;
  }
}
  \n __kernel void parentKernel(__global uint* buf, uint level) {
  if (level) {
    queue_t def_q = get_default_queue();
    ndrange_t ndrange = ndrange_1D(64, 64);
    clk_event_t user_evt = create_user_event();
    clk_event_t block_evt, wait_evt;
    wait_evt = user_evt;

    for (uint i = 0; i < level; i++) {
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 0,
                                   /*&user_evt*/ NULL, &block_evt, ^{
                                     childKernel(buf, level - 1, block_evt);
                                   });

      // wait_evt = block_evt;
    }
    if (is_valid_event(user_evt)) {
      set_user_event_status(user_evt, CL_COMPLETE);
      release_event(user_evt);
    }
  } else {
    int idx = get_global_id(0);
    if (idx < 0) {
      buf[idx] = 0;
    }
  }
}
  \n)};

OCLPerfDeviceEnqueueEvent::OCLPerfDeviceEnqueueEvent() {
  subTests_level = sizeof(levelList) / sizeof(unsigned int);
  subTests_qsize = (sizeof(qsizeList) / sizeof(unsigned int));
  subTests_thread = sizeof(testList) / sizeof(testStruct);
  testListSize = subTests_thread;
  //_numSubTests  = 2*testListSize + subTests_level + subTests_qsize;
  _numSubTests = subTests_level * subTests_qsize * subTests_thread;
  deviceQueue_ = NULL;
  failed_ = false;
  skip_ = false;
  kernel2_ = NULL;
  level = 2;
}

OCLPerfDeviceEnqueueEvent::~OCLPerfDeviceEnqueueEvent() {}

void OCLPerfDeviceEnqueueEvent::open(unsigned int test, char* units, double& conversion,
                                     unsigned int deviceId) {
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;

  threads = testList[testID_ / (subTests_qsize * subTests_level)].threads;
  queueSize = qsizeList[(testID_ / subTests_level) % subTests_qsize] * 1024;
  level = levelList[testID_ % subTests_level];

  lws_value = 64;

  size_t param_size = 0;
  char* strVersion = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strVersion = new char[param_size];
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_VERSION, param_size, strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strVersion[7] < '2') {
    failed_ = true;
    return;
  }
  delete strVersion;

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], "-cl-std=CL2.0", NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "parentKernel", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  kernel2_ = _wrapper->clCreateKernel(program_, "childKernel", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_ALLOC_HOST_PTR, 2048, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

#if defined(CL_VERSION_2_0)
  const cl_queue_properties cprops[] = {
      CL_QUEUE_PROPERTIES,
      static_cast<cl_queue_properties>(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                       CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_ON_DEVICE),
      CL_QUEUE_SIZE, queueSize, 0};
  deviceQueue_ =
      _wrapper->clCreateCommandQueueWithProperties(context_, devices_[deviceId], cprops, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");
#else
  skip_ = true;
  testDescString = "DeviceEnqueue NOT supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfDeviceEnqueueEvent::run(void) {
  CPerfCounter timer;
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return;
  }

  if (failed_) {
    return;
  }

  if (skip_) {
    return;
  }

  cl_mem buffer = buffers()[0];

  size_t gws[1] = {threads};
  size_t lws[1] = {lws_value};

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(unsigned int), &level);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                            NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  _wrapper->clFinish(cmdQueues_[_deviceId]);

  // Try to normalize the amount of work per test
  // unsigned int repeats = (4096 / threads) * 100 ;
  unsigned int repeats = (4096 / threads) * 10;
  // unsigned int repeats = 100;
  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < repeats; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, lws, 0,
                                              NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

    _wrapper->clFinish(cmdQueues_[_deviceId]);
  }
  timer.Stop();

  double sec = timer.GetElapsedTime();

  _perfInfo = (float)(threads * repeats * level) / (float)(sec * 1000000.);
  char buf[256];
  SNPRINTF(buf, sizeof(buf),
           "%5d threads spawning %2d threads, queue size %3dKB (Mdisp/s), level=%2d", threads,
           lws_value, queueSize / 1024, level);
  testDescString = buf;
}

unsigned int OCLPerfDeviceEnqueueEvent::close(void) {
  // FIXME: Re-enable CPU test once bug 10143 is fixed.
  if (type_ == CL_DEVICE_TYPE_CPU) {
    return 0;
  }

  if (deviceQueue_) {
    error_ = _wrapper->clReleaseCommandQueue(deviceQueue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (kernel2_) {
    error_ = _wrapper->clReleaseKernel(kernel2_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  return OCLTestImp::close();
}
