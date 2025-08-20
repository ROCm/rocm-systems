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

#include "OCLSemaphore.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#ifndef CL_DEVICE_MAX_SEMAPHORES_AMD
#define CL_DEVICE_MAX_SEMAPHORES_AMD 0x1041
#else
#error "CL_DEVICE_MAX_SEMAPHORES_AMD is defined somewhere, remove this define!"
#endif
#ifndef CL_DEVICE_MAX_SEMAPHORE_SIZE_AMD
#define CL_DEVICE_MAX_SEMAPHORE_SIZE_AMD 0x1042
#else
#error "CL_DEVICE_MAX_SEMAPHORE_SIZE_AMD is defined somewhere, remove this define!"
#endif
#ifndef CL_KERNEL_MAX_SEMAPHORE_SIZE_AMD
#define CL_KERNEL_MAX_SEMAPHORE_SIZE_AMD 0x1043
#else
#error "CL_KERNEL_MAX_SEMAPHORE_SIZE_AMD is defined somewhere, remove this define!"
#endif

const static unsigned int MaxSemaphores = 1;

const static char* strKernel =
    "#ifdef cl_amd_semaphore\n"
    "#pragma OPENCL EXTENSION cl_amd_semaphore : enable            \n"
    "kernel void sema_test(sema_t lock, global int* a, global int* b, int "
    "value)\n"
    "  {\n"
    "    size_t idx = get_global_id(0);\n"
    "    size_t gdx = get_group_id(0);\n"
    "    size_t ng = get_num_groups(0);\n"
    "    size_t ssize = get_max_semaphore_size();\n"
    "    a[1] = true;\n"
    "    if (gdx >= ssize) {\n"
    "      return;\n"
    "    }\n"
    "    barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "    semaphore_init(lock, ng);\n"
    "    while (a[1]) {\n"
    "      atom_add(a, b[idx]);\n"
    "      atom_inc(a + 2);\n"
    "      if (gdx == (ssize - 1)) {\n"
    "        semaphore_signal(lock);\n"
    "        if (a[0] >= value) {\n"
    "          a[1] = false;\n"
    "        }\n"
    "      } else {\n"
    "        semaphore_wait(lock);\n"
    "        idx += get_global_size(0);\n"
    "      }\n"
    "    }\n"
    "    semaphore_signal(lock);\n"
    "  }\n"
    "#endif\n";

OCLSemaphore::OCLSemaphore() {
  _numSubTests = 1;
  hasSemaphore = false;
}

OCLSemaphore::~OCLSemaphore() {}

void OCLSemaphore::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  char name[1024] = {0};
  size_t size = 0;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 1024, name, &size);
  if (!strstr(name, "cl_amd_semaphore")) {
    error_ = CL_DEVICE_NOT_FOUND;
    hasSemaphore = false;
    printf("Semaphore extension is required for this test!\n");
    return;
  } else {
    hasSemaphore = true;
  }
  _wrapper->clGetDeviceInfo(devices_[deviceId], (cl_device_info)CL_DEVICE_MAX_SEMAPHORES_AMD,
                            sizeof(size), &size, NULL);
  _wrapper->clGetDeviceInfo(devices_[deviceId], (cl_device_info)CL_DEVICE_MAX_SEMAPHORE_SIZE_AMD,
                            sizeof(size), &size, NULL);

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

  kernel_ = _wrapper->clCreateKernel(program_, "sema_test", &error_);
  _wrapper->clGetKernelInfo(kernel_, (cl_kernel_info)CL_KERNEL_MAX_SEMAPHORE_SIZE_AMD, sizeof(size),
                            &size, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  for (unsigned int i = 0; i < MaxSemaphores; ++i) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, 1024 * size * sizeof(cl_uint),
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, 1024 * size * sizeof(cl_uint),
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLSemaphore::run(void) {
  if (!hasSemaphore) {
    return;
  }
  cl_uint initVal[2] = {5, 10};

  for (unsigned int i = 0; i < MaxSemaphores; ++i) {
    cl_mem buffer = buffers()[i];
    error_ = _wrapper->clSetKernelArg(kernel_, i, sizeof(cl_uint), &initVal[i]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
  }

  cl_mem buffer = buffers()[MaxSemaphores];
  error_ = _wrapper->clSetKernelArg(kernel_, MaxSemaphores, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
  buffer = buffers()[MaxSemaphores + 1];
  error_ = _wrapper->clSetKernelArg(kernel_, MaxSemaphores + 1, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  cl_int val = 64;
  error_ = _wrapper->clSetKernelArg(kernel_, MaxSemaphores + 2, sizeof(val), &val);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");

  size_t gws[1] = {64};
  error_ =
      _wrapper->clEnqueueNDRangeKernel(cmdQueues_[0], kernel_, 1, NULL, gws, NULL, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");

  cl_uint outputV[MaxSemaphores] = {0};

  // Find the new counter value
  initVal[0]++;
  initVal[1]--;

  for (unsigned int i = 0; i < MaxSemaphores; ++i) {
    cl_mem buffer = buffers()[i];
    error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[0], buffers()[i], true, 0, sizeof(cl_uint),
                                           &outputV[i], 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
    if (initVal[i] != outputV[i]) {
      printf("%u != %u", initVal[i], outputV[i]);
      CHECK_RESULT(true, " - Incorrect result for counter!\n");
    }
  }

  // Restore the original value to check the returned result in the kernel
  initVal[0]--;
  initVal[1]++;

  buffer = buffers()[MaxSemaphores];
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[0], buffers()[MaxSemaphores], true, 0,
                                         MaxSemaphores * sizeof(cl_uint), outputV, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
  for (unsigned int i = 0; i < MaxSemaphores; ++i) {
    if (initVal[i] != outputV[i]) {
      printf("%u != %u", initVal[i], outputV[i]);
      CHECK_RESULT(true,
                   " - Incorrect result for counter inside kernel. Returned "
                   "value != original.\n");
    }
  }
}

unsigned int OCLSemaphore::close(void) { return OCLTestImp::close(); }
