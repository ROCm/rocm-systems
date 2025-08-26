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

#include "OCLLDS32K.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CL/cl.h"
// #include <stdint.h>
#include <CL/cl.h>

typedef unsigned int uint32_t;
#if EMU_ENV
#define LDS_SIZE 1024
#define A_SIZE 1024
#else
#define LDS_SIZE 32768
#define A_SIZE (8 * 1024 * 1024)
#endif  // EMU_ENV

#define LOCAL_WORK_SIZE 64

// We'll do a 64MB transaction
#define B_SIZE A_SIZE
#define C_SIZE A_SIZE
#define D_SIZE A_SIZE

#define GLOBAL_WORK_SIZE (A_SIZE / LDS_SIZE * LOCAL_WORK_SIZE)

#define TEST_NAME "lds 32K"

// 32K has 8192 elements
// 64 threads each handle 8192/64=128 values
#if EMU_ENV
static const char program_source[] = KERNEL(
    __kernel void the_kernel(__global const uint* a, __global const uint* b, __global const uint* c,
                             __global uint* d, __global uint* e) {
      // Reduce size for the emulator
      __local uint lds[256];
      uint gid = get_global_id(0);
      __global const uint* ta = a + 4 * gid;
      __global const uint* tb = b + 4 * gid;
      __global const uint* tc = c + 4 * gid;
      __global uint* td = d + 4 * gid;
      uint i;

      for (i = 0; i < 4; ++i) lds[ta[i]] = tc[i];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (i = 0; i < 4; ++i) td[i] = lds[tb[i]];
    } __kernel void the_kernel2(__global uint* d) {
      __local uint lds[8192];
      uint i;
      uint gid = get_global_id(0);

      for (i = 0; i < 128; ++i) lds[i] = d[gid];
      barrier(CLK_LOCAL_MEM_FENCE);

      for (i = 0; i < 128; ++i) d[gid] = lds[i];
    });
#else
static const char program_source[] = KERNEL(
    __kernel void the_kernel(__global const uint* a, __global const uint* b, __global const uint* c,
                             __global uint* d, __global uint* e) {
      __local uint lds[8192];
      uint gid = get_global_id(0);
      __global const uint* ta = a + 128 * gid;
      __global const uint* tb = b + 128 * gid;
      __global const uint* tc = c + 128 * gid;
      __global uint* td = d + 128 * gid;
      uint i;

      for (i = 0; i < 128; ++i) lds[ta[i]] = tc[i];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (i = 0; i < 128; ++i) td[i] = lds[tb[i]];
    } __kernel void the_kernel2(__global uint* d) {
      __local uint lds[8192];
      uint i;
      uint gid = get_global_id(0);

      for (i = 0; i < 128; ++i) lds[i] = d[gid];
      barrier(CLK_LOCAL_MEM_FENCE);

      for (i = 0; i < 128; ++i) d[gid] = lds[i];
    });
#endif  // EMU_ENV

static void fill(uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d, uint32_t* e) {
  uint32_t i, j, k, t;
  static uint32_t p[LDS_SIZE / 4];
  static int is_set = 0;

  if (!is_set) {
    for (i = 0; i < LDS_SIZE / 4; ++i) p[i] = i;
    is_set = 1;
  }

  for (j = 0; j < A_SIZE / LDS_SIZE; ++j) {
    for (i = 0; i < LDS_SIZE / 4; ++i) {
      k = rand() % (LDS_SIZE / 4);
      t = p[i];
      p[i] = p[k];
      p[k] = t;

      c[i] = rand();
    }
    memcpy(a, p, LDS_SIZE);

    for (i = 0; i < LDS_SIZE / 4; ++i) {
      k = rand() % (LDS_SIZE / 4);
      t = p[i];
      p[i] = p[k];
      p[k] = t;

      d[i] = 0xfeedbeefU;
    }
    memcpy(b, p, LDS_SIZE);

    a += LDS_SIZE / 4;
    b += LDS_SIZE / 4;
    c += LDS_SIZE / 4;
    d += LDS_SIZE / 4;
  }
}

static int check(const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* d,
                 const uint32_t* e) {
  uint32_t i, j, t;
  uint32_t lds[LDS_SIZE / 4];

  for (j = 0; j < A_SIZE / LDS_SIZE; ++j) {
    for (i = 0; i < LDS_SIZE / 4; ++i) lds[i] = 0xdeadbeef;

    for (i = 0; i < LDS_SIZE / 4; ++i) lds[a[i]] = c[i];

    for (i = 0; i < LDS_SIZE / 4; ++i) {
      t = lds[b[i]];
      if (d[i] != t) {
        printf("mismatch group %u thread %u element %u: %u instead of %u\n", j, i / 128, i % 128,
               d[i], t);
        return EXIT_FAILURE;
      }
    }

    a += LDS_SIZE / 4;
    b += LDS_SIZE / 4;
    c += LDS_SIZE / 4;
    d += LDS_SIZE / 4;
  }
  return EXIT_SUCCESS;
}

#ifndef E_SIZE
#define E_SIZE 32
#endif

void OCLLDS32K::setup_run(const char* cmplr_opt) {
  cl_ulong lsize;
  const char* ps[2];
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lsize),
                                     &lsize, NULL);
  if (lsize < LDS_SIZE) {
    fprintf(stderr, "Passed! Test does not support 32kb of lds space!");
    return;
  }

  // create the program
  ps[0] = program_source;
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, ps, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource failed");

  // build the program
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], cmplr_opt, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char build_log[16384];
    size_t log_sz;
    fprintf(stderr, "build program failed, err=%d\n", error_);
    error_ = _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                             sizeof(build_log), build_log, &log_sz);
    if (error_ != CL_SUCCESS)
      fprintf(stderr, "failed to get build log, err=%d\n", error_);
    else
      fprintf(stderr, "----- Build Log -----\n%s\n----- ----- --- -----\n", build_log);
    return;
  }

  // create the kernel
  kernel_ = _wrapper->clCreateKernel(program_, "the_kernel", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "create a kernel failed");

  // create the kernel
  kernel2_ = _wrapper->clCreateKernel(program_, "the_kernel2", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "create a kernel failed");

  // allocate the buffer memory objects
  a_buf_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, A_SIZE, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "create a buffer a failed");
  buffers_.push_back(a_buf_);

  b_buf_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, B_SIZE, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "create a buffer b failed");
  buffers_.push_back(b_buf_);

  c_buf_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, C_SIZE, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "create a buffer c failed");
  buffers_.push_back(c_buf_);

  d_buf_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, D_SIZE, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "create a buffer d failed");
  buffers_.push_back(d_buf_);

  e_buf_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, E_SIZE, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "create a buffer e failed");
  buffers_.push_back(e_buf_);

  // set the args values
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&a_buf_);
  error_ |= _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&b_buf_);
  error_ |= _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_mem), (void*)&c_buf_);
  error_ |= _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void*)&d_buf_);
  error_ |= _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_mem), (void*)&e_buf_);
  CHECK_RESULT((error_ != CL_SUCCESS), "setkernelArg failed!");

  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), (void*)&d_buf_);
  CHECK_RESULT((error_ != CL_SUCCESS), "setkernelArg failed!");
}

void OCLLDS32K::cleanup_run() {
  if (kernel2_) {
    _wrapper->clReleaseKernel(kernel2_);
  }
}

void OCLLDS32K::exec_kernel(void* a_mem, void* b_mem, void* c_mem, void* d_mem, void* e_mem) {
  size_t global_work_size[1];
  size_t local_work_size[1];

  // Send data to device
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], a_buf_, CL_TRUE, 0, A_SIZE, a_mem,
                                          0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueWritebuffer failed");

  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], b_buf_, CL_TRUE, 0, B_SIZE, b_mem,
                                          0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueWritebuffer failed");

  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], c_buf_, CL_TRUE, 0, C_SIZE, c_mem,
                                          0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueWritebuffer failed");

  // set work-item dimensions
  global_work_size[0] = GLOBAL_WORK_SIZE;
  local_work_size[0] = LOCAL_WORK_SIZE;

  // execute kernel
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            global_work_size, local_work_size, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel failed");

  // execute kernel
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            global_work_size, local_work_size, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel failed");

  // execute kernel
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            global_work_size, local_work_size, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel failed");

  // read results
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], d_buf_, CL_TRUE, 0, D_SIZE, d_mem,
                                         0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");

  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], e_buf_, CL_TRUE, 0, E_SIZE, e_mem,
                                         0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clFinish failed");
}

const char* OCLLDS32K::kernel_src = "";

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

OCLLDS32K::OCLLDS32K() { _numSubTests = 1; }

OCLLDS32K::~OCLLDS32K() {}

void OCLLDS32K::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  _deviceId = deviceId;
  testID_ = test;
  OCLTestImp::open(test, units, conversion, _deviceId);
}

void OCLLDS32K::run(void) {
  void* a;
  void* b;
  void* c;
  void* d;
  void* e;
  const char* cmplr_opt = NULL;
  int j, nj;
  double f, dj, p;

  nj = 5;

  setup_run(cmplr_opt);
  CHECK_RESULT((error_ != CL_SUCCESS), "setup_run failed!");

  p = 10.0;
  dj = 100.0 / (double)nj;

  a = malloc(A_SIZE);
  CHECK_RESULT((a == NULL), "malloc failed");
  memset(a, 0, A_SIZE);

  b = malloc(B_SIZE);
  CHECK_RESULT((b == NULL), "malloc failed");
  memset(b, 0, B_SIZE);

  c = malloc(C_SIZE);
  CHECK_RESULT((c == NULL), "malloc failed");
  memset(c, 0, C_SIZE);

  d = malloc(D_SIZE);
  CHECK_RESULT((d == NULL), "malloc failed");
  memset(d, 0, D_SIZE);

  e = malloc(E_SIZE);
  CHECK_RESULT((e == NULL), "malloc failed");
  memset(e, 0, E_SIZE);

  // printf("Testing " TEST_NAME " on %s\n", argv[1]);
  for (j = 0; j < nj; ++j) {
    fill((uint32_t*)a, (uint32_t*)b, (uint32_t*)c, (uint32_t*)d, (uint32_t*)e);
    // printf("%s Test %d: ", sDevice, j);
    exec_kernel(a, b, c, d, e);
    CHECK_RESULT((error_ != CL_SUCCESS), "exec_kernel failed!");

    CHECK_RESULT((check((uint32_t*)a, (uint32_t*)b, (uint32_t*)c, (uint32_t*)d, (uint32_t*)e) < 0),
                 " Failed!\n");
    f = (j + 1) * dj;
    if (nj > 1 && f >= p) {
      // printf("%.1lf%%...\n", f);
      // fflush(stdout);
      p += 10.0;
    }
  }
}

unsigned int OCLLDS32K::close(void) {
  cleanup_run();
  return OCLTestImp::close();
}
