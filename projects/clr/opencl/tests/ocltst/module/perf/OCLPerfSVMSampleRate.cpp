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

#include "OCLPerfSVMSampleRate.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define NUM_TYPES 3
static const char* types[NUM_TYPES] = {"float", "float2", "float4"};
static const unsigned int typeSizes[NUM_TYPES] = {4, 8, 16};

#define NUM_SIZES 12
static const unsigned int sizes[NUM_SIZES] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

#define NUM_BUFS 6
#define MAX_BUFS (1 << (NUM_BUFS - 1))

#define NUM_READS numBufs_

OCLPerfSVMSampleRate::OCLPerfSVMSampleRate() {
  _numSubTests = NUM_TYPES * NUM_SIZES * NUM_BUFS * 3;
  skip_ = false;
}

OCLPerfSVMSampleRate::~OCLPerfSVMSampleRate() {}

void OCLPerfSVMSampleRate::setKernel(void) {
  shader_.clear();
  shader_ +=
      "kernel void sampleRate(global DATATYPE* outBuffer, unsigned int "
      "inBufSize, unsigned int writeIt,\n";
  char buf[256];
  for (unsigned int i = 0; i < numBufs_; i++) {
    SNPRINTF(buf, sizeof(buf), "global DATATYPE* inBuffer%d", i);
    shader_ += buf;
    if (i < (numBufs_ - 1)) {
      shader_ += ",";
    }
    shader_ += "\n";
  }
  shader_ += ")\n";
  shader_ +=
      "{\n"
      "    uint gid = get_global_id(0);\n"
      "    uint inputIdx = gid % inBufSize;\n"
      "    DATATYPE tmp = (DATATYPE)0.0f;\n";

  for (unsigned int j = 0; j < (NUM_READS / numBufs_); j++) {
    for (unsigned int i = 0; i < numBufs_; i++) {
      SNPRINTF(buf, sizeof(buf), "    tmp += inBuffer%d[inputIdx];\n", i);
      shader_ += buf;
    }
    shader_ += "    inputIdx += writeIt;\n";  // writeIt is 0, so we don't need
                                              // to add a modulo
  }
  if (typeSizes[typeIdx_] > 4) {
    shader_ +=
        "    if (writeIt*(unsigned int)tmp.x) outBuffer[gid] = tmp;\n"
        "}\n";
  } else {
    shader_ +=
        "    if (writeIt*(unsigned int)tmp) outBuffer[gid] = tmp;\n"
        "}\n";
  }
  // printf("Shader -> %s\n", shader_.c_str());
}

void OCLPerfSVMSampleRate::setData(void* buffer, unsigned int val) {
#if defined(CL_VERSION_2_0)
  error_ = _wrapper->clEnqueueSVMMemFill(cmd_queue_, buffer, &val, sizeof(unsigned int), bufSize_,
                                         0, NULL, NULL);
  if ((error_ == CL_MEM_OBJECT_ALLOCATION_FAILURE) || (error_ == CL_OUT_OF_RESOURCES) ||
      (error_ == CL_OUT_OF_HOST_MEMORY)) {
    error_ = CL_SUCCESS;
    skip_ = true;
    testDescString = "Not enough memory, skipped";
    return;
  }
  _wrapper->clFinish(cmd_queue_);
#endif
}

void OCLPerfSVMSampleRate::checkData(void* buffer) {
#if defined(CL_VERSION_2_0)
  error_ =
      _wrapper->clEnqueueSVMMap(cmd_queue_, true, CL_MAP_READ, buffer, outBufSize_, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueSVMMap failed");
  float* data = (float*)buffer;
  for (unsigned int i = 0; i < outBufSize_ / sizeof(float); i++) {
    if (data[i] != (float)numBufs_) {
      printf("Data validation failed at %d! Got %f, expected %f\n", i, data[i], (float)numBufs_);
      break;
    }
  }
  error_ = _wrapper->clEnqueueSVMUnmap(cmd_queue_, buffer, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_);
#endif
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfSVMSampleRate::open(unsigned int test, char* units, double& conversion,
                                unsigned int deviceId) {
  cl_device_id device;
  error_ = CL_SUCCESS;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = 0;
  kernel_ = 0;
  cmd_queue_ = 0;
  inBuffer_ = NULL;
  outBuffer_ = NULL;
  coarseGrainBuffer_ = false;
  fineGrainBuffer_ = false;
  fineGrainSystem_ = false;

  // We compute a square domain
  width_ = sizes[test % NUM_SIZES];
  typeIdx_ = (test / NUM_SIZES) % NUM_TYPES;
  bufSize_ = width_ * width_ * typeSizes[typeIdx_];
  numBufs_ = (1 << ((test / (NUM_SIZES * NUM_TYPES)) % NUM_BUFS));
  svmMode_ = test / (NUM_SIZES * NUM_TYPES * NUM_BUFS);

  device = devices_[deviceId];

#if defined(CL_VERSION_2_0)
  cl_device_svm_capabilities caps;
  error_ = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities),
                           &caps, NULL);
  if (svmMode_ == 0) {
    if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
      coarseGrainBuffer_ = true;
      testdesc = "crs";
    } else {
      skip_ = true;  // Should never happen as OCL 2.0 devices are required to
                     // support coarse grain SVM
      testDescString = "Coarse grain SVM NOT supported. Test Skipped.";
      return;
    }
  } else if (svmMode_ == 1) {
    if (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
      fineGrainBuffer_ = true;
      testdesc = "fgb";
    } else {
      skip_ = true;  // No support for fine grain buffer SVM
      testDescString = "Fine grain buffer SVM NOT supported. Test Skipped.";
      return;
    }
  } else if (svmMode_ == 2) {
    if (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
      fineGrainSystem_ = true;
      testdesc = "fgs";
    } else {
      skip_ = true;  // No support for fine grain system SVM
      testDescString = "Fine grain system SVM NOT supported. Test Skipped.";
      return;
    }
  }

  char charbuf[1024];

  cmd_queue_ = cmdQueues_[_deviceId];

  outBufSize_ = sizes[NUM_SIZES - 1] * sizes[NUM_SIZES - 1] * typeSizes[NUM_TYPES - 1];
  if ((svmMode_ == 0) || (svmMode_ == 1)) {
    inBuffer_ = (void**)malloc(sizeof(void*) * numBufs_);
    memset(inBuffer_, 0, sizeof(void*) * numBufs_);
    cl_mem_flags flags;
    flags = CL_MEM_READ_ONLY;
    if (svmMode_ == 1) flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    for (unsigned int i = 0; i < numBufs_; i++) {
      inBuffer_[i] = _wrapper->clSVMAlloc(context_, flags, bufSize_, 0);
      CHECK_RESULT(inBuffer_[i] == NULL, "clCreateBuffer(inBuffer) failed");
    }

    flags = CL_MEM_WRITE_ONLY;
    if (svmMode_ == 1) flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    outBuffer_ = _wrapper->clSVMAlloc(context_, flags, outBufSize_, 0);
    CHECK_RESULT(outBuffer_ == NULL, "clCreateBuffer(outBuffer) failed");
  } else {
    inBuffer_ = (void**)malloc(sizeof(void*) * numBufs_);
    memset(inBuffer_, 0, sizeof(void*) * numBufs_);
    for (unsigned int i = 0; i < numBufs_; i++) {
      inBuffer_[i] = malloc(bufSize_);
      CHECK_RESULT(inBuffer_[i] == NULL, "malloc(inBuffer) failed");
    }
    outBuffer_ = malloc(outBufSize_);
    CHECK_RESULT(outBuffer_ == NULL, "malloc(outBuffer) failed");
  }

  setKernel();
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  const char* buildOps = NULL;
  // Have to force OCL 2.0 to use SVM
  SNPRINTF(charbuf, sizeof(charbuf), "-cl-std=CL2.0 -D DATATYPE=%s", types[typeIdx_]);
  buildOps = charbuf;
  error_ = _wrapper->clBuildProgram(program_, 1, &device, buildOps, NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "sampleRate", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, 0, outBuffer_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(outBuffer) failed");
  unsigned int sizeDW = width_ * width_;
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(unsigned int), (void*)&sizeDW);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(sizeDW) failed");
  unsigned int writeIt = 0;
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(unsigned int), (void*)&writeIt);
  CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(writeIt) failed");
  for (unsigned int i = 0; i < numBufs_; i++) {
    error_ = _wrapper->clSetKernelArgSVMPointer(kernel_, i + 3, inBuffer_[i]);
    CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg(inBuffer) failed");
    setData(inBuffer_[i], 0x3f800000);
    if (skip_) return;
  }
  setData(outBuffer_, 0xdeadbeef);
#else
  skip_ = true;
  testDescString = "SVM NOT supported for < 2.0 builds. Test Skipped.";
  return;
#endif
}

void OCLPerfSVMSampleRate::run(void) {
  int global = outBufSize_ / typeSizes[typeIdx_];
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};
  unsigned int maxIter = MAX_ITERATIONS * (MAX_BUFS / numBufs_);

  if (skip_) return;

  CPerfCounter timer;

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < maxIter; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
  }

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue_);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Test doesn't write anything, so nothing to check
  // checkData(outBuffer_);
  // Compute GB/s
  double perf = ((double)outBufSize_ * NUM_READS * (double)maxIter * (double)(1e-09)) / sec;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), "Domain %dx%d, %2d %s bufs, %6s, %4dx%4d (GB/s)", sizes[NUM_SIZES - 1],
           sizes[NUM_SIZES - 1], numBufs_, testdesc.c_str(), types[typeIdx_], width_, width_);

  _perfInfo = (float)perf;
  testDescString = buf;
}

unsigned int OCLPerfSVMSampleRate::close(void) {
#if defined(CL_VERSION_2_0)
  if (cmd_queue_) _wrapper->clFinish(cmd_queue_);

  if ((svmMode_ == 0) || (svmMode_ == 1)) {
    if (inBuffer_) {
      for (unsigned int i = 0; i < numBufs_; i++) {
        if (inBuffer_[i]) {
          _wrapper->clSVMFree(context_, inBuffer_[i]);
          CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clSVMFree(inBuffer_) failed");
        }
      }
      free(inBuffer_);
    }
    if (outBuffer_) {
      _wrapper->clSVMFree(context_, outBuffer_);
    }
  } else {
    if (inBuffer_) {
      for (unsigned int i = 0; i < numBufs_; i++) {
        if (inBuffer_[i]) {
          free(inBuffer_[i]);
        }
      }
      free(inBuffer_);
    }
    if (outBuffer_) {
      free(outBuffer_);
    }
  }
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
  }
#endif
  return OCLTestImp::close();
}
