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

#include "OCLP2PBuffer.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "CL/cl.h"

const static size_t ChunkSize = 256 * 1024;
const static int NumSizes = 5;
const static int NumRuns = 4;
const static int NumChunksArray[NumSizes] = {1, 4, 16, 32, 64};
const static size_t MaxSubTests = NumRuns * NumSizes;
const static int NumIterArray[NumSizes] = {20, 15, 10, 10, 10};

OCLP2PBuffer::OCLP2PBuffer() {
#ifdef CL_VERSION_2_0
  _numSubTests = MaxSubTests;
#else
  _numSubTests = 0;
#endif
  failed_ = false;
  maxSize_ = 0;
  context0_ = nullptr;
  context1_ = nullptr;
  cmdQueue0_ = nullptr;
  cmdQueue1_ = nullptr;
}

OCLP2PBuffer::~OCLP2PBuffer() {}

void OCLP2PBuffer::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
#ifdef CL_VERSION_2_0
  cl_uint numPlatforms = 0;
  cl_uint num_devices = 0;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  if (deviceCount_ < 2) {
    printf("\nTwo GPUs are required to run P2P test\n");
    failed_ = true;
    return;
  }

  testID_ = test;
  char name[1024] = {0};
  size_t size = 0;
  _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_EXTENSIONS, 1024, name, &size);
  if (!strstr(name, "cl_amd_copy_buffer_p2p")) {
    printf("P2P extension is required for this test!\n");
    failed_ = true;
    return;
  }

  _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_EXTENSIONS, 1024, name, &size);
  if (!strstr(name, "cl_amd_copy_buffer_p2p")) {
    printf("P2P extension is required for this test!\n");
    failed_ = true;
    return;
  }
  num_p2p_0_ = 0;
  _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_NUM_P2P_DEVICES_AMD, sizeof(num_p2p_0_),
                            &num_p2p_0_, nullptr);
  if (num_p2p_0_ != 0) {
    cl_device_id* p2p = new cl_device_id[num_p2p_0_];
    _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_P2P_DEVICES_AMD,
                              sizeof(cl_device_id) * num_p2p_0_, p2p, nullptr);
    delete[] p2p;
  }
  num_p2p_1_ = 0;
  _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_NUM_P2P_DEVICES_AMD, sizeof(num_p2p_1_),
                            &num_p2p_1_, nullptr);
  if (num_p2p_1_ != 0) {
    cl_device_id* p2p = new cl_device_id[num_p2p_1_];
    _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_P2P_DEVICES_AMD,
                              sizeof(cl_device_id) * num_p2p_1_, p2p, nullptr);
    delete[] p2p;
  }

  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform_, 0};
  context0_ = _wrapper->clCreateContext(props, 1, &devices_[0], NULL, 0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateContext#0 failed");

  context1_ = _wrapper->clCreateContext(props, 1, &devices_[1], NULL, 0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateContext#1 failed");

  NumChunks = NumChunksArray[testID_ % NumSizes];
  NumIter = NumIterArray[testID_ % NumSizes];
  BufferSize = NumChunks * ChunkSize * sizeof(cl_uint);

  p2p_copy_ = (clEnqueueCopyBufferP2PAMD_fn)clGetExtensionFunctionAddressForPlatform(
      platform_, "clEnqueueCopyBufferP2PAMD");
  if (p2p_copy_ == NULL) {
    testDescString = "Failed to initialize P2P extension!\n";
    failed_ = true;
    return;
  }

  cl_queue_properties prop[] = {CL_QUEUE_PROPERTIES, 0, 0};
  cmdQueue0_ = _wrapper->clCreateCommandQueueWithProperties(context0_, devices_[0], prop, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");
  cmdQueue1_ = _wrapper->clCreateCommandQueueWithProperties(context1_, devices_[1], prop, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueueWithProperties() failed");

  size_t chunkSize = ChunkSize;

  cl_mem buf = NULL;
  cl_uint memFlags = 0;
  buf = _wrapper->clCreateBuffer(context0_, CL_MEM_READ_ONLY | memFlags, BufferSize, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buf);

  buf = _wrapper->clCreateBuffer(context1_, memFlags, BufferSize, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buf);
#endif
}

void OCLP2PBuffer::run(void) {
#ifdef CL_VERSION_2_0
  if (failed_) {
    return;
  }
  size_t finalBuf = 0;
  cl_uint subTest = (testID_ / NumSizes) % 2;

  cl_uint* buffer = new cl_uint[NumChunks * ChunkSize];
  cl_uint* buffer2 = new cl_uint[NumChunks * ChunkSize];
  cl_event event;

  memset(buffer, 0x23, BufferSize);
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueue1_, buffers_[1], CL_TRUE, 0, BufferSize, buffer,
                                          0, nullptr, (subTest == 0) ? &event : nullptr);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  memset(buffer2, 0xEB, BufferSize);
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueue0_, buffers_[0], CL_TRUE, 0, BufferSize, buffer2,
                                          0, nullptr, (subTest == 1) ? &event : nullptr);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  CPerfCounter timer;

  double sec = 0.;
  if (subTest == 0) {
    error_ = p2p_copy_(cmdQueue0_, buffers_[0], buffers_[1], 0, 0, BufferSize, 1, &event, nullptr);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueCopyBufferP2PAMD() failed");
    _wrapper->clFinish(cmdQueue0_);
  } else {
    error_ = p2p_copy_(cmdQueue1_, buffers_[1], buffers_[0], 0, 0, BufferSize, 1, &event, nullptr);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueCopyBufferP2PAMD() failed");
    _wrapper->clFinish(cmdQueue1_);
  }
  clReleaseEvent(event);
  cl_command_queue execQueue;
  if (((testID_ / NumSizes) == 0) || ((testID_ / NumSizes) == 3)) {
    execQueue = cmdQueue0_;
  } else {
    execQueue = cmdQueue1_;
  }

  for (int i = 0; i < NumIter; ++i) {
    timer.Reset();
    timer.Start();

    if (subTest == 0) {
      p2p_copy_(execQueue, buffers_[0], buffers_[1], 0, 0, BufferSize, 0, nullptr, nullptr);
    } else {
      p2p_copy_(execQueue, buffers_[1], buffers_[0], 0, 0, BufferSize, 0, nullptr, nullptr);
    }
    _wrapper->clFinish(execQueue);
    timer.Stop();
    double cur = timer.GetElapsedTime();
    if (i == 0) {
      sec = cur;
    } else {
      sec = std::min(cur, sec);
    }
  }
  memset(buffer, 0x20, BufferSize);
  if (subTest == 0) {
    error_ = _wrapper->clEnqueueReadBuffer(cmdQueue1_, buffers_[1], CL_TRUE, 0, BufferSize, buffer,
                                           0, NULL, NULL);
  } else {
    error_ = _wrapper->clEnqueueReadBuffer(cmdQueue0_, buffers_[0], CL_TRUE, 0, BufferSize, buffer,
                                           0, NULL, NULL);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed!");

  cl_uint cmp_value = (subTest == 0) ? 0xEBEBEBEB : 0x23232323;
  for (int c = 0; c < NumChunks; ++c) {
    for (cl_uint i = 0; i < ChunkSize; ++i) {
      if (buffer[c * ChunkSize + i] != cmp_value) {
        CHECK_RESULT(true, "Validation failed!");
      }
    }
  }
  delete[] buffer;
  delete[] buffer2;

  cl_uint* p2p = ((subTest == 0) ? &num_p2p_0_ : &num_p2p_1_);
  static const char* MemTypeStr[] = {"Visible  ", "Remote   ", "Invisible", "Staging"};
  _perfInfo = (float)BufferSize / ((float)sec * 1000.f * 1000.f * 1000.f);
  std::stringstream str;
  if ((testID_ / (2 * NumSizes)) == 0) {
    str << "Write dev" << ((subTest == 0) ? 0 : 1) << "->dev" << ((subTest == 0) ? 1 : 0)
        << ((*p2p != 0) ? " <P2P> " : " ") << "(";
  } else {
    str << "Read  dev" << ((subTest == 0) ? 1 : 0) << "<-dev" << ((subTest == 0) ? 0 : 1)
        << ((*p2p != 0) ? " <P2P> " : " ") << "(";
  }
  str.width(2);
  str << BufferSize / (1000 * 1000);
  str << " MB "
      << ") transfer speed (GB/s):";
  testDescString = str.str();
#endif
}

unsigned int OCLP2PBuffer::close(void) {
#ifdef CL_VERSION_2_0
  if (!failed_) {
    if (cmdQueue0_ != nullptr) {
      _wrapper->clReleaseCommandQueue(cmdQueue0_);
    }
    if (cmdQueue1_ != nullptr) {
      _wrapper->clReleaseCommandQueue(cmdQueue1_);
    }
    if (context0_ != nullptr) {
      _wrapper->clReleaseContext(context0_);
    }
    if (context1_ != nullptr) {
      _wrapper->clReleaseContext(context1_);
    }
  }
  return OCLTestImp::close();
#else
  return CL_SUCCESS;
#endif
}
