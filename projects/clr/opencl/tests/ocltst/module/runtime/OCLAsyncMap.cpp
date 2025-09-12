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

#include "OCLAsyncMap.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

#if EMU_ENV
static const size_t BufSize = 0x800;
static const size_t MapRegion = 0x100;
#else
static const size_t BufSize = 0x800000;
static const size_t MapRegion = 0x100000;
#endif  // EMU_ENV

static const unsigned int NumMaps = BufSize / MapRegion;

OCLAsyncMap::OCLAsyncMap() { _numSubTests = 1; }

OCLAsyncMap::~OCLAsyncMap() {}

void OCLAsyncMap::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  cl_mem buffer;
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, BufSize * sizeof(cl_uint), NULL,
                                    &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLAsyncMap::run(void) {
  cl_uint* values[NumMaps];
  cl_mem mapBuffer = buffers()[0];
  size_t offset = 0;
  size_t region = MapRegion * sizeof(cl_uint);

  for (unsigned int i = 0; i < NumMaps; ++i) {
    values[i] = reinterpret_cast<cl_uint*>(_wrapper->clEnqueueMapBuffer(
        cmdQueues_[_deviceId], mapBuffer, CL_TRUE, (CL_MAP_READ | CL_MAP_WRITE), offset, region, 0,
        NULL, NULL, &error_));
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer() failed");
    offset += region;
  }

  for (unsigned int i = 0; i < NumMaps; ++i) {
    for (unsigned int j = 0; j < MapRegion; ++j) {
      values[i][j] = i;
    }
  }

  for (unsigned int i = 0; i < NumMaps; ++i) {
    error_ = _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], mapBuffer, values[i], 0, NULL,
                                               NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer() failed");
  }

  values[0] = reinterpret_cast<cl_uint*>(
      _wrapper->clEnqueueMapBuffer(cmdQueues_[_deviceId], mapBuffer, CL_TRUE, CL_MAP_READ, 0,
                                   BufSize * sizeof(cl_uint), 0, NULL, NULL, &error_));
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapBuffer() failed");

  for (unsigned int i = 0; i < NumMaps; ++i) {
    values[i] = values[0] + i * MapRegion;
    for (unsigned int j = 0; j < MapRegion; ++j) {
      CHECK_RESULT((values[i][j] != i), "validation failed");
    }
  }

  error_ =
      _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], mapBuffer, values[0], 0, NULL, NULL);

  _wrapper->clFinish(cmdQueues_[_deviceId]);
}

unsigned int OCLAsyncMap::close(void) { return OCLTestImp::close(); }
