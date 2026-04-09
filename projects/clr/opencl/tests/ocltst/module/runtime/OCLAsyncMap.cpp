/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
