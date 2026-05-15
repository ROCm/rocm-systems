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

#include "OCLMapCount.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

OCLMapCount::OCLMapCount() { _numSubTests = 1; }

OCLMapCount::~OCLMapCount() {}

void OCLMapCount::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  size_t size;
  clMemWrapper memObject;

  // Get the address alignment, so we can make sure the sub buffer test later
  // works properly
  cl_uint addressAlign;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                                     sizeof(addressAlign), &addressAlign, NULL);
  if (addressAlign < 128) addressAlign = 128;

  void* void_buffer = malloc(addressAlign * 4);

  // Create a buffer to test against
  memObject = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                       addressAlign * 4, void_buffer, &error_);
  if (error_) {
    free(void_buffer);
    printf("Unable to create buffer to test");
  }

  // Map buffer
  void* mapped = _wrapper->clEnqueueMapBuffer(cmdQueues_[deviceId], memObject, true, CL_MAP_READ, 0,
                                              addressAlign * 4, 0, NULL, NULL, &error_);

  cl_uint mapCount;

  // Find the number of mappings on buffer after map
  error_ =
      _wrapper->clGetMemObjectInfo(memObject, CL_MEM_MAP_COUNT, sizeof(mapCount), &mapCount, &size);
  CHECK_RESULT((error_ != CL_SUCCESS), "Unable to get mem object map count");
  if (mapCount != 1) {
    printf(
        "ERROR: Returned mem object map count does not validate! (expected %d, "
        "got %d)\n",
        1, mapCount);
    return;
  }

  // Unmap buffer
  error_ =
      _wrapper->clEnqueueUnmapMemObject(cmdQueues_[deviceId], memObject, mapped, 0, NULL, NULL);

  // Find the number of mappings on buffer after unmap
  error_ =
      _wrapper->clGetMemObjectInfo(memObject, CL_MEM_MAP_COUNT, sizeof(mapCount), &mapCount, &size);
  CHECK_RESULT((error_ != CL_SUCCESS), "Unable to get mem object map count");
  if (mapCount != 0) {
    printf(
        "ERROR: Returned mem object map count does not validate! (expected %d, "
        "got %d)\n",
        0, mapCount);
    return;
  }
}

void OCLMapCount::run(void) {}

unsigned int OCLMapCount::close(void) { return OCLTestImp::close(); }
