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

#include "OCLCreateContext.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

OCLCreateContext::OCLCreateContext() { _numSubTests = 1; }

OCLCreateContext::~OCLCreateContext() {}

void OCLCreateContext::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLCreateContext::run(void) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;

  int error = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error != CL_SUCCESS, "clGetPlatformIDs failed");
    for (unsigned i = 0; i < numPlatforms; ++i) {
      char pbuf[100];
      error =
          _wrapper->clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
      if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
        platform = platforms[i];
        break;
      }
    }
    delete platforms;
  }

  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  /* Get the number of requested devices */
  error = _wrapper->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  CHECK_RESULT(error != CL_SUCCESS, "clGetDeviceIDs failed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error = _wrapper->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
  CHECK_RESULT(error != CL_SUCCESS, "clGetDeviceIDs failed");

  device = devices[0];

  cl_context gContext = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error);
  CHECK_RESULT(gContext == 0, "clCreateContext failed");

  error = _wrapper->clReleaseContext(gContext);
  CHECK_RESULT(error != CL_SUCCESS, "clReleaseContext failed");
}

unsigned int OCLCreateContext::close(void) { return _crcword; }
