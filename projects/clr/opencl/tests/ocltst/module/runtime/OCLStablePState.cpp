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

#include "OCLStablePState.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "CL/cl_ext.h"

cl_device_id gpu_device;

OCLStablePState::OCLStablePState() {
  _numSubTests = 1;
  failed_ = false;
}

OCLStablePState::~OCLStablePState() {}

void OCLStablePState::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _deviceId = deviceId;

  if (type_ != CL_DEVICE_TYPE_GPU) {
    error_ = CL_DEVICE_NOT_FOUND;
    printf("GPU device is required for this test!\n");
    return;
  }

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
#if 0
    // Get last for default
    platform = platforms[numPlatforms - 1];
    for (unsigned i = 0; i < numPlatforms; ++i) {
#endif
    platform = platforms[_platformIndex];
    char pbuf[100];
    error_ = _wrapper->clGetPlatformInfo(platforms[_platformIndex], CL_PLATFORM_VENDOR,
                                         sizeof(pbuf), pbuf, NULL);
    num_devices = 0;
    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
#if 0
    }
#endif
    delete platforms;
  }
  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find platform with GPU devices, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];
  gpu_device = device;
}

static void CL_CALLBACK notify_callback(cl_event event, cl_int event_command_exec_status,
                                        void* user_data) {}

void OCLStablePState::run(void) {
  if (failed_) {
    return;
  }
  cl_set_device_clock_mode_input_amd setClockModeInput;
  setClockModeInput.clock_mode = CL_DEVICE_CLOCK_MODE_PROFILING_AMD;
  cl_set_device_clock_mode_output_amd setClockModeOutput = {};
  error_ = _wrapper->clSetDeviceClockModeAMD(gpu_device, setClockModeInput, &setClockModeOutput);
#ifdef _WIN32
  CHECK_RESULT(error_ != CL_SUCCESS, "SetClockMode profiling failed\n");
#else
  error_ = CL_SUCCESS;
#endif

  setClockModeInput.clock_mode = CL_DEVICE_CLOCK_MODE_DEFAULT_AMD;
  setClockModeOutput = {};
  error_ = _wrapper->clSetDeviceClockModeAMD(gpu_device, setClockModeInput, &setClockModeOutput);
#ifdef _WIN32
  CHECK_RESULT(error_ != CL_SUCCESS, "SetClockMode default failed\n");
#else
  error_ = CL_SUCCESS;
#endif
}

unsigned int OCLStablePState::close(void) { return OCLTestImp::close(); }
