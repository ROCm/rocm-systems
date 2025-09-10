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

#include "OCLMemoryInfo.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "CL/cl_ext.h"

OCLMemoryInfo::OCLMemoryInfo() {
  // Run the second test with 64 bit only
  _numSubTests = (sizeof(int*) == 8) ? 2 : 1;
  failed_ = false;
}

OCLMemoryInfo::~OCLMemoryInfo() {}

void OCLMemoryInfo::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  _deviceId = deviceId;
  test_ = test;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }

  char name[1024] = {0};
  size_t size = 0;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 1024, name, &size);
  if (!strstr(name, "cl_amd_device_attribute_query")) {
    printf("AMD device attribute  extension is required for this test!\n");
    failed_ = true;
    return;
  }
  // Observed failures with APUs on GSL path due to incorrect available memory,
  // reported for visible heap
  cl_bool is_apu = false;
  error_ = clGetDeviceInfo(devices_[deviceId], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool),
                           &is_apu, nullptr);
  if (is_apu) {
    printf("Test not supported for apus, skipping...\n");
    failed_ = true;
    return;
  }
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLMemoryInfo::run(void) {
  if (failed_) {
    return;
  }
  size_t BufSize = 0x1000000;
  bool succeed = false;
  bool done = false;
  if (test_ == 0) {
    // use multiple loops to make sure the failure case is not caused
    // by reusing the allocation from the cached memory pool
    for (int i = 0; i < 5 && !done; i++) {
      cl_mem buffer;
      size_t memoryInfo[2];
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                                2 * sizeof(size_t), memoryInfo, NULL);

      buffer = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, BufSize * sizeof(cl_int4),
                                        NULL, &error_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
      buffers_.push_back(buffer);

      unsigned int* values;
      values = reinterpret_cast<unsigned int*>(new cl_int4[BufSize]);
#if !EMU_ENV
      // Clear destination buffer
      memset(values, 0, BufSize * sizeof(cl_int4));
      error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                              BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
#endif
      size_t memoryInfo2[2];
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                                2 * sizeof(size_t), memoryInfo2, NULL);

      size_t dif = memoryInfo[0] - memoryInfo2[0];
      if (dif == 0) {  // the buffer memory may come from the cached memory pool
        BufSize *= 2;  // double the size and try again
      } else if ((dif >= (static_cast<size_t>(BufSize * sizeof(cl_int4) * 1.5f) / 1024)) ||
                 (dif <= ((BufSize * sizeof(cl_int4) / 2) / 1024))) {
        done = true;
      } else {
        succeed = true;
        done = true;
      }

      delete[] values;
    }
  } else {
    int i = 0;
    size_t sizeAll;
    size_t memoryInfo[2];
    _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                              2 * sizeof(size_t), memoryInfo, NULL);
    unsigned int* values;
    values = reinterpret_cast<unsigned int*>(new cl_int4[BufSize]);
    memset(values, 0, BufSize * sizeof(cl_int4));
    // Loop a few times to make sure the results are consistent
    for (int k = 0; k < 3; ++k) {
      sizeAll = 0;
      while (true) {
        cl_mem buffer;

        buffer = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, BufSize * sizeof(cl_int4),
                                          NULL, &error_);
        CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
        buffers_.push_back(buffer);
#if !EMU_ENV
        // Clear destination buffer
        error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffer, CL_TRUE, 0,
                                                BufSize * sizeof(cl_int4), values, 0, NULL, NULL);
        CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
#endif
        sizeAll += BufSize * sizeof(cl_int4) / 1024;
        size_t memoryInfo2[2];
        _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                                  2 * sizeof(size_t), memoryInfo2, NULL);
        if (memoryInfo2[0] < (0x50000 + (BufSize * sizeof(cl_int4) / 1024))) {
          break;
        }
        size_t dif = memoryInfo[0] - memoryInfo2[0];
        // extra memory could be allocated/destroyed in the driver
        if (dif == 0) {
          // the buffer memory may come from the cached memory pool
        } else if ((dif / sizeAll) == 1 || (sizeAll / dif) == 1) {
          succeed = true;
        } else {
          succeed = false;
          break;
        }
        ++i;
      }
      for (auto& it : buffers()) {
        error_ = _wrapper->clReleaseMemObject(it);
        CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseMemObject() failed");
      }
      buffers_.clear();
      if (!succeed) {
        break;
      }
    }
    delete[] values;
  }

  if (!succeed) {
    CHECK_RESULT(true, "Reported free memory doesn't match allocated size!");
  }
}

unsigned int OCLMemoryInfo::close(void) { return OCLTestImp::close(); }
