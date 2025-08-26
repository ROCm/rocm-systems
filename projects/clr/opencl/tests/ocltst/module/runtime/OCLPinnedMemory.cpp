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

#include "OCLPinnedMemory.h"

#ifdef _WIN32
#include <VersionHelpers.h>
// Pick up from OCLSVM
size_t getTotalSystemMemory();
#else
#include <sys/sysinfo.h>
size_t getTotalSystemMemory() {
  struct sysinfo info;
  sysinfo(&info);
  return info.totalram;
}
#endif

#include <algorithm>
#include <cmath>
#include <numeric>

OCLPinnedMemory::OCLPinnedMemory() { _numSubTests = 2; }

OCLPinnedMemory::~OCLPinnedMemory() {}

void OCLPinnedMemory::open(unsigned int test, char* units, double& conversion,
                           unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_ERROR(error_, "Error opening test");
  _openTest = test;
  host_memory_ = nullptr;

#ifdef _WIN32
  // Observed failures on Win7
  if (!IsWindows8OrGreater()) {
    printf("Test requires Win10, skipping...\n");
    _openTest = -1;
    return;
  }
#endif

  cl_int status;

  cl_bool is_apu;
  status = clGetDeviceInfo(devices_[deviceId], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool),
                           &is_apu, nullptr);
  CHECK_ERROR(status, "clGetDeviceInfo failed.");
  if (is_apu) {
    printf("Test not supported for apus, skipping...\n");
    _openTest = -1;
    return;
  }

  cl_uint address_bits;
  status = clGetDeviceInfo(devices_[deviceId], CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint),
                           &address_bits, nullptr);
  CHECK_ERROR(status, "clGetDeviceInfo failed.");
  if (address_bits < 64u) {
    printf("GPU VA range size below 4GB, skipping...\n");
    _openTest = -1;
    return;
  }

  row_size_ = getTotalSystemMemory();
  if (row_size_ <= (1ull << 32u)) {
    printf("System memory below 4GB, skipping...\n");
    _openTest = -1;
    return;
  }
  row_size_ *= ratio_;
#if EMU_ENV
  if (row_size_ > 5000) {
    row_size_ = 5000;
  }
#endif
  row_size_ = floor(sqrt(row_size_));
  row_size_ = (row_size_ + row_data_size_ - 1) & ~(row_data_size_ - 1);

  pin_size_ = row_size_ * row_size_ / row_data_size_;
  host_memory_ = new row_data_t[pin_size_];
}

void OCLPinnedMemory::runNoPrepinnedMemory() {
  cl_int status;

  row_data_t* tmp = new row_data_t[row_size_];
  std::iota(tmp, tmp + row_size_, 0);
  std::fill_n(host_memory_, pin_size_, 0);

  cl_mem tmp_buffer =
      clCreateBuffer(context_, CL_MEM_USE_HOST_PTR, row_size_ * row_data_size_, tmp, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");
  cl_mem buffer =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, row_size_ * row_data_size_, nullptr, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");

  status = clEnqueueCopyBuffer(cmdQueues_[_deviceId], tmp_buffer, buffer, 0, 0,
                               row_size_ * row_data_size_, 0, nullptr, nullptr);
  CHECK_ERROR(status, "clEnqueueCopyBuffer failed.");
  clFinish(cmdQueues_[_deviceId]);

  size_t buffer_offset[3] = {0, 0, 0};
  size_t host_offset[3] = {0, 0, 0};
  size_t region[3] = {row_data_size_, row_size_, 1};

  status =
      clEnqueueReadBufferRect(cmdQueues_[_deviceId], buffer, CL_TRUE, buffer_offset, host_offset,
                              region, 0, 0, row_size_, 0, host_memory_, 0, nullptr, nullptr);
  CHECK_ERROR(status, "clEnqueueReadBufferRect failed.");
  status = clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(status, "clFinish failed.");

  for (uint64_t i = 0; i < row_size_; i++) {
    if (tmp[i] != host_memory_[i * row_size_ / row_data_size_]) {
      status = -1;
      break;
    }
  }

  CHECK_RESULT(status == -1, "Error when reading data.");

  status = clReleaseMemObject(buffer);
  CHECK_ERROR(status, "clReleaseMemObject failed.");
  status = clReleaseMemObject(tmp_buffer);
  CHECK_ERROR(status, "clReleaseMemObject failed.");
  delete[] tmp;
}

void OCLPinnedMemory::runPrepinnedMemory() {
  cl_int status;

  row_data_t* tmp = new row_data_t[row_size_];
  std::iota(tmp, tmp + row_size_, 0);
  std::fill_n(host_memory_, pin_size_, 0);

  cl_mem tmp_buffer =
      clCreateBuffer(context_, CL_MEM_USE_HOST_PTR, row_size_ * row_data_size_, tmp, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");
  cl_mem buffer =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, row_size_ * row_data_size_, nullptr, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");

  status = clEnqueueCopyBuffer(cmdQueues_[_deviceId], tmp_buffer, buffer, 0, 0,
                               row_size_ * row_data_size_, 0, nullptr, nullptr);
  CHECK_ERROR(status, "clEnqueueCopyBuffer failed.");

  cl_mem pinned_buffer = clCreateBuffer(context_, CL_MEM_USE_HOST_PTR, pin_size_ * row_data_size_,
                                        host_memory_, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");

  clEnqueueMapBuffer(cmdQueues_[_deviceId], pinned_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                     pin_size_ * row_data_size_, 0, nullptr, nullptr, &status);
  CHECK_ERROR(status, "clEnqueueMapBuffer failed.");

  size_t buffer_offset[3] = {0, 0, 0};
  size_t host_offset[3] = {0, 0, 0};
  size_t region[3] = {row_data_size_, row_size_, 1};

  status =
      clEnqueueReadBufferRect(cmdQueues_[_deviceId], buffer, CL_TRUE, buffer_offset, host_offset,
                              region, 0, 0, row_size_, 0, host_memory_, 0, nullptr, nullptr);
  CHECK_ERROR(status, "clEnqueueReadBufferRect failed.");

  for (uint64_t i = 0; i < row_size_; i++) {
    if (tmp[i] != host_memory_[i * row_size_ / row_data_size_]) {
      status = -1;
      break;
    }
  }

  CHECK_RESULT(status == -1, "Error when reading data.");

  status = clEnqueueUnmapMemObject(cmdQueues_[_deviceId], pinned_buffer, host_memory_, 0, nullptr,
                                   nullptr);
  CHECK_ERROR(status, "clEnqueueUnmap failed.")
  status = clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(status, "clFinish failed.");

  status = clReleaseMemObject(pinned_buffer);
  CHECK_ERROR(status, "clReleaseMemObject failed.");
  status = clReleaseMemObject(buffer);
  CHECK_ERROR(status, "clReleaseMemObject failed.");
  status = clReleaseMemObject(tmp_buffer);
  CHECK_ERROR(status, "clReleaseMemObject failed.");
  delete[] tmp;
}

void OCLPinnedMemory::run() {
  switch (_openTest) {
    case 0:
      runNoPrepinnedMemory();
      break;
    case 1:
      runPrepinnedMemory();
      break;
  }
}

unsigned int OCLPinnedMemory::close() {
  delete[] host_memory_;
  return OCLTestImp::close();
}
