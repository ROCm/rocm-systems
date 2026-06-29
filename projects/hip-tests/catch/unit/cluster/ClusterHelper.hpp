/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

template <typename T> class BasicMemoryAllocator {
 public:
  BasicMemoryAllocator(size_t num_elems) : num_elems_(num_elems) {
    num_size_ = num_elems * sizeof(T);
  }
  ~BasicMemoryAllocator() {}

  int* CreateAndResetHostMemory();
  int* CreateAndResetDeviceMemory();
  void DestroyHostMemory(T* hptr);
  void DestroyDeviceMemory(T* dptr);

  bool ValidateArrays(T* hptr_in, T* hptr_out);


 private:
  size_t num_elems_;
  size_t num_size_;
};

template <typename T> int* BasicMemoryAllocator<T>::CreateAndResetHostMemory() {
  int* hptr = (int*)malloc(num_size_);
  memset(hptr, 0x00, num_size_);
  return hptr;
}

template <typename T> int* BasicMemoryAllocator<T>::CreateAndResetDeviceMemory() {
  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, num_size_));
  HIP_CHECK(hipMemset(dptr, 0x00, num_size_));
  return dptr;
}

template <typename T> void BasicMemoryAllocator<T>::DestroyHostMemory(T* hptr) { free(hptr); }

template <typename T> void BasicMemoryAllocator<T>::DestroyDeviceMemory(T* dptr) {
  HIP_CHECK(hipFree(dptr));
}

template <typename T> bool BasicMemoryAllocator<T>::ValidateArrays(T* hptr_in, T* hptr_out) {
  for (size_t idx = 0; idx < num_elems_; ++idx) {
    if (hptr_in[idx] != hptr_out[idx]) {
      INFO("First failing index @ %d \n" << idx);
      return false;
    } else {
      INFO("hptr_in[" << idx << "]: " << hptr_in[idx] << " hptr_out[" << idx
                      << "]: " << hptr_out[idx]);
    }
  }
  return true;
}

inline bool CheckTargetSupport() {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  return devProp.clusterLaunch != 0;
}