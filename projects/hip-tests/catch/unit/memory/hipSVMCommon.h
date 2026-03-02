// Copyright (c) 2017 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/*
 * Modifications Copyright (C)2023 Advanced
 * Micro Devices, Inc. All rights reserved.
 */
#ifndef __HIPSVMCOMMON_H__
#define __HIPSVMCOMMON_H__

#include <vector>
#include <string>

#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
#include <windows.h>
#endif

// SVM Atomic wrappers.
// Platforms that support SVM atomics (atomics that work across the host and devices) need to
// implement these host side functions correctly. Platforms that do not support SVM atomics can
// simpy implement these functions as empty stubs since the functions will not be called. For now
// only Windows x86 is implemented, add support for other platforms as needed.
unsigned int inline AtomicLoad32(unsigned int* pValue) {
#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
  return (unsigned int)InterlockedExchangeAdd((LONG*)pValue, 0l);
#elif defined(__GNUC__)
  return __sync_add_and_fetch(pValue, 0);
#else
  return -1;
#endif
}

// all the x86 atomics are seq_cst, so don't need to do anything with the memory order parameter.
unsigned int inline AtomicFetchAdd32(unsigned int* object, int operand) {
#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
  return InterlockedExchangeAdd((LONG*)object, operand);
#elif defined(__GNUC__)
  return __sync_fetch_and_add(object, operand);
#else
  return -1;
#endif
}

template <typename T> T inline AtomicFetchAdd64(T* object, T operand) {
#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
  return (T)InterlockedExchangeAdd64((LONG64*)object, (LONG64)operand);
#elif defined(__GNUC__)
  return (T)__sync_fetch_and_add((intptr_t*)object, (intptr_t)operand);
#else
  return -1;
#endif
}

unsigned int inline AtomicExchange32(unsigned int* object, unsigned int desired) {
#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
  return (unsigned int)InterlockedExchange((LONG*)object, (LONG)desired);
#elif defined(__GNUC__)
  return __sync_lock_test_and_set(object, desired);
#else
  return -1;
#endif
}

template <typename T> T inline AtomicExchange64(T* a, T expected) {
#if defined(_MSC_VER) || (defined(__INTEL_COMPILER) && defined(WIN32))
  return (T)InterlockedExchangePointer((PVOID volatile*)a, (PVOID)expected);
#elif defined(__GNUC__)
  return (T)__sync_lock_test_and_set((long long*)a, (long long)expected);
#else
  tmp = 0;
#endif
}

template <typename T> bool AtomicCompareExchange64(T* a, T* expected, T desired) {
#if defined(_MSC_VER) || (defined(__INTEL_COMPILER) && defined(WIN32))
  T tmp = (T)InterlockedCompareExchange64((LONG64*)a, (LONG64)desired, *(LONG64*)expected);
#elif defined(__GNUC__)
  T tmp = (T)__sync_val_compare_and_swap((intptr_t*)a, (intptr_t)(*expected), (intptr_t)desired);
#else
  tmp = 0;
#endif
  if (tmp == *expected) return true;
  *expected = tmp;
  return false;
}

inline void* align_malloc(size_t size, size_t alignment) {
#if defined(_WIN32) && defined(_MSC_VER)
  return _aligned_malloc(size, alignment);
#elif defined(__linux__) || defined(linux) || defined(__APPLE__)
  void* ptr = NULL;
#if defined(__ANDROID__)
  ptr = memalign(alignment, size);
  if (ptr) return ptr;
#else
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }
  if (0 == posix_memalign(&ptr, alignment, size)) return ptr;
#endif
  return NULL;
#elif defined(__MINGW32__)
  return __mingw_aligned_malloc(size, alignment);
#else
#error "Please add support OS for aligned malloc"
#endif
}

inline void align_free(void* ptr) {
#if defined(_WIN32) && defined(_MSC_VER)
  _aligned_free(ptr);
#elif defined(__linux__) || defined(linux) || defined(__APPLE__)
  return free(ptr);
#elif defined(__MINGW32__)
  return __mingw_aligned_free(ptr);
#else
#error "Please add support OS for aligned free"
#endif
}

#endif  // #ifndef __HIPSVMCOMMON_H__
