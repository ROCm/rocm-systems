/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>

constexpr int kMemOrder = __ATOMIC_RELAXED;
constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

// Trivially-copyable class.
class DummyTC {
 public:
  __device__ DummyTC() {}
  __device__ ~DummyTC() = default;
  __device__ DummyTC(const DummyTC&) = default;
  __device__ DummyTC& operator=(const DummyTC&) = default;
  __device__ DummyTC(DummyTC&&) = default;
  __device__ DummyTC& operator=(DummyTC&&) = default;
};

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

__global__ void StoreCompileKernel(int* x) {
  // Valid combinations
  __hip_atomic_store(x, 1, __ATOMIC_RELAXED, kMemScope);
  __hip_atomic_store(x, 1, __ATOMIC_RELEASE, kMemScope);
  __hip_atomic_store(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  __hip_atomic_store(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  __hip_atomic_store(*x, 1, kMemOrder, kMemScope);
  // Consume not allowed by C++1 for store
  __hip_atomic_store(x, 1, __ATOMIC_CONSUME, kMemScope);
  // Acquire not allowed by C++11 for store
  __hip_atomic_store(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  // Acquire-Release not allowed by C++11 for store
  __hip_atomic_store(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  // Memory order is out of bounds
  __hip_atomic_store(x, 1, -1, kMemScope);
  __hip_atomic_store(x, 1, 10, kMemScope);
  // Memory scope is out of bounds
  __hip_atomic_store(x, 1, kMemOrder, -1);
  __hip_atomic_store(x, 1, kMemOrder, 10);

  // Storing an object that is not trivially-copyable
  Dummy dummy_a{};
  Dummy dummy_b{};
  __hip_atomic_store(&dummy_a, dummy_b, kMemOrder, kMemScope);

  // Storing an object that is trivially-copyable
  DummyTC dummytc_a{};
  DummyTC dummytc_b{};
  __hip_atomic_store(&dummytc_a, dummytc_b, kMemOrder, kMemScope);
}

__global__ void LoadCompileKernel(int* x, int* y) {
  // Valid combinations
  *y = __hip_atomic_load(x, __ATOMIC_RELAXED, kMemScope);
  *y = __hip_atomic_load(x, __ATOMIC_CONSUME, kMemScope);
  *y = __hip_atomic_load(x, __ATOMIC_ACQUIRE, kMemScope);
  *y = __hip_atomic_load(x, __ATOMIC_SEQ_CST, kMemScope);

  // Value instead of pointer to the atomic builtin for 1st parameter
  *y = __hip_atomic_load(*x, kMemOrder, kMemScope);
  // Release not allowed by C++11 for load
  *y = __hip_atomic_load(x, __ATOMIC_RELEASE, kMemScope);
  // Acquire-Release not allowed by C++11 for load
  *y = __hip_atomic_load(x, __ATOMIC_ACQ_REL, kMemScope);
  // Memory order is out of bounds
  *y = __hip_atomic_load(x, -1, kMemScope);
  *y = __hip_atomic_load(x, 10, kMemScope);
  // Memory scope is out of bounds
  *y = __hip_atomic_load(x, kMemOrder, -1);
  *y = __hip_atomic_load(x, kMemOrder, 10);

  // Loading an object that is not trivially-copyable
  Dummy dummy_a{};
  Dummy dummy_b{};
  dummy_a = __hip_atomic_load(&dummy_b, kMemOrder, kMemScope);

  // Loading an object that is trivially-copyable
  DummyTC dummytc_a{};
  DummyTC dummytc_b{};
  dummytc_a = __hip_atomic_load(&dummytc_b, kMemOrder, kMemScope);
}

__global__ void CompareWeakCompileKernel(int* x, int* expected) {
  bool res{false};
  // Valid combinations
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_RELAXED,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_CONSUME,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_CONSUME,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_CONSUME,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_CONSUME,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_CONSUME,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQ_REL,
                                           kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST,
                                           kMemScope);

  // Release not allowed on fail by C++11
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, __ATOMIC_RELEASE, kMemScope);
  // Acquire-Release not allowed on fail by C++11
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, __ATOMIC_ACQ_REL, kMemScope);
  // Fail stronger than success
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_SEQ_CST,
                                           kMemScope);
  // Pointer to a non-const type
  res = __hip_atomic_compare_exchange_weak(reinterpret_cast<const int*>(x), expected, 1, kMemOrder,
                                           kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  res = __hip_atomic_compare_exchange_weak(*x, expected, 1, kMemOrder, kMemOrder, kMemScope);
  // Memory order on success is out of bounds
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, -1, kMemOrder, kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, 10, kMemOrder, kMemScope);
  // Memory order on failure is out of bounds
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, -1, kMemScope);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, 10, kMemScope);
  // Memory scope is out of bounds
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, kMemOrder, -1);
  res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, kMemOrder, 10);

  // User-defined class is not trivially-copyable and therefore cannot be atomically copied
  Dummy dummy_a{};
  Dummy dummy_b{};
  Dummy dummy_c{};
  res = __hip_atomic_compare_exchange_weak(&dummy_a, &dummy_b, dummy_c, kMemOrder, kMemOrder,
                                           kMemScope);
  // User-defined class is trivially-copyable and can be atomically copied
  DummyTC dummytc_a{};
  DummyTC dummytc_b{};
  DummyTC dummytc_c{};
  res = __hip_atomic_compare_exchange_weak(&dummytc_a, &dummytc_b, dummytc_c, kMemOrder, kMemOrder,
                                           kMemScope);
}

__global__ void CompareStrongCompileKernel(int* x, int* expected) {
  bool res{false};
  // Valid combinations
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_RELAXED,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_CONSUME,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_CONSUME,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_CONSUME,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_CONSUME,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_CONSUME,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQ_REL,
                                             kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST,
                                             kMemScope);

  // Release not allowed on fail by C++11
  res =
      __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, __ATOMIC_RELEASE, kMemScope);
  // Acquire-Release not allowed on fail by C++11
  res =
      __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, __ATOMIC_ACQ_REL, kMemScope);
  // Fail stronger than success
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_SEQ_CST,
                                             kMemScope);
  // Pointer to a non-const type
  res = __hip_atomic_compare_exchange_strong(reinterpret_cast<const int*>(x), expected, 1,
                                             kMemOrder, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin for 1st parameter
  res = __hip_atomic_compare_exchange_strong(*x, expected, 1, kMemOrder, kMemOrder, kMemScope);
  // Memory order on success is out of bounds
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, -1, kMemOrder, kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, 10, kMemOrder, kMemScope);
  // Memory order on failure is out of bounds
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, -1, kMemScope);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, 10, kMemScope);
  // Memory scope is out of bounds
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, kMemOrder, -1);
  res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, kMemOrder, 10);

  // User-defined class is not trivially-copyable and therefore cannot be atomically copied
  Dummy dummy_a{};
  Dummy dummy_b{};
  Dummy dummy_c{};
  res = __hip_atomic_compare_exchange_strong(&dummy_a, &dummy_b, dummy_c, kMemOrder, kMemOrder,
                                             kMemScope);
  // User-defined class is trivially-copyable and can be atomically copied
  DummyTC dummytc_a{};
  DummyTC dummytc_b{};
  DummyTC dummytc_c{};
  res = __hip_atomic_compare_exchange_strong(&dummytc_a, &dummytc_b, dummytc_c, kMemOrder,
                                             kMemOrder, kMemScope);
}

__global__ void ExchangeCompileKernel(int* x) {
  int old{};
  // Valid combinations
  old = __hip_atomic_exchange(x, 1, __ATOMIC_RELAXED, kMemScope);
  old = __hip_atomic_exchange(x, 1, __ATOMIC_CONSUME, kMemScope);
  old = __hip_atomic_exchange(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  old = __hip_atomic_exchange(x, 1, __ATOMIC_RELEASE, kMemScope);
  old = __hip_atomic_exchange(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  old = __hip_atomic_exchange(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  old = __hip_atomic_exchange(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  old = __hip_atomic_exchange(*x, 1, kMemOrder, kMemScope);
  // Memory order out of bounds
  old = __hip_atomic_exchange(x, 1, -1, kMemScope);
  old = __hip_atomic_exchange(x, 1, 10, kMemScope);
  // Memory scope out of bounds
  old = __hip_atomic_exchange(x, 1, kMemOrder, -1);
  old = __hip_atomic_exchange(x, 1, kMemOrder, 10);

  // User-defined class is not trivially-copyable and therefore cannot be atomically copied
  Dummy dummy_a{};
  Dummy dummy_b{};
  dummy_b = __hip_atomic_exchange(&dummy_a, dummy_b, kMemOrder, kMemScope);

  // User-defined class is trivially-copyable and can be atomically copied
  DummyTC dummytc_a{};
  DummyTC dummytc_b{};
  dummytc_b = __hip_atomic_exchange(&dummytc_a, dummytc_b, kMemOrder, kMemScope);
}

__global__ void FetchAddCompileKernel(int* x) {
  int old{};
  // Valid combinations
  old = __hip_atomic_fetch_add(x, 1, __ATOMIC_RELAXED, kMemScope);
  old = __hip_atomic_fetch_add(x, 1, __ATOMIC_CONSUME, kMemScope);
  old = __hip_atomic_fetch_add(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  old = __hip_atomic_fetch_add(x, 1, __ATOMIC_RELEASE, kMemScope);
  old = __hip_atomic_fetch_add(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  old = __hip_atomic_fetch_add(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  old = __hip_atomic_fetch_add(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  old = __hip_atomic_fetch_add(*x, 1, kMemOrder, kMemScope);
  // Memory order out of bounds
  old = __hip_atomic_fetch_add(x, 1, -1, kMemScope);
  old = __hip_atomic_fetch_add(x, 1, 10, kMemScope);
  // Memory scope out of bounds
  old = __hip_atomic_fetch_add(x, 1, kMemOrder, -1);
  old = __hip_atomic_fetch_add(x, 1, kMemOrder, 10);

  Dummy dummy{};
  old = __hip_atomic_fetch_add(&dummy, 1, kMemOrder, kMemScope);
}

__global__ void FetchAndCompileKernel(int* x) {
  int old{};
  // Valid combinations
  old = __hip_atomic_fetch_and(x, 1, __ATOMIC_RELAXED, kMemScope);
  old = __hip_atomic_fetch_and(x, 1, __ATOMIC_CONSUME, kMemScope);
  old = __hip_atomic_fetch_and(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  old = __hip_atomic_fetch_and(x, 1, __ATOMIC_RELEASE, kMemScope);
  old = __hip_atomic_fetch_and(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  old = __hip_atomic_fetch_and(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  old = __hip_atomic_fetch_and(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  old = __hip_atomic_fetch_and(*x, 1, kMemOrder, kMemScope);
  // Memory order out of bounds
  old = __hip_atomic_fetch_and(x, 1, -1, kMemScope);
  old = __hip_atomic_fetch_and(x, 1, 10, kMemScope);
  // Memory scope out of bounds
  old = __hip_atomic_fetch_and(x, 1, kMemOrder, -1);
  old = __hip_atomic_fetch_and(x, 1, kMemOrder, 10);

  // Value must be an integer
  Dummy dummy{};
  old = __hip_atomic_fetch_and(&dummy, 1, kMemOrder, kMemScope);
  float float_var{1.5f};
  old = __hip_atomic_fetch_and(&float_var, 1, kMemOrder, kMemScope);
  double double_var{1.5};
  old = __hip_atomic_fetch_and(&double_var, 1, kMemOrder, kMemScope);
}

__global__ void FetchOrCompileKernel(int* x) {
  int old{};
  // Valid combinations
  old = __hip_atomic_fetch_or(x, 1, __ATOMIC_RELAXED, kMemScope);
  old = __hip_atomic_fetch_or(x, 1, __ATOMIC_CONSUME, kMemScope);
  old = __hip_atomic_fetch_or(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  old = __hip_atomic_fetch_or(x, 1, __ATOMIC_RELEASE, kMemScope);
  old = __hip_atomic_fetch_or(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  old = __hip_atomic_fetch_or(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  old = __hip_atomic_fetch_or(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  old = __hip_atomic_fetch_or(*x, 1, kMemOrder, kMemScope);
  // Memory order out of bounds
  old = __hip_atomic_fetch_or(x, 1, -1, kMemScope);
  old = __hip_atomic_fetch_or(x, 1, 10, kMemScope);
  // Memory scope out of bounds
  old = __hip_atomic_fetch_or(x, 1, kMemOrder, -1);
  old = __hip_atomic_fetch_or(x, 1, kMemOrder, 10);

  // Value must be an integer
  Dummy dummy{};
  old = __hip_atomic_fetch_or(&dummy, 1, kMemOrder, kMemScope);
  float float_var{1.5f};
  old = __hip_atomic_fetch_or(&float_var, 1, kMemOrder, kMemScope);
  double double_var{1.5};
  old = __hip_atomic_fetch_or(&double_var, 1, kMemOrder, kMemScope);
}

__global__ void FetchXorCompileKernel(int* x) {
  int old{};
  // Valid combinations
  old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_RELAXED, kMemScope);
  old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_CONSUME, kMemScope);
  old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_RELEASE, kMemScope);
  old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  old = __hip_atomic_fetch_xor(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  old = __hip_atomic_fetch_xor(*x, 1, kMemOrder, kMemScope);
  // Memory order out of bounds
  old = __hip_atomic_fetch_xor(x, 1, -1, kMemScope);
  old = __hip_atomic_fetch_xor(x, 1, 10, kMemScope);
  // Memory scope out of bounds
  old = __hip_atomic_fetch_xor(x, 1, kMemOrder, -1);
  old = __hip_atomic_fetch_xor(x, 1, kMemOrder, 10);

  // Value must be an integer
  Dummy dummy{};
  old = __hip_atomic_fetch_xor(&dummy, 1, kMemOrder, kMemScope);
  float float_var{1.5f};
  old = __hip_atomic_fetch_xor(&float_var, 1, kMemOrder, kMemScope);
  double double_var{1.5};
  old = __hip_atomic_fetch_xor(&double_var, 1, kMemOrder, kMemScope);
}

__global__ void FetchMaxCompileKernel(int* x) {
  int old{};
  // Valid combinations
  old = __hip_atomic_fetch_max(x, 1, __ATOMIC_RELAXED, kMemScope);
  old = __hip_atomic_fetch_max(x, 1, __ATOMIC_CONSUME, kMemScope);
  old = __hip_atomic_fetch_max(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  old = __hip_atomic_fetch_max(x, 1, __ATOMIC_RELEASE, kMemScope);
  old = __hip_atomic_fetch_max(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  old = __hip_atomic_fetch_max(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  old = __hip_atomic_fetch_max(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  old = __hip_atomic_fetch_max(*x, 1, kMemOrder, kMemScope);
  // Memory order out of bounds
  old = __hip_atomic_fetch_max(x, 1, -1, kMemScope);
  old = __hip_atomic_fetch_max(x, 1, 10, kMemScope);
  // Memory scope out of bounds
  old = __hip_atomic_fetch_max(x, 1, kMemOrder, -1);
  old = __hip_atomic_fetch_max(x, 1, kMemOrder, 10);

  // Value must be integer or floating point type
  Dummy dummy{};
  old = __hip_atomic_fetch_max(&dummy, 1, kMemOrder, kMemScope);
}

__global__ void FetchMinCompileKernel(int* x) {
  int old{};
  // Valid combinations
  old = __hip_atomic_fetch_min(x, 1, __ATOMIC_RELAXED, kMemScope);
  old = __hip_atomic_fetch_min(x, 1, __ATOMIC_CONSUME, kMemScope);
  old = __hip_atomic_fetch_min(x, 1, __ATOMIC_ACQUIRE, kMemScope);
  old = __hip_atomic_fetch_min(x, 1, __ATOMIC_RELEASE, kMemScope);
  old = __hip_atomic_fetch_min(x, 1, __ATOMIC_ACQ_REL, kMemScope);
  old = __hip_atomic_fetch_min(x, 1, __ATOMIC_SEQ_CST, kMemScope);

  // Pointer to a non-const type
  old = __hip_atomic_fetch_min(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
  // Value instead of pointer to the atomic builtin
  old = __hip_atomic_fetch_min(*x, 1, kMemOrder, kMemScope);
  // Memory order out of bounds
  old = __hip_atomic_fetch_min(x, 1, -1, kMemScope);
  old = __hip_atomic_fetch_min(x, 1, 10, kMemScope);
  // Memory scope out of bounds
  old = __hip_atomic_fetch_min(x, 1, kMemOrder, -1);
  old = __hip_atomic_fetch_min(x, 1, kMemOrder, 10);

  // Value must be integer or floating point type
  Dummy dummy{};
  old = __hip_atomic_fetch_min(&dummy, 1, kMemOrder, kMemScope);
}
