////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 AMD Research and AMD HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

/*
  Helpers to use native types with C++11 atomic operations.
  Fixes GCC builtin functionality for x86 with respect to WC and non-temporal
  stores.
*/
#ifndef HSA_RUNTIME_CORE_UTIL_ATOMIC_HELPERS_H_
#define HSA_RUNTIME_CORE_UTIL_ATOMIC_HELPERS_H_

#include <atomic>

//ALWAYS_CONSERVATIVE will very likely overfence your code.
//For use as a debugging aid only.
#define ALWAYS_CONSERVATIVE 0

#if !ALWAYS_CONSERVATIVE
#if defined(__x86_64__) || defined(_M_X64)
#define X64_ORDER_WC 1
#endif
#if X64_ORDER_WC
#include <xmmintrin.h>
#endif
#endif

namespace rocr {
namespace atomic {

static constexpr int c11ToBuiltInFlags(std::memory_order order)
{
#if ALWAYS_CONSERVATIVE
  return __ATOMIC_RELAXED;
#elif X64_ORDER_WC
  return __ATOMIC_RELAXED;
#else
  return (order == std::memory_order_relaxed) ? __ATOMIC_RELAXED :
    (order == std::memory_order_acquire) ? __ATOMIC_ACQUIRE :
    (order == std::memory_order_release) ? __ATOMIC_RELEASE :
    (order == std::memory_order_seq_cst) ? __ATOMIC_SEQ_CST :
    (order == std::memory_order_consume) ? __ATOMIC_CONSUME :
    (order == std::memory_order_acq_rel) ? __ATOMIC_ACQ_REL :
    __ATOMIC_SEQ_CST;
#endif
}

static __forceinline void PreFence(std::memory_order order) {
#if ALWAYS_CONSERVATIVE
  switch (order) {
    case std::memory_order_release:
    case std::memory_order_seq_cst:
    case std::memory_order_acq_rel:
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
    default:;
  }
#elif X64_ORDER_WC
  switch (order) {
    case std::memory_order_release:
    case std::memory_order_seq_cst:
    case std::memory_order_acq_rel:
      _mm_sfence();
    default:;
  }
#endif
}

static __forceinline void PostFence(std::memory_order order) {
#if ALWAYS_CONSERVATIVE
  switch (order) {
    case std::memory_order_seq_cst:
    case std::memory_order_acq_rel:
    case std::memory_order_acquire:
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
    default:;
  }
#elif X64_ORDER_WC
  switch (order) {
    case std::memory_order_seq_cst:
      return _mm_mfence();
    case std::memory_order_acq_rel:
    case std::memory_order_acquire:
      return _mm_lfence();
    default:;
  }
#endif
}

static __forceinline void Fence(std::memory_order order=std::memory_order_seq_cst) {
#if ALWAYS_CONSERVATIVE
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#elif X64_ORDER_WC
  switch (order) {
    case std::memory_order_seq_cst:
    case std::memory_order_acq_rel:
      return _mm_mfence();
    case std::memory_order_acquire:
      return _mm_lfence();
    case std::memory_order_release:
      return _mm_sfence();
    default:;
  }
#else
  std::atomic_thread_fence(order);
#endif
}

template <class T>
static __forceinline void BasicCheck(const T* ptr) {
  constexpr bool value = __atomic_always_lock_free(sizeof(T), 0);
  static_assert(value, "Atomic type may not be compatible with peripheral atomics.");
};

template <class T>
static __forceinline void BasicCheck(const volatile T* ptr) {
  constexpr bool value = __atomic_always_lock_free(sizeof(T), 0);
  static_assert(value, "Atomic type may not be compatible with peripheral atomics.");
};

/// @brief: Load value of type T atomically with specified memory order.
/// @param: ptr(Input), a pointer to type T.
/// @param: order(Input), memory order with atomic load, relaxed by default.
/// @return: T, loaded value.
template <class T>
static __forceinline T
    Load(const T* ptr, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  T ret;
  PreFence(order);
  __atomic_load(ptr, &ret, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: function overloading, for more info, see previous one.
/// @param: ptr(Input), a pointer to volatile type T.
/// @param: order(Input), memory order with atomic load, relaxed by default.
/// @return: T, loaded value.
template <class T>
static __forceinline T
    Load(const volatile T* ptr,
         std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  T ret;
  PreFence(order);
  __atomic_load(ptr, &ret, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Store value of type T with specified memory order.
/// @param: ptr(Input), a pointer to instance which will be stored.
/// @param: val(Input), value to be stored.
/// @param: order(Input), memory order with atomic store, relaxed by default.
/// @return: void.
template <class T>
static __forceinline void Store(
    T* ptr, T val, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  __atomic_store(ptr, &val, c11ToBuiltInFlags(order));
  PostFence(order);
}

/// @brief: Function overloading, for more info, see previous one.
/// @param: ptr(Input), a pointer to volatile instance which will be stored.
/// @param: val(Input), value to be stored.
/// @param: order(Input), memory order with atomic store, relaxed by default.
/// @return: void.
template <class T>
static __forceinline void Store(
    volatile T* ptr, T val,
    std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  __atomic_store(ptr, &val, c11ToBuiltInFlags(order));
  PostFence(order);
}

/// @brief: Compare and swap value atomically with specified memory order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value to be stored if condition is satisfied.
/// @param: expected(Input), value which is expected.
/// @param: order(Input), memory order with atomic operation.
/// @return: T, observed value of type T.
template <class T>
static __forceinline T
    Cas(T* ptr, T val, T expected,
        std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  __atomic_compare_exchange(ptr, &expected, &val, false, c11ToBuiltInFlags(order), __ATOMIC_RELAXED);
  PostFence(order);
  return expected;
}

/// @brief: Function overloading, for more info, see previous one.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: val(Input), value to be stored if condition is satisfied.
/// @param: expected(Input), value which is expected.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, observed value of type T.
template <class T>
static __forceinline T
    Cas(volatile T* ptr, T val, T expected,
        std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  __atomic_compare_exchange(ptr, &expected, &val, false, c11ToBuiltInFlags(order), __ATOMIC_RELAXED);
  PostFence(order);
  return expected;
}

/// @brief: Exchange the value atomically with specified memory order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value to be stored.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, the value prior to the exchange.
template <class T>
static __forceinline T
    Exchange(T* ptr, T val,
             std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  T ret;
  PreFence(order);
  __atomic_exchange(ptr, &val, &ret, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Function overloading, for more info, see previous one.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value to be stored.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, the value prior to the exchange.
template <class T>
static __forceinline T
    Exchange(volatile T* ptr, T val,
             std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  T ret;
  PreFence(order);
  __atomic_exchange(ptr, &val, &ret, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Add value to variable atomically with specified memory order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value to be added.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, the value of the variable prior to the addition.
template <class T>
static __forceinline T
    Add(T* ptr, T val, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_add(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Subtract value from the variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value to be subtraced.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of the variable prior to the subtraction.
template <class T>
static __forceinline T
    Sub(T* ptr, T val, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_sub(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Bit And operation on variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value which is ANDed with variable.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T
    And(T* ptr, T val, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_and(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Bit Or operation on variable atomically with specified memory order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value which is ORed with variable.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T
    Or(T* ptr, T val, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_or(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Bit Xor operation on variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: val(Input), value which is XORed with variable.
/// @order: order(Input), memory order which is relaxed by default.
/// @return: T, valud of variable prior to the opertaion.
template <class T>
static __forceinline T
    Xor(T* ptr, T val, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_xor(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Increase the value of variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T
    Increment(T* ptr, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_add(ptr, 1, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Decrease the value of the variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to variable which is operated on.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T
    Decrement(T* ptr, std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_sub(ptr, 1, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Add value to variable atomically with specified memory order.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: val(Input), value to be added.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, the value of the variable prior to the addition.
template <class T>
static __forceinline T
    Add(volatile T* ptr, T val,
        std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_add(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Subtract value from the variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: val(Input), value to be subtraced.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of the variable prior to the subtraction.
template <class T>
static __forceinline T
    Sub(volatile T* ptr, T val,
        std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_sub(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Bit And operation on variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: val(Input), value which is ANDed with variable.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T
    And(volatile T* ptr, T val,
        std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_and(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Bit Or operation on variable atomically with specified memory order.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: val(Input), value which is ORed with variable.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T Or(volatile T* ptr, T val,
                          std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_or(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Bit Xor operation on variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: val(Input), value which is XORed with variable.
/// @order: order(Input), memory order which is relaxed by default.
/// @return: T, valud of variable prior to the opertaion.
template <class T>
static __forceinline T
    Xor(volatile T* ptr, T val,
        std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_xor(ptr, val, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Increase the value of variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T
    Increment(volatile T* ptr,
              std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_add(ptr, 1, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}

/// @brief: Decrease the value of the variable atomically with specified memory
/// order.
/// @param: ptr(Input), a pointer to volatile variable which is operated on.
/// @param: order(Input), memory order which is relaxed by default.
/// @return: T, value of variable prior to the operation.
template <class T>
static __forceinline T
    Decrement(volatile T* ptr,
              std::memory_order order = std::memory_order_relaxed) {
  BasicCheck<T>(ptr);
  PreFence(order);
  T ret = __atomic_fetch_sub(ptr, 1, c11ToBuiltInFlags(order));
  PostFence(order);
  return ret;
}
}   //  namespace atomic
}   //  namespace rocr

#ifdef X64_ORDER_WC
#undef X64_ORDER_WC
#endif

#ifdef ALWAYS_CONSERVATIVE
#undef ALWAYS_CONSERVATIVE
#endif

#endif  // HSA_RUNTIME_CORE_UTIL_ATOMIC_HELPERS_H_
