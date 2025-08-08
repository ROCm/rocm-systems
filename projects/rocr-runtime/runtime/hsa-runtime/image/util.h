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

#ifndef HSA_RUNTIME_EXT_IMAGE_UTIL_H
#define HSA_RUNTIME_EXT_IMAGE_UTIL_H

#include "stdint.h"
#include "stddef.h"
#include "stdlib.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>

#include "inc/hsa.h"

namespace rocr {
namespace image {

#if defined(_MSC_VER)
#define ALIGNED_(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define ALIGNED_(x) __attribute__((aligned(x)))
#endif  // __GNUC__
#endif // _MSC_VER

#define MULTILINE(...) # __VA_ARGS__

#define ASSERT_SIZE_UINT32(desc)                                                                   \
  static_assert(sizeof(desc) == sizeof(uint32_t), #desc " size should be 32-bits");

}  // namespace image
}  // namespace rocr


#if defined(__GNUC__)
#include "mm_malloc.h"
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#elif defined(__loongarch64)
#else
#error                                                                                             \
    "Processor not identified.  " \
            "Need to provide a lightweight approximate clock interface (aka __rdtsc())."
#endif

namespace rocr {
namespace image {

#define __forceinline __inline__ __attribute__((always_inline))
static __forceinline void __debugbreak() { __builtin_trap(); }
#define __declspec(x) __attribute__((x))
#undef __stdcall
#define __stdcall  // __attribute__((__stdcall__))
#define __ALIGNED__(x) __attribute__((aligned(x)))

static __forceinline void* _aligned_malloc(size_t size, size_t alignment) {
#ifdef _ISOC11_SOURCE
  return aligned_alloc(alignment, size);
#else
  void* mem = NULL;
  if (0 != posix_memalign(&mem, alignment, size)) return NULL;
  return mem;
#endif
}
static __forceinline void _aligned_free(void* ptr) { return free(ptr); }
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include "intrin.h"
#define __ALIGNED__(x) __declspec(align(x))

namespace rocr {
namespace image {
#else
#error "Compiler and/or processor not identified."
#endif

// A macro to disallow the copy and move constructor and operator= functions
#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                                         \
  TypeName(const TypeName&) = delete;                                                              \
  TypeName(TypeName&&) = delete;                                                                   \
  void operator=(const TypeName&) = delete;                                                        \
  void operator=(TypeName&&) = delete;

template <typename lambda> class ScopeGuard {
 public:
  explicit __forceinline ScopeGuard(const lambda& release) : release_(release), dismiss_(false) {}

  ScopeGuard(ScopeGuard& rhs) { *this = rhs; }

  __forceinline ~ScopeGuard() {
    if (!dismiss_) release_();
  }
  __forceinline ScopeGuard& operator=(ScopeGuard& rhs) {
    dismiss_ = rhs.dismiss_;
    release_ = rhs.release_;
    rhs.dismiss_ = true;
    return *this;
  }
  __forceinline void Dismiss() { dismiss_ = true; }

 private:
  lambda release_;
  bool dismiss_;
};

template <typename lambda> static __forceinline ScopeGuard<lambda> MakeScopeGuard(lambda rel) {
  return ScopeGuard<lambda>(rel);
}

#define MAKE_SCOPE_GUARD_HELPER(lname, sname, ...)                                                 \
  auto lname = __VA_ARGS__;                                                                        \
  ScopeGuard<decltype(lname)> sname(lname);
#define MAKE_SCOPE_GUARD(...)                                                                      \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), PASTE(scopeGuard, __COUNTER__),    \
                          __VA_ARGS__)
#define MAKE_NAMED_SCOPE_GUARD(name, ...)                                                          \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), name, __VA_ARGS__)

/// @brief: Finds out the min one of two inputs, input must support ">"
/// operator.
/// @param: a(Input), a reference to type T.
/// @param: b(Input), a reference to type T.
/// @return: T.
template <class T> static __forceinline T Min(const T& a, const T& b) { return (a > b) ? b : a; }

template <class T, class... Arg> static __forceinline T Min(const T& a, const T& b, Arg... args) {
  return Min(a, Min(b, args...));
}

/// @brief: Find out the max one of two inputs, input must support ">" operator.
/// @param: a(Input), a reference to type T.
/// @param: b(Input), a reference to type T.
/// @return: T.
template <class T> static __forceinline T Max(const T& a, const T& b) { return (b > a) ? b : a; }

template <class T, class... Arg> static __forceinline T Max(const T& a, const T& b, Arg... args) {
  return Max(a, Max(b, args...));
}

/// @brief: Free the memory space which is newed previously.
/// @param: ptr(Input), a pointer to memory space. Can't be NULL.
/// @return: void.
struct DeleteObject {
  template <typename T> void operator()(const T* ptr) const { delete ptr; }
};

/// @brief: Checks if a value is power of two, if it is, return true. Be careful
/// when passing 0.
/// @param: val(Input), the data to be checked.
/// @return: bool.
template <typename T>
static __forceinline bool IsPowerOfTwo(T val) {
  return (val & (val - 1)) == 0;
}

/// @brief: Calculates the floor value aligned based on parameter of alignment.
/// If value is at the boundary of alignment, it is unchanged.
/// @param: value(Input), value to be calculated.
/// @param: alignment(Input), alignment value.
/// @return: T.
template <typename T>
static __forceinline T AlignDown(T value, size_t alignment) {
  assert(IsPowerOfTwo(alignment));
  return (T)(value & ~(alignment - 1));
}

/// @brief: Same as previous one, but first parameter becomes pointer, for more
/// info, see the previous desciption.
/// @param: value(Input), pointer to type T.
/// @param: alignment(Input), alignment value.
/// @return: T*, pointer to type T.
template <typename T>
static __forceinline T* AlignDown(T* value, size_t alignment) {
  return (T*)AlignDown((intptr_t)value, alignment);
}

/// @brief: Calculates the ceiling value aligned based on parameter of
/// alignment.
/// If value is at the boundary of alignment, it is unchanged.
/// @param: value(Input), value to be calculated.
/// @param: alignment(Input), alignment value.
/// @param: T.
template <typename T>
static __forceinline T AlignUp(T value, size_t alignment) {
  return AlignDown((T)(value + alignment - 1), alignment);
}

/// @brief: Same as previous one, but first parameter becomes pointer, for more
/// info, see the previous desciption.
/// @param: value(Input), pointer to type T.
/// @param: alignment(Input), alignment value.
/// @return: T*, pointer to type T.
template <typename T>
static __forceinline T* AlignUp(T* value, size_t alignment) {
  return (T*)AlignDown((intptr_t)((uint8_t*)value + alignment - 1), alignment);
}

/// @brief: Checks if the input value is at the boundary of alignment, if it is,
/// @return true.
/// @param: value(Input), value to be checked.
/// @param: alignment(Input), alignment value.
/// @return: bool.
template <typename T>
static __forceinline bool IsMultipleOf(T value, size_t alignment) {
  return (AlignUp(value, alignment) == value);
}

/// @brief: Same as previous one, but first parameter becomes pointer, for more
/// info, see the previous desciption.
/// @param: value(Input), pointer to type T.
/// @param: alignment(Input), alignment value.
/// @return: bool.
template <typename T>
static __forceinline bool IsMultipleOf(T* value, size_t alignment) {
  return (AlignUp(value, alignment) == value);
}

static __forceinline uint32_t NextPow2(uint32_t value) {
  if (value == 0) return 1;
  uint32_t v = value - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

static __forceinline uint64_t NextPow2(uint64_t value) {
  if (value == 0) return 1;
  uint64_t v = value - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}

static __forceinline bool strIsEmpty(const char* str) noexcept { return str[0] == '\0'; }

static __forceinline std::string& ltrim(std::string& s) {
  auto it = std::find_if(s.begin(), s.end(),
                         [](char c) { return !std::isspace<char>(c, std::locale::classic()); });
  s.erase(s.begin(), it);
  return s;
}

static __forceinline std::string& rtrim(std::string& s) {
  auto it = std::find_if(s.rbegin(), s.rend(),
                         [](char c) { return !std::isspace<char>(c, std::locale::classic()); });
  s.erase(it.base(), s.end());
  return s;
}

static __forceinline std::string& trim(std::string& s) { return ltrim(rtrim(s)); }

template<uint32_t lowBit, uint32_t highBit, typename T>
static __forceinline uint32_t BitSelect(T p) {
  static_assert(sizeof(T) <= sizeof(uintptr_t), "Type out of range.");
  static_assert(highBit < sizeof(uintptr_t)*8, "Bit index out of range.");

  uintptr_t ptr = p;
  if(highBit != (sizeof(uintptr_t)*8-1))
    return (uint32_t)((ptr & ((1ull<<(highBit+1))-1)) >> lowBit);
  else
    return (uint32_t)(ptr >> lowBit);
}

inline uint32_t PtrLow16Shift8(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFULL) >> 8);
}

inline uint32_t PtrHigh64Shift16(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFFFFFFFFF0000ULL) >> 16);
}

inline uint32_t PtrLow40Shift8(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFFFFFFFULL) >> 8);
}

inline uint32_t PtrHigh64Shift40(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFFF0000000000ULL) >> 40);
}

inline uint32_t PtrLow32(const void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

inline uint32_t PtrHigh32(const void* p) {
  uint32_t ptr = 0;
#ifdef HSA_LARGE_MODEL
  ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) >> 32);
#endif
  return ptr;
}

}  // namespace image
}  // namespace rocr

#endif  // HSA_RUNTIME_EXT_IMAGE_UTIL_H
