////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
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

// Generally useful utility functions

#ifndef HSA_RUNTIME_CORE_UTIL_UTILS_H_
#define HSA_RUNTIME_CORE_UTIL_UTILS_H_

#include "stdint.h"
#include "stddef.h"
#include "stdlib.h"
#include "stdarg.h"
#include "unistd.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <thread>

namespace rocr {
extern FILE* log_file;
extern uint8_t log_flags[8];

typedef unsigned int uint;
typedef uint64_t uint64;

#if defined(__GNUC__)
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#endif

#define __forceinline __inline__ __attribute__((always_inline))
#define __declspec(x) __attribute__((x))
#undef __stdcall
#define __stdcall  // __attribute__((__stdcall__))
#define __ALIGNED__(x) __attribute__((aligned(x)))

void log_printf(const char* file, int line, const char* format, ...);

static __forceinline void* _aligned_malloc(size_t size, size_t alignment) {
#ifdef _ISOC11_SOURCE
  return aligned_alloc(alignment, size);
#else
  void *mem = NULL;
  if (0 != posix_memalign(&mem, alignment, size)) return NULL;
  return mem;
#endif
}
static __forceinline void _aligned_free(void* ptr) { return free(ptr); }
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include "intrin.h"
#define __ALIGNED__(x) __declspec(align(x))
#if (_MSC_VER < 1800)  // < VS 2013
static __forceinline unsigned long long int strtoull(const char* str,
                                                     char** endptr, int base) {
  return static_cast<unsigned long long>(_strtoui64(str, endptr, base));
}
#endif
#if (_MSC_VER < 1900)  // < VS 2015
#define thread_local __declspec(thread)
#endif
#else
#error "Compiler and/or processor not identified."
#endif

#define STRING2(x) #x
#define STRING(x) STRING2(x)

#define PASTE2(x, y) x##y
#define PASTE(x, y) PASTE2(x, y)

#ifdef NDEBUG
#define debug_warning_n(exp, limit)                                                                \
  do {                                                                                             \
  } while (false)
#else
#define debug_warning_n(exp, limit)                                                                \
  do {                                                                                             \
    static std::atomic<int> count(0);                                                              \
    if (!(exp) && (limit == 0 || count < limit)) {                                                 \
      fprintf(stderr, "Warning: " STRING(exp) " in %s, " __FILE__ ":" STRING(__LINE__) "\n",       \
              __PRETTY_FUNCTION__);                                                                \
      count++;                                                                                     \
    }                                                                                              \
  } while (false)
#endif
#define debug_warning(exp) debug_warning_n((exp), 0)

#ifdef NDEBUG
#define debug_print(fmt, ...)                                                                      \
  do {                                                                                             \
  } while (false)
#else
#define debug_print(fmt, ...)                                                                      \
  do {                                                                                             \
    fprintf(stderr, fmt, ##__VA_ARGS__);                                                           \
  } while (false)
#endif

#ifdef NDEBUG
#define ifdebug if (false)
#else
#define ifdebug if (true)
#endif

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LogPrint(flag, format, ...)                                                                \
  do {                                                                                             \
    if (hsa_flag_isset64(log_flags, flag))                                                         \
      rocr::log_printf(__FILENAME__, __LINE__, format, ##__VA_ARGS__);                             \
  } while (false);

// A macro to remove unused variable warnings
#define UNUSED(x) (void)(x)

// A macro to disallow the copy and move constructor and operator= functions
#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                                         \
  TypeName(const TypeName&) = delete;                                                              \
  TypeName(TypeName&&) = delete;                                                                   \
  void operator=(const TypeName&) = delete;                                                        \
  void operator=(TypeName&&) = delete;

template <typename lambda>
class ScopeGuard {
 public:
  explicit __forceinline ScopeGuard(const lambda& release)
      : release_(release), dismiss_(false) {}

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

template <typename lambda>
static __forceinline ScopeGuard<lambda> MakeScopeGuard(lambda rel) {
  return ScopeGuard<lambda>(rel);
}

#define MAKE_SCOPE_GUARD_HELPER(lname, sname, ...) \
  auto lname = __VA_ARGS__;                        \
  ScopeGuard<decltype(lname)> sname(lname);
#define MAKE_SCOPE_GUARD(...)                                   \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), \
                          PASTE(scopeGuard, __COUNTER__), __VA_ARGS__)
#define MAKE_NAMED_SCOPE_GUARD(name, ...)                             \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), name, \
                          __VA_ARGS__)

/// @brief: Finds out the min one of two inputs, input must support ">"
/// operator.
/// @param: a(Input), a reference to type T.
/// @param: b(Input), a reference to type T.
/// @return: T.
template <class T>
static __forceinline T Min(const T& a, const T& b) {
  return (a > b) ? b : a;
}

template <class T, class... Arg>
static __forceinline T Min(const T& a, const T& b, Arg... args) {
  return Min(a, Min(b, args...));
}

/// @brief: Find out the max one of two inputs, input must support ">" operator.
/// @param: a(Input), a reference to type T.
/// @param: b(Input), a reference to type T.
/// @return: T.
template <class T>
static __forceinline T Max(const T& a, const T& b) {
  return (b > a) ? b : a;
}

template <class T, class... Arg>
static __forceinline T Max(const T& a, const T& b, Arg... args) {
  return Max(a, Max(b, args...));
}

/// @brief: Free the memory space which is newed previously.
/// @param: ptr(Input), a pointer to memory space. Can't be NULL.
/// @return: void.
struct DeleteObject {
  template <typename T>
  void operator()(const T* ptr) const {
    delete ptr;
  }
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
  return (T)((value / alignment) * alignment);
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

/// @brief: Flush the cachelines associated with the
/// provided address, offset, and length
/// @param: base(Input), base address to flush
/// @param: offset(Input), offset of base address to flush
/// @param: len(Input), length of buffer to flush
inline void FlushCpuCache(const void* base, size_t offset, size_t len) {
  static long cacheline_size = 0;

  if (!cacheline_size) {
    long sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sz <= 0) return;
    cacheline_size = sz;
  }

  const char* cur = (const char*)base;
  cur += offset;
  uintptr_t lastline = (uintptr_t)(cur + len - 1) | (cacheline_size - 1);
  do {
    _mm_clflush((const void*)cur);
    cur += cacheline_size;
  } while (cur <= (const char*)lastline);
}

}  // namespace rocr

template <uint32_t lowBit, uint32_t highBit, typename T>
static __forceinline uint32_t BitSelect(T p) {
  static_assert(sizeof(T) <= sizeof(uintptr_t), "Type out of range.");
  static_assert(highBit < sizeof(uintptr_t) * 8, "Bit index out of range.");

  uintptr_t ptr = p;
  if (highBit != (sizeof(uintptr_t) * 8 - 1))
    return (uint32_t)((ptr & ((1ull << (highBit + 1)) - 1)) >> lowBit);
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

/// @brief: Concatenates two numbers of type InType to a number of type OutType
/// @param: hi(Input), To be placed in the upper bits of the output
/// @param: lo(Input), To be placed in the lower bits of the output
/// @return: OutType, Concatenation of hi and lo
template <typename OutType, typename InType>
typename std::enable_if<std::is_integral<OutType>::value && std::is_integral<InType>::value &&
                            sizeof(OutType) >= 2 * sizeof(InType),
                        OutType>::type
Concat(InType hi, InType lo) {
  OutType res = ((static_cast<OutType>(hi) << sizeof(InType) * 8) | static_cast<OutType>(lo));
  return res;
}


#include "atomic_helpers.h"

#endif  // HSA_RUNTIME_CORE_UTIL_UTILS_H_
