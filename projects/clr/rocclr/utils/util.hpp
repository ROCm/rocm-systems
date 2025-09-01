/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include "top.hpp"

#include <atomic>
#include <string>

#ifdef _WIN32
#include <intrin.h>
#endif

namespace amd {

/*! \addtogroup Utils Utilities
 *  @{
 */

//! \brief Check if the given value \a val is a power of 2.
template <typename T> static inline bool isPowerOfTwo(T val) { return (val & (val - 1)) == 0; }

//! \cond ignore

// Compute the next power of 2 helper.
template <uint N> struct NextPowerOfTwoFunction {
  template <typename T> static T compute(T val) {
    val = NextPowerOfTwoFunction<N / 2>::compute(val);
    return (val >> N) | val;
  }
};

// Specialized version for <1> to break the recursion.
template <> struct NextPowerOfTwoFunction<1> {
  template <typename T> static T compute(T val) { return (val >> 1) | val; }
};

template <uint N, int S> struct NextPowerOfTwoHelper {
  static constexpr uint prev = NextPowerOfTwoHelper<N, S / 2>::value;
  static constexpr uint value = (prev >> S) | prev;
};
template <uint N> struct NextPowerOfTwoHelper<N, 1> {
  static constexpr int value = (N >> 1) | N;
};

template <uint N> struct NextPowerOfTwo {
  static constexpr uint value = NextPowerOfTwoHelper<N - 1, 16>::value + 1;
};

//! \endcond

/*! \brief Return the next power of two for a value of type T.
 *
 *  The compute function is (with n = sizeof(T)*8):
 *
 *    val = (val >> 1) | val;
 *    val = (val >> 2) | val;
 *    ...
 *    val = (val >> n/4) | val;
 *    val = (val >> n/2) | val;
 *
 *  The next power of two is: 1+compute(val-1)
 */
template <typename T> inline T nextPowerOfTwo(T val) {
  return NextPowerOfTwoFunction<sizeof(T) * 4>::compute(val - 1) + 1;
}

// Compute log2(N)
template <uint N> struct Log2 {
  static constexpr uint value = Log2<N / 2>::value + 1;
};

// Break the recursion
template <> struct Log2<1> {
  static constexpr uint value = 0;
};

/*! \brief Return the log2 for a value of type T.
 *
 * The compute function is (with n = sizeof(T)*8):
 *
 *   uint l = 0;
 *   if (val >= 1 << n/2) { val >>= n/2; l |= n/2; }
 *   if (val >= 1 << n/4) { val >>= n/4; l |= n/4; }
 *   ...
 *   if (val >= 1 << 2) { val >>= 2; l |= 2; }
 *   if (val >= 1 << 1) { l |= 1; }
 *   return l;
 */
template <uint N> struct Log2Function {
  template <typename T> static uint compute(T val) {
    uint l = 0;
    if (val >= T(1) << N) {
      val >>= N;
      l = N;
    }
    return l + Log2Function<N / 2>::compute(val);
  }
};

template <> struct Log2Function<1> {
  template <typename T> static uint compute(T val) { return (val >= T(1) << 1) ? 1 : 0; }
};

// log2 helper function
template <typename T> inline uint log2(T val) { return Log2Function<sizeof(T) * 4>::compute(val); }

template <typename T> inline T alignDown(T value, size_t alignment) {
  return (T)(value & ~(alignment - 1));
}

template <typename T> inline T* alignDown(T* value, size_t alignment) {
  return (T*)alignDown((intptr_t)value, alignment);
}

template <typename T> inline T alignUp(T value, size_t alignment) {
  return alignDown((T)(value + alignment - 1), alignment);
}

template <typename T> inline T* alignUp(T* value, size_t alignment) {
  return (T*)alignDown((intptr_t)(value + alignment - 1), alignment);
}

template <typename T> inline bool isMultipleOf(T value, size_t alignment) {
  if (isPowerOfTwo(alignment)) {
    // fast path, using logical operators
    return alignUp(value, alignment) == value;
  }
  return value % alignment == 0;
}

template <typename T> inline bool isMultipleOf(T* value, size_t alignment) {
  intptr_t ptr = reinterpret_cast<intptr_t>(value);
  return isMultipleOf(ptr, alignment);
}

template <class Reference, class Value> struct DeviceMap {
  Reference ref_;
  Value value_;
};


inline uint countBitsSet32(uint32_t value) {
#if __GNUC__ >= 4
  return (uint)__builtin_popcount(value);
#else
  value = value - ((value >> 1) & 0x55555555);
  value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
  return (uint)(((value + (value >> 4) & 0xF0F0F0F) * 0x1010101) >> 24);
#endif
}

inline uint countBitsSet64(uint64_t value) {
#if __GNUC__ >= 4
  return (uint)__builtin_popcountll(value);
#else
  value = value - ((value >> 1) & 0x5555555555555555ULL);
  value = (value & 0x3333333333333333ULL) + ((value >> 2) & 0x3333333333333333ULL);
  value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  return (uint)((uint64_t)(value * 0x0101010101010101ULL) >> 56);
#endif
}

inline uint leastBitSet32(uint32_t value) {
#if defined(_WIN32)
  unsigned long idx;
  return _BitScanForward(&idx, (unsigned long)value) ? idx : (uint)-1;
#else
  return value ? __builtin_ctz(value) : (uint)-1;
#endif
}

inline uint leastBitSet64(uint64_t value) {
#if defined(_WIN64)
  unsigned long idx;
  return _BitScanForward64(&idx, (unsigned __int64)value) ? idx : (uint)-1;
#elif defined(__GNUC__)
  return value ? __builtin_ctzll(value) : (uint)-1;
#else
  static constexpr uint8_t lookup67[67 + 1] = {
      64, 0,  1,  39, 2,  15, 40, 23, 3,  12, 16, 59, 41, 19, 24, 54, 4,  -1, 13, 10, 17, 62, 60,
      28, 42, 30, 20, 51, 25, 44, 55, 47, 5,  32, -1, 38, 14, 22, 11, 58, 18, 53, 63, 9,  61, 27,
      29, 50, 43, 46, 31, 37, 21, 57, 52, 8,  26, 49, 45, 36, 56, 7,  48, 35, 6,  34, 33, -1};

  return (uint)lookup67[((int64_t)value & -(int64_t)value) % 67];
#endif
}

template <typename T> inline uint countBitsSet(T value) {
  return (sizeof(T) == 8) ? countBitsSet64((uint64_t)value) : countBitsSet32((uint32_t)value);
}

template <typename T> inline uint leastBitSet(T value) {
  return (sizeof(T) == 8) ? leastBitSet64((uint64_t)value) : leastBitSet32((uint32_t)value);
}

static inline bool Is32Bits() { return LP64_SWITCH(true, false); }

static inline bool Is64Bits() { return LP64_SWITCH(false, true); }

template <typename lambda> class ScopeGuard {
 public:
  explicit ALWAYSINLINE ScopeGuard(const lambda& release) : release_(release), dismiss_(false) {}

  ScopeGuard(ScopeGuard& rhs) { *this = rhs; }

  ALWAYSINLINE ~ScopeGuard() {
    if (!dismiss_) release_();
  }
  ALWAYSINLINE ScopeGuard& operator=(ScopeGuard& rhs) {
    dismiss_ = rhs.dismiss_;
    release_ = rhs.release_;
    rhs.dismiss_ = true;
  }
  ALWAYSINLINE void Dismiss() { dismiss_ = true; }

 private:
  lambda release_;
  bool dismiss_;
};

#define MAKE_SCOPE_GUARD_HELPER(lname, sname, ...)                                                 \
  auto lname = __VA_ARGS__;                                                                        \
  amd::ScopeGuard<decltype(lname)> sname(lname);
#define MAKE_SCOPE_GUARD(name, ...)                                                                \
  MAKE_SCOPE_GUARD_HELPER(XCONCAT(scopeGuardLambda, __COUNTER__), name, __VA_ARGS__)

// utility function to convert half precision to float to a
// single precision value.
inline float half2float(const uint16_t Val) {
  constexpr uint32_t halfExpoentMask = 0x7c00;
  constexpr uint32_t halfFractionMask = 0x03ff;
  constexpr uint32_t floatExponentBias = 127;
  constexpr uint32_t halfExponentBias = 15;
  constexpr uint32_t signBitShift = 16;
  constexpr uint32_t floatExponentShift = 23;
  uint32_t signBit = ((uint32_t)(Val & 0x8000)) << signBitShift;
  uint32_t exponent = (Val & halfExpoentMask) >> 10;
  uint32_t fraction = ((uint32_t)(Val & halfFractionMask))
                      << 13;  // Aligning half fraction to float
  union {
    uint32_t u32Arg;
    float fArg;
  };
  // Handling special cases
  if (exponent == 0x1f) {  // NaN or Infinity
    // When all exponent bits are 1, the value is either Infinity or NaN
    // For NaN, the fraction part should also be non-zero.
    u32Arg =
        signBit | 0x7f800000 | fraction;  // setting exponent to all 1's and keeping the fraction
    return fArg;
  } else if (exponent == 0) {  // Subnormal numbers or zero
    if (fraction == 0) {
      u32Arg = signBit;  // Plus or minus zero
      return fArg;
    } else {
      // Normalize subnormal number
      while ((fraction & (1 << 23)) == 0) {
        fraction <<= 1;
        exponent--;
      }
      exponent++;
      fraction &= ~(1 << 23);  // Remove leading 1 (implicit for normalized numbers)
    }
  }
  uint32_t floatExponent = ((exponent + floatExponentBias - halfExponentBias) & 0xff)
                           << floatExponentShift;
  u32Arg = signBit | floatExponent | fraction;
  return fArg;
}

/*@}*/  // namespace amd
}  // namespace amd

#endif /*UTIL_HPP_*/
