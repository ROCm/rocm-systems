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

#ifndef HSA_RUNTIME_CORE_UTIL_TIMER_H_
#define HSA_RUNTIME_CORE_UTIL_TIMER_H_

#include "core/util/utils.h"
#include "core/util/os.h"
#include <chrono>
#include <time.h>
#include <type_traits>

namespace rocr {
namespace timer {

// Needed to patch around a mixed arithmetic bug in MSVC's duration_cast as of
// VS 2013.
template <bool isFloat, bool isSigned>
struct wide_type {
  typedef double type;
};
template <>
struct wide_type<false, false> {
  typedef uintmax_t type;
};
template <>
struct wide_type<false, true> {
  typedef intmax_t type;
};

template <typename To, typename Rep, typename Period>
static __forceinline To
    duration_cast(const std::chrono::duration<Rep, Period>& d) {
  typedef typename wide_type<std::is_floating_point<Rep>::value,
                             std::is_signed<Rep>::value>::type wide;
  typedef std::chrono::duration<wide, typename To::period> unit_convert_t;

  unit_convert_t temp = std::chrono::duration_cast<unit_convert_t>(d);
  return To(static_cast<typename To::rep>(temp.count()));
}
// End patch

template <typename Rep, typename Period>
static __forceinline double duration_in_seconds(
    std::chrono::duration<Rep, Period> delta) {
  typedef std::chrono::duration<double, std::ratio<1, 1>> seconds;
  return seconds(delta).count();
}

template <typename rep>
static __forceinline rep duration_from_seconds(double delta) {
  typedef std::chrono::duration<double, std::ratio<1, 1>> seconds;
  return std::chrono::duration_cast<rep>(seconds(delta));
}

// Provices a C++11 standard clock interface to the os::AccurateClock functions
class accurate_clock {
 public:
  typedef double rep;
  typedef std::nano period;
  typedef std::chrono::duration<rep, period> duration;
  typedef std::chrono::time_point<accurate_clock> time_point;

  static const bool is_steady = true;

  static __forceinline time_point now() {
    return time_point(duration(raw_now() * period_ns));
  }

  // These two extra APIs and types let us use clocks without conversion to the
  // arbitrary period unit
  typedef uint64_t raw_rep;
  typedef uint64_t raw_frequency;

  static __forceinline raw_rep raw_now() { return os::ReadAccurateClock(); }
  static __forceinline raw_frequency raw_freq() { return freq; }

 private:
  static double period_ns;
  static raw_frequency freq;

  class init {
   public:
    init();
  };
  static init accurate_clock_init;
};

// Provices a C++11 standard clock interface to the lowest latency approximate
// clock
class fast_clock {
 public:
  typedef double rep;
  typedef std::pico period;
  typedef std::chrono::duration<rep, period> duration;
  typedef std::chrono::time_point<fast_clock> time_point;

  static const bool is_steady = true;

  static __forceinline time_point now() {
    return time_point(duration(raw_now() * period_ps));
  }

  // These two extra APIs and types let us use clocks without conversion to the
  // arbitrary period unit
  typedef uint64_t raw_rep;
  typedef double raw_frequency;

#if defined(__x86_64__) || defined(_M_X64)
  static __forceinline raw_rep raw_now() { return __rdtsc(); }
  static __forceinline raw_frequency raw_freq() { return freq; }
#else
  static __forceinline raw_rep raw_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (raw_rep(ts.tv_sec) * 1000000000 + raw_rep(ts.tv_nsec));
  }
  static __forceinline raw_frequency raw_freq() { return 1.e-9; }
#endif

 private:
  static double period_ps;
  static raw_frequency freq;

  class init {
   public:
    init();
  };
  static init fast_clock_init;
};
}   //  namespace timer
}   //  namespace rocr  

#endif
