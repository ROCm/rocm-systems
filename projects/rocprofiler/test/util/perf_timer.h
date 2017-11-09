/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

<95>    Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
<95>    Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef TEST_UTIL_PERF_TIMER_H_
#define TEST_UTIL_PERF_TIMER_H_

// Will use AMD timer or general Linux timer based on compilation flag
// Need to consider platform is Windows or Linux

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#include <intrin.h>
#include <time.h>
#include <windows.h>
#else
#if defined(__GNUC__)
#include <sys/time.h>
#include <x86intrin.h>
#endif  // __GNUC__
#endif  // _MSC_VER

#include <iostream>
#include <string>
#include <vector>

class PerfTimer {
 public:
  enum { SUCCESS = 0, FAILURE = 1 };

  PerfTimer();
  ~PerfTimer();

  // General Linux timing method
  int CreateTimer();
  int StartTimer(int index);
  int StopTimer(int index);

  // retrieve time
  double ReadTimer(int index);
  // write into a file
  double WriteTimer(int index);

 private:
  struct Timer {
    std::string name; /* name of time object */
    long long freq;   /* frequency */
    double clocks;    /* number of ticks at end */
    double start;     /* start point ticks */
  };

  std::vector<Timer*> timers_; /* vector to Timer objects */
  double freq_in_100mhz_;

  // AMD timing method
  uint64_t CoarseTimestampUs();
  uint64_t MeasureTSCFreqHz();

  void Error(std::string str);
};

#endif  // TEST_UTIL_PERF_TIMER_H_
