/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
*******************************************************************************/

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
