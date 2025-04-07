/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include "common/hsatimer.h"
#include <x86intrin.h>

namespace rocrtst {

static const uint64_t kNanosecondsPerSecond = 1000000000;

PerfTimer::PerfTimer(void) {
  freq_in_100mhz = MeasureTSCFreqHz();
}

PerfTimer::~PerfTimer() {
  while (!_timers.empty()) {
    Timer* temp = _timers.back();
    _timers.pop_back();
    delete temp;
  }
}

int PerfTimer::CreateTimer(void) {
  Timer* newTimer = new Timer;
  newTimer->_start = 0;
  newTimer->_clocks = 0;

  newTimer->_freq = kNanosecondsPerSecond;

  /* Push back the address of new Timer instance created */
  _timers.push_back(newTimer);
  return static_cast<int>(_timers.size() - 1);
}

int PerfTimer::StartTimer(int index) {
  if (index >= static_cast<int>(_timers.size())) {
    Error("Cannot reset timer. Invalid handle.");
    return 1;
  }

// General Linux timing method
#ifndef _AMD
  struct timespec s;
  clock_gettime(CLOCK_MONOTONIC, &s);
  _timers[index]->_start = (uint64_t) s.tv_sec * kNanosecondsPerSecond
                           + (uint64_t) s.tv_nsec;
#else

  // AMD timing method

  unsigned int unused;
  _timers[index]->_start = __rdtscp(&unused);

#endif

  return 0;
}

int PerfTimer::StopTimer(int index) {
  uint64_t n = 0;

  if (index >= static_cast<int>(_timers.size())) {
    Error("Cannot reset timer. Invalid handle.");
    return 1;
  }

  // General Linux timing method
#ifndef _AMD
  struct timespec s;
  clock_gettime(CLOCK_MONOTONIC, &s);
  n = (uint64_t) s.tv_sec * kNanosecondsPerSecond + (uint64_t) s.tv_nsec;
#else
  // AMD Linux timing

  unsigned int unused;
  n = __rdtscp(&unused);
#endif

  n -= _timers[index]->_start;
  _timers[index]->_start = 0;

#ifndef _AMD
  _timers[index]->_clocks += n;
#else
  // convert to ms
  _timers[index]->_clocks += 1.0E-6 * 10 * n / freq_in_100mhz;
  cout << "_AMD is enabled!!!" << endl;
#endif

  return 0;
}

void PerfTimer::Error(std::string str) {
  std::cout << str << std::endl;
}

double PerfTimer::ReadTimer(int index) {
  if (index >= static_cast<int>(_timers.size())) {
    Error("Cannot read timer. Invalid handle.");
    return 1;
  }

  double reading = static_cast<double>(_timers[index]->_clocks);

  reading = static_cast<double>(reading / _timers[index]->_freq);

  return reading;
}

void PerfTimer::ResetTimer(int index) {
  // Check if index value is over the timer's size
  if (index >= static_cast<int>(_timers.size())) {
    Error("Invalid index value\n");
    exit(1);
  }

  _timers[index]->_clocks = 0.0;
  _timers[index]->_start = 0.0;
}

uint64_t PerfTimer::CoarseTimestampUs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

uint64_t PerfTimer::MeasureTSCFreqHz() {
  // Make a coarse interval measurement of TSC ticks for 1 gigacycles.
  unsigned int unused;
  uint64_t tscTicksEnd;

  uint64_t coarseBeginUs = CoarseTimestampUs();
  uint64_t tscTicksBegin = __rdtscp(&unused);

  do {
    tscTicksEnd = __rdtscp(&unused);
  } while (tscTicksEnd - tscTicksBegin < 1000000000);

  uint64_t coarseEndUs = CoarseTimestampUs();

  // Compute the TSC frequency and round to nearest 100MHz.
  uint64_t coarseIntervalNs = (coarseEndUs - coarseBeginUs) * 1000;
  uint64_t tscIntervalTicks = tscTicksEnd - tscTicksBegin;
  return (tscIntervalTicks * 10 + (coarseIntervalNs / 2)) / coarseIntervalNs;
}

}  // namespace rocrtst
