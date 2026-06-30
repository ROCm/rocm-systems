// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "util/perf_timer.h"

PerfTimer::PerfTimer() { freq_in_100mhz_ = MeasureTSCFreqHz(); }

PerfTimer::~PerfTimer() {
  while (!timers_.empty()) {
    Timer* temp = timers_.back();
    timers_.pop_back();
    delete temp;
  }
}

// New cretaed timer instantance index will be returned
int PerfTimer::CreateTimer() {
  Timer* newTimer = new Timer;
  newTimer->start = 0;
  newTimer->clocks = 0;

#ifdef _WIN32
  QueryPerformanceFrequency((LARGE_INTEGER*)&newTimer->freq);
#else
  newTimer->freq = (long long)1.0E3;
#endif

  /* Push back the address of new Timer instance created */
  timers_.push_back(newTimer);
  return (int)(timers_.size() - 1);
}

int PerfTimer::StartTimer(int index) {
  if (index >= (int)timers_.size()) {
    Error("Cannot reset timer. Invalid handle.");
    return FAILURE;
  }

#ifdef _WIN32
// General Windows timing method
#ifndef _AMD
  long long tmpStart;
  QueryPerformanceCounter((LARGE_INTEGER*)&(tmpStart));
  timers_[index]->start = (double)tmpStart;
#else
// AMD Windows timing method
#endif
#else
// General Linux timing method
#ifndef _AMD
  struct timeval s;
  gettimeofday(&s, 0);
  timers_[index]->start = s.tv_sec * 1.0E3 + ((double)(s.tv_usec / 1.0E3));
#else
  // AMD timing method
  unsigned int unused;
  timers_[index]->start = __rdtscp(&unused);
#endif
#endif

  return SUCCESS;
}

int PerfTimer::StopTimer(int index) {
  double n = 0;
  if (index >= (int)timers_.size()) {
    Error("Cannot reset timer. Invalid handle.");
    return FAILURE;
  }
#ifdef _WIN32
#ifndef _AMD
  long long n1;
  QueryPerformanceCounter((LARGE_INTEGER*)&(n1));
  n = (double)n1;
#else
// AMD Window Timing
#endif

#else
// General Linux timing method
#ifndef _AMD
  struct timeval s;
  gettimeofday(&s, 0);
  n = s.tv_sec * 1.0E3 + (double)(s.tv_usec / 1.0E3);
#else
  // AMD Linux timing
  unsigned int unused;
  n = __rdtscp(&unused);
#endif
#endif

  n -= timers_[index]->start;
  timers_[index]->start = 0;

#ifndef _AMD
  timers_[index]->clocks += n;
#else
  // timers_[index]->clocks += 10 * n / freq_in_100mhz_; // unit is ns
  timers_[index]->clocks += 1.0E-6 * 10 * n / freq_in_100mhz_;  // convert to ms
#endif

  return SUCCESS;
}

void PerfTimer::Error(std::string str) { std::cout << str << std::endl; }

double PerfTimer::ReadTimer(int index) {
  if (index >= (int)timers_.size()) {
    Error("Cannot read timer. Invalid handle.");
    return FAILURE;
  }

  double reading = double(timers_[index]->clocks);

  reading = double(reading / timers_[index]->freq);

  return reading;
}

uint64_t PerfTimer::CoarseTimestampUs() {
#ifdef _WIN32
  uint64_t freqHz, ticks;
  QueryPerformanceFrequency((LARGE_INTEGER*)&freqHz);
  QueryPerformanceCounter((LARGE_INTEGER*)&ticks);

  // Scale numerator and divisor until (ticks * 1000000) fits in uint64_t.
  while (ticks > (1ULL << 44)) {
    ticks /= 16;
    freqHz /= 16;
  }

  return (ticks * 1000000) / freqHz;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
#endif
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
