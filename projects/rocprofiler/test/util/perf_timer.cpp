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
