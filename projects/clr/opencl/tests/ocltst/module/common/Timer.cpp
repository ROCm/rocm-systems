/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "Timer.h"

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __linux__
#include <time.h>
#define NANOSECONDS_PER_SEC 1000000000
#endif

CPerfCounter::CPerfCounter() : _clocks(0), _start(0) {
#ifdef _WIN32

  QueryPerformanceFrequency((LARGE_INTEGER*)&_freq);

#endif

#ifdef __linux__
  _freq = NANOSECONDS_PER_SEC;
#endif
}

CPerfCounter::~CPerfCounter() {
  // EMPTY!
}

void CPerfCounter::Start(void) {
#ifdef _WIN32

  if (_start) {
    MessageBox(NULL, "Bad Perf Counter Start", "Error", MB_OK);
    exit(0);
  }
  QueryPerformanceCounter((LARGE_INTEGER*)&_start);

#endif
#ifdef __linux__

  struct timespec s;
  clock_gettime(CLOCK_MONOTONIC, &s);
  _start = (i64)s.tv_sec * NANOSECONDS_PER_SEC + (i64)s.tv_nsec;

#endif
}

void CPerfCounter::Stop(void) {
  i64 n;

#ifdef _WIN32

  if (!_start) {
    MessageBox(NULL, "Bad Perf Counter Stop", "Error", MB_OK);
    exit(0);
  }

  QueryPerformanceCounter((LARGE_INTEGER*)&n);

#endif
#ifdef __linux__

  struct timespec s;
  clock_gettime(CLOCK_MONOTONIC, &s);
  n = (i64)s.tv_sec * NANOSECONDS_PER_SEC + (i64)s.tv_nsec;

#endif

  n -= _start;
  _start = 0;
  _clocks += n;
}

void CPerfCounter::Reset(void) {
#ifdef _WIN32
  if (_start) {
    MessageBox(NULL, "Bad Perf Counter Reset", "Error", MB_OK);
    exit(0);
  }
#endif
  _clocks = 0;
}

double CPerfCounter::GetElapsedTime(void) {
#ifdef _WIN32
  if (_start) {
    MessageBox(NULL, "Trying to get time while still running.", "Error", MB_OK);
    exit(0);
  }
#endif

  return (double)_clocks / (double)_freq;
}
