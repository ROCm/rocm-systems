/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
*/

#pragma once

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

static __device__ void busy_wait(unsigned long long wait_period) {
  unsigned long long time_diff = 0;
#if HT_AMD
  unsigned long long last_clock = wall_clock64();
#else
  unsigned long long last_clock = clock64();
#endif
  while (time_diff < wait_period) {
#if HT_AMD
  unsigned long long cur_clock = wall_clock64();
#else
  unsigned long long cur_clock = clock64();
#endif
    if (cur_clock > last_clock) {
      time_diff += (cur_clock - last_clock);
    }
    last_clock = cur_clock;
  }
}
