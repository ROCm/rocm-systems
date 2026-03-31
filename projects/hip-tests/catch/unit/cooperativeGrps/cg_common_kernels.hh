/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
