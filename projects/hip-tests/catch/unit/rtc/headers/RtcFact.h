/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
This file is being read by a function defined check_headers which is present in
RtcFunctions.cpp file, it requires a function named 'fact' to be present
in a separate file. The -I compiler option uses this function fact's path as an
input to find this file and access it.
*/

#ifndef CATCH_UNIT_RTC_HEADERS_RTCFACT_H_
#define CATCH_UNIT_RTC_HEADERS_RTCFACT_H_

__device__ int fact(int num) {
  int fact = 1;
  for (int i = 1; i <= num; i++) {
    fact *= i;
  }
  return fact;
}

#endif  // CATCH_UNIT_RTC_HEADERS_RTCFACT_H_
