/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "ras_param.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef NCCL_RAS_CLIENT
#include "debug.h"
#include "param.h"
#else
static const char* rasParamGetEnv(const char* name) {
  return getenv(name);
}
#define ncclGetEnv rasParamGetEnv
#endif

static float rasLoadTimeoutFactor() {
  constexpr float kDefault = 1.0f;
  const char* str = ncclGetEnv("NCCL_RAS_TIMEOUT_FACTOR");
  if (str == nullptr || str[0] == '\0') return kDefault;

  errno = 0;
  char* end = nullptr;
  const double value = strtod(str, &end);
  if (errno != 0 || end == str || (end != nullptr && *end != '\0') || !std::isfinite(value) || value <= 0.0) {
#ifndef NCCL_RAS_CLIENT
    INFO(NCCL_ALL, "Invalid value %s for NCCL_RAS_TIMEOUT_FACTOR, using default %g.", str, kDefault);
#else
    fprintf(stderr, "Invalid value %s for NCCL_RAS_TIMEOUT_FACTOR, using default %g.\n", str, kDefault);
#endif
    return kDefault;
  }

#ifndef NCCL_RAS_CLIENT
  INFO(NCCL_ENV, "NCCL_RAS_TIMEOUT_FACTOR set by environment to %g", value);
#endif
  return (float)value;
}

float ncclParamRasTimeoutFactor(void) {
  constexpr float uninitialized = -1.0f;
  static float cache = uninitialized;
  if (cache < 0.0f) cache = rasLoadTimeoutFactor();
  return cache;
}

int64_t rasTimeoutFactorNs(int64_t baseSeconds) {
  const double scaled = (double)baseSeconds * 1000000000.0 * ncclParamRasTimeoutFactor();
  // Saturate before the double->int64 conversion (out of range is undefined behavior), and well below
  // INT64_MAX: callers add the result to clockNano() and to other timeouts (e.g. startTime + timeout +
  // RAS_COLLECTIVE_EXTRA_TIMEOUT), and those sums must not overflow. INT64_MAX/4 ns is ~73 years.
  constexpr int64_t kMaxTimeoutNs = INT64_MAX / 4;
  if (std::isnan(scaled) || scaled >= (double)kMaxTimeoutNs) return kMaxTimeoutNs;
  if (scaled <= -(double)kMaxTimeoutNs) return -kMaxTimeoutNs;
  return (int64_t)scaled;
}

double rasTimeoutFactorSec(int baseSeconds) {
  return baseSeconds * ncclParamRasTimeoutFactor();
}
