/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_ENV_PARSE_H_
#define NCCL_M2N_ENV_PARSE_H_

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>

static inline bool isM2nEnvDigit(char c) {
  return c >= '0' && c <= '9';
}

static inline bool isM2nEnvSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static inline const char* skipM2nEnvSpaces(const char* s) {
  while (isM2nEnvSpace(*s)) {
    s++;
  }
  return s;
}

inline bool parseM2nEnvInt(const char* s, int* out) {
  if (s == nullptr || out == nullptr) {
    return false;
  }
  const char* firstDigit = skipM2nEnvSpaces(s);
  if (*firstDigit == '+' || *firstDigit == '-') {
    firstDigit++;
  }
  if (!isM2nEnvDigit(*firstDigit)) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  long n = strtol(s, &end, 10);
  if (end == s || *end != '\0' || errno == ERANGE || n < INT_MIN || n > INT_MAX) {
    return false;
  }
  *out = static_cast<int>(n);
  return true;
}

inline bool parseM2nPositiveEnvInt(const char* s, int* out) {
  if (out == nullptr) {
    return false;
  }
  int n = 0;
  if (!parseM2nEnvInt(s, &n) || n <= 0) {
    return false;
  }
  *out = n;
  return true;
}

inline bool parseM2nNonNegativeEnvInt(const char* s, int* out) {
  if (out == nullptr) {
    return false;
  }
  int n = 0;
  if (!parseM2nEnvInt(s, &n) || n < 0) {
    return false;
  }
  *out = n;
  return true;
}

inline bool parseM2nEnvSize(const char* s, size_t* out, bool allowZero) {
  if (s == nullptr || out == nullptr) {
    return false;
  }
  const char* firstDigit = skipM2nEnvSpaces(s);
  if (*firstDigit == '+') {
    firstDigit++;
  }
  if (!isM2nEnvDigit(*firstDigit)) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  unsigned long long n = strtoull(s, &end, 10);
  if (end == s || *end != '\0' || errno == ERANGE) {
    return false;
  }
  if (!allowZero && n == 0) {
    return false;
  }
  if (n > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    return false;
  }
  *out = static_cast<size_t>(n);
  return true;
}

#endif /* NCCL_M2N_ENV_PARSE_H_ */
