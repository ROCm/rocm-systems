/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HSAKMT_ENV_H_INCLUDED
#define HSAKMT_ENV_H_INCLUDED

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int hsakmt_safe_env_to_int(const char* envvar, int default_val) {
  if (envvar == NULL) return default_val;
  char* endptr;
  errno = 0;
  long val = strtol(envvar, &endptr, 10);
  if (endptr == envvar) return default_val;
  // Allow trailing whitespace from shell/.env files; reject other trailing content.
  while (*endptr == ' ' || *endptr == '\t' || *endptr == '\n' || *endptr == '\r') {
    ++endptr;
  }
  if (*endptr != '\0') return default_val;
  // On LLP64 (Windows), long is 32-bit so rely on errno for overflow.
  if (errno == ERANGE || val < INT_MIN || val > INT_MAX) return default_val;
  return (int)val;
}

#ifdef __cplusplus
}
#endif

#endif  // HSAKMT_ENV_H_INCLUDED
