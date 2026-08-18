// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UALOE_LOG_H
#define UALOE_LOG_H

#include <stdio.h>

/* External control variable for runtime logging enable/disable */
extern int ualoe_log_enabled;

/* Log error message to stderr if logging is enabled */
#define ualoe_log_error(fmt, ...)          \
  do {                                     \
    if (ualoe_log_enabled) {               \
      fprintf(stderr, fmt, ##__VA_ARGS__); \
    }                                      \
  } while (0)

/* Log warning message to stderr if logging is enabled */
#define ualoe_log_warning(fmt, ...)        \
  do {                                     \
    if (ualoe_log_enabled) {               \
      fprintf(stderr, fmt, ##__VA_ARGS__); \
    }                                      \
  } while (0)

#endif /* UALOE_LOG_H */
