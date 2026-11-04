/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SDXI_LOG_H
#define SDXI_LOG_H

#include <stdio.h>
#include <time.h>

typedef enum {
    SDXI_LOG_ERROR = 0,
    SDXI_LOG_WARN,
    SDXI_LOG_INFO,
    SDXI_LOG_DEBUG,
} sdxi_log_level_t;

void sdxi_log_set_level(sdxi_log_level_t level);
void sdxi_log(sdxi_log_level_t level, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

#define sdxi_error(fmt, ...) sdxi_log(SDXI_LOG_ERROR, "ERROR: " fmt, ##__VA_ARGS__)
#define sdxi_warn(fmt, ...)  sdxi_log(SDXI_LOG_WARN,  "WARN:  " fmt, ##__VA_ARGS__)
#define sdxi_info(fmt, ...)  sdxi_log(SDXI_LOG_INFO,  "INFO:  " fmt, ##__VA_ARGS__)
#define sdxi_debug(fmt, ...) sdxi_log(SDXI_LOG_DEBUG, "DEBUG: " fmt, ##__VA_ARGS__)

#endif /* SDXI_LOG_H */
