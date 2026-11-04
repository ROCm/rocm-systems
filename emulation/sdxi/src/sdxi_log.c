/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "sdxi_log.h"

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

static sdxi_log_level_t g_log_level = SDXI_LOG_INFO;

void sdxi_log_set_level(sdxi_log_level_t level)
{
    g_log_level = level;
}

void sdxi_log(sdxi_log_level_t level, const char *fmt, ...)
{
    if (level > g_log_level)
        return;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[%ld.%06ld] rocm-sdxi: ", (long)ts.tv_sec,
            ts.tv_nsec / 1000L);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
