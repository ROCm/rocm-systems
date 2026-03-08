/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

/*
 * Internal header for wddm_lite backend.
 * Windows replacement for src/libhsakmt.h — uses CRITICAL_SECTION
 * instead of pthread_mutex, no fork detection.
 */

#ifndef WDDM_LITE_INTERNAL_H_INCLUDED
#define WDDM_LITE_INTERNAL_H_INCLUDED

#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <hsakmt/hsakmt.h>

/* Global state — defined in globals.cpp */
extern unsigned long                hsakmt_kfd_open_count;
extern CRITICAL_SECTION             hsakmt_mutex;
extern bool                         hsakmt_is_dgpu;
extern int                          hsakmt_page_size;
extern int                          hsakmt_page_shift;
extern int                          hsakmt_debug_level;
extern HsaVersionInfo               hsakmt_kfd_version_info;

/* Debug levels matching the KFD convention */
#define HSAKMT_DEBUG_LEVEL_ERR      3
#define HSAKMT_DEBUG_LEVEL_WARNING  4
#define HSAKMT_DEBUG_LEVEL_INFO     6
#define HSAKMT_DEBUG_LEVEL_DEBUG    7
#define HSAKMT_DEBUG_LEVEL_DEFAULT  HSAKMT_DEBUG_LEVEL_ERR

/* Debug macros */
#define pr_err(fmt, ...)                                            \
    do {                                                            \
        if (hsakmt_debug_level >= HSAKMT_DEBUG_LEVEL_ERR)           \
            fprintf(stderr, fmt, ##__VA_ARGS__);                    \
    } while (0)

#define pr_warn(fmt, ...)                                           \
    do {                                                            \
        if (hsakmt_debug_level >= HSAKMT_DEBUG_LEVEL_WARNING)       \
            fprintf(stderr, fmt, ##__VA_ARGS__);                    \
    } while (0)

#define pr_info(fmt, ...)                                           \
    do {                                                            \
        if (hsakmt_debug_level >= HSAKMT_DEBUG_LEVEL_INFO)          \
            fprintf(stderr, fmt, ##__VA_ARGS__);                    \
    } while (0)

/* CHECK_KFD_OPEN — no fork detection on Windows */
#define CHECK_KFD_OPEN() \
    do { if (hsakmt_kfd_open_count == 0) return HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED; } while (0)

/* Version init */
HSAKMT_STATUS hsakmt_init_kfd_version(void);

#endif /* WDDM_LITE_INTERNAL_H_INCLUDED */
