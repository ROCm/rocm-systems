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

#include "wddm_lite_internal.h"
#include "wddm_lite_device.h"

/* HSAKMT global state — Windows version */

unsigned long           hsakmt_kfd_open_count = 0;
CRITICAL_SECTION        hsakmt_mutex;
bool                    hsakmt_is_dgpu = false;
int                     hsakmt_page_size = 4096;
int                     hsakmt_page_shift = 12;
int                     hsakmt_debug_level = HSAKMT_DEBUG_LEVEL_DEFAULT;
HsaVersionInfo          hsakmt_kfd_version_info;

/* Global wddm_lite device instance */
struct WddmLiteDevice   g_wddm_lite_dev = {};

/* One-time initialization of the critical section */
static bool cs_initialized = false;

void wddm_lite_init_mutex(void)
{
    if (!cs_initialized) {
        InitializeCriticalSection(&hsakmt_mutex);
        cs_initialized = true;
    }
}

void wddm_lite_fini_mutex(void)
{
    if (cs_initialized) {
        DeleteCriticalSection(&hsakmt_mutex);
        cs_initialized = false;
    }
}
