/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

// Windows sysinfo.h stub header
// This provides minimal sysinfo definitions for compilation compatibility

#ifndef _SYSINFO_H_STUB
#define _SYSINFO_H_STUB

#ifndef _WIN32
// On non-Windows platforms, use the real sysinfo.h
#include_next <sys/sysinfo.h>
#else
#include <windows.h>
// sysinfo implementation for Windows
// Simplified sysinfo struct for Windows compatibility
struct sysinfo {
    long uptime;        // Seconds since boot
    unsigned long loads[3]; // 1, 5, and 15 minute load averages
    unsigned long totalram; // Total usable main memory size
    unsigned long freeram;  // Available memory size
    unsigned long sharedram; // Amount of shared memory
    unsigned long bufferram; // Memory used by buffers
    unsigned long totalswap; // Total swap space size
    unsigned long freeswap;  // swap space still available
    unsigned short procs;    // Number of current processes
    unsigned long totalhigh; // Total high memory size
    unsigned long freehigh;  // Available high memory size
    unsigned int mem_unit;   // Memory unit size in bytes
    char _f[20-2*sizeof(long)-sizeof(int)]; // Padding to 64 bytes
};

inline int sysinfo(struct sysinfo *info) {
    if (!info) return -1;

    // Get system memory information
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        info->totalram = (unsigned long)(memStatus.ullTotalPhys);
        info->freeram = (unsigned long)(memStatus.ullAvailPhys);
        info->totalswap = (unsigned long)(memStatus.ullTotalPageFile);
        info->freeswap = (unsigned long)(memStatus.ullAvailPageFile);
    } else {
        info->totalram = 0;
        info->freeram = 0;
        info->totalswap = 0;
        info->freeswap = 0;
    }

    // Get uptime
    info->uptime = GetTickCount64() / 1000;  // Convert milliseconds to seconds

    // Set other fields to reasonable defaults
    info->loads[0] = info->loads[1] = info->loads[2] = 0;
    info->sharedram = 0;
    info->bufferram = 0;
    info->procs = 0;
    info->totalhigh = 0;
    info->freehigh = 0;
    info->mem_unit = 1;

    return 0;
}
#endif
#endif // _SYSINFO_H_STUB
