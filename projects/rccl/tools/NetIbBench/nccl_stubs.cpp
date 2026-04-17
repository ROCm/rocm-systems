/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Minimal stubs for RCCL internal symbols required by the net_vXX.cc
// object files.  The INFO() macro in those translation units expands to
// ncclDebugLog(); we forward everything to stderr so that plugin load
// messages are still visible.

#include <cstdio>
#include <cstdarg>
#include "nccl_common.h"

thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1] = {};

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/,
                  const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
    if (level > NCCL_LOG_WARN) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}
