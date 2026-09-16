/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shim: shadows rocshmem's log.hpp when compiling anvil.cpp into librccl.so.
// Maps rocshmem LOG_* macros to RCCL's native logging infrastructure.

#ifndef RCCL_GIN_LOG_SHIM_HPP
#define RCCL_GIN_LOG_SHIM_HPP

// Host-side: when building librccl.so, map to RCCL native logging.
// When installed (consumed by rccl-tests), host macros are not needed
// — only device-side stubs below are used by anvil_device.hpp.
#if __has_include("debug.h")
#include "debug.h"
#define LOG_ERROR(fmt, ...)       WARN("GIN: " fmt, ##__VA_ARGS__)
#define LOG_ERROR_EXIT(fmt, ...)  do { WARN("GIN FATAL: " fmt, ##__VA_ARGS__); abort(); } while(0)
#define LOG_ERROR_ABORT(fmt, ...) do { WARN("GIN ABORT: " fmt, ##__VA_ARGS__); abort(); } while(0)
#define LOG_WARN(fmt, ...)        INFO(NCCL_INIT, "GIN: " fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)        INFO(NCCL_INIT, "GIN: " fmt, ##__VA_ARGS__)
#ifdef ENABLE_TRACE
#define LOG_TRACE(fmt, ...)       TRACE(NCCL_INIT, "GIN: " fmt, ##__VA_ARGS__)
#else
#define LOG_TRACE(...)            do {} while(0)
#endif
#else
// Installed context: no RCCL internals available.
#include <cstdio>
#define LOG_ERROR(fmt, ...)       fprintf(stderr, "GIN: " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR_EXIT(fmt, ...)  do { fprintf(stderr, "GIN FATAL: " fmt "\n", ##__VA_ARGS__); abort(); } while(0)
#define LOG_ERROR_ABORT(fmt, ...) do { fprintf(stderr, "GIN ABORT: " fmt "\n", ##__VA_ARGS__); abort(); } while(0)
#define LOG_WARN(fmt, ...)        fprintf(stderr, "GIN: " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)        ((void)0)
#define LOG_TRACE(...)            do {} while(0)
#endif

// Device-side logging stubs (anvil_device.hpp uses LOGD_ERROR_ABORT for
// unsupported-architecture guards — map to __builtin_trap on device).
#define LOGD_ERROR_ABORT(fmt, ...) __builtin_trap()
#define LOGD_ERROR(fmt, ...)       ((void)0)
#define LOGD_WARN(fmt, ...)        ((void)0)
#define LOGD_INFO(fmt, ...)        ((void)0)
#define LOGD_TRACE(fmt, ...)       ((void)0)

#endif // RCCL_GIN_LOG_SHIM_HPP
