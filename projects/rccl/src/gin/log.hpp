/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side log shim: maps rocshmem LOG_* macros to RCCL native logging.
// Internal to librccl.so — never installed. Used by anvil.cpp via per-file
// include path ordering (src/gin/ before rocshmem/src/).

#ifndef RCCL_GIN_LOG_HOST_HPP
#define RCCL_GIN_LOG_HOST_HPP

#include "debug.h"
#include <cstdlib>

#define LOG_ERROR(fmt, ...)       WARN("GIN: " fmt, ##__VA_ARGS__)
#define LOG_ERROR_EXIT(fmt, ...)  do { WARN("GIN FATAL: " fmt, ##__VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#define LOG_ERROR_ABORT(fmt, ...) do { WARN("GIN ABORT: " fmt, ##__VA_ARGS__); abort(); } while(0)
#define LOG_WARN(fmt, ...)        INFO(NCCL_INIT, "GIN: " fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)        INFO(NCCL_INIT, "GIN: " fmt, ##__VA_ARGS__)
#ifdef ENABLE_TRACE
#define LOG_TRACE(fmt, ...)       TRACE(NCCL_INIT, "GIN: " fmt, ##__VA_ARGS__)
#else
#define LOG_TRACE(...)            do {} while(0)
#endif

// Device-side logging stubs for rocshmem device code that may include log.hpp.
// LOGD_ERROR_ABORT maps to __builtin_trap on device; others are no-ops.
#define LOGD_ERROR_ABORT(fmt, ...) __builtin_trap()
#define LOGD_ERROR(fmt, ...)       ((void)0)
#define LOGD_WARN(fmt, ...)        ((void)0)
#define LOGD_INFO(fmt, ...)        ((void)0)
#define LOGD_TRACE(fmt, ...)       ((void)0)

#endif // RCCL_GIN_LOG_HOST_HPP
