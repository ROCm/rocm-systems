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
#include "rocshmem/rocshmem_config.h"  // BUILD_DEBUG_DEVICE (guarded)
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

// Device-side logging for rocshmem device code compiled into librccl.so. Like
// the installed shim: constmem-free device printf gated on BUILD_DEBUG_DEVICE
// (no __constant__ logd_constants, no PE/rank prefix), else no-ops.
#define LOGD_ERROR_ABORT(fmt, ...) do { LOGD_ERROR(fmt, ##__VA_ARGS__); __builtin_trap(); } while (0)
#if defined(BUILD_DEBUG_DEVICE)
#define LOGD_ERROR(fmt, ...)       printf("GIN-GDA ERROR: " fmt " (" __FILE__ ":%d)\n", ##__VA_ARGS__, __LINE__)
#define LOGD_WARN(fmt, ...)        printf("GIN-GDA WARN: "  fmt " (" __FILE__ ":%d)\n", ##__VA_ARGS__, __LINE__)
#define LOGD_INFO(fmt, ...)        printf("GIN-GDA INFO: "  fmt " (" __FILE__ ":%d)\n", ##__VA_ARGS__, __LINE__)
#define LOGD_TRACE(fmt, ...)       printf("GIN-GDA TRACE: " fmt " (" __FILE__ ":%d)\n", ##__VA_ARGS__, __LINE__)
#else
#define LOGD_ERROR(fmt, ...)       ((void)0)
#define LOGD_WARN(fmt, ...)        ((void)0)
#define LOGD_INFO(fmt, ...)        ((void)0)
#define LOGD_TRACE(fmt, ...)       ((void)0)
#endif

#endif // RCCL_GIN_LOG_HOST_HPP
