/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// GIN device log shim (installed). Device macros only — no host dependencies.
// Inside librccl.so, src/gin/log.hpp shadows this with RCCL native logging.

#ifndef RCCL_GIN_LOG_DEVICE_HPP
#define RCCL_GIN_LOG_DEVICE_HPP

// For BUILD_DEBUG_DEVICE. Guarded, so double-inclusion with the QP headers that
// also pull it is harmless; including it here makes the LOGD gating below robust
// regardless of consumer include order.
#include "nccl_device/gin/rocshmem_gda/gda_config.h"

// Host-side: no-ops (device headers should not emit host log calls).
#define LOG_ERROR(...)             ((void)0)
#define LOG_ERROR_EXIT(...)        ((void)0)
#define LOG_ERROR_ABORT(...)       ((void)0)
#define LOG_WARN(...)              ((void)0)
#define LOG_INFO(...)              ((void)0)
#define LOG_TRACE(...)             ((void)0)

// Device-side. Unlike rocSHMEM's LOGD (which reads __constant__ logd_constants
// for the PE/flags), the GIN build carries no device globals, so these emit a
// plain constmem-free device printf gated at compile time on BUILD_DEBUG_DEVICE.
// Output is the raw error message + source location (no PE/rank prefix).
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
#define LOGD_TRACE(...)            ((void)0)
#endif

#endif // RCCL_GIN_LOG_DEVICE_HPP
