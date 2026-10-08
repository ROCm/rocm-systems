/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// GIN device log shim (installed). Device macros only — no host dependencies.
// Inside librccl.so, src/gin/log.hpp shadows this with RCCL native logging.

#ifndef RCCL_GIN_LOG_DEVICE_HPP
#define RCCL_GIN_LOG_DEVICE_HPP

// Host-side: no-ops (device headers should not emit host log calls).
#define LOG_ERROR(...)             ((void)0)
#define LOG_ERROR_EXIT(...)        ((void)0)
#define LOG_ERROR_ABORT(...)       ((void)0)
#define LOG_WARN(...)              ((void)0)
#define LOG_INFO(...)              ((void)0)
#define LOG_TRACE(...)             ((void)0)

// Device-side: LOGD_ERROR_ABORT traps; others are no-ops.
#define LOGD_ERROR_ABORT(fmt, ...) __builtin_trap()
#define LOGD_ERROR(fmt, ...)       ((void)0)
#define LOGD_WARN(fmt, ...)        ((void)0)
#define LOGD_INFO(fmt, ...)        ((void)0)
#define LOGD_TRACE(...)            ((void)0)

#endif // RCCL_GIN_LOG_DEVICE_HPP
