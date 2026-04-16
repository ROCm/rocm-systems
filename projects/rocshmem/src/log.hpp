/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_LOG_HPP_
#define LIBRARY_SRC_LOG_HPP_

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "constants.hpp"
#include "envvar.hpp"

/**
 * @file log.hpp
 * @brief Leveled logging macros for host and device code.
 *
 * Output format:
 *   Host:   L<PE> <message> <func>@<file>:<line>
 *   Device: L<PE>w<WG>t<TH> <message> <file>:<line>
 *
 * Where L is a single-letter level: E(rror), W(arn), I(nfo), T(race).
 * PE, WG, TH are 4-digit zero-padded PE number, flat workgroup id, and
 * flat thread id respectively. PE is -1 before initialization.
 * Device macros embed __FILE__ via string concatenation (no %s needed)
 * and use %d for __LINE__; __func__ is not available on device.
 * Callers should NOT include a trailing newline — the macros append one.
 *
 * Host macros:
 *   LOG_ERROR       — prints error, does NOT terminate
 *   LOG_ERROR_EXIT  — prints error, calls exit(EXIT_FAILURE)
 *   LOG_ERROR_ABORT — prints error, calls abort()
 *   LOG_WARN        — prints if debug_level >= WARN
 *   LOG_INFO        — prints if debug_level >= INFO
 *   LOG_TRACE       — compiled with BUILD_DEBUG_TRACE_HOST
 *
 * Device macros:
 *   LOGD_ERROR       — prints error (gated by BUILD_DEBUG_DEVICE)
 *   LOGD_ERROR_ABORT — prints + abort() (abort unconditional)
 *   LOGD_WARN        — gated by BUILD_DEBUG_DEVICE
 *   LOGD_INFO        — gated by BUILD_DEBUG_DEVICE
 *   LOGD_TRACE       — gated by BUILD_DEBUG_DEVICE + BUILD_DEBUG_TRACE_DEVICE
 */

namespace rocshmem {
  inline int log_pe_number = -1;

  // __attribute__((error(...))) causes a compile error at the call site
  // (not at the definition), so it is safe to declare in a header.
  // __HIP_DEVICE_COMPILE__ is defined only during the device compilation
  // pass, giving true host-vs-device enforcement regardless of whether the
  // call site is __host__, __device__, or __host__ __device__.
#ifdef __HIP_DEVICE_COMPILE__
  __attribute__((error("host-only macro used in device code")))
  void static_assert_host_only();
  __device__ inline void static_assert_device_only() {}
#else
  __host__ inline void static_assert_host_only() {}
  __attribute__((error("device-only macro used in host code")))
  void static_assert_device_only();
#endif
}  // namespace rocshmem

/*****************************************************************************
 * Host-side logging macros
 *
 * When :color modifier is active (default), the level letter and PE are
 * colored and the func@file:line suffix is printed in gray.
 * Use :nocolor to disable.  Single fprintf per call site; the ternary
 * selects between colored and plain format strings (same args).
 *****************************************************************************/

#define LOG_ERROR(fmt, ...) do {                                              \
  rocshmem::static_assert_host_only();                                        \
  if (rocshmem::envvar::log_flags.show_error)                                 \
    fprintf(stderr, rocshmem::envvar::log_flags.show_color                    \
        ? "\033[1;91mE%04dh rocSHMEM\033[0m " fmt "\t\033[90m%s@%s:%d\033[0m\n" \
        : "E%04dh rocSHMEM " fmt "\t%s@%s:%d\n",                              \
        rocshmem::log_pe_number,                                              \
        __VA_OPT__(__VA_ARGS__,)                                              \
        __func__, __FILE__, __LINE__);                                        \
} while (0)

#define LOG_ERROR_EXIT(fmt, ...) do {                                         \
  rocshmem::static_assert_host_only();                                        \
  if (rocshmem::envvar::log_flags.show_error)                                 \
    fprintf(stderr, rocshmem::envvar::log_flags.show_color                    \
        ? "\033[1;91mE%04dh rocSHMEM\033[0m " fmt "\t\033[90m%s@%s:%d\033[0m\n" \
        : "E%04dh rocSHMEM " fmt "\t%s@%s:%d\n",                              \
        rocshmem::log_pe_number,                                              \
        __VA_OPT__(__VA_ARGS__,)                                              \
        __func__, __FILE__, __LINE__);                                        \
  exit(EXIT_FAILURE);                                                         \
} while (0)

#define LOG_ERROR_ABORT(fmt, ...) do {                                        \
  rocshmem::static_assert_host_only();                                        \
  if (rocshmem::envvar::log_flags.show_error)                                 \
    fprintf(stderr, rocshmem::envvar::log_flags.show_color                    \
        ? "\033[1;91mE%04dh rocSHMEM\033[0m " fmt "\t\033[90m%s@%s:%d\033[0m\n" \
        : "E%04dh rocSHMEM " fmt "\t%s@%s:%d\n",                              \
        rocshmem::log_pe_number,                                              \
        __VA_OPT__(__VA_ARGS__,)                                              \
        __func__, __FILE__, __LINE__);                                        \
  abort();                                                                    \
} while (0)

#define LOG_WARN(fmt, ...) do {                                               \
  rocshmem::static_assert_host_only();                                        \
  if (rocshmem::envvar::log_flags.show_warn)                                  \
    fprintf(stderr, rocshmem::envvar::log_flags.show_color                    \
        ? "\033[1;93mW%04dh rocSHMEM\033[0m " fmt "\t\033[90m%s@%s:%d\033[0m\n" \
        : "W%04dh rocSHMEM " fmt "\t%s@%s:%d\n",                              \
        rocshmem::log_pe_number,                                              \
        __VA_OPT__(__VA_ARGS__,)                                              \
        __func__, __FILE__, __LINE__);                                        \
} while (0)

#define LOG_INFO(fmt, ...) do {                                               \
  rocshmem::static_assert_host_only();                                        \
  if (rocshmem::envvar::log_flags.show_info)                                  \
    fprintf(stdout, rocshmem::envvar::log_flags.show_color                    \
        ? "\033[32mI%04dh rocSHMEM\033[0m " fmt "\t\033[90m%s@%s:%d\033[0m\n" \
        : "I%04dh rocSHMEM " fmt "\t%s@%s:%d\n",                              \
        rocshmem::log_pe_number,                                              \
        __VA_OPT__(__VA_ARGS__,)                                              \
        __func__, __FILE__, __LINE__);                                        \
} while (0)

#ifdef BUILD_DEBUG_TRACE_HOST
#define LOG_API(fmt, ...) do {                                                \
  rocshmem::static_assert_host_only();                                        \
  if (rocshmem::envvar::log_flags.show_api)                                   \
    fprintf(stdout, rocshmem::envvar::log_flags.show_color                    \
        ? "\033[35mA%04dh rocSHMEM\033[0m " fmt "\t\033[90m%s@%s:%d\033[0m\n" \
        : "A%04dh rocSHMEM " fmt "\t%s@%s:%d\n",                              \
        rocshmem::log_pe_number,                                              \
        __VA_OPT__(__VA_ARGS__,)                                              \
        __func__, __FILE__, __LINE__);                                        \
} while (0)
#define LOG_TRACE(fmt, ...) do {                                              \
  rocshmem::static_assert_host_only();                                        \
  if (rocshmem::envvar::log_flags.show_trace)                                 \
    fprintf(stdout, rocshmem::envvar::log_flags.show_color                    \
        ? "\033[34mT%04dh rocSHMEM\033[0m " fmt "\t\033[90m%s@%s:%d\033[0m\n" \
        : "T%04dh rocSHMEM " fmt "\t%s@%s:%d\n",                              \
        rocshmem::log_pe_number,                                              \
        __VA_OPT__(__VA_ARGS__,)                                              \
        __func__, __FILE__, __LINE__);                                        \
} while (0)
#else
#define LOG_API(...) do { rocshmem::static_assert_host_only(); } while (0)
#define LOG_TRACE(...) do { rocshmem::static_assert_host_only(); } while (0)
#endif

/*****************************************************************************
 * Device-side printf
 *
 * The first 3 printf arguments are the PE number, flat workgroup id,
 * and flat thread id (consumed by the leading format specifiers that
 * callers place in the format string).
 *****************************************************************************/

namespace rocshmem {

struct logd_constants {
  static constexpr uint32_t SHOW_ERROR = 1u << 0;
  static constexpr uint32_t SHOW_WARN  = 1u << 1;
  static constexpr uint32_t SHOW_INFO  = 1u << 2;
  static constexpr uint32_t SHOW_API   = 1u << 3;
  static constexpr uint32_t SHOW_TRACE = 1u << 4;
  static constexpr uint32_t SHOW_COLOR = 1u << 5;

  int      pe_number;
  uint32_t flags;
} __attribute__((aligned(16)));
extern __constant__ logd_constants logd_constants;

template <typename... Args>
[[maybe_unused]] __device__ __attribute__((noinline))
void dprintf(const char* fmt, const Args&... args) {
  int flat_thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
                       hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
  int flat_wg_id = hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
                   hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
  printf(fmt, logd_constants.pe_number, flat_wg_id, flat_thread_id, args...);
}

}  // namespace rocshmem

/*****************************************************************************
 * Device-side logging macros
 *
 * LOGD_ERROR / LOGD_ERROR_ABORT are always compiled in so that error
 * paths are never silently compiled away.
 * LOGD_WARN / LOGD_INFO are gated by BUILD_DEBUG_DEVICE.
 * LOGD_API / LOGD_TRACE are gated by BUILD_DEBUG_TRACE_DEVICE
 * (and additionally require BUILD_DEBUG_DEVICE for WARN/INFO).
 *
 * All device macros call rocshmem::dprintf() (defined above), which is
 * marked __attribute__((noinline)).  This confines printf's register and
 * scratch memory pressure to dprintf's own stack frame, so the calling
 * kernel's occupancy is not reduced by logging code — even when the
 * show_* branch is never taken at runtime.  The first 3 printf arguments
 * are PE number, flat workgroup id, and flat thread id.
 *
 * Each macro uses a ternary to select between a colored and plain format
 * string (same args for both) — one branch + two .rodata string constants
 * per call site, no extra registers.
 *
 * Callers should NOT include a trailing newline — the macros append one.
 *****************************************************************************/

/* LOGD_ERROR and LOGD_ERROR_ABORT are always compiled in (not gated by
 * BUILD_DEBUG_DEVICE) so that error paths are never silently
 * compiled away.  Single dprintf call; ternary selects format string. */

#define LOGD_ERROR(fmt, ...) do {                                             \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::logd_constants.flags & rocshmem::logd_constants::SHOW_ERROR)  \
    rocshmem::dprintf(rocshmem::logd_constants.flags                          \
        & rocshmem::logd_constants::SHOW_COLOR                                \
        ? "\033[1;91mE%04dw%04ut%04u\033[0m " fmt                             \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "E%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)

#define LOGD_ERROR_ABORT(fmt, ...) do {                                       \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::logd_constants.flags & rocshmem::logd_constants::SHOW_ERROR)  \
    rocshmem::dprintf(rocshmem::logd_constants.flags                          \
        & rocshmem::logd_constants::SHOW_COLOR                                \
        ? "\033[1;91mE%04dw%04ut%04u\033[0m " fmt                             \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "E%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
  abort();                                                                    \
} while (0)

#ifdef BUILD_DEBUG_DEVICE

#define LOGD_WARN(fmt, ...) do {                                              \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::logd_constants.flags & rocshmem::logd_constants::SHOW_WARN)   \
    rocshmem::dprintf(rocshmem::logd_constants.flags                          \
        & rocshmem::logd_constants::SHOW_COLOR                                \
        ? "\033[1;93mW%04dw%04ut%04u\033[0m " fmt                             \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "W%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)

#define LOGD_INFO(fmt, ...) do {                                              \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::logd_constants.flags & rocshmem::logd_constants::SHOW_INFO)   \
    rocshmem::dprintf(rocshmem::logd_constants.flags                          \
        & rocshmem::logd_constants::SHOW_COLOR                                \
        ? "\033[32mI%04dw%04ut%04u\033[0m " fmt                               \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "I%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)

#if defined(BUILD_DEBUG_TRACE_DEVICE)
#define LOGD_API(fmt, ...) do {                                               \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::logd_constants.flags & rocshmem::logd_constants::SHOW_API)    \
    rocshmem::dprintf(rocshmem::logd_constants.flags                          \
        & rocshmem::logd_constants::SHOW_COLOR                                \
        ? "\033[95mA%04dw%04ut%04u\033[0m " fmt                               \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "A%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)

#define LOGD_TRACE(fmt, ...) do {                                             \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::logd_constants.flags & rocshmem::logd_constants::SHOW_TRACE)  \
    rocshmem::dprintf(rocshmem::logd_constants.flags                          \
        & rocshmem::logd_constants::SHOW_COLOR                                \
        ? "\033[94mT%04dw%04ut%04u\033[0m " fmt                               \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "T%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)
#else
#define LOGD_API(...) do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_TRACE(...) do { rocshmem::static_assert_device_only(); } while (0)
#endif

#else  /* !BUILD_DEBUG_DEVICE */

#define LOGD_WARN(...)        do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_INFO(...)        do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_API(...)         do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_TRACE(...)       do { rocshmem::static_assert_device_only(); } while (0)

#endif  /* BUILD_DEBUG_DEVICE */

#endif  // LIBRARY_SRC_LOG_HPP_
