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
 *   LOG_TRACE       — compiled with BUILD_DEBUG_LEVEL_TRACE_HOST
 *
 * Device macros:
 *   LOGD_ERROR       — prints error (gated by BUILD_DEBUG_LEVEL_DEVICE)
 *   LOGD_ERROR_ABORT — prints + abort() (abort unconditional)
 *   LOGD_WARN        — gated by BUILD_DEBUG_LEVEL_DEVICE
 *   LOGD_INFO        — gated by BUILD_DEBUG_LEVEL_DEVICE
 *   LOGD_TRACE       — gated by BUILD_DEBUG_LEVEL_DEVICE + BUILD_DEBUG_LEVEL_TRACE_DEVICE
 */

namespace rocshmem {
  inline int log_pe_number = -1;

  // Calling static_assert_host_only from __device__/__global__ code produces a
  // compile error ("reference to __host__ function in __device__").
  __host__ inline void static_assert_host_only() {}

  // Calling static_assert_device_only from __host__ code produces a
  // compile error ("reference to __device__ function in __host__").
  __device__ inline void static_assert_device_only() {}
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

#ifdef BUILD_DEBUG_LEVEL_TRACE_HOST
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

struct log_state_device_t {
  int  pe_number;
  bool show_error;
  bool show_warn;
  bool show_info;
  bool show_api;
  bool show_trace;
  bool show_color;
};
extern __constant__ log_state_device_t log_device;

template <typename... Args>
[[maybe_unused]] __device__ __attribute__((noinline))
void dprintf(const char* fmt, const Args&... args) {
  int flat_thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
                       hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
  int flat_wg_id = hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
                   hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
  printf(fmt, log_device.pe_number, flat_wg_id, flat_thread_id, args...);
}

}  // namespace rocshmem

/*****************************************************************************
 * Device-side logging macros
 *
 * The printf portion is gated by BUILD_DEBUG_LEVEL_DEVICE.
 * abort() in ABORT_DEVICE and ERROR_DEVICE is unconditional.
 * Device printf uses rocshmem::dprintf() (defined above) which prepends
 * as the first 6 printf arguments, so the format string must start with
 * 6 %u specifiers to consume them.
 *
 * Callers should NOT include a trailing newline — the macros append one.
 *
 * LOGD_TRACE additionally requires BUILD_DEBUG_LEVEL_TRACE_DEVICE;
 * when BUILD_DEBUG_LEVEL_DEVICE is ON but BUILD_DEBUG_LEVEL_TRACE_DEVICE is OFF,
 * device trace is compiled away.
 *****************************************************************************/

#ifdef BUILD_DEBUG_LEVEL_DEVICE

/* Single dprintf call per macro; the ternary selects between two format
 * string literals (same args for both).  Only cost is one branch + two
 * string constants in .rodata per call site — no extra registers. */

#define LOGD_ERROR(fmt, ...) do {                                             \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::log_device.show_error)                                        \
  rocshmem::dprintf(rocshmem::log_device.show_color                           \
      ? "\033[1;91mE%04dw%04ut%04u\033[0m " fmt                               \
        "\t\033[90m" __FILE__ ":%d\033[0m\n"                                  \
      : "E%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                         \
      __VA_OPT__(__VA_ARGS__,) __LINE__);                                     \
} while (0)

#define LOGD_ERROR_ABORT(fmt, ...) do {                                       \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::log_device.show_error)                                        \
  rocshmem::dprintf(rocshmem::log_device.show_color                           \
      ? "\033[1;91mE%04dw%04ut%04u\033[0m " fmt                               \
        "\t\033[90m" __FILE__ ":%d\033[0m\n"                                  \
      : "E%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                         \
      __VA_OPT__(__VA_ARGS__,) __LINE__);                                     \
  abort();                                                                    \
} while (0)

#define LOGD_WARN(fmt, ...) do {                                              \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::log_device.show_warn)                                         \
    rocshmem::dprintf(rocshmem::log_device.show_color                         \
        ? "\033[1;93mW%04dw%04ut%04u\033[0m " fmt                             \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "W%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)

#define LOGD_INFO(fmt, ...) do {                                              \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::log_device.show_info)                                         \
    rocshmem::dprintf(rocshmem::log_device.show_color                         \
        ? "\033[32mI%04dw%04ut%04u\033[0m " fmt                               \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "I%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)

#if defined(BUILD_DEBUG_LEVEL_TRACE_DEVICE)
#define LOGD_API(fmt, ...) do {                                               \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::log_device.show_api)                                          \
    rocshmem::dprintf(rocshmem::log_device.show_color                         \
        ? "\033[95mA%04dw%04ut%04u\033[0m " fmt                               \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "A%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)

#define LOGD_TRACE(fmt, ...) do {                                             \
  rocshmem::static_assert_device_only();                                      \
  if (rocshmem::log_device.show_trace)                                        \
    rocshmem::dprintf(rocshmem::log_device.show_color                         \
        ? "\033[94mT%04dw%04ut%04u\033[0m " fmt                               \
          "\t\033[90m" __FILE__ ":%d\033[0m\n"                                \
        : "T%04dw%04ut%04u " fmt "\t" __FILE__ ":%d\n",                       \
        __VA_OPT__(__VA_ARGS__,) __LINE__);                                   \
} while (0)
#else
#define LOGD_API(...) do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_TRACE(...) do { rocshmem::static_assert_device_only(); } while (0)
#endif

#else  /* !BUILD_DEBUG_LEVEL_DEVICE */

#define LOGD_ERROR(...)       do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_ERROR_ABORT(...) do { rocshmem::static_assert_device_only(); abort(); } while (0)
#define LOGD_WARN(...)        do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_INFO(...)        do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_API(...)         do { rocshmem::static_assert_device_only(); } while (0)
#define LOGD_TRACE(...)       do { rocshmem::static_assert_device_only(); } while (0)

#endif  /* BUILD_DEBUG_LEVEL_DEVICE */

#endif  // LIBRARY_SRC_LOG_HPP_
