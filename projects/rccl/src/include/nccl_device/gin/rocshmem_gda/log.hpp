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

#include <cstdlib>

#include <hip/hip_runtime.h>

#ifndef LIBRARY_SRC_LOG_HPP_
#define LIBRARY_SRC_LOG_HPP_

namespace rocshmem {
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

  template <typename... T>
  __host__ __device__ constexpr void unused_args([[maybe_unused]] T&& ...) { }
}  // namespace rocshmem



/*****************************************************************************
 * Host-side logging macros
 *
 * TODO: implement using RCCL logging
 *****************************************************************************/

#define LOG_ERROR(fmt, ...) do {                                              \
  rocshmem::static_assert_host_only();                                        \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOG_ERROR_EXIT(fmt, ...) do {                                         \
  rocshmem::static_assert_host_only();                                        \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
  exit(EXIT_FAILURE);                                                         \
} while (0)

#define LOG_ERROR_ABORT(fmt, ...) do {                                        \
  rocshmem::static_assert_host_only();                                        \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
  abort();                                                                    \
} while (0)

#define LOG_WARN(fmt, ...) do {                                               \
  rocshmem::static_assert_host_only();                                        \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOG_INFO(fmt, ...) do {                                               \
  rocshmem::static_assert_host_only();                                        \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOG_API(fmt, ...) do {                                                \
  rocshmem::static_assert_host_only();                                        \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOG_TRACE(fmt, ...) do {                                              \
  rocshmem::static_assert_host_only();                                        \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)



/*****************************************************************************
 * Device-side logging macros
 *
 * TODO: implement using RCCL logging
 *****************************************************************************/

#define LOGD_ERROR(fmt, ...) do {                                             \
  rocshmem::static_assert_device_only();                                      \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOGD_ERROR_ABORT(fmt, ...) do {                                       \
  rocshmem::static_assert_device_only();                                      \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
  abort();                                                                    \
} while (0)

#define LOGD_WARN(fmt, ...) do {                                              \
  rocshmem::static_assert_device_only();                                      \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOGD_INFO(fmt, ...) do {                                              \
  rocshmem::static_assert_device_only();                                      \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOGD_API(fmt, ...) do {                                               \
  rocshmem::static_assert_device_only();                                      \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)

#define LOGD_TRACE(fmt, ...) do {                                             \
  rocshmem::static_assert_device_only();                                      \
  rocshmem::unused_args(fmt __VA_OPT__(,) __VA_ARGS__);                       \
} while (0)



#endif  // LIBRARY_SRC_LOG_HPP_
