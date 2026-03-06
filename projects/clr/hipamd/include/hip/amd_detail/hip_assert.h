/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if defined(__clang__) && defined(__HIP__)

// abort
extern "C" __device__ inline __attribute__((weak)) void abort() { __builtin_trap(); }

// The noinline attribute helps encapsulate the printf expansion,
// which otherwise has a performance impact just by increasing the
// size of the calling function. Additionally, the weak attribute
// allows the function to exist as a global although its definition is
// included in every compilation unit.
#if defined(_WIN32) || defined(_WIN64)
extern "C" __device__ __attribute__((noinline)) __attribute__((weak)) void _wassert(
    const wchar_t* _msg, const wchar_t* _file, unsigned _line) {
  // FIXME: Need `wchar_t` support to generate assertion message.
  __builtin_trap();
}
#else /* defined(_WIN32) || defined(_WIN64) */
extern "C" __device__ __attribute__((noinline)) __attribute__((weak)) void __assert_fail(
    const char* assertion, const char* file, unsigned int line, const char* function) {
  const char fmt[] = "%s:%u: %s: Device-side assertion `%s' failed.\n";

  // strlen is not available as a built-in yet, so we create our own
  // loop in a macro. With a string literal argument, the compiler
  // usually manages to replace the loop with a constant.
  //
  // The macro does not check for null pointer, since all the string
  // arguments are defined to be constant literals when called from
  // the assert() macro.
  //
  // NOTE: The loop below includes the null terminator in the length
  // as required by append_string_n().
#define __hip_get_string_length(LEN, STR)                                                          \
  do {                                                                                             \
    const char* tmp = STR;                                                                         \
    while (*tmp++);                                                                                \
    LEN = tmp - STR;                                                                               \
  } while (0)

  auto msg = __ockl_fprintf_stderr_begin();
  int len = 0;
  __hip_get_string_length(len, fmt);
  msg = __ockl_fprintf_append_string_n(msg, fmt, len, 0);
  __hip_get_string_length(len, file);
  msg = __ockl_fprintf_append_string_n(msg, file, len, 0);
  msg = __ockl_fprintf_append_args(msg, 1, line, 0, 0, 0, 0, 0, 0, 0);
  __hip_get_string_length(len, function);
  msg = __ockl_fprintf_append_string_n(msg, function, len, 0);
  __hip_get_string_length(len, assertion);
  __ockl_fprintf_append_string_n(msg, assertion, len, /* is_last = */ 1);

#undef __hip_get_string_length

  __builtin_trap();
}

extern "C" __device__ __attribute__((noinline)) __attribute__((weak)) void __assertfail() {
  // ignore all the args for now.
  __builtin_trap();
}
#endif /* defined(_WIN32) || defined(_WIN64) */

#if defined(NDEBUG)
#define __hip_assert(COND)
#else
#define __hip_assert(COND)                                                                         \
  do {                                                                                             \
    if (!(COND)) __builtin_trap();                                                                 \
  } while (0)
#endif

#endif  // defined(__clang__) and defined(__HIP__)
