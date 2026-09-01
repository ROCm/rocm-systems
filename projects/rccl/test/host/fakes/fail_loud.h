/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// One spelling of "this path must not be reachable" for every fakes file.
//
// The stub floors and the argument checks inside the fakes had grown three
// different helper names and two message formats, so the same concept was not
// greppable. Splitting the floor by production TU multiplied the copies rather
// than reducing them, which is what made a shared helper worth its own header.
//
// Output is a single line, always `[<tag>] <what>`, so a CI log can be swept
// with one pattern regardless of which fakes file aborted.

#ifndef RCCL_TEST_HOST_FAIL_LOUD_H_
#define RCCL_TEST_HOST_FAIL_LOUD_H_

#include <cstdio>
#include <cstdlib>

// `what` is a free-form message, not necessarily a function name: the stub
// floors pass the unfaked symbol, the argument checks pass what was violated.
[[noreturn]] inline void FailLoud(const char* tag, const char* what) {
  std::fprintf(stderr, "[%s] %s\n", tag, what);
  std::fflush(stderr);
  ::abort();
}

#endif  // RCCL_TEST_HOST_FAIL_LOUD_H_
