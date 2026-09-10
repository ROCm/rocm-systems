/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// One spelling of "this path must not be reachable", for the fakes files this
// refactor introduced or touched.
//
// SCOPE, stated so the charter is not read as wider than it is: the older
// fail-loud floors (collective_stubs.cc, nccl_stubs.cc, topo_stubs.cc,
// bootstrap_stubs.cc, and the pre-existing entries in transport_stubs.cc) still
// use bare ::abort(), which prints nothing at all. Converting roughly 90 sites
// is a separate change; this header is what any NEW site should use, and what
// those sites should converge on.
//
// Output is always a single line, so a CI log can be swept with one pattern
// regardless of which fakes file aborted.

#ifndef RCCL_TEST_HOST_FAIL_LOUD_H_
#define RCCL_TEST_HOST_FAIL_LOUD_H_

#include <cstdio>
#include <cstdlib>

// `what` is a free-form message: use this when the abort is an argument check or
// anything other than "nobody faked this symbol".
[[noreturn]] inline void FailLoud(const char* tag, const char* what) {
  std::fprintf(stderr, "[%s] %s\n", tag, what);
  std::fflush(stderr);
  ::abort();
}

// The stub-floor case. Keeps the verb in the message: a log line naming only a
// symbol does not say what went wrong, and every caller would otherwise have to
// remember to spell the qualifier itself.
[[noreturn]] inline void FailLoudUnfaked(const char* tag, const char* symbol) {
  std::fprintf(stderr, "[%s] unfaked call: %s\n", tag, symbol);
  std::fflush(stderr);
  ::abort();
}

#endif  // RCCL_TEST_HOST_FAIL_LOUD_H_
