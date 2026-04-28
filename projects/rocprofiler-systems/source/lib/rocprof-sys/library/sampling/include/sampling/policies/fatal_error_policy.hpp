// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// FatalErrorPolicy — named requirement
// ============================================================================
// Required member functions:
//   template <class... Args>
//   [[noreturn]] void fatal(char const* file, int line,
//                           std::string_view fmt, Args const&... args) noexcept;
//       Report a fatal error and terminate. In production, logs LOG_CRITICAL
//       then calls std::exit(1). In tests, throws SamplingFatalError.
//
//   The three fatal sites routed through this policy (NFR-FM-1):
//   1. pthread_sigmask failure (EC-8)
//   2. TmpFileOffloadStore::write — three sub-sites (L12, L13, L14 strings)
//   3. RealOverflowTrigger::configure on EPERM/ENOSYS (EC-3)
//
// Production: rocprofsys::sampling::RealFatalErrorPolicy
//             - LOG_CRITICAL(fmt, args...) then std::exit(1)
//             - serialized via FatalMutex (DEC-5, order-index 0) for atomic log line
// Test double: rocprofsys::sampling::test::ThrowingFatalErrorPolicy
//             - logs then throws SamplingFatalError (defined in
//             throwing_fatal_error_policy.hpp)

namespace rocprofsys::sampling
{
class real_fatal_error_policy;
}
namespace rocprofsys::sampling::test
{
struct throwing_fatal_error_policy;
}
