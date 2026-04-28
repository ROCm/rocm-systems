// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// SignalDispatcherPolicy — named requirement
// ============================================================================
// Required member functions:
//   int sigmask(int how, sigset_t const* set, sigset_t* oldset) noexcept;
//       pthread_sigmask passthrough. how is SIG_BLOCK, SIG_UNBLOCK, or SIG_SETMASK.
//       Returns 0 on success, errno value on failure.
//       On non-zero return, SignalMaskGuard calls FatalErrorPolicy::fatal() with
//       the matching message (EC-8 / NFR-FM-1).
//
// Production: rocprofsys::sampling::RealSignalDispatcher
//             - directly calls ::pthread_sigmask
// Test double: rocprofsys::sampling::test::RecordingSignalDispatcher
//             - records (how, set) per call; returns scripted errno or 0

namespace rocprofsys::sampling
{
class real_signal_dispatcher;
}
namespace rocprofsys::sampling::test
{
struct recording_signal_dispatcher;
}
