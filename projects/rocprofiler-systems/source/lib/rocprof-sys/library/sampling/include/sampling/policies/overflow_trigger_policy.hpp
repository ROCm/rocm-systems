// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// OverflowTriggerPolicy — named requirement
// ============================================================================
// Required member functions:
//   void configure(int64_t tid, pid_t sys_tid, int signum,
//                  perf_event_attr const& attr);
//       Open a perf event and arm for overflow-signal delivery.
//       On EPERM/ENOSYS, calls FatalErrorPolicy::fatal().
//   void start() noexcept;
//       Enable the perf event counter (PERF_EVENT_IOC_ENABLE).
//   void stop() noexcept;
//       Disable the perf event counter (PERF_EVENT_IOC_DISABLE). Idempotent.
//   bool is_open() const noexcept;
//       Return true if the perf_event fd is valid.
//
// Production: rocprofsys::sampling::RealOverflowTrigger
//             - replaces library/perf.hpp perf_event class
//             - perf_event_open + mmap + fcntl(F_SETSIG + F_SETOWN_EX)
//             - destructor closes fd and munmaps ring
//             - Requires (Linux): perf_event_open(2), PERF_EVENT_IOC_*
// Test double: rocprofsys::sampling::test::MockOverflowTrigger
//             - records configure() arguments; start/stop/is_open tracking

namespace rocprofsys::sampling
{
class real_overflow_trigger;
}
namespace rocprofsys::sampling::test
{
struct mock_overflow_trigger;
}
