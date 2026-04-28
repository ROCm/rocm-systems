// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// TimerTriggerPolicy — named requirement
// ============================================================================
// Required member functions:
//   void configure(int64_t tid, pid_t sys_tid, int signum,
//                  clockid_t clock, double freq_hz, double delay_sec) noexcept;
//       Configure a POSIX interval timer for the thread. Does not arm it.
//   void start() noexcept;
//       Arm the timer (timer_settime). Idempotent.
//   void stop() noexcept;
//       Disarm the timer. Idempotent. Safe to call from destructor.
//   bool is_armed() const noexcept;
//       Return true if the timer is currently armed.
//
// Production: rocprofsys::sampling::RealTimerTrigger
//             - wraps timer_create/timer_settime with SIGEV_THREAD_ID
//             - per-instance timer_t handle; destructor calls stop() then timer_delete
//             - Requires (Linux): timer_create(2), timer_settime(2), SIGEV_THREAD_ID
// Test double: rocprofsys::sampling::test::MockTimerTrigger
//             - records (tid, sys_tid, signum, clock, freq_hz, delay_sec) per configure()
//             - flags armed/disarmed via start()/stop()

namespace rocprofsys::sampling
{
class real_timer_trigger;
}
namespace rocprofsys::sampling::test
{
struct mock_timer_trigger;
}
