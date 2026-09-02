////////////////////////////////////////////////////////////////////////////////
//
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Escalating backoff for userspace polling fallbacks.
//
// Some wait paths cannot sleep in the kernel (e.g. AsyncEventsLoop when the
// thunk exposes no interrupt-backed signal events, as on the WSL/dxg thunk).
// Those paths re-scan signal values in userspace and would otherwise burn a
// full CPU core. To keep idle cost near zero without penalizing a wait that
// completes quickly, the nap between scans starts at a small floor and doubles
// up to a ceiling; callers reset to the floor whenever a wait makes progress so
// the escalation only compounds within a single idle wait.

#ifndef HSA_RUNTIME_CORE_UTIL_POLL_BACKOFF_H_
#define HSA_RUNTIME_CORE_UTIL_POLL_BACKOFF_H_

#include <algorithm>

namespace rocr {
namespace core {

// Floor (microseconds) for the polling-fallback nap.
inline constexpr int kPollNapFloorUs = 20;

// Ceiling (microseconds) when the runtime has no interrupt-backed signal
// events at all (g_use_interrupt_wait == false, e.g. the WSL/dxg thunk).
// Every signal is polling-only, so the only cost of a long nap is observation
// latency of the napping wait itself.
inline constexpr int kPollNapCeilingUs = 2000;

// Ceiling (microseconds) when interrupts are available globally but the wait
// batch was forced into polling by a signal with no EopEvent (an IPC signal
// or an internal DefaultSignal, e.g. gang copies). One such signal drags every
// interrupt-backed signal on the shared async-events thread into this polling
// scan, so the nap here bounds the added callback latency of unrelated
// interrupt-backed handlers. Kept at the interrupt path's 200us active-poll
// window (see AsyncEventsLoop) so that bound stays at the noise floor.
inline constexpr int kPollNapCeilingMixedUs = 200;

// Given the current nap duration, return the next one: double it, capped at
// ceiling_us. Saturating at the ceiling is a fixed point, so repeated calls
// converge to and stay at ceiling_us. The multiply is only evaluated when
// current_us <= ceiling_us/2, so current_us*2 <= ceiling_us and can never
// overflow even for a ceiling_us up to INT_MAX.
constexpr int NextPollNapUs(int current_us, int ceiling_us = kPollNapCeilingUs) {
  return (current_us > ceiling_us / 2) ? ceiling_us
                                       : std::min(current_us * 2, ceiling_us);
}

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_POLL_BACKOFF_H_
