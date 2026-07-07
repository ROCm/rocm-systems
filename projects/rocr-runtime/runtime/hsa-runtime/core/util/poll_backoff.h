////////////////////////////////////////////////////////////////////////////////
//
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

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

// Default floor and ceiling (microseconds) for the polling-fallback nap.
constexpr int kPollNapFloorUs = 20;
constexpr int kPollNapCeilingUs = 2000;

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
