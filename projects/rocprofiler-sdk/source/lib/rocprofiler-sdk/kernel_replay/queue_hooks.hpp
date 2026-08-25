// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <cstdint>
#include <functional>

namespace rocprofiler
{
namespace hsa
{
class Queue;
}
namespace kernel_replay
{
/// @brief Everything the replay window needs from the AQL packet interceptor.
///
/// The window is a multi-pass, blocking sequence: drain the agent, snapshot device memory, run the
/// dispatch N times, restore between passes. The interceptor's own job is per-packet rewriting for
/// every other service. Handing the window these few callables keeps the two apart, so a change to
/// either does not have to be read as a change to the other.
struct replay_dispatch_t
{
    const hsa::Queue*              queue  = nullptr;
    const hsa::rocprofiler_packet* packet = nullptr;

    rocprofiler_thread_id_t thread_id               = 0;
    uint64_t                internal_correlation_id = 0;
    uint64_t                ancestor_correlation_id = 0;

    /// Reserved by the caller before CONFIG runs, so the whole logical dispatch -- the CONFIG
    /// callbacks, every pass, and the single run when replay is declined -- shares one id and the
    /// dispatch counter is bumped exactly once.
    rocprofiler_dispatch_id_t dispatch_id = 0;

    /// @brief Submit this dispatch once through the caller's ordinary packet-transform path, so
    /// counter collection, the serializer, the async completion handler and record delivery all
    /// behave exactly as they do for an un-replayed dispatch.
    /// @param is_replay_pass Suppress the application's completion signal. The window fires that
    /// signal itself, once, after the loop, so the application observes a single execution
    /// regardless of pass count, early exit, or an indefinite loop.
    std::function<void(bool is_replay_pass)> submit_dispatch{};

    /// @brief Put a bare barrier-AND packet on the queue that decrements @p completion once prior
    /// work on the queue has retired. Used for the pre-snapshot fence and for the application's
    /// single completion.
    std::function<void(hsa_signal_t completion)> submit_barrier{};
};

/// @brief Run one dispatch through the replay window: the CONFIG callbacks, admission control, the
/// queue and agent drains, the snapshot/restore pass loop, and every decline path.
///
/// The lock this takes exclusively, and the one an ordinary dispatch must take shared while a
/// replay service is active, are both in replay_diagnostics.hpp -- see dispatch_lock_for().
///
/// @retval true The dispatch has been submitted here -- replayed, or declined and run exactly once
/// -- and the caller must not submit it again.
/// @retval false The tool did not ask for this dispatch to be replayed. The caller submits it on
/// its ordinary path, reusing @c dispatch_id.
bool
run_replay_window(const replay_dispatch_t& dispatch);
}  // namespace kernel_replay
}  // namespace rocprofiler
