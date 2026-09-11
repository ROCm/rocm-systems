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

// Localized context control for kernel replay (see experimental/kernel_replay.h). During a replay
// loop the tool may enable/disable individual contexts per pass through the SDK-provided
// replay_local_enable_context / replay_local_disable_context callbacks. Those decisions are
// recorded in a thread-local override that lives only for the loop and is never written to global
// context state, so only the replaying thread's dispatches (this agent, serialized by the per-agent
// replay lock) observe them. Service consumers query local_context_override() at dispatch time to
// honor it.
//
// Two thread-local scopes are involved, both managed by the SDK (never by the tool):
//  - loop scope (scoped_local_context_control): the override map is live for the whole replay loop,
//    so services can query it while a pass dispatches. The map persists across passes, which gives
//    the "sticky" semantics (a toggle stays in effect until changed within the same loop).
//  - arm window (set_toggles_armed): the enable/disable callbacks are only legal while the tool's
//  PASS
//    PHASE_ENTER callback runs; execute_pass_phase_enter() arms them around that callback and
//    disarms after, so a call made outside that window fails.

#include "lib/rocprofiler-sdk/context/context.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace rocprofiler
{
namespace kernel_replay
{
struct local_context_control_t
{
    // context -> forced active (true) / inactive (false). An absent entry means "defer to global
    // context state". Owned by scoped_local_context_control for one replay loop and pointed to by
    // the thread-local routing while the loop runs.
    std::unordered_map<rocprofiler_context_id_t, bool> overrides{};

    // The contexts globally active when the replay loop began. A local start/stop is honored only
    // for these; toggling any other context is rejected, so a local start cannot promote a
    // globally-stopped context.
    std::unordered_set<rocprofiler_context_id_t> pre_active{};
};

// RAII: owns this replay loop's override map and installs it as the thread's active routing for the
// guard's lifetime, clearing the routing on destruction; global context state is never touched.
// Replays on one agent are serialized by the per-agent replay lock, so a loop never nests on a
// thread -- the destructor just clears the routing. Real teardown that can fail (e.g. PC sampling
// hardware) is done explicitly by the loop.
class scoped_local_context_control
{
public:
    // active_contexts = the contexts globally active when the replay loop begins (typically
    // context::get_active_contexts()). These become the pre-active mask: a local start/stop is only
    // honored for one of these (see local_context_control_t::pre_active).
    explicit scoped_local_context_control(const context::context_array_t& active_contexts);
    ~scoped_local_context_control();

    scoped_local_context_control(const scoped_local_context_control&) = delete;
    scoped_local_context_control& operator=(const scoped_local_context_control&) = delete;
    scoped_local_context_control(scoped_local_context_control&&)                 = delete;
    scoped_local_context_control& operator=(scoped_local_context_control&&) = delete;

private:
    local_context_control_t m_control{};  // this loop's overrides (owned)
};

// Arm/disarm the tool-facing toggle callbacks. execute_pass_phase_enter() brackets the tool's PASS
// PHASE_ENTER callback with set_toggles_armed(true)/(false); the callbacks below reject calls made
// while disarmed (i.e. outside that window).
void
set_toggles_armed(bool armed);

// Tool-facing callbacks (signatures match the function pointers in
// rocprofiler_callback_tracing_kernel_replay_data_t). Legal only while a loop scope is active and
// toggles are armed (i.e. during PASS PHASE_ENTER); otherwise they return
// ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR and record nothing. A toggle for a context that was not
// globally active when the loop began returns ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED and
// records nothing, so a local start cannot promote a globally-stopped context.
rocprofiler_status_t
replay_local_enable_context(rocprofiler_context_id_t context_id);

rocprofiler_status_t
replay_local_disable_context(rocprofiler_context_id_t context_id);

// Cheap fast-path gate: true only while a replay loop on this thread has recorded at least one
// override. Per-dispatch consumers check this first so normal dispatches (and replay passes with no
// toggles) pay a single thread-local read instead of a per-context scan.
bool
local_context_has_overrides();

// Consumer query: this thread's replay override for `context_id`, or std::nullopt when there is no
// active replay loop on this thread or the context has not been toggled. true = forced active,
// false = forced inactive. Callers fall back to the global/default active check on nullopt.
std::optional<bool>
local_context_override(rocprofiler_context_id_t context_id);
}  // namespace kernel_replay
}  // namespace rocprofiler
