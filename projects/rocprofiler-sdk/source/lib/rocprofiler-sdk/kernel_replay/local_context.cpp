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

#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"

namespace rocprofiler
{
namespace kernel_replay
{
namespace
{
// The current replay loop's override map for this thread, or null when no loop is active. Points at
// a caller-owned local_context_control_t (a stack local in the WriteInterceptor replay loop); the
// scope guards below manage this pointer, so it is always valid while non-null.
thread_local local_context_control_t* tl_control = nullptr;

// Whether the tool-facing toggle callbacks are currently legal (true only during the tool's PASS
// PHASE_ENTER callback; set by set_toggles_armed()).
thread_local bool tl_toggles_armed = false;

rocprofiler_status_t
record_override(rocprofiler_context_id_t context_id, bool active)
{
    // Legal only inside a replay loop (tl_control set) and during the PASS PHASE_ENTER window
    // (armed). A stray call outside either fails and records nothing.
    if(tl_control == nullptr || !tl_toggles_armed) return ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR;
    // Only contexts globally active when the loop began may be toggled: a local start must not
    // promote a globally-stopped context (its service/callback thread is stopped), and there is
    // nothing to stop for one either. Reject and record nothing otherwise. (C3)
    if(tl_control->pre_active.count(context_id) == 0)
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
    tl_control->overrides[context_id] = active;
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

scoped_local_context_control::scoped_local_context_control(
    const context::context_array_t& active_contexts)
{
    for(const auto* ctx : active_contexts)
        if(ctx != nullptr)
            m_control.pre_active.emplace(rocprofiler_context_id_t{.handle = ctx->context_idx});
    tl_control = &m_control;
}

scoped_local_context_control::~scoped_local_context_control() { tl_control = nullptr; }

void
set_toggles_armed(bool armed)
{
    tl_toggles_armed = armed;
}

rocprofiler_status_t
replay_local_enable_context(rocprofiler_context_id_t context_id)
{
    return record_override(context_id, true);
}

rocprofiler_status_t
replay_local_disable_context(rocprofiler_context_id_t context_id)
{
    return record_override(context_id, false);
}

bool
local_context_has_overrides()
{
    return tl_control != nullptr && !tl_control->overrides.empty();
}

std::optional<bool>
local_context_override(rocprofiler_context_id_t context_id)
{
    if(tl_control == nullptr) return std::nullopt;
    auto itr = tl_control->overrides.find(context_id);
    if(itr == tl_control->overrides.end()) return std::nullopt;
    return itr->second;
}
}  // namespace kernel_replay
}  // namespace rocprofiler
