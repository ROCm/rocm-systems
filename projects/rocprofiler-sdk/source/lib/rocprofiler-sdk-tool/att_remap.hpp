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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "config.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace att_remap
{
/// Whether --att-remap is on (ROCPROF_ATT_REMAP).
bool
enabled();

/// Load the remapper library. A failure is fatal: the run was asked to remap.
void
init(const std::string& output_path);

/// Record each remapped dispatch in ROCPROF_ATT_REMAP_INDEX and unload the remapper.
void
fini(const std::string& trace_path);

/// The SIMDs of a WGP; a remapped dispatch is traced on each in turn.
constexpr uint64_t num_simds = 4;

/// Configure the device thread trace contexts that trace `agent`: one per SIMD, because a thread
/// trace context traces the target it was configured with. Each uses `parameters` with the SIMD
/// replaced and every one of the agent's `num_shader_engines` selected, and delivers its trace to
/// `shader_callback`. The contexts stay stopped until a pass traces through them.
void
configure_agent(rocprofiler_agent_id_t                                   agent,
                uint64_t                                                 num_shader_engines,
                const std::vector<rocprofiler_thread_trace_parameter_t>& parameters,
                rocprofiler_thread_trace_shader_data_callback_t          shader_callback);

/// Whether configure_agent() set `agent` up for tracing.
bool
traces_agent(rocprofiler_agent_id_t agent);

/// Keep a loaded code object's bytes; the remapper rewrites them.
void
add_code_object(uint64_t code_object_id, std::vector<uint8_t> bytes);

/// The application dispatch a replay loop is about to run.
struct dispatch_source
{
    const char* kernel_name    = nullptr;
    uint64_t    code_object_id = 0;
    uint32_t    agent_node_id  = 0;
};

/// CONFIG PHASE_ENTER: take over the dispatch if the remapper wants it. On success the payload's
/// pass count and continue callbacks drive an indefinite loop and `user_data` holds the session.
bool
begin(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
      rocprofiler_user_data_t*                           user_data,
      const dispatch_source&                             source);

/// Whether a remap loop is running on this thread.
bool
active();

/// PASS PHASE_ENTER: submit what the remapper planned for this pass and, for a traced pass, start
/// the device thread trace context of its SIMD.
void
pass_enter(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
           rocprofiler_user_data_t                            user_data);

/// PASS PHASE_EXIT: stop the pass's thread trace context, which delivers its trace, and hand the
/// pass's results to the remapper.
void
pass_exit(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
          rocprofiler_user_data_t                            user_data);

/// CONFIG PHASE_EXIT: end the session.
void
end(rocprofiler_user_data_t user_data);

/// A replay pass whose thread trace is being collected or delivered.
struct traced_pass
{
    uint64_t          dispatch_id = 0;
    uint64_t          pass        = 0;
    att_replay_target target      = {};
};

/// The pass `agent` is tracing, from its PASS PHASE_ENTER until its trace has been delivered. An
/// agent replays one dispatch at a time, so there is at most one.
std::optional<traced_pass>
traced_pass_on(rocprofiler_agent_id_t agent);

/// The bytes of a loaded code object: the memory it was loaded from, or, for a file-backed one, the
/// range its `file://PATH#offset=N&size=M` URI names, read through `file_descriptor` (the file
/// at PATH when the descriptor cannot be read).
std::optional<std::vector<uint8_t>>
code_object_bytes(const rocprofiler_callback_tracing_code_object_load_data_t& code_object);
}  // namespace att_remap
}  // namespace tool
}  // namespace rocprofiler
