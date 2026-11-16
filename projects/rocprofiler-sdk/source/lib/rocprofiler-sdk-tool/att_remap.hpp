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
      const dispatch_source&                             source,
      rocprofiler_context_id_t                           att_context);

/// Whether a remap loop is running on this thread.
bool
active();

/// PASS PHASE_ENTER: submit and trace what the remapper planned for this pass.
void
pass_enter(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
           rocprofiler_user_data_t                            user_data);

/// PASS PHASE_EXIT: hand the pass's results to the remapper.
void
pass_exit(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
          rocprofiler_user_data_t                            user_data);

/// CONFIG PHASE_EXIT: end the session.
void
end(rocprofiler_user_data_t user_data);

/// The current pass's thread trace target, if the remapper traces it.
std::optional<att_replay_target>
current_trace_target();

/// The bytes of a loaded code object: the memory it was loaded from, or, for a file-backed one, the
/// range its `file://PATH#offset=N&size=M` URI names, read through `file_descriptor` (the file
/// at PATH when the descriptor cannot be read).
std::optional<std::vector<uint8_t>>
code_object_bytes(const rocprofiler_callback_tracing_code_object_load_data_t& code_object);
}  // namespace att_remap
}  // namespace tool
}  // namespace rocprofiler
