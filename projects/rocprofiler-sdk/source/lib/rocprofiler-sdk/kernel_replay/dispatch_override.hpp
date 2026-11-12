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

// A replay pass may submit a tool-supplied dispatch instead of the application's
// (rocprofiler_callback_tracing_kernel_replay_data_t::replay_set_dispatch). These helpers convert
// between AQL dispatch packets and rocprofiler_kernel_replay_dispatch_t. They touch no GPU or
// runtime state, so they are unit tested directly.

#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <optional>

namespace rocprofiler
{
namespace kernel_replay
{
// The dispatch `pkt` submits, or std::nullopt when it is not a kernel dispatch (plain or gfx1250
// extended).
std::optional<rocprofiler_kernel_replay_dispatch_t>
describe_dispatch(const hsa::rocprofiler_packet& pkt);

// Build the packet a replay pass submits for `dispatch`. It starts from the application's packet
// `app`, so the packet type, header (fences, barrier bit), dependency signal and reserved fields
// are kept, and only the dispatch's own fields change. Returns
// ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT, leaving `out` untouched, when `dispatch` cannot be
// expressed in a packet of that type.
rocprofiler_status_t
build_dispatch_packet(const hsa::rocprofiler_packet&              app,
                      const rocprofiler_kernel_replay_dispatch_t& dispatch,
                      hsa::rocprofiler_packet&                    out);
}  // namespace kernel_replay
}  // namespace rocprofiler
