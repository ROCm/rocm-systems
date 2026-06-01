// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/cid_manager.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
#    include <hsa/hsa_ven_amd_pc_sampling.h>
#endif

#include <memory>
#include <vector>

namespace rocprofiler
{
namespace pc_sampling
{
// forward declaration to avoid circular dependency
class PCSCIDManager;

struct PCSAgentSession
{
    const rocprofiler_agent_t*       agent     = nullptr;
    rocprofiler_pc_sampling_method_t method    = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
    rocprofiler_pc_sampling_unit_t   unit      = ROCPROFILER_PC_SAMPLING_UNIT_NONE;
    uint64_t                         interval  = 0;
    rocprofiler_buffer_id_t          buffer_id = {.handle = 0};
    // hsa relevant information
    std::optional<hsa_agent_t> hsa_agent = std::nullopt;
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    hsa_ven_amd_pcs_t hsa_pc_sampling = {};
#endif
    hsa::ClientID intercept_cb_id = -1;
    // ioctl relevant information
    uint32_t ioctl_pcs_id = 0;
    // PC sampling parser
    std::unique_ptr<PCSamplingParserContext> parser = {};
    // Manager responsible for retiring CIDs
    std::unique_ptr<PCSCIDManager> cid_manager = {};
    // Context that owns this session
    rocprofiler_context_id_t context_id = {.handle = 0};
    // Client index from the context (tool's id)
    uint32_t client_idx = 0;

    // v2 API: which record kinds the client requested.
    // Empty means this session was created via the old (v1) API.
    std::vector<rocprofiler_pc_sampling_record_kind_t> requested_record_kinds = {};

    // v2 API: whether the client opted in to receiving invalid samples (i.e. requested the
    // INVALID_SAMPLE record kind). Computed once at configure time from requested_record_kinds and
    // forwarded to parser->parse() per call. False for v1 sessions.
    bool deliver_invalid = false;

    // Returns true if this session was created via the v2 API
    bool is_v2_api() const { return !requested_record_kinds.empty(); }

    // Returns the valid version record kind (V0-V4) if configured via v2 API,
    // or ROCPROFILER_PC_SAMPLING_RECORD_NONE if not applicable.
    rocprofiler_pc_sampling_record_kind_t get_valid_record_kind() const
    {
        for(auto kind : requested_record_kinds)
        {
            if(kind != ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE) return kind;
        }
        return ROCPROFILER_PC_SAMPLING_RECORD_NONE;
    }

    // Returns true if the client wants to receive invalid samples
    bool wants_invalid_samples() const
    {
        for(auto kind : requested_record_kinds)
        {
            if(kind == ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE) return true;
        }
        return false;
    }
};

// TODO static assertions

}  // namespace pc_sampling
}  // namespace rocprofiler
