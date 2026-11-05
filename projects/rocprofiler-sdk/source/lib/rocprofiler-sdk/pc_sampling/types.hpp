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

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

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

    // Lifecycle, in order:
    // 1. configure_pc_sampling_service creates the session and records the owning context.
    // 2. start_service / stop_service are the tool's global intent (context start/stop).
    // 3. sync_replay_service / restore_replay_service suspend and restore one agent's ROCr
    //    session around a kernel-replay pass without changing that global intent.
    // 4. service_fini tears the session down.
    // Transitions are serialized. ROCr start/stop and the stop-time flush must finish before a
    // replay pass uses another clock-gating-sensitive service. replay_suspended is read under
    // state_mutex so a global start cannot resurrect sampling during a counter pass.
    std::mutex              state_mutex{};
    std::condition_variable state_cv{};
    bool                    transition_in_progress = false;
    bool                    transition_target      = false;
    std::thread::id         transition_owner{};
    std::atomic<bool>       active{false};
    std::atomic<bool>       replay_suspended{false};
};

// TODO static assertions

}  // namespace pc_sampling
}  // namespace rocprofiler
