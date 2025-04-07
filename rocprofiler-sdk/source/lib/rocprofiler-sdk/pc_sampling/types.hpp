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
};

// TODO static assertions

}  // namespace pc_sampling
}  // namespace rocprofiler
