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

#include "lib/rocprofiler-sdk/kernel_replay/utils.hpp"

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa_ext_amd.h>

#include <cstdint>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_tracker
{
namespace
{
// HSA_EXT_POINTER_TYPE_HSA_VMEM is absent from HSA headers older than 1.12 (the same reason
// memory_tracker.cpp defines memory_pool_executable_flag by hand). Comparing the numeric value
// keeps this file buildable against an older external header; on such a runtime the field simply
// never reports 6 and the classification degrades to the hsa_amd_vmem_map interception alone.
constexpr uint32_t pointer_type_hsa_vmem = 6;
}  // namespace

alloc_query_t
query_alloc(void* ptr)
{
    alloc_query_t q = {};
    if(ptr == nullptr) return q;

    // Resolve pointer_info from rocprofiler's captured HSA table (the original, un-wrapped
    // function). Same accessor pattern memory_snapshot uses for hsa_memory_copy.
    auto* ext = hsa::get_amd_ext_table();
    if(ext == nullptr || ext->hsa_amd_pointer_info_fn == nullptr) return q;

    hsa_amd_pointer_info_t info = {};
    info.size                   = sizeof(info);
    if(ext->hsa_amd_pointer_info_fn(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS)
        return q;

    q.agent   = info.agentOwner;
    q.kernarg = (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0;
    q.coarse  = (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0;
    // ROCr does not reliably tag IPC-opened memory with HSA_EXT_POINTER_TYPE_IPC, so this is a
    // best-effort signal only; it never promotes an allocation into the snapshot, it only keeps one
    // out and raises the untracked count.
    q.ipc_shared = (info.type == HSA_EXT_POINTER_TYPE_IPC);
    q.vmem       = (static_cast<uint32_t>(info.type) == pointer_type_hsa_vmem);
    q.gpu_owned  = is_gpu_agent(info.agentOwner);
    q.trackable  = q.coarse && !q.kernarg && !q.ipc_shared && !q.vmem;
    return q;
}

bool
is_gpu_agent(hsa_agent_t agent)
{
    if(agent.handle == 0) return false;

    auto* core = hsa::get_core_table();
    if(core == nullptr || core->hsa_agent_get_info_fn == nullptr) return false;

    hsa_device_type_t type{};
    if(core->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
        return false;
    return type == HSA_DEVICE_TYPE_GPU;
}
}  // namespace memory_tracker
}  // namespace kernel_replay
}  // namespace rocprofiler
