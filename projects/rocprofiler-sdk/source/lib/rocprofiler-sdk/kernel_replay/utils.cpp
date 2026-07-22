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

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_tracker
{
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

    q.agent               = info.agentOwner;
    const bool is_kernarg = (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0;
    const bool is_coarse =
        (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0;
    q.trackable = is_coarse && !is_kernarg;
    return q;
}
}  // namespace memory_tracker
}  // namespace kernel_replay
}  // namespace rocprofiler
