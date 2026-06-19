// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/memory_snapshot.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/memory_tracker.hpp"

#include <hsa/hsa.h>

namespace rocprofiler
{
namespace hsa
{
namespace memory_snapshot
{
namespace
{
hsa_status_t
dma_copy(void* dst, const void* src, size_t n)
{
    auto* core = get_core_table();
    if(!core || !core->hsa_memory_copy_fn) return HSA_STATUS_ERROR;
    return core->hsa_memory_copy_fn(dst, src, n);
}
}  // namespace

size_t
Snapshot::snap()
{
    auto inventory = memory_tracker::snap_inventory();

    blocks_.clear();
    blocks_.reserve(inventory.size());

    for(const auto& [ptr, size] : inventory)
    {
        if(size == 0) continue;

        mem_block blk;
        blk.gpu_addr = ptr;
        blk.host_copy.resize(size);

        if(dma_copy(blk.host_copy.data(), ptr, size) != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << "replay snapshot: device->host copy failed for region " << ptr;
            continue;
        }

        blocks_.push_back(std::move(blk));
    }

    ROCP_INFO << "replay snapshot: captured " << blocks_.size() << "/" << inventory.size()
              << " regions";
    return blocks_.size();
}

size_t
Snapshot::restore()
{
    size_t ok = 0;
    for(const auto& blk : blocks_)
    {
        if(dma_copy(blk.gpu_addr, blk.host_copy.data(), blk.host_copy.size()) != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << "replay restore: host->device copy failed for region " << blk.gpu_addr;
            continue;
        }
        ++ok;
    }

    ROCP_INFO << "replay restore: restored " << ok << "/" << blocks_.size() << " regions";
    return ok;
}
}  // namespace memory_snapshot
}  // namespace hsa
}  // namespace rocprofiler
