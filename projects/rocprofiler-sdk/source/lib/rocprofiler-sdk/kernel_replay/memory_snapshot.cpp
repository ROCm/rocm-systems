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

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <new>
#include <optional>
#include <string_view>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_snapshot
{
namespace
{
hsa_status_t
dma_copy(void* dst, const void* src, size_t n)
{
    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_memory_copy_fn) return HSA_STATUS_ERROR;
    return core->hsa_memory_copy_fn(dst, src, n);
}
}  // namespace

size_t
Snapshot::snap()
{
    auto inventory = memory_tracker::snap_inventory();

    blocks_.clear();
    ok_ = true;
    blocks_.reserve(inventory.size());

    auto capture = [&](void* gpu_addr, size_t size, std::string_view what) -> bool {
        if(size == 0) return true;

        mem_block blk;
        blk.gpu_addr = gpu_addr;
        try
        {
            blk.host_copy.resize(size);
        } catch(const std::bad_alloc&)
        {
            ROCP_WARNING << fmt::format("kernel-replay snapshot: host allocation of {} bytes "
                                        "failed for {} {} (memory pressure)",
                                        size,
                                        what,
                                        gpu_addr);
            return false;
        }

        if(dma_copy(blk.host_copy.data(), gpu_addr, size) != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << fmt::format(
                "kernel-replay snapshot: device->host copy failed for {} {} ({}B)",
                what,
                gpu_addr,
                size);
            return false;
        }

        blocks_.push_back(std::move(blk));
        return true;
    };

    for(const auto& [ptr, size] : inventory)
    {
        if(!capture(ptr, size, "region"))
        {
            ok_ = false;
            blocks_.clear();
            return 0;
        }
    }

    ROCP_INFO << fmt::format("kernel-replay snapshot: captured {}/{} regions",
                             blocks_.size(),
                             inventory.size());
    return blocks_.size();
}

size_t
Snapshot::restore()
{
    size_t ok = 0;
    for(const auto& blk : blocks_)
    {
        const auto status = memory_tracker::restore_tracked_region(
            blk.gpu_addr, blk.host_copy.data(), blk.host_copy.size());

        if(!status)
        {
            ROCP_WARNING << fmt::format(
                "kernel-replay restore: skipping region {} ({}B) that is no longer live",
                blk.gpu_addr,
                blk.host_copy.size());
            continue;
        }

        if(*status != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << fmt::format(
                "kernel-replay restore: host->device copy failed for region {} ({}B)",
                blk.gpu_addr,
                blk.host_copy.size());
            continue;
        }

        ++ok;
    }

    ROCP_INFO << fmt::format("kernel-replay restore: restored {}/{} regions", ok, blocks_.size());
    return ok;
}
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
