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

#include "lib/rocprofiler-sdk/kfd/kfd_memory.hpp"
#include "lib/common/logging.hpp"

#include <hsakmt/hsakmt.h>
#include <hsakmt/hsakmttypes.h>

#include <cstring>
#include <mutex>
#include <utility>

namespace rocprofiler
{
namespace kfd
{
namespace
{
constexpr uint64_t page_size = 4096;

uint64_t
align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

std::once_flag kfd_init_flag;
bool           kfd_initialized = false;

void
ensure_kfd_open()
{
    std::call_once(kfd_init_flag, []() {
        auto status = hsaKmtOpenKFD();
        if(status == HSAKMT_STATUS_SUCCESS)
        {
            kfd_initialized = true;
            ROCP_TRACE << "KFD opened for direct memory allocation";
        }
        else
        {
            ROCP_ERROR << "Failed to open KFD: status=" << status;
        }
    });
}

HsaMemFlags
make_flags(kfd_memory_type type)
{
    HsaMemFlags flags;
    memset(&flags, 0, sizeof(flags));

    switch(type)
    {
        case kfd_memory_type::host_uncached:
            flags.ui32.HostAccess  = 1;
            flags.ui32.Uncached    = 1;
            flags.ui32.CoarseGrain = 0;
            break;
        case kfd_memory_type::host_coherent:
            flags.ui32.HostAccess  = 1;
            flags.ui32.CoarseGrain = 0;
            break;
        case kfd_memory_type::host_executable:
            flags.ui32.HostAccess    = 1;
            flags.ui32.ExecuteAccess = 1;
            flags.ui32.CoarseGrain   = 0;
            break;
        case kfd_memory_type::device_coarse:
            flags.ui32.NonPaged     = 1;
            flags.ui32.NoSubstitute = 1;
            flags.ui32.CoarseGrain  = 1;
            break;
    }

    return flags;
}
}  // namespace

void
kfd_memory_region::release()
{
    if(!ptr) return;

    if(mapped)
    {
        auto status = hsaKmtUnmapMemoryToGPU(ptr);
        if(status != HSAKMT_STATUS_SUCCESS)
        {
            ROCP_ERROR << "Failed to unmap KFD memory " << ptr << ": status=" << status;
        }
        mapped = false;
    }

    auto status = hsaKmtFreeMemory(ptr, size);
    if(status != HSAKMT_STATUS_SUCCESS)
    {
        ROCP_ERROR << "Failed to free KFD memory " << ptr << " (size=" << size
                   << "): status=" << status;
    }

    ptr  = nullptr;
    size = 0;
}

kfd_memory_region::~kfd_memory_region() { release(); }

kfd_memory_region::kfd_memory_region(kfd_memory_region&& other) noexcept
: ptr{std::exchange(other.ptr, nullptr)}
, size{std::exchange(other.size, 0)}
, node_id{std::exchange(other.node_id, 0)}
, mapped{std::exchange(other.mapped, false)}
{}

kfd_memory_region&
kfd_memory_region::operator=(kfd_memory_region&& other) noexcept
{
    if(this != &other)
    {
        release();
        ptr     = std::exchange(other.ptr, nullptr);
        size    = std::exchange(other.size, 0);
        node_id = std::exchange(other.node_id, 0);
        mapped  = std::exchange(other.mapped, false);
    }
    return *this;
}

unique_kfd_memory_t
allocate(uint32_t gpu_node_id, uint64_t size, kfd_memory_type type)
{
    ensure_kfd_open();

    if(!kfd_initialized)
    {
        ROCP_ERROR << "KFD not initialized, cannot allocate memory";
        return nullptr;
    }

    uint64_t aligned_size = align_up(size, page_size);
    if(aligned_size == 0)
    {
        ROCP_ERROR << "Cannot allocate zero-size KFD memory";
        return nullptr;
    }

    auto  flags = make_flags(type);
    void* ptr   = nullptr;

    auto status = hsaKmtAllocMemory(gpu_node_id, aligned_size, flags, &ptr);
    if(status != HSAKMT_STATUS_SUCCESS || !ptr)
    {
        ROCP_ERROR << "hsaKmtAllocMemory failed: node=" << gpu_node_id << " size=" << aligned_size
                   << " type=" << static_cast<int>(type) << " status=" << status;
        return nullptr;
    }

    auto region     = std::make_unique<kfd_memory_region>();
    region->ptr     = ptr;
    region->size    = aligned_size;
    region->node_id = gpu_node_id;
    region->mapped  = false;

    ROCP_TRACE << "KFD allocated " << aligned_size << " bytes at " << ptr << " on node "
               << gpu_node_id;

    return region;
}

bool
map_to_gpu(kfd_memory_region& region, const std::vector<uint32_t>& gpu_node_ids)
{
    if(!region.ptr || gpu_node_ids.empty()) return false;

    HsaMemMapFlags map_flags;
    memset(&map_flags, 0, sizeof(map_flags));

    auto node_ids_copy = gpu_node_ids;

    auto status = hsaKmtMapMemoryToGPUNodes(
        region.ptr, region.size, nullptr, map_flags, node_ids_copy.size(), node_ids_copy.data());
    if(status != HSAKMT_STATUS_SUCCESS)
    {
        ROCP_ERROR << "hsaKmtMapMemoryToGPUNodes failed: ptr=" << region.ptr
                   << " size=" << region.size << " n_nodes=" << gpu_node_ids.size()
                   << " status=" << status;
        return false;
    }

    region.mapped = true;

    ROCP_TRACE << "KFD mapped " << region.ptr << " to " << gpu_node_ids.size() << " GPU node(s)";

    return true;
}

}  // namespace kfd
}  // namespace rocprofiler
