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

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace kfd
{
/// Memory types matching SDK allocation patterns.
enum class kfd_memory_type
{
    host_uncached = 0,  /// Counter data buffers (GTT, coherent, uncached)
    host_coherent,      /// Host-accessible trace output (GTT, coherent)
    host_executable,    /// Command buffers / PM4 (GTT, coherent, executable)
    device_coarse,      /// ATT trace buffers (VRAM, coarse-grained)
};

/// RAII wrapper for KFD-allocated memory. Automatically unmaps and frees on destruction.
struct kfd_memory_region
{
    void*    ptr     = nullptr;
    uint64_t size    = 0;
    uint32_t node_id = 0;
    bool     mapped  = false;

    kfd_memory_region() = default;
    ~kfd_memory_region();

    kfd_memory_region(kfd_memory_region&& other) noexcept;
    kfd_memory_region& operator=(kfd_memory_region&& other) noexcept;

    kfd_memory_region(const kfd_memory_region&) = delete;
    kfd_memory_region& operator=(const kfd_memory_region&) = delete;

private:
    void release();
};

using unique_kfd_memory_t = std::unique_ptr<kfd_memory_region>;

/// Allocate GPU-accessible memory via KFD (libhsakmt).
/// @param gpu_node_id  KFD node ID for the target GPU (0 for system/CPU node).
/// @param size         Allocation size in bytes (will be page-aligned internally).
/// @param type         Memory type controlling flags (coherency, caching, etc.).
/// @return unique_ptr to kfd_memory_region, or nullptr on failure.
unique_kfd_memory_t
allocate(uint32_t gpu_node_id, uint64_t size, kfd_memory_type type);

/// Map an existing allocation to additional GPU node(s).
/// @param region       The memory region to map.
/// @param gpu_node_ids KFD node IDs to map the memory to.
/// @return true on success.
bool
map_to_gpu(kfd_memory_region& region, const std::vector<uint32_t>& gpu_node_ids);

}  // namespace kfd
}  // namespace rocprofiler
