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

#include <rocprofiler-sdk/agent.h>

#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocprofiler
{
namespace thread_trace
{
enum class kfd_memory_kind_t
{
    host,
    device,
};

/// Direct /dev/kfd memory owner for one GPU. It deliberately does not use
/// libhsakmt: all BO allocation, mapping, and teardown is performed with KFD
/// ioctls, and the KFD/DRM descriptors are retained past ROCr shutdown.
class kfd_memory_pool_t
{
public:
    explicit kfd_memory_pool_t(const rocprofiler_agent_t& agent);
    ~kfd_memory_pool_t();

    kfd_memory_pool_t(const kfd_memory_pool_t&) = delete;
    kfd_memory_pool_t& operator=(const kfd_memory_pool_t&) = delete;
    kfd_memory_pool_t(kfd_memory_pool_t&&)                 = delete;
    kfd_memory_pool_t& operator=(kfd_memory_pool_t&&) = delete;

    void* allocate(size_t size, kfd_memory_kind_t kind, size_t alignment = 4096);
    void  deallocate(void* ptr);
    bool  is_device_pointer(const void* ptr) const;

    uint32_t gpu_id() const;
    uint32_t gfx_target_version() const;
    uint32_t cwsr_size() const;
    uint32_t ctl_stack_size() const;
    uint32_t num_xcc() const;
    uint32_t debug_memory_size() const;
    uint32_t max_cu_id() const;
    uint32_t max_wave_id() const;
    int      kfd_fd() const;

private:
    struct impl;
    std::unique_ptr<impl> _impl;
};

/// Completion signal stored in KFD-allocated, GPU-mapped memory. The HSA signal
/// type is used only as the frozen packet ABI; no HSA signal API is used.
class kfd_signal_t
{
public:
    explicit kfd_signal_t(std::shared_ptr<kfd_memory_pool_t> memory);
    ~kfd_signal_t();

    kfd_signal_t(const kfd_signal_t&) = delete;
    kfd_signal_t& operator=(const kfd_signal_t&) = delete;
    kfd_signal_t(kfd_signal_t&&)                 = delete;
    kfd_signal_t& operator=(kfd_signal_t&&) = delete;

    hsa_signal_t  handle() const;
    amd_signal_t* abi() const { return _signal; }

    void reset();
    void wait() const;

private:
    std::shared_ptr<kfd_memory_pool_t> _memory{};
    amd_signal_t*                      _signal{nullptr};
};

/// Direct KFD AQL compute queue for aqlprofile packets and CP DMA fallback copies.
class kfd_aql_queue_t
{
public:
    explicit kfd_aql_queue_t(std::shared_ptr<kfd_memory_pool_t> memory, size_t max_copy_size = 0);
    ~kfd_aql_queue_t();

    kfd_aql_queue_t(const kfd_aql_queue_t&) = delete;
    kfd_aql_queue_t& operator=(const kfd_aql_queue_t&) = delete;

    void submit(const hsa_ext_amd_aql_pm4_packet_t& packet, hsa_signal_t completion);
    void copy(void* dst, const void* src, size_t size, kfd_signal_t& completion);

private:
    struct impl;
    std::unique_ptr<impl> _impl;
};

/// KFD copy path for thread trace. SDMA is preferred and CP DMA on the existing
/// AQL queue is the fallback.
class kfd_copy_queue_t
{
public:
    kfd_copy_queue_t(const std::shared_ptr<kfd_memory_pool_t>& memory, size_t max_copy_size);
    ~kfd_copy_queue_t();

    kfd_copy_queue_t(const kfd_copy_queue_t&) = delete;
    kfd_copy_queue_t& operator=(const kfd_copy_queue_t&) = delete;

    void submit(const hsa_ext_amd_aql_pm4_packet_t& packet, hsa_signal_t completion);
    void copy(void* dst, const void* src, size_t size);

private:
    struct impl;
    std::unique_ptr<impl> _impl;
};

}  // namespace thread_trace
}  // namespace rocprofiler
