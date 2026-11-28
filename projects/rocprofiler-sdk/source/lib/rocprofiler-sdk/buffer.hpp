// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/container/record_header_buffer.hpp"
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/demangle.hpp"

#include <fmt/format.h>

#include <sys/types.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

namespace rocprofiler
{
namespace buffer
{
struct instance
{
    using buffer_t             = common::container::record_header_buffer;
    static constexpr auto size = 3;  // triple buffering

    mutable std::array<buffer_t, size>         buffers = {};
    mutable std::array<std::atomic_flag, size> syncer  = {
        {ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT}};
    mutable std::atomic<uint64_t>   buffer_idx    = {};  // generation, not just slot
    std::mutex                      flush_mutex   = {};
    mutable std::atomic<uint64_t>   drop_count    = {};
    uint64_t                        watermark     = 0;
    uint64_t                        context_id    = 0;  // rocprofiler_context_id_t value
    uint64_t                        buffer_id     = 0;  // rocprofiler_buffer_id_t value
    uint64_t                        task_group_id = 0;  // thread-pool assignment
    rocprofiler_buffer_tracing_cb_t callback      = nullptr;
    void*                           callback_data = nullptr;
    rocprofiler_buffer_policy_t     policy        = ROCPROFILER_BUFFER_POLICY_NONE;

    template <typename Tp>
    bool emplace(uint32_t, uint32_t, Tp&);

    buffer_t& get_internal_buffer();
    buffer_t& get_internal_buffer(size_t);
};

using unique_buffer_vec_t = common::container::stable_vector<std::unique_ptr<instance>, 16>;

bool
is_valid_buffer_id(rocprofiler_buffer_id_t id);

std::optional<rocprofiler_buffer_id_t>
allocate_buffer();

unique_buffer_vec_t*
get_buffers();

instance*
get_buffer(rocprofiler_buffer_id_t buffer_id);

instance*
get_buffer(uint64_t buffer_idx);

rocprofiler_status_t
flush(rocprofiler_buffer_id_t buffer_id, bool wait);

rocprofiler_status_t
flush(uint64_t buffer_idx, bool wait);
}  // namespace buffer
}  // namespace rocprofiler

inline rocprofiler::buffer::instance::buffer_t&
rocprofiler::buffer::instance::get_internal_buffer()
{
    auto idx = buffer_idx.load() % buffers.size();
    return buffers.at(idx);
}

inline rocprofiler::buffer::instance::buffer_t&
rocprofiler::buffer::instance::get_internal_buffer(size_t idx)
{
    return buffers.at(idx % buffers.size());
}

inline rocprofiler::buffer::instance*
rocprofiler::buffer::get_buffer(uint64_t buffer_idx)
{
    return get_buffer(rocprofiler_buffer_id_t{buffer_idx});
}

inline rocprofiler_status_t
rocprofiler::buffer::flush(uint64_t buffer_idx, bool wait)
{
    return flush(rocprofiler_buffer_id_t{buffer_idx}, wait);
}

template <typename Tp>
inline bool
rocprofiler::buffer::instance::emplace(uint32_t category, uint32_t kind, Tp& value)
{
    for(;;)
    {
        auto  generation = buffer_idx.load(std::memory_order_relaxed);
        auto  idx        = generation % buffers.size();
        auto& flag       = syncer.at(idx);
        if(flag.test_and_set(std::memory_order_acquire))
        {
            std::this_thread::yield();
            continue;
        }
        bool success      = false;
        bool should_flush = false;
        {
            auto release =
                common::scope_destructor{[&flag]() { flag.clear(std::memory_order_release); }};
            // A delayed producer must not write into a previously flushed generation.
            if(generation != buffer_idx.load(std::memory_order_relaxed)) continue;
            auto& target = buffers.at(idx);
            success      = target.emplace(category, kind, value);
            if(!success && target.capacity() < sizeof(value))
            {
                ROCP_CI_LOG(ERROR) << fmt::format(
                    "buffer {} too small (size={}) to hold an object of type {} with size {}",
                    buffer_id,
                    target.capacity(),
                    common::cxx_demangle(typeid(value).name()),
                    sizeof(value));
                return false;
            }
            should_flush = target.count() >= watermark;
        }
        if(!success && policy == ROCPROFILER_BUFFER_POLICY_LOSSLESS)
        {
            auto status = buffer::flush(buffer_id, false);
            if(status != ROCPROFILER_STATUS_SUCCESS &&
               status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            {
                ROCP_ERROR << "buffer " << buffer_id << " flush failed: " << status;
                return false;
            }
            std::this_thread::yield();
            continue;
        }
        if(!success) ++drop_count;
        if(should_flush) buffer::flush(buffer_id, false);
        return success;
    }
}
