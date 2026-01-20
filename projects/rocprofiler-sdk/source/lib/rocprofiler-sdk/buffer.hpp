// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace rocprofiler
{
namespace buffer
{
struct instance
{
    using buffer_t = common::container::record_header_buffer;

    // Buffer pool design:
    // - buffers: pool of empty buffers ready for writing, front() is active
    // - flushing: FIFO queue of buffers awaiting/undergoing flush
    // When active buffer fills, it moves to flushing and flush task is launched.
    // Flush task processes buffers in order and returns them to the pool.
    mutable std::deque<buffer_t>    buffers       = {};       // empty buffers, front() is active
    mutable std::deque<buffer_t>    flushing      = {};       // buffers being flushed (FIFO)
    mutable std::mutex              mutex         = {};       // protects both deques
    mutable std::condition_variable buffer_avail  = {};       // signaled when buffer returned
    mutable std::atomic<bool>       flush_running = {false};  // flush task active
    mutable std::atomic<uint64_t>   drop_count    = {};
    size_t                          buffer_size   = 0;  // size of each buffer
    size_t                          pool_size     = 2;  // number of buffers in pool
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
};

using unique_buffer_vec_t = common::container::stable_vector<std::unique_ptr<instance>, 4>;

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

// Internal: launch flush task if not already running
void
launch_flush_task(rocprofiler_buffer_id_t buffer_id);
}  // namespace buffer
}  // namespace rocprofiler

inline rocprofiler::buffer::instance::buffer_t&
rocprofiler::buffer::instance::get_internal_buffer()
{
    // Return the active buffer (front of the pool)
    // Caller must hold mutex
    return buffers.front();
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
    auto lock = std::unique_lock<std::mutex>{mutex};

    // Helper to launch flush task with unlock/relock pattern
    auto trigger_flush = [this, &lock](bool relock) {
        lock.unlock();
        buffer::launch_flush_task(rocprofiler_buffer_id_t{buffer_id});
        if(relock) lock.lock();
    };

    // Wait for a buffer to be available if pool is empty
    while(buffers.empty())
    {
        if(policy != ROCPROFILER_BUFFER_POLICY_LOSSLESS)
        {
            // LOSSY policy: drop the record
            ++drop_count;
            return false;
        }
        // LOSSLESS policy: wait for a buffer to be returned from flush
        buffer_avail.wait(lock);
    }

    // Try to write to the active buffer (front of pool)
    bool success = buffers.front().emplace(category, kind, value);

    if(success)
    {
        // Check watermark - if reached, move buffer to flush queue
        if(buffers.front().count() >= watermark)
        {
            flushing.push_back(std::move(buffers.front()));
            buffers.pop_front();
            trigger_flush(false);
        }
        return true;
    }

    // Emplace failed - check why
    if(buffer_size < sizeof(value))
    {
        // Buffer too small for this record type
        ROCP_CI_LOG(ERROR) << fmt::format(
            "buffer {} too small (size={}) to hold an object of type {} with size {}",
            buffer_id,
            buffer_size,
            common::cxx_demangle(typeid(value).name()),
            sizeof(value));
        return false;
    }

    // Buffer is full - move to flush queue and try next buffer
    flushing.push_back(std::move(buffers.front()));
    buffers.pop_front();

    // Retry with next available buffer (may need to wait for flush to complete)
    while(buffers.empty())
    {
        if(policy != ROCPROFILER_BUFFER_POLICY_LOSSLESS)
        {
            // LOSSY policy: launch flush and drop the record
            trigger_flush(false);
            ++drop_count;
            return false;
        }
        // LOSSLESS policy: launch flush, then wait for buffer to be returned
        trigger_flush(true);
        if(buffers.empty())
        {
            buffer_avail.wait(lock);
        }
    }

    // Retry emplace with the new active buffer
    success = buffers.front().emplace(category, kind, value);

    if(success && buffers.front().count() >= watermark)
    {
        flushing.push_back(std::move(buffers.front()));
        buffers.pop_front();
        trigger_flush(false);
    }

    return success;
}
