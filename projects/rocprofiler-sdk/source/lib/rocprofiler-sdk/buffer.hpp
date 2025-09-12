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

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>

namespace rocprofiler
{
namespace buffer
{
struct instance
{
    using buffer_t = common::container::record_header_buffer;

    mutable std::array<buffer_t, 2> buffers       = {};
    mutable std::atomic_flag        syncer        = ATOMIC_FLAG_INIT;  // writer and reader lock.
    mutable std::atomic<uint32_t>   buffer_idx    = {};                // array index
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
    // get the index of the current buffer
    auto get_idx = [this]() { return buffer_idx.load(std::memory_order_acquire) % buffers.size(); };

    while(syncer.test_and_set())
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::microseconds{10});
    }
    auto idx     = get_idx();
    auto success = buffers.at(idx).emplace(category, kind, value);
    syncer.clear();
    if(!success)
    {
        if(buffers.at(idx).capacity() < sizeof(value))
        {
            ROCP_CI_LOG(ERROR) << "buffer " << buffer_id
                               << " too small (size=" << buffers.at(idx).capacity()
                               << ") to hold an object of type "
                               << common::cxx_demangle(typeid(value).name()) << " with size "
                               << sizeof(value);
            return false;
        }

        if(policy == ROCPROFILER_BUFFER_POLICY_LOSSLESS)
        {
            // blocks until buffer is flushed
            do
            {
                buffer::flush(buffer_id, true);
                while(syncer.test_and_set())
                {
                    std::this_thread::yield();
                    std::this_thread::sleep_for(std::chrono::microseconds{10});
                }
                idx     = get_idx();
                success = buffers.at(idx).emplace(category, kind, value);
                syncer.clear();
            } while(!success);
        }
        else
        {
            ++drop_count;
        }
    }

    if(buffers.at(idx).count() >= watermark)
    {
        // flush without syncing
        buffer::flush(buffer_id, false);
    }

    return success;
}
