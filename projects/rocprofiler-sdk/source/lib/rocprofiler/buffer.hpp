// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <bits/stdint-uintn.h>
#include <rocprofiler/buffer.h>
#include <rocprofiler/fwd.h>

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

    mutable std::array<buffer_t, 2>     buffers       = {};
    mutable std::atomic<unsigned short> buffer_idx    = {};
    mutable std::atomic_flag            syncer        = ATOMIC_FLAG_INIT;
    mutable std::atomic<uint64_t>       drop_count    = {};
    uint64_t                            watermark     = 0;
    uint64_t                            context_id    = 0;
    uint64_t                            buffer_id     = 0;
    uint64_t                            task_group_id = 0;
    rocprofiler_buffer_tracing_cb_t     callback      = nullptr;
    void*                               callback_data = nullptr;
    rocprofiler_buffer_policy_t         policy        = ROCPROFILER_BUFFER_POLICY_NONE;

    template <typename Tp>
    void emplace(uint32_t, uint32_t, Tp&);
};

using unique_buffer_vec_t = common::container::stable_vector<std::unique_ptr<instance>, 4>;

std::optional<rocprofiler_buffer_id_t>
allocate_buffer();

unique_buffer_vec_t&
get_buffers();

rocprofiler_status_t
flush(rocprofiler_buffer_id_t buffer_id, bool wait);

inline rocprofiler_status_t
flush(uint64_t buffer_idx, bool wait)
{
    return flush(rocprofiler_buffer_id_t{buffer_idx}, wait);
}
}  // namespace buffer
}  // namespace rocprofiler

template <typename Tp>
inline void
rocprofiler::buffer::instance::emplace(uint32_t category, uint32_t kind, Tp& value)
{
    // get the index of the current buffer
    auto get_idx = [this]() { return buffer_idx.load(std::memory_order_acquire) % buffers.size(); };

    auto idx = get_idx();
    if(!buffers.at(idx).emplace(category, kind, value))
    {
        if(buffers.at(idx).size() < sizeof(value))
        {
            auto msg = std::stringstream{};
            msg << "buffer " << buffer_id << " to small (size=" << buffers.at(idx).size()
                << ") to hold an object of type " << common::cxx_demangle(typeid(value).name())
                << " with size " << sizeof(value);
            throw std::runtime_error(msg.str());
        }

        if(policy == ROCPROFILER_BUFFER_POLICY_LOSSLESS)
        {
            // blocks until buffer is flushed
            bool success = false;
            while(!success)
            {
                buffer::flush(buffer_id, true);
                idx     = get_idx();
                success = buffers.at(idx).emplace(category, kind, value);
            }
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
}
