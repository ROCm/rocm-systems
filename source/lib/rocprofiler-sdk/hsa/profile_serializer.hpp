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

#include "lib/common/container/small_vector.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa_barrier.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <fmt/format.h>

#include <deque>
#include <string>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
/*This is a profiler serializer. It should be instantiated
only once for the profiler. The following is the
description of each field.
1. _dispatch_queue - The queue to which the currently dispatched kernel
        belongs to.
        At any given time, in serialization only one kernel
        can be executing.
2. _dispatch_ready- It is a software data structure which holds
        the queues which have a kernel ready to be dispatched.
        This stores the queues in FIFO order.
3. serializer_mutex - The mutex is used for thread synchronization
        while accessing the singleton instance of this structure.
Currently, in case of profiling kernels are serialized by default.
*/
class profiler_serializer
{
public:
    enum class Status
    {
        ENABLED,
        DISABLED,
    };

    struct barrier_with_state
    {
        barrier_with_state(Status _state, std::unique_ptr<hsa_barrier> _barrier)
        : state(_state)
        , barrier(std::move(_barrier))
        {}
        Status                       state;
        std::unique_ptr<hsa_barrier> barrier;
    };

    void kernel_completion_signal(const Queue&);
    // Signal a kernel dispatch is taking place, generates packets needed to be
    // inserted to support kernel dispatch
    common::container::small_vector<hsa::rocprofiler_packet, 3> kernel_dispatch(const Queue&) const;

    void queue_ready(hsa_queue_t* hsa_queue, const Queue& queue);
    // Enable the serializer
    void enable(const hsa_barrier::queue_map_ptr_t& queues);
    // Disable the serializer
    void disable(const hsa_barrier::queue_map_ptr_t& queues);

    void destroy_queue(hsa_queue_t* id, const Queue& queue);

    static void add_queue(hsa_queue_t** hsa_queues, const Queue& queue);

    // Returns a string containing all class variable data
    std::string to_string() const;

private:
    const Queue*                   _dispatch_queue{nullptr};
    std::deque<const Queue*>       _dispatch_ready;
    std::atomic<Status>            _serializer_status{Status::DISABLED};
    std::deque<barrier_with_state> _barrier;
    mutable std::atomic<int64_t> _enqueued_packets{0};
    mutable std::atomic<int64_t> _completed_packets{0};
};

}  // namespace hsa
}  // namespace rocprofiler

namespace fmt
{
// fmt::format support for profiler_serializer
template <>
struct formatter<rocprofiler::hsa::profiler_serializer>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::hsa::profiler_serializer const& serializer, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", serializer.to_string());
    }
};
}  // namespace fmt
