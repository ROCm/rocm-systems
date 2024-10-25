// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace rocprofiler
{
namespace hsa
{
class Queue;

class hsa_barrier
{
public:
    using queue_map_ptr_t = std::unordered_map<hsa_queue_t*, Queue*>;
    hsa_barrier(std::function<void()>&& finished, CoreApiTable core_api);
    ~hsa_barrier();

    void set_barrier(const queue_map_ptr_t& q);

    std::optional<rocprofiler_packet> enqueue_packet(const Queue* queue);
    bool                              register_completion(const Queue* queue);

    bool complete() const { return _core_api.hsa_signal_load_scacquire_fn(_barrier_signal) == 0; }

    // Removes a queue from the barrier dependency list
    void remove_queue(const Queue* queue);

private:
    std::function<void()>                                      _barried_finished = {};
    CoreApiTable                                               _core_api         = {};
    common::Synchronized<std::unordered_map<int64_t, int64_t>> _queue_waiting    = {};
    common::Synchronized<std::unordered_set<int64_t>>          _barrier_enqueued = {};
    std::atomic<bool>                                          _complete         = {false};

    // Blocks all queues from executing until the barrier is lifted
    hsa_signal_t _barrier_signal = {};
};

}  // namespace hsa
}  // namespace rocprofiler
