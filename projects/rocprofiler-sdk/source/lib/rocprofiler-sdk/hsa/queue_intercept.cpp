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

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"
#include "lib/common/static_object.hpp"

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
queue_registry_t&
get_queue_registry()
{
    static auto*& _v = common::static_object<queue_registry_t>::construct();
    return *_v;
}

doorbell_map_t&
get_doorbell_map()
{
    static auto*& _v = common::static_object<doorbell_map_t>::construct();
    return *_v;
}

QueueState*
lookup_queue_state(const hsa_queue_t* queue)
{
    QueueState* result = nullptr;
    get_queue_registry().rlock([&](const auto& registry) {
        auto it = registry.find(queue);
        if(it != registry.end())
        {
            result = it->second.get();
        }
    });
    return result;
}

QueueState*
lookup_queue_state_by_doorbell(hsa_signal_t signal)
{
    QueueState* result = nullptr;
    get_doorbell_map().rlock([&](const auto& doorbell_map) {
        auto it = doorbell_map.find(signal.handle);
        if(it != doorbell_map.end())
        {
            result = it->second;
        }
    });
    return result;
}

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell)
{
    QueueState* state = lookup_queue_state(queue);
    if(state)
    {
        get_doorbell_map().wlock(
            [&](auto& doorbell_map) { doorbell_map[doorbell.handle] = state; });
    }
}

void
unregister_doorbell(hsa_signal_t doorbell)
{
    get_doorbell_map().wlock([&](auto& doorbell_map) { doorbell_map.erase(doorbell.handle); });
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
