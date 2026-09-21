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

// Windows stand-ins for the HSA sources excluded from this build. Queue interposition
// installs a write interceptor on the HSA soft queue so AQL packets can be rewritten;
// the ETW tracing path observes the runtime from outside the process instead, so no
// queue is ever intercepted and the API tables are never captured.

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/profile_serializer.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"

#include <optional>
#include <string_view>

namespace rocprofiler
{
namespace hsa
{
// stands in for hsa.cpp
hsa_core_table_t*
get_core_table()
{
    return nullptr;
}

hsa_amd_ext_table_t*
get_amd_ext_table()
{
    return nullptr;
}

std::string_view get_hsa_status_string(hsa_status_t)
{
    return "HSA is not available on this platform";
}

// stands in for queue.cpp
queue_state
Queue::get_state() const
{
    return queue_state::normal;
}

void
Queue::destroy_signal(pooled_signal_t*)
{}

// stands in for queue_controller.cpp
QueueController*
get_queue_controller()
{
    return nullptr;
}

ClientID QueueController::add_callback(std::optional<rocprofiler_agent_t>, queue_callbacks_t)
{
    return -1;
}

void QueueController::remove_callback(ClientID) {}

const Queue*
QueueController::get_queue(const hsa_queue_t&) const
{
    return nullptr;
}

void
QueueController::set_queue_state(queue_state, hsa_queue_t*)
{}

common::Synchronized<profiler_serializer>&
QueueController::serializer(const Queue*)
{
    static auto _v = common::Synchronized<profiler_serializer>{};
    return _v;
}

void
QueueController::enable_serialization()
{}

void
QueueController::disable_serialization()
{}

// stands in for queue_interposition.cpp
namespace queue_interposition
{
void
notify_queue_interposition_consumer_context_started(const context::context*)
{}

void
notify_queue_interposition_consumer_context_stopped(const context::context*)
{}
}  // namespace queue_interposition
}  // namespace hsa
}  // namespace rocprofiler
