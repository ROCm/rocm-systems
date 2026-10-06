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

#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler;

namespace
{
class FakeQueue : public hsa::Queue
{
public:
    FakeQueue(const hsa::AgentCache& a, rocprofiler_queue_id_t id)
    : hsa::Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    const hsa::AgentCache& get_agent() const final { return _agent; }
    rocprofiler_queue_id_t get_id() const final { return _id; }

    ~FakeQueue() override = default;

private:
    const hsa::AgentCache& _agent;
    rocprofiler_queue_id_t _id = {};
};

}  // namespace

namespace rocprofiler
{
void
test_init();  // att_packet_test.cpp
}

TEST(thread_trace, resource_deinit_does_not_release_foreign_serialization_owner)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto* controller = hsa::get_queue_controller();
    ASSERT_NE(controller, nullptr);
    ASSERT_FALSE(controller->get_supported_agents().empty());

    const auto& agent = controller->get_supported_agents().begin()->second;
    ASSERT_NE(agent.get_rocp_agent(), nullptr);

    FakeQueue queue{agent, {.handle = 9}};
    controller->serializer(&queue);

    thread_trace::DispatchThreadTracer tracer{};

    controller->enable_serialization();
    tracer.start_context();
    tracer.stop_context();
    tracer.resource_deinit();

    EXPECT_TRUE(controller->is_serialization_enabled(agent.get_rocp_agent()->id))
        << "resource_deinit must not release another owner's serialization hold";

    controller->disable_serialization();
    EXPECT_FALSE(controller->is_serialization_enabled(agent.get_rocp_agent()->id));
}
