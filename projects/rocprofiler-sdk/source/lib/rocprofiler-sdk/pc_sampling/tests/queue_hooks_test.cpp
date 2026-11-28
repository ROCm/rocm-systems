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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
#    include "lib/rocprofiler-sdk/agent.hpp"
#    include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#    include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#    include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#    include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#    include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/cid_manager.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/types.hpp"
#endif

#include <gtest/gtest.h>

#include <memory>

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
using namespace rocprofiler::counters::test_constants;

namespace rocprofiler
{
namespace hsa
{
class PCSamplingQueueHooksFakeQueue : public Queue
{
public:
    PCSamplingQueueHooksFakeQueue(const AgentCache& agent, rocprofiler_queue_id_t id)
    : Queue(agent, get_api_table())
    , m_agent(agent)
    , m_id(id)
    {}

    const AgentCache&      get_agent() const final { return m_agent; }
    rocprofiler_queue_id_t get_id() const final { return m_id; }

private:
    const AgentCache&      m_agent;
    rocprofiler_queue_id_t m_id = {};
};
}  // namespace hsa
}  // namespace rocprofiler
#endif

namespace
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
void
test_init()
{
    auto table       = HsaApiTable{};
    table.amd_ext_   = &get_ext_table();
    table.core_      = &get_api_table();
    auto* controller = rocprofiler::hsa::get_queue_controller();

    rocprofiler::agent::construct_agent_cache(&table);
    ASSERT_NE(controller, nullptr);
    controller->init(get_api_table(), get_ext_table());
}
#endif

// is_configured_on_agent replaces the queue.get_notifiers() signal that the old
// per-queue callback registration used to provide to the HSA write interceptor
// gate. With no PC sampling service configured, any agent id must report as not
// configured. This is always linkable regardless of HSA PC sampling support and
// requires no GPU / HSA runtime.
TEST(pc_sampling_queue_hooks, is_configured_on_agent_unconfigured)
{
    rocprofiler_agent_id_t agent_id;
    agent_id.handle = 123456;
    EXPECT_FALSE(rocprofiler::pc_sampling::is_configured_on_agent(agent_id));
}

TEST(pc_sampling_queue_hooks, exit_hook_unmarked_packet_is_noop)
{
    rocprofiler::hsa::rocprofiler_packet                    kern_pkt{};
    std::shared_ptr<rocprofiler::hsa::queue_info_session_t> null_session;
    rocprofiler::hsa::packet_data_t                         packet{};
    rocprofiler::hsa::inst_pkt_t                            inst_pkt{};

    EXPECT_FALSE(packet.has_pc_sampling_ref);

    rocprofiler::pc_sampling::kernel_dispatch_phase_exit_hook(
        nullptr,
        kern_pkt,
        null_session,
        packet,
        inst_pkt,
        rocprofiler::kernel_dispatch::profiling_time{});

    SUCCEED();
}

TEST(pc_sampling_queue_hooks, exit_hook_marked_packet_with_null_session_is_noop)
{
    rocprofiler::hsa::rocprofiler_packet                    kern_pkt{};
    std::shared_ptr<rocprofiler::hsa::queue_info_session_t> null_session;
    rocprofiler::hsa::packet_data_t                         packet{};
    rocprofiler::hsa::inst_pkt_t                            inst_pkt{};

    packet.has_pc_sampling_ref = true;

    rocprofiler::pc_sampling::kernel_dispatch_phase_exit_hook(
        nullptr,
        kern_pkt,
        null_session,
        packet,
        inst_pkt,
        rocprofiler::kernel_dispatch::profiling_time{});

    SUCCEED();
}

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
TEST(pc_sampling_queue_hooks, dispatch_provenance_controls_reference_retirement)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    auto hsa_cleanup = rocprofiler::common::scope_destructor{[]() { (void) hsa_shut_down(); }};
    test_init();

    auto* controller = rocprofiler::hsa::get_queue_controller();
    ASSERT_NE(controller, nullptr);
    auto& agents = controller->get_supported_agents();
    ASSERT_FALSE(agents.empty());

    const auto& agent      = agents.begin()->second;
    const auto* rocp_agent = agent.get_rocp_agent();
    ASSERT_NE(rocp_agent, nullptr);

    auto pcs_session    = std::make_shared<rocprofiler::pc_sampling::PCSAgentSession>();
    pcs_session->agent  = rocp_agent;
    pcs_session->parser = std::make_unique<PCSamplingParserContext>();
    pcs_session->cid_manager =
        std::make_unique<rocprofiler::pc_sampling::PCSCIDManager>(pcs_session->parser.get());

    auto previous_session =
        rocprofiler::pc_sampling::get_global_pc_sampling_sessions().wlock([&](auto& sessions) {
            auto previous = std::shared_ptr<rocprofiler::pc_sampling::PCSAgentSession>{};
            if(auto itr = sessions.find(rocp_agent->id); itr != sessions.end())
                previous = itr->second;
            sessions[rocp_agent->id] = pcs_session;
            return previous;
        });
    auto session_cleanup =
        rocprofiler::common::scope_destructor{[agent_id = rocp_agent->id, previous_session]() {
            rocprofiler::pc_sampling::get_global_pc_sampling_sessions().wlock([&](auto& sessions) {
                if(previous_session)
                    sessions[agent_id] = previous_session;
                else
                    sessions.erase(agent_id);
            });
        }};

    auto queue =
        rocprofiler::hsa::PCSamplingQueueHooksFakeQueue(agent, rocprofiler_queue_id_t{.handle = 1});

    const auto run_case = [&](bool marked, uint64_t internal_id) {
        auto correlation_id =
            rocprofiler::context::correlation_id{1, rocprofiler_thread_id_t{1}, internal_id};
        auto packet = rocprofiler::hsa::packet_data_t{};
        if(marked)
        {
            correlation_id.add_ref_count();
            packet.has_pc_sampling_ref = true;
        }

        auto session = std::make_shared<rocprofiler::hsa::queue_info_session_t>(
            rocprofiler::hsa::queue_info_session_t{.queue          = queue,
                                                   .correlation_id = &correlation_id});
        auto kernel_packet = rocprofiler::hsa::rocprofiler_packet{};
        auto inst_packets  = rocprofiler::hsa::inst_pkt_t{};

        rocprofiler::pc_sampling::kernel_dispatch_phase_exit_hook(
            &queue,
            kernel_packet,
            session,
            packet,
            inst_packets,
            rocprofiler::kernel_dispatch::profiling_time{});

        const auto flush = []() {};
        pcs_session->cid_manager->manage_cids_explicit(flush);
        EXPECT_EQ(correlation_id.get_ref_count(), 1)
            << "completion must release exactly the dispatch's PC-sampling reference";

        pcs_session->cid_manager->manage_cids_explicit(flush);
        EXPECT_EQ(correlation_id.get_ref_count(), 1)
            << "a second flush must not release the same reference again";
    };

    run_case(false, 1001);
    run_case(true, 1002);
}

// The unconfigured case above can only show the gate stays shut. Seed the two pieces of
// state is_pc_sample_service_configured reads -- the HSA-init flag and the global session
// map -- and the gate must open. Configuring a real service would take a GPU, but the map
// lookup is a find(), so the session is never dereferenced and a null entry is enough.
TEST(pc_sampling_queue_hooks, is_configured_on_agent_configured)
{
    rocprofiler_agent_id_t agent_id;
    agent_id.handle = 424242;

    auto&      hsa_ready  = rocprofiler::pc_sampling::is_hsa_initialized();
    const bool prev_ready = hsa_ready.exchange(true);

    rocprofiler::pc_sampling::get_global_pc_sampling_sessions().wlock(
        [agent_id](auto& sessions) { sessions[agent_id] = nullptr; });

    EXPECT_TRUE(rocprofiler::pc_sampling::is_configured_on_agent(agent_id));

    // Drop the session but leave the HSA-init flag set, so this shows the session lookup is
    // what closed the gate again rather than the flag doing all the work.
    rocprofiler::pc_sampling::get_global_pc_sampling_sessions().wlock(
        [agent_id](auto& sessions) { sessions.erase(agent_id); });

    EXPECT_FALSE(rocprofiler::pc_sampling::is_configured_on_agent(agent_id));

    // pcs-test runs every PC sampling suite in one process.
    hsa_ready.store(prev_ready);
}
#endif
}  // namespace
