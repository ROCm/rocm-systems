#include "lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp"

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/tests/fake_queue.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <gtest/gtest.h>

namespace
{
void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &rocprofiler::counters::test_constants::get_ext_table();
    table.core_    = &rocprofiler::counters::test_constants::get_api_table();
    rocprofiler::agent::construct_agent_cache(&table);
    ASSERT_TRUE(rocprofiler::hsa::get_queue_controller() != nullptr);
    rocprofiler::hsa::get_queue_controller()->init(
        rocprofiler::counters::test_constants::get_api_table(),
        rocprofiler::counters::test_constants::get_ext_table());
}

TEST(PCSamplingQueueHooks, IsConfiguredOnAgentReturnsFalseWithoutConfiguredAgent)
{
    rocprofiler_agent_id_t fake_agent{.handle = 0};
    EXPECT_FALSE(rocprofiler::pc_sampling::is_configured_on_agent(fake_agent));
}

// Link-sanity: both queue_hooks symbols must resolve in both
// ROCPROFILER_SDK_HSA_PC_SAMPLING==0 and ==1 builds.
TEST(PCSamplingQueueHooks, SymbolsExist)
{
    rocprofiler_agent_id_t fake_agent{.handle = 0};
    EXPECT_FALSE(rocprofiler::pc_sampling::is_configured_on_agent(fake_agent));

    auto* marker_fn_ptr = &rocprofiler::pc_sampling::maybe_marker_packet;
    EXPECT_NE(marker_fn_ptr, nullptr);
}

TEST(PCSamplingQueueHooks, MaybeMarkerReturnsNulloptWithoutConfiguredAgent)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    ASSERT_TRUE(rocprofiler::hsa::get_queue_controller() != nullptr);
    auto agents = rocprofiler::hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    const auto&            agent = agents.begin()->second;
    rocprofiler::hsa::FakeQueue fq(agent, rocprofiler_queue_id_t{.handle = 0});

    rocprofiler::hsa::queue_info_session_t::external_corr_id_map_t extern_ids{};
    auto result = rocprofiler::pc_sampling::maybe_marker_packet(
        fq, /*dispatch_id*/ 0, extern_ids, /*correlation_id*/ nullptr);

    EXPECT_FALSE(result.has_value());
}
}  // namespace
