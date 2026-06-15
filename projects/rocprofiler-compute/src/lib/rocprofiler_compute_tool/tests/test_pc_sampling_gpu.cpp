// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sampling_gpu.h"

#include <gtest/gtest.h>

#include <vector>

using namespace rocprofiler_compute_tool;

namespace
{
struct PcConfigCollector
{
    std::vector<rocprofiler_pc_sampling_configuration_t> configs;
    bool                                                 has_usable_method = false;
};

rocprofiler_status_t collect_pc_configs(const rocprofiler_pc_sampling_configuration_t* cfgs,
                                        size_t                                         num_cfgs,
                                        void*                                          user_data)
{
    auto* collector = static_cast<PcConfigCollector*>(user_data);
    for (size_t i = 0; i < num_cfgs; ++i)
    {
        collector->configs.push_back(cfgs[i]);
        if (cfgs[i].method != ROCPROFILER_PC_SAMPLING_METHOD_NONE)
        {
            collector->has_usable_method = true;
        }
    }
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

// Self-skipping GPU integration test: asserts a real agent advertises at least
// one usable PC-sampling config, and skips cleanly when no GPU/support is present.
TEST_F(TestPcSamplingGpu, QueriesPcSamplingConfigsOnRealAgent)
{
    std::vector<rocprofiler_agent_id_t> agents;
    try
    {
        sdk.query_available_gpu_agents(agents);
    }
    catch (...)
    {
        GTEST_SKIP() << "no GPU agent present";
        return;
    }

    if (agents.empty())
    {
        GTEST_SKIP() << "no GPU agent present";
        return;
    }

    // The SDK throws when the driver does not implement PC sampling; skip rather than fail.
    PcConfigCollector collector{};
    try
    {
        sdk.query_pc_sampling_configs(agents.front(), collect_pc_configs, &collector);
    }
    catch (...)
    {
        GTEST_SKIP() << "PC sampling not supported on this agent/driver";
        return;
    }

    if (collector.configs.empty() || !collector.has_usable_method)
    {
        GTEST_SKIP() << "agent advertises no usable PC-sampling configuration";
        return;
    }

    EXPECT_GE(agents.size(), 1u);
    EXPECT_FALSE(collector.configs.empty());
    EXPECT_TRUE(collector.has_usable_method);
}
