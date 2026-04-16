// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_sdk_callbacks.h"

#include "fmt/format.h"
#include "rocprofiler_compute_tool.h"

using namespace rocprofiler_compute_tool;

TEST_F(TestSdkCallbacks, ProvidedSameKernelWithMultiplexingDisabled_ReturnsFirstPmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::DISABLED;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc0);
}

TEST_F(TestSdkCallbacks, ProvidedDifferentKernelsWithMultiplexingDisabled_ReturnsFirstPmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::DISABLED;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(2, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc0);
}

TEST_F(TestSdkCallbacks, ProvidedSameKernelWithKernelMultiplexing_ReturnsEachPmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::KERNEL;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_LT(config_index_0, created_config_info.size());
    EXPECT_LT(config_index_1, created_config_info.size());
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedDifferentKernelsWithKernelMultiplexing_ReturnsFirstPmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::KERNEL;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(2, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_2 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_3 = dispatch_kernel_with_id(2, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_2].counter_names, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index_3].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedSameKernelSameParamsWithLaunchMultiplexing_ReturnsEachPmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::LAUNCH;
    constexpr kernel_dispatch_info_t info    = {1, 2, {3, 3, 3}, {4, 4, 4}, 5};

    const auto config_index_0 = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedSameKernelDifferentParamsWithLaunchMultiplexing_ReturnsSamePmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode   = iteration_multiplexing_mode_t::LAUNCH;
    kernel_dispatch_info_t info                = {1, 2, {3, 3, 3}, {4, 4, 4}, 5};
    const auto&            created_config_info = m_sdk_wrapper->get_create_counter_config_info();

    auto config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.kernel_id++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.queue_id++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.workgroup_size.x++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.grid_size.x++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.LDS_memory_size++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);
}

TEST_F(TestSdkCallbacks, ProvidedRequestedCountersAvaiable_AllConfiguredForCollection)
{
    dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info.size(), 2);
    EXPECT_EQ(created_config_info[0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[1].counter_names, m_counters_pmc1);
}

TEST_P(TestSdkCallbacksMultiplexing, DISABLED_ProvidedCountersNotAvailable_ReturnsNoConfig)
{
    m_tool_data->requested_counters = convert_counters_per_pmc_to_str({m_counters_pmc0, m_counters_pmc1});

    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    dispatch_data.dispatch_info.kernel_id                      = 1;
    dispatch_data.dispatch_info.agent_id.handle                = 0xff;
    rocprofiler_counter_config_id_t config{m_invalid_config_id};
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);

    EXPECT_EQ(config.handle, m_invalid_config_id);
    
}

TEST_P(TestSdkCallbacksMultiplexing, ProvidedKernelIdsOfInterest_ReturnsResultForThemOnly)
{
    m_tool_data->target_kernel_ids.insert(2);

    m_tool_data->iteration_multiplexing_mode = m_multiplexing_mode;
    const auto config_index_0                = dispatch_kernel_with_id(1, m_counters_pmc0);
    const auto config_index_1                = dispatch_kernel_with_id(2, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(config_index_0, m_invalid_config_id);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_P(TestSdkCallbacksMultiplexing, ProvidedKernelDispatchRanges_ReturnsResultForThemOnly)
{
    m_tool_data->kernel_filter_ranges.emplace_back(2, 2);

    m_tool_data->iteration_multiplexing_mode = m_multiplexing_mode;
    const auto config_index_0                = dispatch_kernel_with_id(1, m_counters_pmc0);
    const auto config_index_1                = dispatch_kernel_with_id(1, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(config_index_0, m_invalid_config_id);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

//////////////////////////////////////////////////////////////////////////
/// TestSdkCallbacks
void TestSdkCallbacks::SetUp()
{
    m_sdk_wrapper = std::make_shared<MockSdkWrapper>();
    test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    m_sdk_callbacks = std::make_shared<SdkCallbacksImpl>(m_sdk_wrapper);
    m_tool_data     = std::make_unique<tool_data_t>();
}

uint64_t TestSdkCallbacks::dispatch_kernel_with_dispatch_info(const kernel_dispatch_info_t& dispatch_info,
                                                              const std::vector<std::string>& counters_pmc0,
                                                              const std::vector<std::string>& counters_pmc1)
{
    m_tool_data->requested_counters = convert_counters_per_pmc_to_str({counters_pmc0, counters_pmc1});
    m_sdk_wrapper->set_available_counters(concat_counters(counters_pmc0, counters_pmc1));

    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    dispatch_data.dispatch_info.kernel_id                      = dispatch_info.kernel_id;
    dispatch_data.dispatch_info.queue_id.handle                = dispatch_info.queue_id;
    dispatch_data.dispatch_info.workgroup_size                 = dispatch_info.workgroup_size;
    dispatch_data.dispatch_info.grid_size                      = dispatch_info.grid_size;
    dispatch_data.dispatch_info.group_segment_size             = dispatch_info.LDS_memory_size;
    dispatch_data.dispatch_info.agent_id.handle                = 0xff;
    rocprofiler_counter_config_id_t config{m_invalid_config_id};
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);
    return config.handle;
}

uint64_t TestSdkCallbacks::dispatch_kernel_with_id(uint64_t                        kernel_id,
                                                   const std::vector<std::string>& counters_pmc0,
                                                   const std::vector<std::string>& counters_pmc1)
{
    kernel_dispatch_info_t dispatch_info = {};
    dispatch_info.kernel_id              = kernel_id;
    return dispatch_kernel_with_dispatch_info(dispatch_info, counters_pmc0, counters_pmc1);
}

std::string TestSdkCallbacks::convert_counters_per_pmc_to_str(
    const std::vector<std::vector<std::string>>& counters_per_pmc)
{
    std::string result;
    for (const auto& counters : counters_per_pmc)
    {
        result += convert_counters_to_str(counters);
        result += ",";
    }
    return remove_trailing_comma(result);
}

std::string TestSdkCallbacks::convert_counters_to_str(const std::vector<std::string>& counters)
{
    if (counters.empty())
        return "";

    std::string result = "pmc: ";
    for (const auto& counter : counters)
    {
        result += fmt::format("{} ", counter);
    }
    return result;
}

std::string TestSdkCallbacks::remove_trailing_comma(const std::string& str)
{
    std::string result = str;
    if (!result.empty() && result.back() == ',')
    {
        result.pop_back();
    }
    return result;
}

std::vector<std::string> TestSdkCallbacks::concat_counters(const std::vector<std::string>& v0,
                                                           const std::vector<std::string>& v1)
{
    auto result = v0;
    result.insert(result.end(), v1.begin(), v1.end());
    return result;
}

//////////////////////////////////////////////////////////////////////////
/// TestSdkCallbacksMultiplexing
void TestSdkCallbacksMultiplexing::SetUp()
{
    TestSdkCallbacks::SetUp();
    m_multiplexing_mode = GetParam();
}