// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_sdk_callbacks.h"

#include "fmt/format.h"
#include "rocprofiler_compute_tool.h"

TEST_F(TestSdkCallbacks, ProvidedCountersAvaiableWithMultiplexingDisabled_ReturnsFirstPmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode = rocprofiler_compute_tool::iteration_multiplexing_mode_t::DISABLED;
    const auto config_index_0 = dispatch_kernel(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_LT(config_index_0, created_config_info.size());
    EXPECT_LT(config_index_1, created_config_info.size());
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc0);
}

TEST_F(TestSdkCallbacks, ProvidedCountersAvaiableWithKernelMultiplexing_ReturnsEachPmcForCollection)
{
    m_tool_data->iteration_multiplexing_mode = rocprofiler_compute_tool::iteration_multiplexing_mode_t::KERNEL;
    const auto config_index_0 = dispatch_kernel(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_LT(config_index_0, created_config_info.size());
    EXPECT_LT(config_index_1, created_config_info.size());
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedRequestedCountersAvaiable_AllConfiguredForCollection)
{
    dispatch_kernel(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info.size(), 2);
    EXPECT_EQ(created_config_info[0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedKernelIdsWithDisabledMultiplexing_ReturnsResultForThemOnly)
{
    m_tool_data->iteration_multiplexing_mode = rocprofiler_compute_tool::iteration_multiplexing_mode_t::DISABLED;
    m_tool_data->target_kernel_ids.insert(2);

    dispatch_kernel(1, m_counters_pmc0);
    dispatch_kernel(2, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info.size(), 1);
    EXPECT_EQ(created_config_info[0].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedKernelDispatchRangesWithDisabledMultiplexing_ReturnsResultForThemOnly)
{
    m_tool_data->iteration_multiplexing_mode = rocprofiler_compute_tool::iteration_multiplexing_mode_t::DISABLED;
    m_tool_data->kernel_filter_ranges.emplace_back(2, 2);

    const auto config_index_0 = dispatch_kernel(11, m_counters_pmc0);
    const auto config_index_1 = dispatch_kernel(11, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info.size(), 1);
    EXPECT_EQ(config_index_0, m_invalid_config_id);
    EXPECT_EQ(config_index_1, 0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedMultiplexingDisabled_ReturnsFirstPmcConfig)
{
    const auto selected_config = dispatch_kernel(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_GE(created_config_info.size(), selected_config);
    EXPECT_EQ(created_config_info[selected_config].counter_names, m_counters_pmc0);
}

//////////////////////////////////////////////////////////////////////////
/// TestSdkCallbacks
void TestSdkCallbacks::SetUp()
{
    m_sdk_wrapper = std::make_shared<MockSdkWrapper>();
    rocprofiler_compute_tool::test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    m_sdk_callbacks = std::make_shared<rocprofiler_compute_tool::SdkCallbacksImpl>(m_sdk_wrapper);
    m_tool_data     = std::make_unique<rocprofiler_compute_tool::tool_data_t>();
}

uint64_t TestSdkCallbacks::dispatch_kernel(uint64_t                        kernel_id,
                                           const std::vector<std::string>& counters_pmc0,
                                           const std::vector<std::string>& counters_pmc1)
{
    m_tool_data->requested_counters = convert_counters_per_pmc_to_str({counters_pmc0, counters_pmc1});
    m_sdk_wrapper->set_available_counters(concat_counters(counters_pmc0, counters_pmc1));

    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    dispatch_data.dispatch_info.kernel_id                      = kernel_id;
    dispatch_data.dispatch_info.agent_id.handle                = 0xff;
    rocprofiler_counter_config_id_t config{m_invalid_config_id};
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);
    return config.handle;
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