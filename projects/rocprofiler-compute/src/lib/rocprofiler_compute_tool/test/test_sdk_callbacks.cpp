// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_sdk_callbacks.h"

#include "fmt/format.h"
#include "rocprofiler_compute_tool.h"

TEST_F(TestSdkCallbacks, ProvidedRequestedCountersAvaiable_AllConfiguredForCollection)
{
    const std::vector<std::string> counters_pmc0 = {"counter0", "counter1"};
    const std::vector<std::string> counters_pmc1 = {"counter2"};

    dispatch_kernel(1, counters_pmc0, counters_pmc1);

    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info().size(), 2);
    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info()[0].counter_names, counters_pmc0);
    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info()[1].counter_names, counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedKernelIds_ReturnsResultForThemOnly)
{
    const std::vector<std::string> counters_pmc0 = {"counter0", "counter1"};
    const std::vector<std::string> counters_pmc1 = {"counter2"};

    m_tool_data->target_kernel_ids.insert(2);
    dispatch_kernel(1, counters_pmc0);
    dispatch_kernel(2, counters_pmc1);

    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info().size(), 1);
    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info()[0].counter_names, counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedKernelDispatchRanges_ReturnsResultForThemOnly)
{
    const std::vector<std::string> counters_pmc0 = {"counter0", "counter1"};
    const std::vector<std::string> counters_pmc1 = {"counter2"};

    m_tool_data->kernel_filter_ranges.emplace_back(2, 2);
    dispatch_kernel(11, counters_pmc0);
    dispatch_kernel(11, counters_pmc1);

    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info().size(), 1);
    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info()[0].counter_names, counters_pmc1);
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

void TestSdkCallbacks::dispatch_kernel(uint64_t                        kernel_id,
                                       const std::vector<std::string>& counters_pmc0,
                                       const std::vector<std::string>& counters_pmc1)
{
    m_tool_data->requested_counters = convert_counters_per_pmc_to_str({counters_pmc0, counters_pmc1});
    m_sdk_wrapper->set_available_counters(concat_counters(counters_pmc0, counters_pmc1));

    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    dispatch_data.dispatch_info.kernel_id                      = kernel_id;
    dispatch_data.dispatch_info.agent_id.handle                = 0xff;
    rocprofiler_counter_config_id_t config;
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);
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