// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_sdk_callbacks.h"

#include "rocprofiler_compute_tool.h"

TEST_F(TestSdkCallbacks, ProvidedRequestedCountersAvaiable_AllConfiguredForCollection)
{
    m_tool_data->requested_counters = "pmc: counter0 counter1, pmc: counter2";
    m_sdk_wrapper->set_available_counters({"counter0", "counter1", "counter2"});

    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    dispatch_data.dispatch_info.kernel_id                      = 1;
    dispatch_data.dispatch_info.agent_id.handle                = 0xff;
    rocprofiler_counter_config_id_t config;
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);

    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info().size(), 2);
    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info()[0].counter_names,
              (std::vector<std::string>{"counter0", "counter1"}));
    EXPECT_EQ(m_sdk_wrapper->get_create_counter_config_info()[1].counter_names,
              std::vector<std::string>{"counter2"});
}

TEST_F(TestSdkCallbacks, ProvidedKernelIds_ReturnsResultForThemOnly)
{
    m_tool_data->target_kernel_ids.insert(10);
}

TEST_F(TestSdkCallbacks, ProvidedKernelDispatchRanges_ReturnsResultForThemOnly)
{
    m_tool_data->kernel_filter_ranges.push_back({1, 2});
}

TEST_F(TestSdkCallbacks, ProvidedCountersNotSupported_DoesntReturnThem) {}

//////////////////////////////////////////////////////////////////////////
/// TestSdkCallbacks
void TestSdkCallbacks::SetUp()
{
    m_sdk_wrapper = std::make_shared<MockSdkWrapper>();
    rocprofiler_compute_tool::test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    m_sdk_callbacks = std::make_shared<rocprofiler_compute_tool::SdkCallbacksImpl>(m_sdk_wrapper);
    m_tool_data     = std::make_unique<rocprofiler_compute_tool::tool_data_t>();
}