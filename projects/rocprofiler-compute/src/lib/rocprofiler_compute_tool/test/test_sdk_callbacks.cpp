// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_sdk_callbacks.h"

#include "rocprofiler_compute_tool.h"

TEST_F(TestSdkCallbacks, Simple)
{
    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    rocprofiler_counter_config_id_t              config;
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);
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