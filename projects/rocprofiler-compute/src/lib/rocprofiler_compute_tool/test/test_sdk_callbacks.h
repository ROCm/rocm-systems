// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "mocks.h"

class TestSdkCallbacks : public ::testing::Test
{
protected:
    void                                                        SetUp() override;
    std::shared_ptr<MockSdkWrapper>                             m_sdk_wrapper;
    std::shared_ptr<rocprofiler_compute_tool::SdkCallbacksImpl> m_sdk_callbacks;
    std::unique_ptr<rocprofiler_compute_tool::tool_data_t>      m_tool_data;
};
