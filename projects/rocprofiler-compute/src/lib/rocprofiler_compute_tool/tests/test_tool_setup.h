// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "mocks.h"
#include "rocprofiler_compute_tool.h"

#include <memory>

class TestToolSetUp : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

    rocprofiler_client_id_t              m_client_id{};
    std::shared_ptr<MockInputParameters> m_input_parameters;
    std::shared_ptr<MockSdkWrapper>      m_sdk_wrapper;
    std::shared_ptr<MockCountersWriter>  m_counters_writer;
    std::shared_ptr<MockToolSetUp>       m_tool_setup;
};

class TestEnvironmentSetUp : public ::testing::Test
{
protected:
    void SetUp() override;

    std::shared_ptr<MockEnvironmentSetUp> m_tool_setup;
};
