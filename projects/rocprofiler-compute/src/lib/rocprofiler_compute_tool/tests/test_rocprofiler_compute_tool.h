// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "mocks.h"
#include "rocprofiler_compute_tool.h"

#include <memory>

class TestRocprofilerComputeTool : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;
    static rocprofiler_compute_tool::tool_data_t* get_tool_data(const rocprofiler_tool_configure_result_t* cfg);
    static void compare_counter_config_ids(const std::vector<uint64_t>& expected,
                                           const std::vector<uint64_t>& actual);
    static rocprofiler_compute_tool::counter_info_record_t create_counter_record(uint64_t counter_id,
                                                                                 uint64_t kernel_id);

    rocprofiler_client_id_t              m_client_id{};
    std::shared_ptr<MockInputParameters> m_input_parameters;
    std::shared_ptr<MockSdkWrapper>      m_sdk_wrapper;
    std::shared_ptr<MockCountersWriter>  m_counters_writer;
    std::shared_ptr<MockToolSetUp>       m_tool_setup;
};
