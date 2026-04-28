// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "gtest/gtest.h"
#include "mocks.h"
#include "rocprofiler_compute_tool.h"

#include <memory>

class test_rocprofiler_compute_tool_t : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;
    static rocm_compute::tool_data_t* get_tool_data(const rocprofiler_tool_configure_result_t* cfg);
    static void compare_counter_config_ids(const std::vector<uint64_t>& expected,
                                           const std::vector<uint64_t>& actual);
    static rocm_compute::counter_info_record_t create_counter_record(uint64_t counter_id,
                                                                                 uint64_t kernel_id);
    rocprofiler_callback_tracing_record_t create_code_object_load_info_with_payload(
        rocprofiler_callback_tracing_code_object_load_data_t& payload);

    rocprofiler_client_id_t                              m_client_id{};
    rocm_compute::tool_data_t                m_tool_data{};
    rocprofiler_callback_tracing_code_object_load_data_t m_payload{};
    rocprofiler_callback_tracing_record_t                m_pc_sampling_record{};
    std::shared_ptr<mock_env_parameters_t>               m_env_parameters;
    std::shared_ptr<mock_sdk_wrapper_t>                  m_sdk_wrapper;
    std::shared_ptr<mock_counters_writer_t>              m_counters_writer;
    std::shared_ptr<mock_sdk_callbacks_t>                m_sdk_callbacks;
    std::shared_ptr<mock_pc_sampling_collector_t>        m_pc_sampling_collector;
};
