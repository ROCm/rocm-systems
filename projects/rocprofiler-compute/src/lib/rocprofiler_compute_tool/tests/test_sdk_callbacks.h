// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "mocks.h"
#include "gtest/gtest.h"

class test_sdk_callbacks_t : public ::testing::Test
{
protected:
    void SetUp() override;
    uint64_t dispatch_kernel_with_dispatch_info(const rocprofiler_compute_tool::kernel_dispatch_info_t& dispatch_info,
                                                const std::vector<std::string>& counters_pmc0,
                                                const std::vector<std::string>& counters_pmc1) const;

    uint64_t dispatch_kernel_with_id(uint64_t                        kernel_id,
                                     const std::vector<std::string>& counters_pmc0,
                                     const std::vector<std::string>& counters_pmc1 = {}) const;
    void invoke_record_callback(uint64_t counter_id, const std::string& counter_name, double counter_value) const;
    void invoke_tool_tracing_callback(uint64_t kernel_id, const std::string& kernel_name = "default") const;

    static std::string convert_counters_per_pmc_to_str(const std::vector<std::vector<std::string>>& counters_per_pmc);
    static std::string convert_counters_to_str(const std::vector<std::string>& counters);
    static std::string remove_trailing_comma(const std::string& str);
    static std::vector<std::string> concat_counters(const std::vector<std::string>& v0,
                                                    const std::vector<std::string>& v1);

    std::shared_ptr<mock_sdk_wrapper_t>                                 m_sdk_wrapper;
    std::shared_ptr<rocprofiler_compute_tool::sdk_callbacks_impl_t> m_sdk_callbacks;
    std::unique_ptr<rocprofiler_compute_tool::tool_data_t>          m_tool_data;
    const std::vector<std::string> m_counters_pmc0     = {"counter0", "counter1"};
    const std::vector<std::string> m_counters_pmc1     = {"counter2"};
    const uint64_t                 m_invalid_config_id = ~0u;
};

class test_sdk_callbacks_multiplexing_t
    : public test_sdk_callbacks_t
    , public testing::WithParamInterface<rocprofiler_compute_tool::IterationMultiplexingMode>
{
protected:
    void SetUp() override;

    rocprofiler_compute_tool::IterationMultiplexingMode m_multiplexing_mode =
        rocprofiler_compute_tool::IterationMultiplexingMode::Disabled;
};

struct kernel_filtering_test_params_t
{
    std::string mangled_kernel_name;
    std::string demangled_kernel_name;
    std::string kernel_regex;
};

class test_sdk_callbacks_kernel_filtering_t
    : public test_sdk_callbacks_t
    , public ::testing::WithParamInterface<kernel_filtering_test_params_t>
{
protected:
    void                           SetUp() override;
    kernel_filtering_test_params_t m_filtering_params = {};
};
