// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "mocks.h"

class TestSdkCallbacks : public ::testing::Test
{
protected:
    void                     SetUp() override;
    MockPcSamplingCollector& enable_pc_sampling();
    uint64_t dispatch_kernel_with_dispatch_info(const rocprofiler_compute_tool::kernel_dispatch_info_t& dispatch_info,
                                                const std::vector<std::string>& counters_pmc0,
                                                const std::vector<std::string>& counters_pmc1);

    uint64_t dispatch_kernel_with_id(uint64_t                        kernel_id,
                                     const std::vector<std::string>& counters_pmc0,
                                     const std::vector<std::string>& counters_pmc1 = {});
    void invoke_record_callback(uint64_t counter_id, const std::string& counter_name, double counter_value);
    void invoke_tool_tracing_callback(uint64_t kernel_id, const std::string& kernel_name = "default");
    void invoke_kernel_symbol_register(size_t code_object_id, uint64_t kernel_id, const char* kernel_name);

    static std::string convert_counters_per_pmc_to_str(const std::vector<std::vector<std::string>>& counters_per_pmc);
    static std::string convert_counters_to_str(const std::vector<std::string>& counters);
    static std::string remove_trailing_comma(const std::string& str);
    static std::vector<std::string> concat_counters(const std::vector<std::string>& v0,
                                                    const std::vector<std::string>& v1);

    std::shared_ptr<MockSdkWrapper>                             m_sdk_wrapper;
    std::shared_ptr<rocprofiler_compute_tool::SdkCallbacksImpl> m_sdk_callbacks;
    std::shared_ptr<MockPcSamplingCollector>                    m_pc_sampling_collector;
    std::shared_ptr<MockSourceSnapshotter>                      m_source_snapshotter;
    std::unique_ptr<rocprofiler_compute_tool::tool_data_t>      m_tool_data;
    const std::vector<std::string> m_counters_pmc0     = {"counter0", "counter1"};
    const std::vector<std::string> m_counters_pmc1     = {"counter2"};
    const uint64_t                 m_invalid_config_id = ~0u;
};

struct kernel_symbol_register_test_params_t
{
    const char* kernel_name = nullptr;
    std::string expected_name;
};

class TestSdkCallbacksKernelSymbolRegister
    : public TestSdkCallbacks
    , public ::testing::WithParamInterface<kernel_symbol_register_test_params_t>
{
};

class TestSdkCallbacksMultiplexing
    : public TestSdkCallbacks
    , public ::testing::WithParamInterface<rocprofiler_compute_tool::iteration_multiplexing_mode_t>
{
protected:
    void SetUp() override;

    rocprofiler_compute_tool::iteration_multiplexing_mode_t m_multiplexing_mode =
        rocprofiler_compute_tool::iteration_multiplexing_mode_t::DISABLED;
};

struct kernel_filtering_test_params_t
{
    std::string mangled_kernel_name;
    std::string demangled_kernel_name;
    std::string kernel_regex;
};

class TestSdkCallbacksKernelFiltering
    : public TestSdkCallbacks
    , public ::testing::WithParamInterface<kernel_filtering_test_params_t>
{
protected:
    void                           SetUp() override;
    kernel_filtering_test_params_t m_filtering_params = {};
};
