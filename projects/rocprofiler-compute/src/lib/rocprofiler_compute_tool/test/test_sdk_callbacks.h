// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "mocks.h"

class TestSdkCallbacks : public ::testing::Test
{
protected:
    void SetUp() override;

    void dispatch_kernel(uint64_t                        kernel_id,
                         const std::vector<std::string>& counters_pmc0,
                         const std::vector<std::string>& counters_pmc1 = {});

    static std::string convert_counters_per_pmc_to_str(const std::vector<std::vector<std::string>>& counters_per_pmc);
    static std::string convert_counters_to_str(const std::vector<std::string>& counters);
    static std::string remove_trailing_comma(const std::string& str);
    static std::vector<std::string> concat_counters(const std::vector<std::string>& v0,
                                                    const std::vector<std::string>& v1);

    std::shared_ptr<MockSdkWrapper>                             m_sdk_wrapper;
    std::shared_ptr<rocprofiler_compute_tool::SdkCallbacksImpl> m_sdk_callbacks;
    std::unique_ptr<rocprofiler_compute_tool::tool_data_t>      m_tool_data;
};
