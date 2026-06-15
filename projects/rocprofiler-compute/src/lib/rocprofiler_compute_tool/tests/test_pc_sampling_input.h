// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#define ROCPROFILER_SDK_EXPERIMENTAL

#include "mocks.h"
#include "rocprofiler_compute_tool.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

// Fixture for the PC-sampling tool-side wiring tests: injects the Mock* objects
// through test_knobs so rocprofiler_configure()/initialize() drive the real tool
// code against them, and exposes Envp for injecting an EnvironCache.
class TestPcSamplingInput : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

    // Owns the storage backing a NULL-terminated char** envp; non-copyable/movable
    // so tests must hold it as a named local while data() is in use.
    class Envp
    {
    public:
        explicit Envp(const std::vector<std::string>& entries)
        {
            m_owned.reserve(entries.size());
            m_pointers.reserve(entries.size() + 1);

            for (const auto& entry : entries)
            {
                m_owned.emplace_back(entry.begin(), entry.end());
                m_owned.back().push_back('\0');
                m_pointers.push_back(m_owned.back().data());
            }

            m_pointers.push_back(nullptr);
        }

        Envp(const Envp&)            = delete;
        Envp& operator=(const Envp&) = delete;
        Envp(Envp&&)                 = delete;
        Envp& operator=(Envp&&)      = delete;

        char** data() { return m_pointers.data(); }

    private:
        std::vector<std::vector<char>> m_owned;
        std::vector<char*>             m_pointers;
    };

    static rocprofiler_compute_tool::tool_data_t* get_tool_data(const rocprofiler_tool_configure_result_t* cfg);

    // Drives the on_hsa_runtime_loaded path: registers the HSA intercept by
    // running initialize(), then fires the recorded intercept callback once.
    void drive_hsa_runtime_loaded();

    // Configures one supporting agent with the given env interval and advertised
    // [min,max] range, returning the interval handed to the SDK.
    uint64_t configured_interval_for(const std::string& env_interval,
                                     uint64_t           min_interval,
                                     uint64_t           max_interval);

    rocprofiler_client_id_t              m_client_id{};
    std::shared_ptr<MockInputParameters> m_input_parameters;
    std::shared_ptr<MockSdkWrapper>      m_sdk_wrapper;
    std::shared_ptr<MockCountersWriter>  m_counters_writer;
};
