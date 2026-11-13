// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "output_registry.h"
#include "sdk_callbacks.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string_view>

/// Counts its calls so a test can tell how often the registry ran it.
class CountingWriter : public rocprofiler_compute_tool::OutputWriter
{
public:
    void write(rocprofiler_compute_tool::tool_data_t& tool_data) override;

    std::string_view name() const override { return "counting"; }

    int write_count() const { return m_write_count.load(); }

private:
    std::atomic<int> m_write_count{0};
};

/// Fails every time, so a test can check the registry keeps going.
class ThrowingWriter : public rocprofiler_compute_tool::OutputWriter
{
public:
    void write(rocprofiler_compute_tool::tool_data_t& tool_data) override;

    std::string_view name() const override { return "throwing"; }
};

class TestOutputRegistry : public ::testing::Test
{
protected:
    void SetUp() override;

    rocprofiler_compute_tool::OutputRegistry m_registry;
    rocprofiler_compute_tool::tool_data_t    m_tool_data;
    std::shared_ptr<CountingWriter>          m_first;
    std::shared_ptr<CountingWriter>          m_second;
};
