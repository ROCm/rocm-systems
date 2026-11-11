// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "agent_writer.h"
#include "sdk_callbacks.h"

#include <gtest/gtest.h>

#include <string>

class TestAgentWriter : public ::testing::Test
{
protected:
    std::string format();

    void add_agent(uint64_t handle, uint32_t node_id, const std::string& product_name);

    rocprofiler_compute_tool::tool_data_t m_tool_data;
};
