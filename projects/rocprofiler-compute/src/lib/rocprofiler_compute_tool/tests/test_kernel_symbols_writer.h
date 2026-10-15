// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "kernel_symbols_writer.h"
#include "sdk_callbacks.h"

#include <gtest/gtest.h>

#include <string>

class TestKernelSymbolsWriter : public ::testing::Test
{
protected:
    std::string format();

    void add_symbol(uint64_t kernel_id, const std::string& kernel_name);

    rocprofiler_compute_tool::tool_data_t m_tool_data;
};
