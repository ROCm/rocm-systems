// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "counters_writer.h"
#include "sdk_callbacks.h"

#include <gtest/gtest.h>

#include <string>

class TestCountersWriter : public ::testing::Test
{
protected:
    std::string format();

    static rocprofiler_compute_tool::counter_info_record_t make_record(uint64_t    dispatch_id,
                                                                       std::string counter_name,
                                                                       double      counter_value);

    rocprofiler_compute_tool::tool_data_t m_tool_data;

    int m_batches = 0;
};
