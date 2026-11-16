// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "dispatch_writer.h"
#include "sdk_callbacks.h"

#include <gtest/gtest.h>

#include <string>

class TestDispatchWriter : public ::testing::Test
{
protected:
    std::string format();

    static rocprofiler_compute_tool::dispatch_record_t make_record(uint64_t dispatch_id,
                                                                   uint64_t kernel_id);

    rocprofiler_compute_tool::tool_data_t m_tool_data;

    int m_batches = 0;
};
