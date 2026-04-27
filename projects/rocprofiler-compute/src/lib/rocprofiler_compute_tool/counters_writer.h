// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "sdk_callbacks.h"

namespace rocprofiler_compute_tool
{

class counters_writer_t
{
public:
    virtual ~counters_writer_t()                           = default;
    virtual void write_counters(const tool_data_t& tool_data) = 0;
};

class csv_counters_writer_t : public counters_writer_t
{
public:
    void write_counters(const tool_data_t& tool_data) override;
};
}  // namespace rocprofiler_compute_tool
