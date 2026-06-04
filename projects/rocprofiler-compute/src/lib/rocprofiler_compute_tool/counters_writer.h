// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "sdk_callbacks.h"

namespace rocprofiler_compute_tool
{

class CountersWriter
{
public:
    virtual ~CountersWriter()                           = default;
    virtual void write_counters(tool_data_t* tool_data) = 0;
};

class CsvCountersWriter : public CountersWriter
{
public:
    void write_counters(tool_data_t* tool_data) override;
};
}  // namespace rocprofiler_compute_tool
