// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "tool_data.h"

#include <string>
#include <vector>

namespace rocm_compute
{

class counters_writer_t
{
public:
    virtual ~counters_writer_t()                                                     = default;
    virtual void write_counters(const std::string&                        output_file,
                                const std::vector<counter_info_record_t>& records) = 0;
};

class csv_counters_writer_t : public counters_writer_t
{
public:
    void write_counters(const std::string&                        output_file,
                        const std::vector<counter_info_record_t>& records) override;
};
}  // namespace rocm_compute
