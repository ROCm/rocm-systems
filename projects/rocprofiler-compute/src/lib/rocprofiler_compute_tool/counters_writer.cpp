// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "counters_writer.h"

#include <fstream>
#include <iostream>

void rocprofiler_compute_tool::CsvCountersWriter::write_counters(tool_data_t* tool_data)
{
    std::ofstream ofs(tool_data->output_filename);
    if (!ofs.is_open())
    {
        std::cerr << "Failed to open output file: " << tool_data->output_filename << std::endl;
        return;
    }
    // Write header at the beginning of the file
    ofs << "dispatch_id,gpu_id,kernel_id,lds_per_workgroup,"
           "counter_id,counter_name,counter_value\n";
    for (const auto& r : tool_data->counter_records)
        ofs << r.dispatch_id << ',' << r.agent_id << "," << r.kernel_id << ',' << r.LDS_memory_size
            << ',' << r.counter_id << ',' << r.counter_name << ',' << r.counter_value << '\n';
    ofs.flush();
    std::clog << "[rocprofiler-compute] [" << __FUNCTION__
              << "] Counter collection data has been written to: " << tool_data->output_filename
              << std::endl;
}
