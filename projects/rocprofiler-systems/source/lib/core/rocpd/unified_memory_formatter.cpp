// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "unified_memory_formatter.hpp"
#include "logger/debug.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace rocprofsys
{
namespace rocpd
{

// ===================================================================
// Public Methods
// ===================================================================

void
unified_memory_formatter::format_text_output(const unified_memory_data& data, int pid,
                                              std::ostream& output)
{
    if(!data.has_data())
    {
        return;  // No unified memory data to display
    }

    print_header(output, pid);

    // Print per-device migration statistics
    for(const auto& device : data.device_summaries)
    {
        print_device_migrations(output, device);
    }

    // Print page fault summary
    if(!data.fault_summary_by_type.empty())
    {
        print_page_fault_summary(output, data.fault_summary_by_type);
    }

    // Print overall statistics (optional, if available)
    if(data.overall_summary)
    {
        print_overall_statistics(output, *data.overall_summary);
    }

    output << std::endl;
}

void
unified_memory_formatter::format_text_output_from_db(const std::string& db_path,
                                                      int                pid,
                                                      std::ostream&      output)
{
    try
    {
        unified_memory_aggregator aggregator(db_path);

        // Check if there's any unified memory data
        if(!aggregator.has_unified_memory_data())
        {
            return;  // No KFD events found
        }

        auto data = aggregator.aggregate();
        format_text_output(data, pid, output);
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to format unified memory output from database: {}", e.what());
    }
}

bool
unified_memory_formatter::write_to_file(const unified_memory_data& data, int pid,
                                         const std::string& filepath)
{
    try
    {
        std::ofstream file(filepath);
        if(!file.is_open())
        {
            LOG_ERROR("Failed to open file for writing: {}", filepath);
            return false;
        }

        format_text_output(data, pid, file);
        file.close();

        LOG_INFO("Unified memory report written to: {}", filepath);
        return true;
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to write unified memory report to file: {}", e.what());
        return false;
    }
}

// ===================================================================
// Private Helper Methods - Formatting
// ===================================================================

std::string
unified_memory_formatter::format_bytes(double bytes)
{
    if(bytes == 0.0) return "0.0000B";

    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int         unit_idx = 0;
    double      value = bytes;

    while(value >= 1024.0 && unit_idx < 4)
    {
        value /= 1024.0;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value << units[unit_idx];
    return oss.str();
}

std::string
unified_memory_formatter::format_time(double nanoseconds)
{
    if(nanoseconds == 0.0) return "0.0000ns";

    double      value;
    const char* unit;

    if(nanoseconds < 1000.0)
    {
        value = nanoseconds;
        unit  = "ns";
    }
    else if(nanoseconds < 1000000.0)
    {
        value = nanoseconds / 1000.0;
        unit  = "us";
    }
    else if(nanoseconds < 1000000000.0)
    {
        value = nanoseconds / 1000000.0;
        unit  = "ms";
    }
    else
    {
        value = nanoseconds / 1000000000.0;
        unit  = "s";
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value << unit;
    return oss.str();
}

std::string
unified_memory_formatter::format_bandwidth(double gbps)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << gbps << " GB/s";
    return oss.str();
}

// ===================================================================
// Private Helper Methods - Printing
// ===================================================================

void
unified_memory_formatter::print_header(std::ostream& output, int pid)
{
    output << "==" << pid << "== Unified Memory profiling result:\n";
}

void
unified_memory_formatter::print_device_migrations(
    std::ostream& output, const device_migration_summary& device)
{
    if(device.migrations.empty()) return;

    // Device header
    output << " Device \"" << device.agent_name << " (" << device.agent_id << ")\"\n";

    // Table header
    output << "    " << std::setw(8) << std::right << "Count"
           << "  " << std::setw(10) << std::right << "Avg Size"
           << "  " << std::setw(10) << std::right << "Min Size"
           << "  " << std::setw(10) << std::right << "Max Size"
           << "  " << std::setw(12) << std::right << "Total Size"
           << "  " << std::setw(12) << std::right << "Total Time"
           << "  " << std::setw(13) << std::right << "Bandwidth"
           << "  " << std::left << "Name"
           << "\n";

    // Migration statistics rows
    for(const auto& migration : device.migrations)
    {
        output << "    " << std::setw(8) << std::right << migration.count << "  "
               << std::setw(10) << std::right << format_bytes(migration.avg_size_bytes)
               << "  " << std::setw(10) << std::right
               << format_bytes(migration.min_size_bytes) << "  " << std::setw(10)
               << std::right << format_bytes(migration.max_size_bytes) << "  "
               << std::setw(12) << std::right << format_bytes(migration.total_size_bytes)
               << "  " << std::setw(12) << std::right
               << format_time(migration.total_time_ns) << "  " << std::setw(13)
               << std::right << format_bandwidth(migration.bandwidth_gbps) << "  "
               << std::left << migration.direction << "\n";
    }

    output << "\n";
}

void
unified_memory_formatter::print_page_fault_summary(
    std::ostream& output, const std::vector<page_fault_summary_by_type>& faults)
{
    if(faults.empty()) return;

    for(const auto& fault : faults)
    {
        output << " Total " << fault.agent_type << " Page faults: " << fault.total_faults;

        // Add breakdown if available
        if(fault.total_read_faults > 0 || fault.total_write_faults > 0)
        {
            output << " (Read: " << fault.total_read_faults
                   << ", Write: " << fault.total_write_faults << ")";
        }

        output << "\n";
    }

    output << "\n";
}

void
unified_memory_formatter::print_overall_statistics(std::ostream& output,
                                                    const unified_memory_summary& summary)
{
    output << " Overall Statistics:\n";
    output << "   Total Migrations:        " << summary.total_migrations << "\n";
    output << "   Total Bytes Migrated:    " << format_bytes(summary.total_bytes_migrated)
           << "\n";
    output << "   Total Migration Time:    " << format_time(summary.total_migration_time_ns)
           << "\n";
    output << "   Overall Bandwidth:       "
           << format_bandwidth(summary.overall_bandwidth_gbps) << "\n";

    output << "\n   Migration Breakdown:\n";
    output << "     Host → Device:         " << summary.host_to_device_count << " migrations, "
           << format_bytes(summary.host_to_device_bytes) << "\n";
    output << "     Device → Host:         " << summary.device_to_host_count << " migrations, "
           << format_bytes(summary.device_to_host_bytes) << "\n";
    output << "     Device → Device:       " << summary.device_to_device_count
           << " migrations, " << format_bytes(summary.device_to_device_bytes) << "\n";

    output << "\n";
}

}  // namespace rocpd
}  // namespace rocprofsys
