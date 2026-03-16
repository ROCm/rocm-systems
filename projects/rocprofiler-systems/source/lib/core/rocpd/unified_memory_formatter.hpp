// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "unified_memory_aggregator.hpp"

#include <iosfwd>
#include <string>

namespace rocprofsys
{
namespace rocpd
{

/**
 * @brief Text output formatter for unified memory profiling
 *
 * Generates nvprof-style text output for unified memory statistics.
 * Output format matches NVIDIA's unified memory profiling for familiarity.
 */
class unified_memory_formatter
{
public:
    /**
     * @brief Generate formatted text output for unified memory data
     * @param data Unified memory profiling data from aggregator
     * @param pid Process ID to display in header
     * @param output Output stream (defaults to stdout)
     */
    static void format_text_output(const unified_memory_data& data, int pid,
                                    std::ostream& output);

    /**
     * @brief Generate formatted text output directly from database
     * @param db_path Path to database file
     * @param pid Process ID to display in header
     * @param output Output stream (defaults to stdout)
     */
    static void format_text_output_from_db(const std::string& db_path, int pid,
                                            std::ostream& output);

    /**
     * @brief Write unified memory report to file
     * @param data Unified memory profiling data
     * @param pid Process ID
     * @param filepath Output file path
     * @return true if successful, false otherwise
     */
    static bool write_to_file(const unified_memory_data& data, int pid,
                              const std::string& filepath);

private:
    /**
     * @brief Format byte size with appropriate units (B, KB, MB, GB)
     * @param bytes Size in bytes
     * @return Formatted string with unit suffix
     */
    static std::string format_bytes(double bytes);

    /**
     * @brief Format time with appropriate units (ns, us, ms, s)
     * @param nanoseconds Time in nanoseconds
     * @return Formatted string with unit suffix
     */
    static std::string format_time(double nanoseconds);

    /**
     * @brief Format bandwidth in GB/s
     * @param gbps Bandwidth in gigabytes per second
     * @return Formatted string
     */
    static std::string format_bandwidth(double gbps);

    /**
     * @brief Print header with process ID
     */
    static void print_header(std::ostream& output, int pid);

    /**
     * @brief Print migration statistics for a device
     */
    static void print_device_migrations(std::ostream&                       output,
                                        const device_migration_summary& device);

    /**
     * @brief Print page fault summary
     */
    static void print_page_fault_summary(std::ostream& output,
                                         const std::vector<page_fault_summary_by_type>& faults);

    /**
     * @brief Print overall statistics
     */
    static void print_overall_statistics(std::ostream&                         output,
                                         const unified_memory_summary& summary);
};

}  // namespace rocpd
}  // namespace rocprofsys
