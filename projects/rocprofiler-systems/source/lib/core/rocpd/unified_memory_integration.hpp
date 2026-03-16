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

#include <cstdint>
#include <string>

// Forward declaration
struct sqlite3;

namespace rocprofsys
{
namespace rocpd
{

/**
 * @brief Integration point for unified memory profiling output generation
 *
 * This module coordinates the generation of all unified memory profiling outputs:
 * - Text report (nvprof-style)
 * - JSON statistics file
 * - Perfetto counter tracks
 *
 * Called during rocprofiler-systems finalization if unified memory profiling is enabled.
 */
class unified_memory_integration
{
public:
    /**
     * @brief Generate all unified memory profiling outputs
     * @param db_path Path to the rocpd database file
     * @param output_dir Output directory for generated files
     * @param pid Process ID for report headers
     *
     * This is the main entry point called during finalization.
     * Checks if unified memory profiling is enabled and if data exists.
     */
    static void generate_outputs(const std::string& db_path,
                                  const std::string& output_dir, int pid);

    /**
     * @brief Check XNACK environment variable and warn if not set
     *
     * XNACK must be enabled for proper unified memory profiling.
     * Prints warning if HSA_XNACK is not set to 1.
     */
    static void check_xnack_environment();

    /**
     * @brief Check if unified memory profiling is enabled via configuration
     * @return true if enabled, false otherwise
     */
    static bool is_enabled();

private:
    /**
     * @brief Generate text output file
     * @param db_path Database file path
     * @param output_path Output file path
     * @param pid Process ID
     */
    static void generate_text_output(const std::string& db_path,
                                      const std::string& output_path, int pid);

    /**
     * @brief Generate JSON output file
     * @param db_path Database file path
     * @param output_path Output file path
     */
    static void generate_json_output(const std::string& db_path,
                                      const std::string& output_path);

    /**
     * @brief Generate Perfetto counter tracks
     * @param db Database handle
     * @param upid Unique process ID for table suffixes
     */
    static void generate_perfetto_counters(sqlite3* db, const std::string& upid);

    /**
     * @brief Get unique process ID (UPID) from database
     * @param db_path Database file path
     * @return UPID string
     */
    static std::string get_upid_from_database(const std::string& db_path);
};

}  // namespace rocpd
}  // namespace rocprofsys
