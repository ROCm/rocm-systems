// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "config.hpp"

#include <mutex>
#include <vector>

namespace rocprofsys
{

/**
 * Thread-safe registry of output files generated during profiling.
 * Each subsystem registers its output files after successful write.
 * The parent process reads the accumulated list at finalization to
 * print the output summary.
 */
class output_file_registry
{
public:
    static output_file_registry& get_instance();

    void register_file(std::string label, std::string path, std::string viewer);
    void register_file(config::output_file entry);

    [[nodiscard]] std::vector<config::output_file> get_files() const;

    void clear();

private:
    output_file_registry()                                       = default;
    output_file_registry(const output_file_registry&)            = delete;
    output_file_registry& operator=(const output_file_registry&) = delete;
    output_file_registry(output_file_registry&&)                 = delete;
    output_file_registry& operator=(output_file_registry&&)      = delete;

    mutable std::mutex               m_mutex;
    std::vector<config::output_file> m_files;
};

}  // namespace rocprofsys
