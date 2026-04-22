// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace rocprofsys
{

struct output_file
{
    std::string label;
    std::string path;
    std::string viewer;
};

enum class output_format
{
    perfetto,
    rocpd,
    json,
    text,
    causal_json,
    causal_text
};

/**
 * Thread-safe registry of output files generated during profiling.
 * Each subsystem registers its output files after successful write.
 * The registry is created in rocprofsys_finalize_hidden and passed
 * to subsystems that register their output files.
 */
class output_file_registry
{
public:
    void register_file(std::string path, output_format format);
    void register_file(std::string path, output_format format,
                       std::string component_name);

    void print_summary() const;
    void clear();

private:
    static output_file make_entry(std::string path, output_format format,
                                  const std::string& component_name = {});

    mutable std::mutex       m_mutex;
    std::vector<output_file> m_files;
};

}  // namespace rocprofsys
