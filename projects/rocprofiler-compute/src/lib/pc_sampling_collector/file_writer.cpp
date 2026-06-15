// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "file_writer.h"

#include "gsl_assert.h"

#include <fstream>
#include <memory>
#include <stdexcept>

namespace rocprofiler_compute_tool
{
std::shared_ptr<file_writer_t> file_writer_t::create()
{
    return std::make_shared<file_writer_json_t>();
}

void file_writer_json_t::write(const std::filesystem::path& output_file_path, const std::string& contents)
{
    Expects(!output_file_path.empty());
    Expects(output_file_path.has_parent_path());

    std::error_code error;
    std::filesystem::create_directories(output_file_path.parent_path(), error);
    if (error)
    {
        throw std::runtime_error("Failed to create output directory: " + output_file_path.string() +
                                 ", error: " + error.message());
    }

    std::ofstream out_file(output_file_path, std::ios::out);
    if (!out_file.is_open())
    {
        throw std::runtime_error("Failed to open output file: " + output_file_path.string());
    }
    out_file << contents;
}
}  // namespace rocprofiler_compute_tool
