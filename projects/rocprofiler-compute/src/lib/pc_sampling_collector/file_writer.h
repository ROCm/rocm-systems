// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <filesystem>
#include <memory>
#include <string>

namespace rocprofiler_compute_tool
{
class file_writer_t
{
public:
    static std::shared_ptr<file_writer_t> create();

    virtual ~file_writer_t() = default;
    virtual void write(const std::filesystem::path& output_file_path, const std::string& contents) = 0;
};

class file_writer_json_t : public file_writer_t
{
public:
    void write(const std::filesystem::path& output_file_path, const std::string& contents) override;
};
}  // namespace rocprofiler_compute_tool
