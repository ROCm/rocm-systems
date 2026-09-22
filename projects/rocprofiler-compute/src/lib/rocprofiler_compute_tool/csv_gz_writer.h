// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "compression/gzip_output_stream.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace rocprofiler_compute_tool
{

/// Quotes a CSV field per RFC 4180. Kernel names carry commas in their
/// template arguments, so any field holding one has to go through here.
inline std::string csv_quote(std::string_view field)
{
    std::string quoted;
    quoted.reserve(field.size() + 2);
    quoted.push_back('"');
    for (char c : field)
    {
        if (c == '"')
            quoted.push_back('"');
        quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
}

/// Writes one gzip CSV artifact. format receives a sink it feeds text to and
/// returns false if it could not write everything.
///
/// The file is built under a ".tmp" name and renamed once it is complete, so a
/// reader never sees a partial artifact and a failed write leaves nothing
/// behind.
template<typename FormatFn>
void write_csv_gz(const std::string& final_path, std::string_view artifact, FormatFn&& format)
{
    const std::string temp_path = final_path + ".tmp";

    compression::GzipFileOutputStream stream(temp_path);
    if (!stream.is_open())
    {
        std::cerr << "Failed to open output file: " << final_path << std::endl;
        return;
    }

    const auto wrote = format([&stream](std::string_view text) { return stream.write(text); });

    std::error_code ec;
    if (!stream.close() || !wrote)
    {
        std::filesystem::remove(temp_path, ec);
        std::cerr << "Failed to write output file: " << final_path << std::endl;
        return;
    }

    std::filesystem::rename(temp_path, final_path, ec);
    if (ec)
    {
        std::filesystem::remove(temp_path, ec);
        std::cerr << "Failed to write output file: " << final_path << std::endl;
        return;
    }

    std::clog << "[rocprofiler-compute] " << artifact << " has been written to: " << final_path
              << std::endl;
}

}  // namespace rocprofiler_compute_tool
