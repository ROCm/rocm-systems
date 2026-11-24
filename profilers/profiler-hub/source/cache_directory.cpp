// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cache_directory.hpp"

#include <profiler-hub/cpp/reader.hpp>
#include <profiler-hub/cpp/reader_types.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <string_view>

namespace profiler_hub
{
namespace
{

constexpr size_t k_max_stem_bytes = 64;
constexpr double k_kib            = 1024.0;

// Replaces characters that Windows does not allow in a file name, and trims
// the trailing dots and spaces that Windows does not allow at the end.
std::string
safe_stem(const std::string& stem)
{
    std::string text = stem;
    for(char& character : text)
    {
        const auto byte = static_cast<unsigned char>(character);
        if(byte < 0x20 ||
           std::string_view{ "<>:\"/\\|?*" }.find(character) != std::string_view::npos)
        {
            character = '_';
        }
    }
    if(text.size() > k_max_stem_bytes)
    {
        text.resize(k_max_stem_bytes);
        // Do not leave a UTF-8 sequence cut in half.
        while(!text.empty() && (static_cast<unsigned char>(text.back()) & 0xC0) == 0x80)
        {
            text.pop_back();
        }
    }
    while(!text.empty() && (text.back() == '.' || text.back() == ' '))
    {
        text.pop_back();
    }
    return text.empty() ? std::string{ "trace" } : text;
}

// Local time of a file-clock timestamp, to the second.
std::string
local_timestamp(std::filesystem::file_time_type time)
{
    // file_time_type has no portable conversion to system_clock before C++20 clock_cast,
    // so shift it by the offset between the two clocks as they are now.
    const auto system_time =
        std::chrono::system_clock::now() +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            time - std::filesystem::file_time_type::clock::now());
    const std::time_t seconds = std::chrono::system_clock::to_time_t(system_time);

    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &local);
    return buffer;
}

std::string
human_size(uint64_t bytes)
{
    const double value      = static_cast<double>(bytes);
    char         buffer[32] = {};
    if(value >= k_kib * k_kib * k_kib)
    {
        std::snprintf(buffer, sizeof(buffer), "%.1fGB", value / (k_kib * k_kib * k_kib));
    }
    else if(value >= k_kib * k_kib)
    {
        std::snprintf(
            buffer, sizeof(buffer), "%.0fMB", std::ceil(value / (k_kib * k_kib)));
    }
    else
    {
        std::snprintf(buffer, sizeof(buffer), "%.0fKB", std::ceil(value / k_kib));
    }
    return buffer;
}

std::string
format_label(reader_types::trace_schema_t schema)
{
    return schema == reader_types::trace_schema_t::rocpd ? "rocpd" : "unknown";
}

}  // namespace

std::string
trace_cache_directory(std::string_view                cache_root,
                      const std::vector<std::string>& source_paths)
{
    if(cache_root.empty() || source_paths.empty()) return {};

    uint64_t                        total_size = 0;
    std::filesystem::file_time_type newest_time{};
    for(const std::string& source_path : source_paths)
    {
        const std::filesystem::path file{ source_path };
        if(!std::filesystem::is_regular_file(file)) return {};
        total_size += static_cast<uint64_t>(std::filesystem::file_size(file));
        const auto write_time = std::filesystem::last_write_time(file);
        if(write_time > newest_time) newest_time = write_time;
    }

    // The probe gives the format and version of the first source, the one the
    // reader opens first. A file it cannot read is labelled unknown, as before.
    const reader_types::file_probe_result_t probe = probe_file(source_paths.front());
    const reader_types::trace_schema_t      schema =
        probe.status == PH_RESULT_SUCCESS ? probe.trace_schema
                                               : reader_types::trace_schema_t::unknown;
    const version_t version =
        probe.status == PH_RESULT_SUCCESS ? probe.schema_version : version_t{ 0, 0, 0 };

    const std::filesystem::path first{ source_paths.front() };
    std::string                 name = safe_stem(first.stem().string());
    if(source_paths.size() > 1)
    {
        name += "+" + std::to_string(source_paths.size() - 1);
    }

    const std::string folder =
        local_timestamp(newest_time) + "_" + format_label(schema) + "_v" +
        std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
        std::to_string(version.patch) + "_" + name + "_" + human_size(total_size);
    return (std::filesystem::path{ cache_root } / folder).string();
}

}  // namespace profiler_hub
