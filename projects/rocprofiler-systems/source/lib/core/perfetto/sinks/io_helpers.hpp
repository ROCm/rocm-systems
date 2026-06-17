// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/units.hpp"
#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>

namespace rocprofsys::core::perfetto_sink_detail
{
[[nodiscard]] inline bool
stderr_log_allowed() noexcept
{
    try
    {
        return config::get_verbose() >= 0 &&
               config::output_filtering::is_log_output_enabled_for_current_mpi_rank();
    } catch(const std::exception&)
    {
        return false;
    }
}

inline void
emit_size_line(const std::string& filename, std::size_t bytes)
{
    if(!stderr_log_allowed()) return;
    std::fprintf(
        stderr, "[rocprofsys][%i]> %s perfetto (%.2f KB / %.2f MB / %.2f GB)... Done\n",
        dmp::rank(), filename.c_str(), static_cast<double>(bytes) / units::kilobyte,
        static_cast<double>(bytes) / units::megabyte,
        static_cast<double>(bytes) / units::gigabyte);
    std::fflush(stderr);
}

inline void
emit_open_error_line(const std::string& filename)
{
    if(!stderr_log_allowed()) return;
    std::fprintf(stderr, "[rocprofsys][%i]> Error opening '%s'...\n", dmp::rank(),
                 filename.c_str());
    std::fflush(stderr);
}

inline bool
write_proto_to(const std::string& filename, const char* data, std::size_t size,
               output_file_registry& registry, bool emit_status = true)
{
    const auto parent = std::filesystem::path{ filename }.parent_path();
    if(!parent.empty())
    {
        std::error_code ec{};
        std::filesystem::create_directories(parent, ec);
        if(ec)
        {
            LOG_ERROR("write_proto_to: could not create directory '{}': {}",
                      parent.string(), ec.message());
            if(emit_status) emit_open_error_line(filename);
            return false;
        }
    }

    std::ofstream ofs{ filename, std::ios::out | std::ios::binary };
    if(!ofs.is_open() || !ofs.good())
    {
        if(emit_status) emit_open_error_line(filename);
        return false;
    }
    ofs.write(data, static_cast<std::streamsize>(size));
    if(emit_status) emit_size_line(filename, size);
    registry.register_file(filename, output_format::perfetto);
    return true;
}
}  // namespace rocprofsys::core::perfetto_sink_detail
