// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "logger.hpp"

#include <spdlog/fmt/fmt.h>

#include <string>
#include <string_view>
#include <unistd.h>

namespace rocprofsys
{
namespace logger_detail
{
std::string
include_process_id_in_filename(std::string_view filename)
{
    if(filename.empty()) return std::string{};

    auto last_sep       = filename.find_last_of('/');
    auto filename_start = (last_sep == std::string_view::npos) ? 0 : last_sep + 1;
    auto dot_pos        = filename.find_last_of('.');
    bool has_extension =
        (dot_pos != std::string_view::npos) && (dot_pos > filename_start);

    if(!has_extension) return fmt::format("{}_{}", filename, getpid());
    return fmt::format("{}_{}{}", filename.substr(0, dot_pos), getpid(),
                       filename.substr(dot_pos));
}
}  // namespace logger_detail
}  // namespace rocprofsys
