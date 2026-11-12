// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "output_stream.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <string_view>
#include <unordered_set>

namespace rocprofiler
{
namespace tool
{
namespace fs = common::filesystem;

namespace
{
const auto stdout_names = std::unordered_set<std::string_view>{"stdout", "STDOUT"};
const auto stderr_names = std::unordered_set<std::string_view>{"stderr", "STDERR"};

// the file name and every directory that has to exist to write it. The prefix
// may contain a subdirectory, so that is not always the output path itself
struct resolved_output
{
    std::string             filename    = {};
    std::array<fs::path, 2> directories = {};
};

resolved_output
resolve_output(const output_config& cfg, std::string_view fname, std::string_view ext)
{
    // add a period to provided file extension if necessary
    constexpr auto period   = std::string_view{"."};
    constexpr auto noperiod = std::string_view{};
    const auto     _ext =
        fmt::format("{}{}", (!ext.empty() && ext.find('.') != 0) ? period : noperiod, ext);

    auto output_path   = fs::path{tool::format_path(cfg.output_path)};
    auto output_prefix = tool::format_path(cfg.output_file);
    auto _ofname =
        tool::format_path(output_path / fmt::format("{}_{}{}", output_prefix, fname, _ext));

    return resolved_output{_ofname, {output_path, fs::path{_ofname}.parent_path()}};
}

std::optional<std::string>
invalid_directory(const resolved_output& resolved)
{
    for(const auto& itr : resolved.directories)
    {
        if(fs::exists(itr) && !fs::is_directory(fs::status(itr))) return itr.string();
    }

    return std::nullopt;
}
}  // namespace

std::optional<std::string>
check_output_path(const output_config& cfg, std::string_view fname, std::string_view ext)
{
    return invalid_directory(resolve_output(cfg, fname, ext));
}

std::string
get_output_filename(const output_config& cfg, std::string_view fname, std::string_view ext)
{
    auto resolved = resolve_output(cfg, fname, ext);

    if(auto invalid = invalid_directory(resolved))
    {
        ROCP_FATAL << fmt::format(
            "ROCPROFILER_OUTPUT_PATH ({}) already exists and is not a directory", *invalid);
    }

    for(const auto& itr : resolved.directories)
    {
        if(!fs::exists(itr)) fs::create_directories(itr);
    }

    return resolved.filename;
}

output_stream
get_output_stream(const output_config& cfg,
                  std::string_view     fname,
                  std::string_view     ext,
                  std::ios::openmode   mode)
{
    auto cfg_output_path = tool::format_path(cfg.output_path);

    if(stdout_names.count(cfg_output_path) > 0 || stdout_names.count(fname) > 0)
        return {&std::cout, [](auto*&) {}};
    else if(stderr_names.count(cfg_output_path) > 0 || stderr_names.count(fname) > 0)
        return {&std::cout, [](auto*&) {}};
    else if(cfg_output_path.empty() || fname.empty())
        return {&std::clog, [](auto*&) {}};

    auto  output_file = get_output_filename(cfg, fname, ext);
    auto* _ofs        = new(std::nothrow) std::ofstream{output_file, mode};

    LOG_IF(FATAL, !_ofs) << fmt::format("Failed to allocate ofstream for output file '{}'",
                                        output_file);
    LOG_IF(FATAL, _ofs && !*_ofs) << fmt::format("Failed to open '{}' for output", output_file);

    ROCP_ERROR << "Opened result file: " << output_file;

    return {_ofs, [](std::ostream*& v) {
                if(v) dynamic_cast<std::ofstream*>(v)->close();
                delete v;
                v = nullptr;
            }};
}
}  // namespace tool
}  // namespace rocprofiler
