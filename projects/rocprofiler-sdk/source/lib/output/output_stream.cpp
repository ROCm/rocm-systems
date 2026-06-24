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

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <string_view>
#include <unordered_set>

#include <unistd.h>

namespace rocprofiler
{
namespace tool
{
namespace fs = common::filesystem;

namespace
{
const auto stdout_names = std::unordered_set<std::string_view>{"stdout", "STDOUT"};
const auto stderr_names = std::unordered_set<std::string_view>{"stderr", "STDERR"};

// True if `pid` appears in `str` as a whole number -- i.e. its digits are not part of a
// longer digit run (so pid 23 does NOT match "out_1234", but DOES match "out_23" or the
// "23" left by a %pid% expansion). Used to avoid double-suffixing a name that already
// embeds this process's pid, without false-matching coincidental digit substrings.
bool
contains_pid_token(const std::string& str, const std::string& pid)
{
    auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    for(auto pos = str.find(pid); pos != std::string::npos; pos = str.find(pid, pos + 1))
    {
        bool digit_before = pos > 0 && is_digit(str[pos - 1]);
        auto end          = pos + pid.size();
        bool digit_after  = end < str.size() && is_digit(str[end]);
        if(!digit_before && !digit_after) return true;
    }
    return false;
}
}  // namespace

std::string
get_output_filename(const output_config& cfg, std::string_view fname, std::string_view ext)
{
    auto cfg_output_path = tool::format_path(cfg.output_path);

    // add a period to provided file extension if necessary
    constexpr auto period   = std::string_view{"."};
    constexpr auto noperiod = std::string_view{};
    const auto     _ext =
        fmt::format("{}{}", (!ext.empty() && ext.find('.') != 0) ? period : noperiod, ext);

    auto output_path   = fs::path{cfg_output_path};
    auto output_prefix = tool::format_path(cfg.output_file);

    // Descendants of the root append their PID so they don't overwrite the root's
    // output; recomputed per call so a process that forks later still gets its PID.
    // We check the already-expanded prefix for this PID rather than scanning raw
    // tokens: that skips the suffix only when the name truly embeds THIS process's
    // pid (%pid%/%p/%nid%), not a fork-shared token like %ppid%/%pgid% that would
    // collide across siblings.
    auto _pid           = std::to_string(getpid());
    auto _root_pid      = common::get_env_optional("ROCPROF_OUTPUT_ROOT_PID");
    bool _is_descendant = _root_pid.has_value() && *_root_pid != _pid;
    if(_is_descendant && !contains_pid_token(output_prefix, _pid))
    {
        output_prefix += fmt::format("_{}", _pid);
    }

    if(fs::exists(output_path) && !fs::is_directory(fs::status(output_path)))
    {
        ROCP_FATAL << fmt::format(
            "ROCPROFILER_OUTPUT_PATH ({}) already exists and is not a directory",
            output_path.string());
    }
    else if(!fs::exists(output_path))
    {
        fs::create_directories(output_path);
    }

    auto _ofname =
        tool::format_path(output_path / fmt::format("{}_{}{}", output_prefix, fname, _ext));

    // the prefix may contain a subdirectory
    if(auto _ofname_path = fs::path{_ofname}.parent_path(); !fs::exists(_ofname_path))
    {
        fs::create_directories(_ofname_path);
    }
    else if(fs::exists(_ofname_path) && !fs::is_directory(fs::status(_ofname_path)))
    {
        ROCP_FATAL << fmt::format(
            "ROCPROFILER_OUTPUT_PATH ({}) already exists and is not a directory",
            output_path.string());
    }

    return _ofname;
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
