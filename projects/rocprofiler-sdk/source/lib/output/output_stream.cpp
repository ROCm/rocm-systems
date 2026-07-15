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
#include "lib/output/format_path.hpp"
#include "lib/output/output_key.hpp"

#include <fmt/format.h>

#include <string>
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

// Control-char sentinel: contains no %, {, or $, so it cannot perturb
// format_path's token/env parsing, and cannot appear in real path text.
constexpr auto pid_sentinel = std::string_view{"\001ROCPROF_PID\001"};

// True if the output-file format contains a token that expands to THIS process's
// pid. Rather than re-implement (and keep in sync with) format_path's token
// grammar -- the %p shorthand vs. %ppid%/%pgid% delimiters, shared-% overlaps,
// etc. -- in a second parser, we ask output_keys() for the same table format_path
// uses, redirect every key whose value IS our pid to a sentinel, and let
// format_path expand: if the sentinel survives, some pid token rendered our pid.
//
// Matching by value (not a hard-coded token list) keeps this in lockstep with
// output_key.cpp -- any present/future pid alias (%pid%, {pid}, %p, %nid%, ...)
// is covered automatically, and %nid% is treated as a pid only when it actually
// resolves to one (i.e. not under MPI). It is also coincidence-proof: a hostname
// or %ppid% can never equal the sentinel, and a token that renders our (unique)
// pid necessarily yields a unique filename.
bool
output_name_encodes_pid(std::string_view output_file)
{
    const auto _pid  = std::to_string(getpid());
    auto       _keys = output_keys();

    bool _has_pid_key = false;
    for(auto& _key : _keys)
    {
        if(_key.value == _pid)
        {
            _key.value   = std::string{pid_sentinel};
            _has_pid_key = true;
        }
    }

    if(!_has_pid_key) return false;
    return format_path(std::string{output_file}, _keys).find(pid_sentinel) != std::string::npos;
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
    // We skip the suffix only when the output-file format already resolves to this
    // process's pid (%pid%/{pid}/%p/%nid%); detection is delegated to format_path
    // so a coincidental digit (e.g. in %hostname%) can never suppress the suffix.
    auto _pid           = std::to_string(getpid());
    auto _root_pid      = common::get_env_optional("ROCPROF_OUTPUT_ROOT_PID");
    bool _is_descendant = _root_pid.has_value() && *_root_pid != _pid;
    if(_is_descendant && !output_name_encodes_pid(cfg.output_file))
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
