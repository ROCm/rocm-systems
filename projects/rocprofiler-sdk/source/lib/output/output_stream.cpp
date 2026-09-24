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

#include <unistd.h>
#include <cerrno>
#include <fstream>
#include <mutex>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace rocprofiler
{
namespace tool
{
namespace fs = common::filesystem;

namespace
{
const auto stdout_names = std::unordered_set<std::string_view>{"stdout", "STDOUT"};
const auto stderr_names = std::unordered_set<std::string_view>{"stderr", "STDERR"};

// Filesystem time when the tool was loaded into this process. Existing output last written after
// this point was produced during this run by another profiled process.
const auto tool_load_time = fs::file_time_type::clock::now();

// Output file stream that is built privately and published when closed (see publish_output).
struct published_ofstream : std::ofstream
{
    published_ofstream(std::string        _private_stem,
                       std::string        _output_stem,
                       std::string        _suffix,
                       std::ios::openmode _mode)
    : std::ofstream{_private_stem + _suffix, _mode}
    , private_stem{std::move(_private_stem)}
    , output_stem{std::move(_output_stem)}
    , suffix{std::move(_suffix)}
    {}

    std::string private_stem = {};
    std::string output_stem  = {};
    std::string suffix       = {};
};

// Output names this process has published. Publishing one of them again replaces it.
auto published_names_mutex = std::mutex{};
auto published_names       = std::unordered_set<std::string>{};

bool
published_by_this_process(const std::string& name)
{
    auto _lk = std::lock_guard<std::mutex>{published_names_mutex};
    return published_names.count(name) > 0;
}

void
mark_published(const std::string& name)
{
    auto _lk = std::lock_guard<std::mutex>{published_names_mutex};
    published_names.emplace(name);
}

void
move_output_part(const fs::path& from, const fs::path& to)
{
    if(!fs::exists(from)) return;
    auto ec = std::error_code{};
    fs::remove_all(to, ec);
    fs::rename(from, to);
}
}  // namespace

std::string
get_private_output_stem(std::string_view output_stem)
{
    auto stem = fs::path{std::string{output_stem}};
    return (stem.parent_path() / fmt::format(".{}.{}.tmp", stem.filename().string(), getpid()))
        .string();
}

std::string
publish_output(std::string_view                     private_stem,
               std::string_view                     output_stem,
               const std::vector<std::string_view>& suffixes)
{
    auto _private = std::string{private_stem};
    auto _stem    = std::string{output_stem};
    auto _claim   = _stem + std::string{suffixes.front()};

    // link() never replaces an existing file, so exactly one process claims the output name.
    if(::link((_private + std::string{suffixes.front()}).c_str(), _claim.c_str()) == 0)
    {
        fs::remove(_private + std::string{suffixes.front()});
    }
    else
    {
        auto link_errno = errno;
        auto ec         = std::error_code{};
        auto mtime      = fs::last_write_time(_claim, ec);
        if(link_errno == EEXIST && !ec && mtime >= tool_load_time &&
           !published_by_this_process(_claim))
        {
            auto _unique = fmt::format("{}_{}", _stem, getpid());
            ROCP_WARNING << fmt::format("{} was written by another process during this run; "
                                        "writing {} instead. Include %pid% in the output file "
                                        "name to give each process its own output.",
                                        _claim,
                                        _unique + std::string{suffixes.front()});
            _stem = std::move(_unique);
        }
        // Otherwise the output is from an earlier run, or the filesystem has no hard links.
        move_output_part(_private + std::string{suffixes.front()},
                         _stem + std::string{suffixes.front()});
    }

    for(size_t i = 1; i < suffixes.size(); ++i)
        move_output_part(_private + std::string{suffixes.at(i)},
                         _stem + std::string{suffixes.at(i)});

    mark_published(_stem + std::string{suffixes.front()});
    return _stem;
}

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
    auto  output_stem = get_output_filename(cfg, fname, std::string_view{});
    auto  suffix      = output_file.substr(output_stem.size());
    auto* _ofs        = new(std::nothrow)
        published_ofstream{get_private_output_stem(output_stem), output_stem, suffix, mode};

    LOG_IF(FATAL, !_ofs) << fmt::format("Failed to allocate ofstream for output file '{}'",
                                        output_file);
    LOG_IF(FATAL, _ofs && !*_ofs) << fmt::format("Failed to open '{}' for output", output_file);

    ROCP_ERROR << "Opened result file: " << output_file;

    return {_ofs, [](std::ostream*& v) {
                if(auto* _pofs = dynamic_cast<published_ofstream*>(v))
                {
                    _pofs->close();
                    publish_output(_pofs->private_stem, _pofs->output_stem, {_pofs->suffix});
                }
                delete v;
                v = nullptr;
            }};
}
}  // namespace tool
}  // namespace rocprofiler
