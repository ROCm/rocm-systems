// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

#include "lib/common/environment.hpp"

#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
enum class config_context
{
    global = 0,
    att_plugin,
    cli_plugin,
    ctf_plugin,
    file_plugin,
    perfetto_plugin,
};

int
get_mpi_size();

int
get_mpi_rank();

struct config
{
    bool        demangle    = get_env("ROCP_DEMANGLE_KERNELS", true);
    bool        truncate    = get_env("ROCP_TRUNCATE_KERNELS", false);
    int         mpi_size    = get_mpi_size();
    int         mpi_rank    = get_mpi_rank();
    std::string output_path = get_env<std::string>("ROCP_OUTPUT_PATH", ".");
    std::string output_file = get_env<std::string>("ROCP_OUTPUT_FILE", "results");
    std::string output_ext  = {};
};

template <config_context ContextT = config_context::global>
config&
get_config()
{
    if constexpr(ContextT == config_context::global)
    {
        static auto _v = config{};
        return _v;
    }
    else
    {
        // context specific config copied from global config
        static auto _v = get_config<config_context::global>();
        return _v;
    }
}

struct output_key
{
    output_key(std::string _key, std::string _val, std::string _desc = {});

    operator std::pair<std::string, std::string>() const;

    std::string key         = {};
    std::string value       = {};
    std::string description = {};
};

std::vector<output_key>
output_keys(std::string _tag = {});

std::string
compose_filename(const config&);

std::string
format(std::string _fpath, const std::string& _tag = {});

std::string
format_name(std::string_view _name, const config& = get_config<>());

void
initialize();
}  // namespace common
}  // namespace rocprofiler
