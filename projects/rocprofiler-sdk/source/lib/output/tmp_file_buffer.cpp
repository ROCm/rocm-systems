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

#include "tmp_file_buffer.hpp"
#include "domain_type.hpp"

#include <fmt/format.h>

#include <utility>

namespace rocprofiler
{
namespace tool
{
std::string
compose_tmp_file_name(const output_config& cfg, domain_type buffer_type)
{
    return rocprofiler::tool::format_path(fmt::format("{}/.rocprofv3/{}-{}.dat",
                                                      cfg.tmp_directory,
                                                      "%ppid%-%pid%",
                                                      get_domain_trace_file_name(buffer_type)));
}

tmp_file_name_callback_t&
get_tmp_file_name_callback()
{
    static tmp_file_name_callback_t val = [](domain_type type) -> std::string {
        ROCP_CI_LOG(WARNING) << "rocprofv3 does not have a tmp file name callback defined for "
                             << get_domain_column_name(type) << ".";
        auto _cfg = output_config::load_from_env();
        return compose_tmp_file_name(_cfg, type);
    };
    return val;
}
}  // namespace tool
}  // namespace rocprofiler
