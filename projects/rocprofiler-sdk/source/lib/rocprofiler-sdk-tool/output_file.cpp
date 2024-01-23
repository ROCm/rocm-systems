// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "output_file.hpp"
#include "config.hpp"

#include <fmt/format.h>

namespace rocprofiler
{
namespace tool
{
namespace fs = common::filesystem;

std::pair<std::ostream*, void (*)(std::ostream*&)>
get_output_stream(const std::string& fname, const std::string& ext)
{
    auto output_path      = fs::path{tool::format(tool::get_config().output_path)};
    auto output_file_name = tool::format(tool::get_config().output_file);

    if(output_path.string().empty()) return {&std::clog, [](auto*&) {}};

    if(fs::exists(output_path) && !fs::is_directory(fs::status(output_path)))
        throw std::runtime_error{
            fmt::format("ROCPROFILER_OUTPUT_PATH ({}) already exists and is not a directory",
                        output_path.string())};
    if(!fs::exists(output_path)) fs::create_directories(output_path);

    auto  output_file = tool::format(output_path / (output_file_name + "_" + fname + ext));
    auto* _ofs        = new std::ofstream{output_file};
    if(!_ofs && !*_ofs)
        throw std::runtime_error{fmt::format("Failed to open {} for output", output_file)};

    LOG(ERROR) << "Results File: " << output_file;

    return {_ofs, [](std::ostream*& v) {
                if(v) dynamic_cast<std::ofstream*>(v)->close();
                delete v;
                v = nullptr;
            }};
}
}  // namespace tool
}  // namespace rocprofiler
