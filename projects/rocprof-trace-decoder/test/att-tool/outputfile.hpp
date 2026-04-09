// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// output_directory: "@CMAKE_CURRENT_BINARY_DIR@/roctracer-roctx-trace"

#pragma once

#include "att_lib_wrapper.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace rocprofiler
{
namespace att_wrapper
{
class OutputFile
{
public:
    OutputFile(const std::string& str)
    {
        static const char* bench = getenv("ATT_BENCHMARK");
        static bool bench_mode = bench ? (atoi(bench)!=0) : false;

        if(bench_mode && str.find(".csv") == std::string::npos) return;
        ofs = std::ofstream(str, std::ofstream::out);
        CHECK_TRUE(ofs.is_open());
    }

    template <typename T>
    OutputFile& operator<<(const T& v)
    {
        if(ofs.is_open()) ofs << v;
        return *this;
    }

    std::ofstream ofs;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
