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

#include "lib/att-tool/att_lib_wrapper.hpp"
#include "lib/common/logging.hpp"

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
        if(!Enabled()) return;
        ofs = std::ofstream(str, std::ofstream::out);
        ROCP_FATAL_IF(!ofs.is_open()) << "Could not open output file " << str;
    }

    template <typename T>
    OutputFile& operator<<(const T& v)
    {
        if(Enabled()) ofs << v;
        return *this;
    }

    static bool& Enabled()
    {
        static bool _enabled = true;
        return _enabled;
    }

    std::ofstream ofs;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
