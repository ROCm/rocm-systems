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

#pragma once

#include "att_lib_wrapper.hpp"

#include <map>
#include <nlohmann/json.hpp>
#include <vector>
#include "util.hpp"

namespace rocprofiler
{
namespace att_wrapper
{
class SPMFile
{
public:
    SPMFile(const Fspath& output_dir)
    : filename(output_dir / "spm.json"){};
    ~SPMFile();

    void addSpm(std::vector<char>& descriptor, std::vector<std::vector<char>>& spm_data);

    void setCounters(const std::map<rocprofiler_counter_id_t, std::string>& _counters)
    {
        counters = _counters;
    }

protected:
    std::map<rocprofiler_counter_id_t, std::string> counters;

    const Fspath filename;
    // Maps counter -> shader -> instance -> values
    nlohmann::json json{};
};
}  // namespace att_wrapper
}  // namespace rocprofiler
