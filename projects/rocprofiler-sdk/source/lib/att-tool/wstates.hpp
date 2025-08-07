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
class WstatesFile
{
    using event_t = std::pair<int64_t, int64_t>;

public:
    WstatesFile(int state, const Fspath& dir);
    ~WstatesFile();

    void add(int64_t time, int64_t duration)
    {
        events.emplace_back(time, 1);
        events.emplace_back(time + duration, -1);
    };

    Fspath               filename{};
    std::vector<event_t> events{};

    static constexpr size_t NUM_WSTATES = 5;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
