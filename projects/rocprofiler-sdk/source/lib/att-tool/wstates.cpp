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

#include "wstates.hpp"
#include "outputfile.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>

namespace rocprofiler
{
namespace att_wrapper
{
WstatesFile::WstatesFile(int state, const Fspath& dir)
: filename(dir / ("wstates" + std::to_string(state) + ".json"))
{}

WstatesFile::~WstatesFile()
{
    if(events.empty() || !GlobalDefs::get().has_format("json")) return;

    std::sort(events.begin(), events.end(), [](const event_t& a, const event_t& b) {
        return a.first < b.first;
    });

    int64_t        accum     = 0;
    int64_t        prev_time = INT64_MIN;
    nlohmann::json jtime;
    nlohmann::json jstate;

    for(auto& [time, value] : events)
    {
        accum += value;
        if(jtime.empty() || time != prev_time)
        {
            jtime.push_back(time);
            jstate.push_back(accum);
        }
        else
        {
            jstate.back() = accum;
        }
        prev_time = time;
    };

    nlohmann::json jfile;
    jfile["time"]  = jtime;
    jfile["state"] = jstate;
    jfile["name"]  = filename.filename();

    OutputFile(filename) << jfile;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
