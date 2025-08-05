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

#include "perfcounter.hpp"
#include "outputfile.hpp"
#include "wave.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace rocprofiler
{
namespace att_wrapper
{
void
PerfcounterFile(WaveConfig& config, const perfevent_t* events, size_t event_count)
{
    nlohmann::json data;
    for(size_t i = 0; i < event_count; i++)
    {
        const auto& event = events[i];

        nlohmann::json json_event;
        json_event.push_back(event.time);
        json_event.push_back(event.events0);
        json_event.push_back(event.events1);
        json_event.push_back(event.events2);
        json_event.push_back(event.events3);
        json_event.push_back(event.CU);
        json_event.push_back(event.bank);

        data.push_back(json_event);
    }

    const nlohmann::json json{{"data", data}};

    std::string filename = "se" + std::to_string(config.shader_engine) + "_perfcounter.json";
    OutputFile(config.filemgr->dir / filename) << json;
}
}  // namespace att_wrapper
}  // namespace rocprofiler
