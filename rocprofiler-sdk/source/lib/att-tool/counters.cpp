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

#include "counters.hpp"
#include "outputfile.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>
#include "util.hpp"

namespace rocprofiler
{
namespace att_wrapper
{
CountersFile::CountersFile(const Fspath& _dir, const std::vector<std::string>& _names)
: dir(_dir)
, names(_names)
{}

void
CountersFile::AddShaderEngine(int se, const att_perfevent_t* events, size_t num_events)
{
    if(!num_events || !GlobalDefs::get().has_format("json")) return;

    nlohmann::json js;

    for(size_t i = 0; i < num_events; i++)
    {
        auto& ev = events[i];
        js.emplace_back({ev.time, ev.events0, ev.events1, ev.events2, ev.events3, ev.CU, ev.bank});
    }

    auto filename = dir / ("se" + std::to_string(se) + "_perfcounter.json");

    OutputFile(filename) << nlohmann::json{"data", js};
    shaders.emplace_back(filename);
}

CountersFile::~CountersFile()
{
    nlohmann::json counters_names;
    for(auto& name : names)
        counters_names.emplace_back(name);

    nlohmann::json perfcounter_filenames;
    for(auto& name : shaders)
        perfcounter_filenames.emplace_back(name);

    nlohmann::json js;
    js["counters"] = counters_names;
    js["shaders"]  = perfcounter_filenames;

    OutputFile(dir / "graph_options.json") << nlohmann::json{"data", js};
}

}  // namespace att_wrapper
}  // namespace rocprofiler
