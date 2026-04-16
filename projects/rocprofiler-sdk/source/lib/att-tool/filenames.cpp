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

#include "filenames.hpp"
#include "other_simd.hpp"
#include "outputfile.hpp"
#include "shaderdata.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace rocprofiler
{
namespace att_wrapper
{
void
FilenameMgr::addwave(const Fspath& name, Coord coord, size_t begint, size_t endt)
{
    streams.emplace(coord, WaveName{name.filename(), begint, endt});
}

void
FilenameMgr::add_other_simd_data(
    int                                                                    se,
    const std::vector<rocprofiler_thread_trace_decoder_inst_other_simd_t>& records)
{
    if(records.empty()) return;

    // Compute begin/end from instructions
    int64_t begin = records.front().time;
    int64_t end   = records.back().time + records.back().cycles;

    auto&        vec = other_simd_files[se];
    const size_t idx = vec.size();
    Fspath       file =
        dir / ("other_simd_se" + std::to_string(se) + "_" + std::to_string(idx) + ".json");

    vec.emplace_back(
        OtherSIMDName{file.filename(), static_cast<size_t>(begin), static_cast<size_t>(end)});

    write_other_simd_json(records, file, begin, end);
}

void
FilenameMgr::add_shaderdata_data(int                                                  se,
                                 const rocprofiler_thread_trace_decoder_shaderdata_t* records,
                                 size_t                                               count)
{
    if(records == nullptr || count == 0 || !GlobalDefs::get().has_format("json")) return;

    int64_t begin = records[0].time;
    int64_t end   = records[count - 1].time;

    auto&        vec = shaderdata_files[se];
    const size_t idx = vec.size();
    Fspath file = dir / ("shaderdata_" + std::to_string(se) + "_" + std::to_string(idx) + ".json");
    vec.emplace_back(
        ShaderDataName{file.filename(), static_cast<size_t>(begin), static_cast<size_t>(end)});

    write_shaderdata_json(records, count, file, begin, end);
}

FilenameMgr::~FilenameMgr()
{
    if(!GlobalDefs::get().has_format("json")) return;

    using std::to_string;
    nlohmann::json namelist;
    for(auto& [coord, data] : streams)
    {
        namelist[to_string(coord.se)][to_string(coord.sm)][to_string(coord.sl)]
                [to_string(coord.id)] = {data.name, data.begin, data.end};
    }

    nlohmann::json other_simd;
    for(auto& [se, vec] : other_simd_files)
    {
        nlohmann::json::array_t arr;
        arr.reserve(vec.size());
        for(const auto& w : vec)
            arr.push_back({w.name, w.begin, w.end});
        other_simd[to_string(se)] = std::move(arr);
    }

    nlohmann::json shaderdata;
    for(auto& [se, vec] : shaderdata_files)
    {
        nlohmann::json::array_t arr;
        arr.reserve(vec.size());
        for(const auto& entry : vec)
            arr.push_back({entry.name, entry.begin, entry.end});
        shaderdata[to_string(se)] = std::move(arr);
    }

    const nlohmann::json metadata = {{"global_begin_time", 0},
                                     {"gfxv", (gfxip > 9) ? "navi" : "vega"},
                                     {"gfxip", gfxip},
                                     {"version", TOOL_VERSION},
                                     {"counter_names", perfcounters},
                                     {"wave_filenames", namelist},
                                     {"other_simd_filenames", other_simd},
                                     {"shaderdata_filenames", shaderdata}};

    OutputFile(filename) << metadata;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
