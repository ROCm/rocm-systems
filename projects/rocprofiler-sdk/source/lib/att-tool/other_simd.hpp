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
#include <rocprof_trace_decoder/trace_decoder_types.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "outputfile.hpp"

namespace rocprofiler
{
namespace att_wrapper
{
inline void
write_other_simd_json(
    const std::vector<rocprofiler_thread_trace_decoder_inst_other_simd_t>& records,
    const Fspath&                                                          filepath,
    int64_t                                                                begin_time,
    int64_t                                                                end_time)
{
    nlohmann::ordered_json out;
    out["type"]                = "OTHER_SIMD_INSTRUCTIONS";
    out["begin_time"]          = begin_time;
    out["end_time"]            = end_time;
    out["wgp"]                 = records.front().wgp;
    out["instructions_schema"] = {"time", "duration", "category"};

    nlohmann::json::array_t events;
    events.reserve(records.size());

    for(const auto& in : records)
    {
        events.push_back({
            in.time,
            static_cast<int>(in.cycles),
            static_cast<int>(in.category),
        });
    }

    out["instructions_count"] = events.size();
    out["instructions"]       = std::move(events);
    OutputFile(filepath.string()) << out;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
