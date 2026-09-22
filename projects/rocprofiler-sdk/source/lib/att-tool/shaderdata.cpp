// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "shaderdata.hpp"

#include "outputfile.hpp"

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>
#include <nlohmann/json.hpp>

namespace rocprofiler
{
namespace att_wrapper
{
void
write_shaderdata_json(const rocprofiler_thread_trace_decoder_shaderdata_t* records,
                      size_t                                               count,
                      const Fspath&                                        filepath,
                      int64_t                                              begin_time,
                      int64_t                                              end_time)
{
    if(records == nullptr || count == 0) return;

    nlohmann::ordered_json out;
    out["type"]           = "shaderdata";
    out["begin_time"]     = begin_time;
    out["end_time"]       = end_time;
    out["records_schema"] = {"time", "value", "cu", "simd", "wave_id", "flags"};

    nlohmann::json::array_t events;
    events.reserve(count);
    for(size_t i = 0; i < count; i++)
    {
        const auto& entry = records[i];
        events.push_back({entry.time,
                          entry.value,
                          static_cast<int>(entry.cu),
                          static_cast<int>(entry.simd),
                          static_cast<int>(entry.wave_id),
                          static_cast<int>(entry.flags)});
    }

    out["records_count"] = events.size();
    out["records"]       = std::move(events);

    OutputFile(filepath.string()) << out;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
