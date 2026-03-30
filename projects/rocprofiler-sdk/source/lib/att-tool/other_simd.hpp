#pragma once
#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>
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
