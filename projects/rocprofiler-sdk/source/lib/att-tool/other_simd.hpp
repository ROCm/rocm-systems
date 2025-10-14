#pragma once
#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>
#include <filesystem>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <vector>
#include "outputfile.hpp"

namespace rocprofiler
{
namespace att_wrapper
{
inline void
write_other_simd_json(const rocprofiler_thread_trace_decoder_other_simd_t& rec,
                      const Fspath&                                        filepath)
{
    nlohmann::json out;
    out["type"]       = "OTHER_SIMD_INSTRUCTIONS";
    out["cu"]         = rec.cu;
    out["simd_sel"]   = rec.simd;
    out["begin_time"] = rec.begin_time;
    out["end_time"]   = rec.end_time;

    nlohmann::json::array_t events;

    if(rec.instructions_array && rec.instructions_size > 0)
    {
        events.reserve(rec.instructions_size);
        for(uint64_t k = 0; k < rec.instructions_size; ++k)
        {
            const auto& in = rec.instructions_array[k];
            events.push_back({
                {"time", in.time},
                {"duration", in.duration},
                {"category", static_cast<uint8_t>(in.category)},
            });
        }
    }

    out["events"] = std::move(events);

    OutputFile(filepath.string()) << out;
}

}  // namespace att_wrapper
}  // namespace rocprofiler