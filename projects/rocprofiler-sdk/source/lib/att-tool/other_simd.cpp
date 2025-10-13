#include "other_simd.hpp"
#include "outputfile.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>

namespace rocprofiler
{
namespace att_wrapper
{

void
OtherSimdFile::Ingest(const rocprofiler_thread_trace_decoder_other_simd_t& rec)
{
    if (rec.base.instructions_size == 0 || rec.base.instructions_array == nullptr) return;

    std::lock_guard<std::mutex> lk(mtx_);
    events_.reserve(events_.size() + static_cast<size_t>(rec.base.instructions_size));

    for (uint64_t i = 0; i < rec.base.instructions_size; ++i)
    {
        const auto& in = rec.base.instructions_array[i];
        OtherSimdInstEvent ev{};
        ev.cu    = rec.base.cu;
        ev.simd  = rec.base.simd;
        ev.other = rec.other_simd;

        ev.time     = in.time;
        ev.duration = in.duration;
        ev.stall    = in.stall;

        ev.category = in.category;

        events_.push_back(ev);
    }
}

void
OtherSimdFile::WriteJson(const std::filesystem::path& filepath) const
{
    std::lock_guard<std::mutex> lk(mtx_);

    if (events_.empty())
    {
        nlohmann::json out = {
            {"type", "OTHER_SIMD_GLOBAL_INSTRUCTIONS"},
            {"events", nlohmann::json::array()},
            {"count", 0}
        };

        OutputFile(filepath.string()) << out;

        return;
    }

    std::vector<OtherSimdInstEvent> sorted = events_;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.time < b.time;
    });

    nlohmann::json arr = nlohmann::json::array();
    arr.reserve(sorted.size());
    for (const auto& e : sorted)
    {
        arr.push_back({
            {"time",      e.time},
            {"duration",  e.duration},
            {"stall",     e.stall},
            {"category",  e.category},
            {"cu",        e.cu},
            {"simd_sel",  e.simd},
            {"simd_other",e.other}
        });
    }

    nlohmann::json out = {
        {"type", "OTHER_SIMD_INSTRUCTIONS"},
        {"events", arr},
        {"count", arr.size()}
    };

    OutputFile(filepath.string()) << out;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
