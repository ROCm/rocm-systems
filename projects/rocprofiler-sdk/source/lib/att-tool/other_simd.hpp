#pragma once

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <filesystem>

namespace rocprofiler
{
namespace att_wrapper
{

struct OtherSimdInstEvent
{
    uint8_t cu    = 0;
    uint8_t simd  = 0;
    uint8_t other = 0;

    int64_t time      = 0;
    int32_t duration  = 0;
    uint32_t stall    = 0;

    uint32_t category = 0;
};

class OtherSimdFile
{
public:
    explicit OtherSimdFile() = default;

    void Ingest(const rocprofiler_thread_trace_decoder_other_simd_t& rec);
    void WriteJson(const std::filesystem::path& filepath) const;

private:
    mutable std::mutex mtx_;
    std::vector<OtherSimdInstEvent> events_;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
