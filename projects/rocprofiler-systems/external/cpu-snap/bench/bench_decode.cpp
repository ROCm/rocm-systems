// bench_decode.cpp
//
// Microbenchmark for the perf_event_open record decoder. Generates
// PERF_RECORD_SAMPLE byte streams in memory and times how fast they
// can be parsed into raw_sample. Isolates the userspace decode cost
// from the kernel sample-IRQ cost (which dominates real-world
// numbers).
//
// What this measures: ns spent per sample inside process_sample
// (parse + filter context-mask + push into vector).
// What this does NOT measure: perf_event_open setup, mmap, ring
// poll, kernel callchain walk, IRQ entry/exit.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <ratio>
#include <string>
#include <vector>

#include <linux/perf_event.h>

namespace {

// Same record layout as sampler::process_sample (frame_pointer mode):
//   header                  (perf_event_header)
//   u64 ip
//   u32 pid, u32 tid
//   u64 time
//   u64 nr_ips
//   u64 ips[nr_ips]
struct rec_layout
{
    perf_event_header h;
    std::uint64_t     ip;
    std::uint32_t     pid;
    std::uint32_t     tid;
    std::uint64_t     time;
    std::uint64_t     nr_ips;
    // followed by nr_ips * u64
};

std::vector<std::uint8_t> build_stream(std::size_t n_records,
                                       std::uint64_t frames_per_record,
                                       std::uint32_t tid)
{
    const std::size_t one_size =
        sizeof(rec_layout) + frames_per_record * sizeof(std::uint64_t);
    std::vector<std::uint8_t> buf(one_size * n_records, 0);

    std::mt19937_64 rng(0xC0FFEE);
    for (std::size_t i = 0; i < n_records; ++i) {
        auto* rec = reinterpret_cast<rec_layout*>(buf.data() + i * one_size);
        rec->h.type   = PERF_RECORD_SAMPLE;
        rec->h.misc   = 0;
        rec->h.size   = static_cast<std::uint16_t>(one_size);
        rec->ip       = rng();
        rec->pid      = tid;
        rec->tid      = tid;
        rec->time     = static_cast<std::uint64_t>(i) * 1000;
        rec->nr_ips   = frames_per_record;
        auto* frames  = reinterpret_cast<std::uint64_t*>(rec + 1);
        for (std::uint64_t f = 0; f < frames_per_record; ++f) {
            // Mix of user-space and PERF_CONTEXT_* / kernel sentinels
            // so the filter branch exercises both arms.
            const std::uint64_t r = rng() & 0x0000'7fff'ffff'ffffULL;
            frames[f] = (f == 0) ? 0xffffffff'ffffff80ULL /*PERF_CONTEXT_USER*/
                                 : r;
        }
    }
    return buf;
}

// Same inner loop as sampler::process_sample (frame_pointer
// mode). Kept stand-alone here so the bench measures decoder cost
// without holding the rest of the recorder.
struct decoded_sample
{
    std::uint32_t tid;
    std::uint64_t timestamp_ns;
    std::uint64_t ip;
    std::uint32_t n_frames;
    static constexpr std::uint32_t k_max_frames = 32;
    std::uint64_t frames[k_max_frames];
};

template <std::uint32_t MaxDepth>
[[gnu::noinline]] decoded_sample decode_one(const perf_event_header* h) noexcept
{
    const auto* p = reinterpret_cast<const std::uint8_t*>(h + 1);
    decoded_sample ds{};
    ds.ip            = *reinterpret_cast<const std::uint64_t*>(p); p += 8;
    p += 4;                                            // pid (unused)
    ds.tid           = *reinterpret_cast<const std::uint32_t*>(p); p += 4;
    ds.timestamp_ns  = *reinterpret_cast<const std::uint64_t*>(p); p += 8;
    const std::uint64_t nr_ips =
        *reinterpret_cast<const std::uint64_t*>(p); p += 8;

    for (std::uint64_t i = 0; i < nr_ips; ++i) {
        const std::uint64_t ipi = *reinterpret_cast<const std::uint64_t*>(p);
        p += 8;
        if (ipi >= 0x0000800000000000ULL) continue;
        if (ds.n_frames >= MaxDepth) continue;
        ds.frames[ds.n_frames++] = ipi;
    }
    return ds;
}

double run_one(std::size_t n_records, std::uint64_t frames_per_record)
{
    const auto stream = build_stream(n_records, frames_per_record, 12345);
    const std::size_t one_size =
        sizeof(rec_layout) + frames_per_record * sizeof(std::uint64_t);

    // Warmup
    std::uint64_t sink = 0;
    for (std::size_t i = 0; i < n_records; ++i) {
        const auto* h = reinterpret_cast<const perf_event_header*>(
            stream.data() + i * one_size);
        sink ^= decode_one<32>(h).ip;
    }

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for (std::size_t i = 0; i < n_records; ++i) {
        const auto* h = reinterpret_cast<const perf_event_header*>(
            stream.data() + i * one_size);
        sink ^= decode_one<32>(h).ip;
    }
    const auto t1 = clock::now();
    asm volatile("" :: "r"(sink) : "memory");

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / static_cast<double>(n_records);
}

}  // namespace

int main(int argc, char** argv)
{
    std::size_t   n_records          = 2'000'000;
    std::vector<std::uint64_t> depths{2, 4, 8, 16, 32};

    for (int i = 1; i < argc; ++i) {
        std::string const a = argv[i];
        if (a.starts_with("--records=")) {
            n_records = std::stoull(a.substr(10));
}
        if (a.starts_with("--depths=")) {
            depths.clear();
            std::string const s = a.substr(9);
            std::size_t pos = 0;
            while (pos < s.size()) {
                const auto comma = s.find(',', pos);
                depths.push_back(std::stoull(s.substr(pos,
                    comma == std::string::npos ? std::string::npos : comma - pos)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
    }

    std::printf("# bench_decode: %zu records per measurement\n", n_records);
    std::printf("%-10s %-14s %-14s\n", "depth", "ns/record", "records/sec");
    for (std::uint64_t const d : depths) {
        const double ns = run_one(n_records, d);
        std::printf("%-10llu %-14.2f %-14.0f\n",
                    static_cast<unsigned long long>(d), ns, 1e9 / ns);
    }
    return 0;
}
