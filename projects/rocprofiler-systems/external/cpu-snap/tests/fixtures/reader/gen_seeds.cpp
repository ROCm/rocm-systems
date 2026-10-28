// Standalone generator for the reader fixture corpus; not part of the build.
// Re-run when the wire format changes:
//   c++ -std=c++20 -I.. -I../../.. -O2 \
//       tests/fixtures/reader/gen_seeds.cpp \
//       src/streaming_pipeline.cpp src/dl_module_watcher.cpp \
//       -ldl -pthread -o /tmp/gen_seeds
//   /tmp/gen_seeds tests/fixtures/reader

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>

#include "cpu_snap/sampling/streaming_pipeline.hpp"

using namespace cpu_snap::core;
using cpu_snap::sampling::internal::streaming_pipeline;

namespace
{

struct sink_capture
{
    std::vector<std::byte> bytes{};
    sink_fn                as_fn()
    {
        return [this](std::span<const std::byte> b) {
            bytes.insert(bytes.end(), b.begin(), b.end());
        };
    }
};

streaming_pipeline::config
cfg(sink_fn sink, std::size_t ckpt, std::size_t maps, std::uint32_t cb)
{
    streaming_pipeline::config c;
    c.sink                = std::move(sink);
    c.chunk_bytes         = cb;
    c.flags               = 0x2;
    c.pid                 = 1234;
    c.start_unix_ns       = 1700000000000000000ULL;
    c.process_name        = "callstack_fixture";
    c.checkpoint_interval = ckpt;
    c.maps_poll_interval  = maps;
    return c;
}

void
write_file(const std::filesystem::path& p, const std::vector<std::byte>& b)
{
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(b.data()),
              static_cast<std::streamsize>(b.size()));
}

void
gen_minimal(const std::filesystem::path& dir)
{
    sink_capture s;
    {
        streaming_pipeline p(cfg(s.as_fn(), 1000, 10000, 4096));
        p.finalize();
    }
    write_file(dir / "minimal.bin", s.bytes);
}

void
gen_steady_state(const std::filesystem::path& dir)
{
    sink_capture s;
    {
        streaming_pipeline  p(cfg(s.as_fn(), 25, 10000, 4096));
        const std::uint64_t frames[] = { 0x1111, 0x2222, 0x3333 };
        for(int i = 0; i < 50; ++i)
            p.on_sample(1, i, std::span<const std::uint64_t>(frames, 3));
        p.finalize();
    }
    write_file(dir / "steady_state.bin", s.bytes);
}

void
gen_multi_tid(const std::filesystem::path& dir)
{
    sink_capture s;
    {
        streaming_pipeline  p(cfg(s.as_fn(), 20, 10000, 4096));
        const std::uint64_t f1[] = { 0xAA, 0xBB };
        const std::uint64_t f2[] = { 0xCC, 0xDD };
        const std::uint64_t f3[] = { 0xEE };
        for(int i = 0; i < 15; ++i)
        {
            p.on_sample(1, 100 + i, std::span<const std::uint64_t>(f1, 2));
            p.on_sample(2, 200 + i, std::span<const std::uint64_t>(f2, 2));
            p.on_sample(3, 300 + i, std::span<const std::uint64_t>(f3, 1));
        }
        p.on_lost(7);
        p.finalize();
    }
    write_file(dir / "multi_tid.bin", s.bytes);
}

}  // namespace

int
main(int argc, char** argv)
{
    std::filesystem::path dir = (argc > 1) ? argv[1] : ".";
    std::filesystem::create_directories(dir);
    gen_minimal(dir);
    gen_steady_state(dir);
    gen_multi_tid(dir);
    return 0;
}
