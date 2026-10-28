#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/types.hpp"

namespace cpu_snap::test_support
{

/** Captures every span the sink receives, copying out the bytes. */
struct sink_capture
{
    std::vector<std::vector<std::byte>> chunks;

    core::sink_fn as_fn()
    {
        return [this](std::span<const std::byte> bytes) {
            chunks.emplace_back(bytes.begin(), bytes.end());
        };
    }

    /** @return All captured bytes concatenated in capture order. */
    [[nodiscard]] std::vector<std::byte> all() const
    {
        std::vector<std::byte> out;
        for(const auto& c : chunks)
        {
            out.insert(out.end(), c.begin(), c.end());
        }
        return out;
    }
};

/** Accumulates every byte the sink receives into one flat buffer. */
struct byte_buffer
{
    std::vector<std::byte> bytes;

    core::sink_fn as_sink()
    {
        return [this](std::span<const std::byte> b) {
            bytes.insert(bytes.end(), b.begin(), b.end());
        };
    }
};

inline core::chunk_framer::config
make_framer_cfg(std::uint32_t chunk_bytes = 4096, std::string_view process_name = "test_process",
                std::uint32_t pid = 1234)
{
    core::chunk_framer::config c;
    c.chunk_bytes   = chunk_bytes;
    c.flags         = 0x2;
    c.pid           = pid;
    c.start_unix_ns = 1700000000000000000ULL;
    c.process_name  = process_name;
    return c;
}

}  // namespace cpu_snap::test_support
