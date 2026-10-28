// Unit tests for cpu_snap::core::chunk_framer.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cpu_snap/error.hpp"
#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
#include "support/stream_capture.hpp"

namespace
{

using namespace cpu_snap::core;
using cpu_snap::core::chunk_framer;
using cpu_snap::test_support::sink_capture;

/** Minimal cursor-style decoder for verification. */
class wire_cursor
{
public:
    wire_cursor(const std::byte* data, std::size_t len) noexcept
    : m_data(data)
    , m_end(data + len)
    {}

    template <typename T>
    T read() noexcept
    {
        T v{};
        std::memcpy(&v, m_data, sizeof(T));
        m_data += sizeof(T);
        return v;
    }

    void read_into(void* dst, std::size_t n) noexcept
    {
        std::memcpy(dst, m_data, n);
        m_data += n;
    }

    void skip(std::size_t n) noexcept { m_data += n; }

    [[nodiscard]] std::size_t remaining() const noexcept
    {
        return static_cast<std::size_t>(m_end - m_data);
    }

    [[nodiscard]] const std::byte* ptr() const noexcept { return m_data; }

private:
    const std::byte* m_data;
    const std::byte* m_end;
};

chunk_framer::config
make_cfg(std::uint32_t chunk_bytes = 4096)
{
    chunk_framer::config c;
    c.chunk_bytes   = chunk_bytes;
    c.flags         = 0x3;
    c.pid           = 4242;
    c.start_unix_ns = 1700000000000000000ULL;
    c.process_name  = "callstack_test";
    return c;
}

}  // namespace

// file_header + chunk emission

TEST(ChunkFramer, ByteExactSingleSampleChunk)
{
    sink_capture sink;
    chunk_framer fr(make_cfg(4096), sink.as_fn());
    fr.start();

    sample_record s;
    s.timestamp_ns = 0x1122334455667788ULL;
    s.tid          = 0xAABBCCDDU;
    s.leaf_node_id = 7U;
    fr.write(s);
    fr.flush();

    ASSERT_EQ(sink.chunks.size(), 2U);

    // file_header lands first as a raw 64-byte blob.
    ASSERT_EQ(sink.chunks[0].size(), sizeof(file_header));
    file_header hdr{};
    std::memcpy(&hdr, sink.chunks[0].data(), sizeof(hdr));
    EXPECT_EQ(hdr.magic[0], 'C');
    EXPECT_EQ(hdr.magic[1], 'S');
    EXPECT_EQ(hdr.magic[2], 'N');
    EXPECT_EQ(hdr.magic[3], 'P');
    EXPECT_EQ(hdr.version_major, k_record_format_version_major);
    EXPECT_EQ(hdr.version_minor, k_record_format_version_minor);
    EXPECT_EQ(hdr.pid, 4242U);
    EXPECT_EQ(hdr.chunk_bytes, 4096U);
    EXPECT_EQ(hdr.flags, 0x3U);

    // The chunk: header (16) + record header (4) + body (16) = 36.
    const std::size_t expected_chunk_bytes = sizeof(chunk_header) + 4 + 16;
    ASSERT_EQ(sink.chunks[1].size(), expected_chunk_bytes);

    wire_cursor  cur(sink.chunks[1].data(), sink.chunks[1].size());
    chunk_header ch{};
    cur.read_into(&ch, sizeof(ch));
    EXPECT_EQ(ch.magic[0], 'C');
    EXPECT_EQ(ch.magic[1], 'H');
    EXPECT_EQ(ch.records_byte_count, 4U + 16U);
    EXPECT_EQ(ch.chunk_seq, 0U);

    auto record_type_byte = cur.read<std::uint8_t>();
    EXPECT_EQ(record_type_byte, static_cast<std::uint8_t>(record_type::sample));
    EXPECT_EQ(cur.read<std::uint8_t>(), 0U);  // reserved
    EXPECT_EQ(cur.read<std::uint16_t>(), 16U);

    EXPECT_EQ(cur.read<std::uint64_t>(), s.timestamp_ns);
    EXPECT_EQ(cur.read<std::uint32_t>(), s.tid);
    EXPECT_EQ(cur.read<std::uint32_t>(), s.leaf_node_id);
}

// chunk boundary behavior

TEST(ChunkFramer, ChunkSeqMonotonicAcrossMultipleFlushes)
{
    sink_capture sink;
    // Tight 128-byte chunks: 16B header = 112B body budget;
    // each sample record costs 4+16 = 20B, so 5 fit per chunk.
    chunk_framer fr(make_cfg(128), sink.as_fn());
    fr.start();

    // 15 samples -> 3 chunks (5 per chunk).
    for(std::uint32_t i = 0; i < 15; ++i)
    {
        sample_record s;
        s.timestamp_ns = i;
        s.tid          = 100 + i;
        s.leaf_node_id = i;
        fr.write(s);
    }
    fr.flush();

    ASSERT_EQ(sink.chunks.size(), 1U + 3U);  // file_header + 3 chunks

    for(std::size_t i = 1; i <= 3; ++i)
    {
        chunk_header ch{};
        std::memcpy(&ch, sink.chunks[i].data(), sizeof(ch));
        EXPECT_EQ(ch.chunk_seq, i - 1) << "chunk index " << i;
    }
}

// error / robustness

TEST(ChunkFramer, SinkExceptionCountedNotPropagated)
{
    int     call_count = 0;
    sink_fn const throwing   = [&](std::span<const std::byte>) {
        ++call_count;
        throw std::runtime_error("simulated sink failure");
    };

    chunk_framer fr(make_cfg(256), throwing);
    EXPECT_NO_THROW(fr.start());
    EXPECT_NO_THROW(fr.write(sample_record{}));
    EXPECT_NO_THROW(fr.flush());
    EXPECT_GE(fr.sink_errors(), 2U);  // file_header emit + chunk emit
    EXPECT_GE(call_count, 2);
}

TEST(ChunkFramer, OversizeRecordThrows)
{
    sink_capture sink;
    chunk_framer fr(make_cfg(1U << 20), sink.as_fn());
    fr.start();

    // body_size = 16 + 8 + n_entries * 24; pick n that pushes past 65535.
    checkpoint_record cp;
    cp.entries.resize(3000);  // 16 + 3000*24 = 72,016 > 0xFFFF
    EXPECT_THROW(fr.write(cp), cpu_snap::resource_error);
}

TEST(ChunkFramer, WriteBeforeStartThrows)
{
    sink_capture sink;
    chunk_framer fr(make_cfg(), sink.as_fn());
    EXPECT_THROW(fr.write(sample_record{}), cpu_snap::usage_error);
}

TEST(ChunkFramer, DoubleStartThrows)
{
    sink_capture sink;
    chunk_framer fr(make_cfg(), sink.as_fn());
    fr.start();
    EXPECT_THROW(fr.start(), cpu_snap::usage_error);
}

TEST(ChunkFramer, RejectsNullSinkAndTinyChunk)
{
    EXPECT_THROW(chunk_framer(make_cfg(), nullptr), cpu_snap::config_error);
    EXPECT_THROW(chunk_framer(make_cfg(8), sink_capture{}.as_fn()),
                 cpu_snap::config_error);
}

// variable-size records round-trip

namespace
{

struct decoded_record_header
{
    std::uint8_t  type;
    std::uint8_t  reserved;
    std::uint16_t body_size;
};

/**
 * Parse a chunk into a list of (record_type, body_bytes) pairs and
 * verify the framing invariants. Skips file_header (caller hands raw
 * chunk bytes).
 */
struct parsed_chunk
{
    chunk_header                                                           header;
    std::vector<std::pair<decoded_record_header, std::vector<std::byte>>> records;
};

parsed_chunk
parse_chunk(const std::vector<std::byte>& bytes)
{
    parsed_chunk pc{};
    wire_cursor  cur(bytes.data(), bytes.size());
    cur.read_into(&pc.header, sizeof(pc.header));

    const std::size_t body_end = sizeof(chunk_header) + pc.header.records_byte_count;

    while(static_cast<std::size_t>(cur.ptr() - bytes.data()) < body_end)
    {
        decoded_record_header rh{};
        cur.read_into(&rh, sizeof(rh));
        std::vector<std::byte> body(rh.body_size);
        if(rh.body_size > 0)
        {
            cur.read_into(body.data(), rh.body_size);
        }
        pc.records.emplace_back(rh, std::move(body));

        const std::uint32_t consumed = sizeof(rh) + rh.body_size;
        const std::uint32_t padded   = (consumed + 3U) & ~3U;
        cur.skip(padded - consumed);
    }

    return pc;
}

}  // namespace

TEST(ChunkFramer, VariableSizeRecordsRoundtrip)
{
    sink_capture sink;
    chunk_framer fr(make_cfg(1U << 20), sink.as_fn());
    fr.start();

    maps_init_record mi;
    mi.wall_ns = 1000;
    for(int i = 0; i < 3; ++i)
    {
        module_info m;
        m.base_va     = 0x4000'0000ULL + i * 0x1000;
        m.size        = 0x1000;
        m.file_offset = 0;
        m.build_id    = { static_cast<std::uint8_t>(0xDE), static_cast<std::uint8_t>(0xAD), static_cast<std::uint8_t>(0xBE),
                          static_cast<std::uint8_t>(0xEF) };
        m.path        = "/lib/mod" + std::to_string(i) + ".so";
        mi.modules.push_back(std::move(m));
    }
    fr.write(mi);

    maps_delta_record md;
    md.wall_ns        = 2000;
    md.operation      = maps_delta_record::op::add;
    md.module.base_va = 0xFEED'0000ULL;
    md.module.size    = 0x2000;
    md.module.path    = "/dynamic.so";
    fr.write(md);

    checkpoint_record cp;
    cp.wall_ns = 3000;
    for(std::uint32_t i = 0; i < 5; ++i)
    {
        cp.entries.push_back({ 100 + i, i, 1000U * i, 0 });
    }
    fr.write(cp);

    fr.flush();

    ASSERT_EQ(sink.chunks.size(), 2U);  // file_header + 1 chunk
    const auto pc = parse_chunk(sink.chunks[1]);
    ASSERT_EQ(pc.records.size(), 3U);

    EXPECT_EQ(pc.records[0].first.type,
              static_cast<std::uint8_t>(record_type::maps_init));
    EXPECT_EQ(pc.records[1].first.type,
              static_cast<std::uint8_t>(record_type::maps_delta));
    EXPECT_EQ(pc.records[2].first.type,
              static_cast<std::uint8_t>(record_type::checkpoint));

    // Spot-check maps_init body: wall_ns + count + pad + 3 modules.
    {
        wire_cursor mc(pc.records[0].second.data(), pc.records[0].second.size());
        EXPECT_EQ(mc.read<std::uint64_t>(), 1000U);
        EXPECT_EQ(mc.read<std::uint32_t>(), 3U);
        EXPECT_EQ(mc.read<std::uint32_t>(), 0U);
        for(int i = 0; i < 3; ++i)
        {
            const auto base = mc.read<std::uint64_t>();
            EXPECT_EQ(base, 0x4000'0000ULL + i * 0x1000);
            EXPECT_EQ(mc.read<std::uint64_t>(), 0x1000U);
            EXPECT_EQ(mc.read<std::uint64_t>(), 0U);
            const auto bid_len  = mc.read<std::uint32_t>();
            const auto path_len = mc.read<std::uint32_t>();
            EXPECT_EQ(bid_len, 4U);
            std::vector<std::uint8_t> bid(bid_len);
            mc.read_into(bid.data(), bid_len);
            EXPECT_EQ(bid[0], 0xDEU);
            EXPECT_EQ(bid[3], 0xEFU);
            std::string path(path_len, '\0');
            mc.read_into(path.data(), path_len);
            EXPECT_EQ(path, "/lib/mod" + std::to_string(i) + ".so");
            // Skip tail padding for this module.
            const std::uint32_t consumed = 3 * sizeof(std::uint64_t) +
                                           2 * sizeof(std::uint32_t) + bid_len + path_len;
            const std::uint32_t padded = (consumed + 3U) & ~3U;
            mc.skip(padded - consumed);
        }
    }

    // checkpoint body: wall_ns, n=5, pad, 5 entries.
    {
        wire_cursor cc(pc.records[2].second.data(), pc.records[2].second.size());
        EXPECT_EQ(cc.read<std::uint64_t>(), 3000U);
        EXPECT_EQ(cc.read<std::uint32_t>(), 5U);
        EXPECT_EQ(cc.read<std::uint32_t>(), 0U);
        for(std::uint32_t i = 0; i < 5; ++i)
        {
            EXPECT_EQ(cc.read<std::uint32_t>(), 100U + i);
            EXPECT_EQ(cc.read<std::uint32_t>(), i);
            EXPECT_EQ(cc.read<std::uint64_t>(), 1000ULL * i);
            EXPECT_EQ(cc.read<std::uint64_t>(), 0U);
        }
    }

}

namespace
{

struct split_readback
{
    std::size_t   records         = 0;
    std::size_t   items           = 0;
    std::uint64_t parse_errors    = 0;
};

template <typename Record, typename Count>
split_readback
read_back(const sink_capture& sink, Count count)
{
    split_readback out;
    reader         rd;
    rd.on_event([&](const record_variant& rec) {
        if(const auto* r = std::get_if<Record>(&rec))
        {
            ++out.records;
            out.items += count(*r);
        }
    });
    const auto bytes = sink.all();
    rd.feed(bytes);
    out.parse_errors = rd.counters().parse_errors;
    return out;
}

}  // namespace

TEST(ChunkFramerSplit, CheckpointWithMoreEntriesThanOneRecordHoldsIsSplit)
{
    sink_capture sink;
    chunk_framer fr(chunk_framer::config{}, sink.as_fn());
    fr.start();

    checkpoint_record cp;
    cp.wall_ns = 7;
    cp.entries.resize(3000);
    for(std::uint32_t i = 0; i < cp.entries.size(); ++i) cp.entries[i].tid = i;
    fr.write_split(cp);
    fr.flush();

    const auto got = read_back<checkpoint_record>(
        sink, [](const checkpoint_record& r) { return r.entries.size(); });
    EXPECT_EQ(got.parse_errors, 0U);
    EXPECT_EQ(got.records, 2U);
    EXPECT_EQ(got.items, 3000U);
}

TEST(ChunkFramerSplit, MapsInitWithMoreModulesThanOneRecordHoldsIsSplit)
{
    sink_capture sink;
    chunk_framer fr(chunk_framer::config{}, sink.as_fn());
    fr.start();

    maps_init_record init;
    init.wall_ns = 7;
    for(std::uint64_t i = 0; i < 700; ++i)
    {
        module_info mod;
        mod.base_va = 0x1000 * (i + 1);
        mod.size    = 0x1000;
        mod.path    = "/usr/lib/x86_64-linux-gnu/some/long/directory/libmodule_" +
                   std::to_string(i) + ".so";
        init.modules.push_back(std::move(mod));
    }
    fr.write_split(init);
    fr.flush();

    const auto got = read_back<maps_init_record>(
        sink, [](const maps_init_record& r) { return r.modules.size(); });
    EXPECT_EQ(got.parse_errors, 0U);
    EXPECT_GE(got.records, 2U);
    EXPECT_EQ(got.items, 700U);
}

TEST(ChunkFramerSplit, SmallRecordsAreWrittenAsOne)
{
    sink_capture sink;
    chunk_framer fr(chunk_framer::config{}, sink.as_fn());
    fr.start();

    checkpoint_record cp;
    cp.entries.resize(3);
    fr.write_split(cp);
    fr.write_split(maps_init_record{});
    fr.flush();

    EXPECT_EQ(sink.chunks.size(), 1U + 1U);  // file_header + one chunk
}
