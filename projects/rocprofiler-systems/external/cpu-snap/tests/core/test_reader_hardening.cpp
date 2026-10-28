// Hardening tests for cpu_snap::core::reader against malformed / adversarial
// wire input (PR2 reader hardening). Each TEST corresponds to one item in
// the hardening pass; see the commit that introduces it for the exact bug.

#include <gtest/gtest.h>

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
#include "support/stream_capture.hpp"

namespace
{

using namespace cpu_snap::core;
using cpu_snap::test_support::byte_buffer;
using cpu_snap::test_support::make_framer_cfg;

}  // namespace

// cyclic NEW_NODE / resolve_stack infinite loop

namespace
{

/** Build a stream containing a single, self-referential NEW_NODE
 *  (node_id == parent_id), which the writer invariant (frame_trie) can
 *  never produce but a crafted stream can. */
byte_buffer
build_self_cycle_stream()
{
    byte_buffer buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();
    new_node_record nn;
    nn.node_id   = 1;
    nn.parent_id = 1;  // self-loop
    nn.ip        = 0xDEAD;
    fr.write(nn);
    fr.flush();
    return buf;
}

}  // namespace

TEST(ReaderHardening, CyclicNewNodeDoesNotHangResolveStack)
{
    auto buf = build_self_cycle_stream();

    // Run the actual repro in a child process with a hard wall-clock alarm:
    // pre-fix, trie_resolver::resolve() spins forever on the self-loop and
    // push_back()s without bound; the SIGALRM default action (terminate)
    // bounds the damage to ~1s of RSS growth instead of hanging the suite.
    pid_t child = fork();
    ASSERT_NE(child, -1);
    if(child == 0)
    {
        alarm(2);
        reader r;
        r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));
        [[maybe_unused]] const auto chain = r.resolve_stack(1);
        _exit(0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status)) << "resolve_stack hung on a cyclic trie (killed by alarm)";
    if(WIFEXITED(status))
    {
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST(ReaderHardening, CyclicNewNodeIsRejectedAtDecode)
{
    auto buf = build_self_cycle_stream();

    reader r;
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_GE(r.counters().parse_errors, 1U);
    EXPECT_TRUE(r.resolve_stack(1).empty());
}

// unbounded chunk body allocation via records_byte_count

TEST(ReaderHardening, OversizeRecordsByteCountIsRejectedNotAllocated)
{
    byte_buffer buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();  // emits just the 64-byte file_header (chunk_bytes == 4096)

    // Craft a chunk header claiming a records_byte_count far larger than the
    // file header's chunk_bytes budget allows, with no matching body bytes.
    // A vulnerable reader assigns m_chunk_buf to that count before any
    // truncation check, forcing a multi-GB allocation from 16 bytes of input.
    chunk_header ch{};
    ch.magic              = { 'C', 'H' };
    ch.version             = 0;
    ch.records_byte_count = 0xFFFFFFF0U;
    ch.chunk_seq          = 1;
    const auto* raw       = reinterpret_cast<const std::byte*>(&ch);
    buf.bytes.insert(buf.bytes.end(), raw, raw + sizeof(ch));

    reader r;
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_GE(r.counters().chunks_rejected, 1U);
    EXPECT_FALSE(r.is_fatal());
}

TEST(ReaderHardening, OversizeFileHeaderChunkBytesIsFatal)
{
    byte_buffer buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();

    // chunk_bytes sits right after magic(4)+version_major(2)+version_minor(2)+
    // start_unix_ns(8)+pid(8) == offset 24 in file_header.
    std::uint32_t bogus_chunk_bytes = 0xFFFFFFFFU;
    std::memcpy(buf.bytes.data() + 24, &bogus_chunk_bytes, sizeof(bogus_chunk_bytes));

    reader r;
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_TRUE(r.is_fatal());
    EXPECT_GE(r.counters().parse_errors, 1U);
}

// reject wire-format majors other than the current one

TEST(ReaderHardening, UnsupportedMajorVersionIsFatal)
{
    byte_buffer buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();

    // version_major sits right after magic(4) in file_header.
    std::uint16_t bogus_major = static_cast<std::uint16_t>(k_record_format_version_major + 1);
    std::memcpy(buf.bytes.data() + 4, &bogus_major, sizeof(bogus_major));

    reader r;
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_TRUE(r.is_fatal());
    EXPECT_GE(r.counters().parse_errors, 1U);
}

// resync after a bad chunk header via forward-scanning

TEST(ReaderHardening, ResyncAfterGarbageAndBadLengthKeepsIntactChunksFlowing)
{
    byte_buffer buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();

    lost_record before;
    before.wall_ns          = 1;
    before.tid              = 1;
    before.lost_count_delta = 1;
    fr.write(before);
    fr.flush();

    // Inject a run of garbage bytes containing no 'CH' magic between two
    // otherwise-valid chunks. A reader that only drops one byte per
    // rejected header would eventually get through this too, but must not
    // get stuck, must not treat it as fatal, and must still deliver the
    // valid chunk that follows.
    buf.bytes.insert(buf.bytes.end(), 64, std::byte{ 0xAB });

    lost_record after;
    after.wall_ns          = 2;
    after.tid              = 2;
    after.lost_count_delta = 2;
    fr.write(after);
    fr.flush();

    reader r;
    unsigned lost_seen = 0;
    r.on_event([&](const record_variant& rec) {
        if(std::holds_alternative<lost_record>(rec)) ++lost_seen;
    });
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_FALSE(r.is_fatal());
    EXPECT_EQ(lost_seen, 2U);
    EXPECT_GE(r.counters().resyncs, 1U);
    EXPECT_GE(r.counters().bytes_skipped, 64U);
}

// decode_maps_init resize before bounds check

TEST(ReaderHardening, MapsInitHugeModuleCountIsRejectedNotAllocated)
{
    byte_buffer buf;
    chunk_framer fr(make_framer_cfg(1U << 16), buf.as_sink());
    fr.start();

    // Emit a valid, tiny maps_init (0 modules) via the framer, then doctor
    // the wire n_mods field to a huge count without providing the matching
    // module payload. A vulnerable decoder resizes the modules vector to
    // n_mods entries (each with heap-owned vectors/strings) before ever
    // checking whether the body actually contains that many modules.
    maps_init_record rec;
    rec.wall_ns = 1;
    fr.write(rec);
    fr.flush();

    // Body layout after the 4-byte record header: u64 wall_ns; u32 n_mods; u32 reserved.
    const std::size_t n_mods_off =
        sizeof(file_header) + sizeof(chunk_header) + 4 /*record header*/ + 8 /*wall_ns*/;
    std::uint32_t bogus_n_mods = 100'000'000U;
    std::memcpy(buf.bytes.data() + n_mods_off, &bogus_n_mods, sizeof(bogus_n_mods));

    reader r;
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_GE(r.counters().parse_errors, 1U);
}

// unknown record types must be skipped, not fatal for the chunk

TEST(ReaderHardening, UnknownRecordTypeIsSkippedNotAbandoned)
{
    byte_buffer buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();
    lost_record before;
    before.wall_ns          = 1;
    before.tid              = 1;
    before.lost_count_delta = 1;
    fr.write(before);
    fr.flush();

    // Manually append one more chunk containing: an unknown record type
    // (0x20) with an 8-byte body, followed by a valid LOST record. A
    // compliant reader must skip the unknown record by body_size and still
    // decode the LOST record that follows it in the same chunk.
    std::vector<std::byte> extra_chunk;
    auto push_u16 = [&](std::uint16_t v) {
        std::array<std::byte, 2> b{};
        std::memcpy(b.data(), &v, 2);
        extra_chunk.insert(extra_chunk.end(), b.begin(), b.end());
    };
    auto push_u32 = [&](std::uint32_t v) {
        std::array<std::byte, 4> b{};
        std::memcpy(b.data(), &v, 4);
        extra_chunk.insert(extra_chunk.end(), b.begin(), b.end());
    };
    auto push_u64 = [&](std::uint64_t v) {
        std::array<std::byte, 8> b{};
        std::memcpy(b.data(), &v, 8);
        extra_chunk.insert(extra_chunk.end(), b.begin(), b.end());
    };

    // chunk_header placeholder (filled below).
    extra_chunk.insert(extra_chunk.end(), sizeof(chunk_header), std::byte{ 0 });

    const std::size_t body_start = extra_chunk.size();
    // Unknown record: type=0x20, reserved=0, body_size=8, then 8 bytes body.
    extra_chunk.push_back(std::byte{ 0x20 });
    extra_chunk.push_back(std::byte{ 0 });
    push_u16(8);
    push_u64(0xAAAABBBBCCCCDDDDULL);

    // Valid LOST record after it.
    extra_chunk.push_back(static_cast<std::byte>(record_type::lost));
    extra_chunk.push_back(std::byte{ 0 });
    push_u16(16);
    push_u64(2);   // wall_ns
    push_u32(9);   // tid
    push_u32(9);   // lost_count_delta

    const auto records_byte_count = static_cast<std::uint32_t>(extra_chunk.size() - body_start);

    chunk_header ch{};
    ch.magic              = { 'C', 'H' };
    ch.version            = 0;
    ch.records_byte_count = records_byte_count;
    ch.chunk_seq          = 1;
    std::memcpy(extra_chunk.data(), &ch, sizeof(ch));

    std::vector<std::byte> full = buf.bytes;
    full.insert(full.end(), extra_chunk.begin(), extra_chunk.end());

    reader r;
    unsigned lost_seen = 0;
    r.on_event([&](const record_variant& rec) {
        if(std::holds_alternative<lost_record>(rec)) ++lost_seen;
    });
    r.feed(std::span<const std::byte>(full.data(), full.size()));

    EXPECT_FALSE(r.is_fatal());
    EXPECT_EQ(lost_seen, 2U);
    EXPECT_GE(r.counters().unknown_records, 1U);
}

// trie integrity: dangling parents, idempotent redefinition

namespace
{

new_node_record
make_node(std::uint32_t id, std::uint32_t parent, std::uint64_t ip)
{
    new_node_record nn;
    nn.node_id   = id;
    nn.parent_id = parent;
    nn.ip        = ip;
    return nn;
}

sample_record
make_sample(std::uint32_t leaf)
{
    sample_record s;
    s.timestamp_ns = 1;
    s.tid          = 7;
    s.leaf_node_id = leaf;
    return s;
}

std::span<const std::byte>
as_span(const byte_buffer& buf)
{
    return { buf.bytes.data(), buf.bytes.size() };
}

}  // namespace

TEST(ReaderHardening, NewNodeWithUnknownParentIsRejectedAndChunkContinues)
{
    byte_buffer  buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();
    fr.write(make_node(1, 0, 0x10));
    fr.write(make_node(5, 3, 0x50));
    fr.write(make_sample(1));
    fr.flush();

    reader                     r;
    std::vector<sample_record> samples;
    r.on_event([&](const record_variant& rec) {
        if(const auto* s = std::get_if<sample_record>(&rec)) samples.push_back(*s);
    });
    r.feed(as_span(buf));

    EXPECT_TRUE(r.resolve_stack(5).empty());
    EXPECT_EQ(r.counters().parse_errors, 1U);
    EXPECT_EQ(samples.size(), 1U);
}

TEST(ReaderHardening, IdenticalNewNodeRedefinitionIsAccepted)
{
    byte_buffer  buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();
    fr.write(make_node(1, 0, 0x10));
    fr.write(make_node(2, 1, 0x20));
    fr.write(make_node(1, 0, 0x10));
    fr.write(make_node(2, 1, 0x20));
    fr.write(make_sample(2));
    fr.flush();

    reader                     r;
    std::vector<sample_record> samples;
    r.on_event([&](const record_variant& rec) {
        if(const auto* s = std::get_if<sample_record>(&rec)) samples.push_back(*s);
    });
    r.feed(as_span(buf));

    EXPECT_EQ(r.counters().parse_errors, 0U);
    EXPECT_EQ(samples.size(), 1U);
    EXPECT_EQ(r.resolve_stack(2), (std::vector<std::uint64_t>{ 0x20, 0x10 }));
}

TEST(ReaderHardening, ConflictingNewNodeRedefinitionIsRejected)
{
    byte_buffer  buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();
    fr.write(make_node(1, 0, 0x10));
    fr.write(make_node(1, 0, 0x99));
    fr.flush();

    reader r;
    r.feed(as_span(buf));

    EXPECT_GE(r.counters().parse_errors, 1U);
    EXPECT_EQ(r.resolve_stack(1), (std::vector<std::uint64_t>{ 0x10 }));
}

TEST(ReaderHardening, FileHeaderChunkBytesAboveWriterCapacityIsFatal)
{
    byte_buffer  buf;
    chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
    fr.start();

    const std::uint32_t above_writer_cap = k_default_chunk_bytes + 1;
    std::memcpy(buf.bytes.data() + 24, &above_writer_cap, sizeof(above_writer_cap));

    reader r;
    r.feed(as_span(buf));

    EXPECT_TRUE(r.is_fatal());
}
