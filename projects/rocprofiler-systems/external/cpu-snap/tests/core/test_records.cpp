// Unit tests for the v1 streaming-pipeline wire-format Types in
// cpu_snap/records.hpp and cpu_snap/types.hpp.

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <type_traits>
#include <variant>

#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"

namespace
{

// Repeat the sizeof asserts at translation-unit scope so a regression
// fires inside this test binary too (catches header-vs-test ABI drift).
static_assert(sizeof(cpu_snap::core::new_node_record) == 16);
static_assert(sizeof(cpu_snap::core::sample_record) == 16);
static_assert(sizeof(cpu_snap::core::lost_record) == 16);
static_assert(sizeof(cpu_snap::core::eof_record) == 16);
static_assert(sizeof(cpu_snap::core::per_tid_checkpoint_entry) == 24);
static_assert(sizeof(cpu_snap::core::file_header) == 64);
static_assert(sizeof(cpu_snap::core::chunk_header) == 16);
}  // namespace

TEST(Records, PackedFramingSizes)
{
    EXPECT_EQ(sizeof(cpu_snap::core::file_header), 64U);
    EXPECT_EQ(sizeof(cpu_snap::core::chunk_header), 16U);
}

TEST(Records, FixedRecordSizes)
{
    EXPECT_EQ(sizeof(cpu_snap::core::new_node_record), 16U);
    EXPECT_EQ(sizeof(cpu_snap::core::sample_record), 16U);
    EXPECT_EQ(sizeof(cpu_snap::core::lost_record), 16U);
    EXPECT_EQ(sizeof(cpu_snap::core::eof_record), 16U);
    EXPECT_EQ(sizeof(cpu_snap::core::per_tid_checkpoint_entry), 24U);
}

TEST(Records, VersionConstants)
{
    EXPECT_EQ(cpu_snap::core::k_record_format_version_major, 2U);
    EXPECT_EQ(cpu_snap::core::k_record_format_version_minor, 0U);
}

TEST(Records, TypeOfEachVariantArm)
{
    using namespace cpu_snap::core;

    record_variant v;

    v = new_node_record{};
    EXPECT_EQ(type_of(v), record_type::new_node);

    v = sample_record{};
    EXPECT_EQ(type_of(v), record_type::sample);

    v = checkpoint_record{};
    EXPECT_EQ(type_of(v), record_type::checkpoint);

    v = maps_init_record{};
    EXPECT_EQ(type_of(v), record_type::maps_init);

    v = maps_delta_record{};
    EXPECT_EQ(type_of(v), record_type::maps_delta);

    v = lost_record{};
    EXPECT_EQ(type_of(v), record_type::lost);

    v = eof_record{};
    EXPECT_EQ(type_of(v), record_type::eof);
}

TEST(Records, VisitDispatchesToActiveAlternative)
{
    using namespace cpu_snap::core;

    auto tag_via_visit = [](const record_variant& v) -> record_type {
        return std::visit(
            [](const auto& r) -> record_type {
                using t = std::decay_t<decltype(r)>;
                if constexpr(std::is_same_v<t, new_node_record>) {
                    return record_type::new_node;
                } else if constexpr(std::is_same_v<t, sample_record>) {
                    return record_type::sample;
                } else if constexpr(std::is_same_v<t, checkpoint_record>) {
                    return record_type::checkpoint;
                } else if constexpr(std::is_same_v<t, maps_init_record>) {
                    return record_type::maps_init;
                } else if constexpr(std::is_same_v<t, maps_delta_record>) {
                    return record_type::maps_delta;
                } else if constexpr(std::is_same_v<t, lost_record>) {
                    return record_type::lost;
                } else if constexpr(std::is_same_v<t, eof_record>) {
                    return record_type::eof;
                } else {
                    return record_type::invalid;
}
            },
            v);
    };

    EXPECT_EQ(tag_via_visit(record_variant{ new_node_record{} }), record_type::new_node);
    EXPECT_EQ(tag_via_visit(record_variant{ sample_record{} }), record_type::sample);
    EXPECT_EQ(tag_via_visit(record_variant{ checkpoint_record{} }),
              record_type::checkpoint);
    EXPECT_EQ(tag_via_visit(record_variant{ maps_init_record{} }),
              record_type::maps_init);
    EXPECT_EQ(tag_via_visit(record_variant{ maps_delta_record{} }),
              record_type::maps_delta);
    EXPECT_EQ(tag_via_visit(record_variant{ lost_record{} }), record_type::lost);
    EXPECT_EQ(tag_via_visit(record_variant{ eof_record{} }), record_type::eof);
}

TEST(Records, NewNodeRoundTrip)
{
    cpu_snap::core::new_node_record r{};
    r.node_id   = 7;
    r.parent_id = 3;
    r.ip        = 0xdeadbeefcafe0000ULL;

    EXPECT_EQ(r.node_id, 7U);
    EXPECT_EQ(r.parent_id, 3U);
    EXPECT_EQ(r.ip, 0xdeadbeefcafe0000ULL);
}

TEST(Records, SampleRoundTrip)
{
    cpu_snap::core::sample_record r{};
    r.timestamp_ns = 1'700'000'000'000ULL;
    r.tid          = 1234;
    r.leaf_node_id = 99;

    EXPECT_EQ(r.timestamp_ns, 1'700'000'000'000ULL);
    EXPECT_EQ(r.tid, 1234U);
    EXPECT_EQ(r.leaf_node_id, 99U);
}

TEST(Records, LostRoundTrip)
{
    cpu_snap::core::lost_record r{};
    r.wall_ns          = 42;
    r.tid              = 0;  // aggregate
    r.lost_count_delta = 17;

    EXPECT_EQ(r.wall_ns, 42U);
    EXPECT_EQ(r.tid, 0U);
    EXPECT_EQ(r.lost_count_delta, 17U);
}

TEST(Records, EofRoundTrip)
{
    cpu_snap::core::eof_record r{};
    r.wall_ns               = 99;
    r.total_samples_emitted = 1'000'000;

    EXPECT_EQ(r.wall_ns, 99U);
    EXPECT_EQ(r.total_samples_emitted, 1'000'000U);
}

TEST(Records, CheckpointHoldsEntries)
{
    cpu_snap::core::checkpoint_record c{};
    c.wall_ns = 5;
    c.entries.push_back({ /*tid*/ 10, /*leaf*/ 20, /*samples*/ 30, /*lost*/ 4 });
    c.entries.push_back({ 11, 21, 31, 5 });

    ASSERT_EQ(c.entries.size(), 2U);
    EXPECT_EQ(c.entries[0].tid, 10U);
    EXPECT_EQ(c.entries[0].leaf_id, 20U);
    EXPECT_EQ(c.entries[0].samples, 30U);
    EXPECT_EQ(c.entries[0].lost, 4U);
    EXPECT_EQ(c.entries[1].tid, 11U);
}

TEST(Records, ModuleInfoRoundTrip)
{
    cpu_snap::core::module_info m{};
    m.base_va     = 0x400000;
    m.size        = 0x10000;
    m.file_offset = 0;
    m.build_id    = { 0x01, 0x02, 0x03, 0x04 };
    m.path        = "/lib/x86_64-linux-gnu/libc.so.6";

    EXPECT_EQ(m.base_va, 0x400000U);
    EXPECT_EQ(m.size, 0x10000U);
    EXPECT_EQ(m.file_offset, 0U);
    ASSERT_EQ(m.build_id.size(), 4U);
    EXPECT_EQ(m.build_id[0], 0x01U);
    EXPECT_EQ(m.path, "/lib/x86_64-linux-gnu/libc.so.6");
}

TEST(Records, MapsInitHoldsModules)
{
    cpu_snap::core::maps_init_record r{};
    r.wall_ns = 1;
    r.modules.push_back({ 0x1000, 0x100, 0, {}, "/a" });
    r.modules.push_back({ 0x2000, 0x200, 0, {}, "/b" });

    ASSERT_EQ(r.modules.size(), 2U);
    EXPECT_EQ(r.modules[0].path, "/a");
    EXPECT_EQ(r.modules[1].base_va, 0x2000U);
}

TEST(Records, MapsDeltaCarriesOpAndModule)
{
    cpu_snap::core::maps_delta_record r{};
    r.wall_ns        = 7;
    r.operation      = cpu_snap::core::maps_delta_record::op::remove;
    r.module.base_va = 0xff00;
    r.module.size    = 0x10;
    r.module.path    = "/tmp/jit.so";

    EXPECT_EQ(r.wall_ns, 7U);
    EXPECT_EQ(r.operation, cpu_snap::core::maps_delta_record::op::remove);
    EXPECT_EQ(r.module.base_va, 0xff00U);
    EXPECT_EQ(r.module.path, "/tmp/jit.so");
}

TEST(Records, PackedFileHeaderFieldLayout)
{
    cpu_snap::core::file_header h{};
    h.magic         = { 'C', 'S', 'N', 'P' };
    h.version_major = cpu_snap::core::k_record_format_version_major;
    h.version_minor = cpu_snap::core::k_record_format_version_minor;
    h.start_unix_ns = 0x1122334455667788ULL;
    h.pid           = 4242;
    h.chunk_bytes   = 1U << 20;
    h.flags         = 0x1;  // reserved bit
    h.process_name  = { 'd', 'e', 'm', 'o' };

    // Verify packed offsets match the wire spec.
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, magic), 0U);
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, version_major), 4U);
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, version_minor), 6U);
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, start_unix_ns), 8U);
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, pid), 16U);
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, chunk_bytes), 24U);
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, flags), 28U);
    EXPECT_EQ(offsetof(cpu_snap::core::file_header, process_name), 32U);

    EXPECT_EQ(h.magic[0], 'C');
    EXPECT_EQ(h.version_major, 2U);
}

TEST(Records, PackedChunkHeaderFieldLayout)
{
    EXPECT_EQ(offsetof(cpu_snap::core::chunk_header, magic), 0U);
    EXPECT_EQ(offsetof(cpu_snap::core::chunk_header, version), 2U);
    EXPECT_EQ(offsetof(cpu_snap::core::chunk_header, records_byte_count), 4U);
    EXPECT_EQ(offsetof(cpu_snap::core::chunk_header, chunk_seq), 8U);
}

TEST(Types, SinkFnAcceptsByteSpan)
{
    std::size_t       total_bytes = 0;
    cpu_snap::core::sink_fn const sink        = [&](std::span<const std::byte> bytes) {
        total_bytes += bytes.size();
    };

    std::byte buf[16]{};
    sink(std::span<const std::byte>{ buf, 16 });
    sink(std::span<const std::byte>{ buf, 4 });
    EXPECT_EQ(total_bytes, 20U);
}

TEST(Types, EventFnDispatchesRecords)
{
    int                n_samples = 0;
    int                n_other   = 0;
    cpu_snap::core::event_fn const cb        = [&](const cpu_snap::core::record_variant& v) {
        if(cpu_snap::core::type_of(v) == cpu_snap::core::record_type::sample) {
            ++n_samples;
        } else {
            ++n_other;
}
    };

    cpu_snap::core::record_variant const a{ cpu_snap::core::sample_record{} };
    cpu_snap::core::record_variant const b{ cpu_snap::core::sample_record{} };
    cpu_snap::core::record_variant const c{ cpu_snap::core::new_node_record{} };
    cb(a);
    cb(b);
    cb(c);
    EXPECT_EQ(n_samples, 2);
    EXPECT_EQ(n_other, 1);
}
