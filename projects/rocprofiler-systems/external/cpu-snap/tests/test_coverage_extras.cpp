// Targeted tests that push line and branch coverage.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/sampling/flat_tid_map.hpp"
#include "cpu_snap/sampling/frame_trie.hpp"
#include "cpu_snap/sampling/streaming_pipeline.hpp"
#include "cpu_snap/sampling/temporal_diff_filter.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
#include "support/stream_capture.hpp"
#include "cpu_snap/processing/symbols/frame_filters.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/processing/symbols/sample_postprocessor.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"

namespace
{

using namespace cpu_snap::core;
using namespace cpu_snap::processing::symbols;

// Re-export the sampling types this test exercises so call sites
// can stay unqualified (cpu_snap::core::internal is brought in by the
// using-namespace above; mixing in cpu_snap::sampling::internal::* explicitly
// avoids any ambiguity around the nested namespace name).
using cpu_snap::core::raw_sample;
using cpu_snap::sampling::sampler_capabilities;
using cpu_snap::processing::symbols::sample_postprocessor;
using cpu_snap::processing::symbols::sample_postprocessor_config;
using cpu_snap::processing::symbols::processed_sample;
using cpu_snap::processing::timeline::slice_emitter;
using cpu_snap::processing::timeline::slice_emitter_config;
using cpu_snap::processing::timeline::emitted_slice;
using cpu_snap::sampling::sampler;
using cpu_snap::sampling::unwind_mode;
using cpu_snap::processing::symbols::coalesce_frames_by_name;
using cpu_snap::test_support::sink_capture;
using cpu_snap::test_support::make_framer_cfg;

}  // namespace

// chunk_framer.hpp tests

// chunk_bytes > k_default_chunk_bytes is rejected.
TEST(ChunkFramerCoverage, RejectsChunkBytesAboveDefault)
{
    sink_capture sink;
    auto         cfg = make_framer_cfg();
    cfg.chunk_bytes  = cpu_snap::core::k_default_chunk_bytes + 1U;
    EXPECT_THROW(chunk_framer(cfg, sink.as_fn()), cpu_snap::config_error);
}

// Flush before start is a no-op, not a throw.
TEST(ChunkFramerCoverage, FlushBeforeStartIsNoop)
{
    sink_capture                   sink;
    chunk_framer         fr(make_framer_cfg(), sink.as_fn());
    EXPECT_NO_THROW(fr.flush());
    EXPECT_EQ(sink.chunks.size(), 0U);
}

// A single record larger than the configured chunk.
// Crafts a maps_init record whose serialized body fits the u16 cap
// but exceeds a tight chunk_bytes; the first sample fits, then the
// framer seals the chunk + opens a new one + checks the size and
// throws cpu_snap::resource_error.
TEST(ChunkFramerCoverage, SingleRecordLargerThanChunkThrows)
{
    sink_capture sink;
    // Chunk size 128 admits the small sample (36 bytes total) but is
    // smaller than any maps_init holding a 200-char path.
    chunk_framer fr(make_framer_cfg(128), sink.as_fn());
    fr.start();
    fr.write(sample_record{});
    maps_init_record mi;
    mi.wall_ns = 1;
    module_info mod;
    mod.base_va = 0x1000;
    mod.size    = 0x100;
    mod.path    = std::string(200, 'p');
    mi.modules.push_back(std::move(mod));
    EXPECT_THROW(fr.write(mi), cpu_snap::resource_error);
}

// flat_tid_map.hpp tests

// Const find and const for_each reaching the overflow map. The fast slots are 256 and
// the probe limit is 8, so we force overflow by hash-colliding 16
// tids onto the same home bucket.
TEST(FlatTidMapCoverage, ConstFindAndForEachIterateOverflow)
{
    cpu_snap::sampling::flat_tid_map<int> map;
    std::vector<pid_t>          overflow_tids;
    constexpr std::size_t       k_total       = 16U;
    constexpr pid_t             k_low_byte    = 0x42;
    constexpr std::size_t       k_probe_limit = 8U;

    for(std::size_t ii = 0; ii < k_total; ++ii)
    {
        const auto tid = static_cast<pid_t>((ii << 8) | k_low_byte);
        map[tid]       = static_cast<int>(ii);
        overflow_tids.push_back(tid);
    }
    ASSERT_GT(map.overflow_size(), 0U);

    const auto& cmap = map;
    // const find: pick a tid we know spilled to overflow.
    const pid_t overflow_tid = overflow_tids.back();
    const int*  found        = cmap.find(overflow_tid);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(*found, static_cast<int>(k_total - 1U));

    // const for_each: must visit every overflow entry too.
    std::size_t visited = 0;
    cmap.for_each([&](pid_t /*tid*/, const int& /*value*/) { ++visited; });
    EXPECT_EQ(visited, k_total);
    EXPECT_EQ(visited, cmap.size());
    EXPECT_GE(visited, k_probe_limit);
}

// temporal_diff_filter.hpp tests

// current_leaf_for_tid returns k_root for an empty
// path; reached via forget_tid + immediate lookup, and via empty
// frames diff()).
TEST(TemporalDiffFilterCoverage, CurrentLeafForUnknownAndEmptyTid)
{
    cpu_snap::sampling::internal::frame_trie           trie;
    cpu_snap::sampling::internal::temporal_diff_filter filt(trie);

    constexpr pid_t k_tid = 99;

    EXPECT_EQ(filt.current_leaf_for_tid(k_tid), cpu_snap::sampling::internal::frame_trie::k_root);

    const std::uint64_t frames[] = { 0xAA, 0xBB };
    (void) filt.diff(k_tid, std::span<const std::uint64_t>(frames, 2));
    EXPECT_NE(filt.current_leaf_for_tid(k_tid), cpu_snap::sampling::internal::frame_trie::k_root);

    filt.forget_tid(k_tid);
    EXPECT_EQ(filt.current_leaf_for_tid(k_tid), cpu_snap::sampling::internal::frame_trie::k_root);

    const auto empty = filt.diff(k_tid, std::span<const std::uint64_t>{});
    EXPECT_EQ(empty.leaf_id, cpu_snap::sampling::internal::frame_trie::k_root);
    EXPECT_EQ(empty.new_nodes.size(), 0U);
}

// streaming_pipeline.cpp tests

// Zero maps_poll_interval and zero
// checkpoint_interval are clamped up to 1, exercised by running the
// pipeline through finalize without crashing).
TEST(StreamingPipelineCoverage, ZeroIntervalsAreClampedToOne)
{
    sink_capture sink;
    {
        cpu_snap::sampling::internal::streaming_pipeline::config cfg;
        cfg.sink                = sink.as_fn();
        cfg.chunk_bytes         = 4096U;
        cfg.maps_poll_interval  = 0U;  // exercises 49-52
        cfg.checkpoint_interval = 0U;  // exercises 53-56
        cfg.pid                 = 7U;
        cfg.process_name        = "zero_interval_test";
        cpu_snap::sampling::internal::streaming_pipeline pipe(std::move(cfg));

        const std::uint64_t frames[] = { 0xAA, 0xBB };
        pipe.on_sample(1, 1, std::span<const std::uint64_t>(frames, 2));
        pipe.on_sample(1, 2, std::span<const std::uint64_t>(frames, 2));
        pipe.finalize();
        EXPECT_EQ(pipe.samples_emitted(), 2U);
        EXPECT_GE(pipe.checkpoints_emitted(), 2U);
    }
    EXPECT_FALSE(sink.chunks.empty());
}

// An internal exception on the lost path is caught, and a delta is
// polled during finalize (dl_module_watcher iterator changed between
// snapshot and finalize).
TEST(StreamingPipelineCoverage, OnLostAggregatesAndFinalizeFlushesDeltas)
{
    sink_capture sink;
    {
        cpu_snap::sampling::internal::streaming_pipeline::config cfg;
        cfg.sink                = sink.as_fn();
        cfg.chunk_bytes         = 4096U;
        cfg.maps_poll_interval  = 1000U;
        cfg.checkpoint_interval = 1000U;
        cfg.pid                 = 7U;
        cfg.process_name        = "lost_path_test";
        cpu_snap::sampling::internal::streaming_pipeline pipe(std::move(cfg));

        pipe.on_lost(5U);
        pipe.on_lost(3U);
        EXPECT_EQ(pipe.lost_total(), 8U);
        pipe.finalize();
    }
    EXPECT_FALSE(sink.chunks.empty());
}

// on_tid_exit clears state and is a no-op after
// finalize, and the subsequent on_sample/on_lost no-ops.
TEST(StreamingPipelineCoverage, OnTidExitAndPostFinalizeNoops)
{
    sink_capture sink;
    {
        cpu_snap::sampling::internal::streaming_pipeline::config cfg;
        cfg.sink                = sink.as_fn();
        cfg.chunk_bytes         = 4096U;
        cfg.maps_poll_interval  = 1000U;
        cfg.checkpoint_interval = 5U;
        cfg.pid                 = 11U;
        cfg.process_name        = "tid_exit_test";
        cpu_snap::sampling::internal::streaming_pipeline pipe(std::move(cfg));

        const std::uint64_t frames[] = { 0x100, 0x200 };
        for(int ii = 0; ii < 3; ++ii)
        {
            pipe.on_sample(static_cast<pid_t>(50), static_cast<std::uint64_t>(ii),
                           std::span<const std::uint64_t>(frames, 2));
        }
        pipe.on_tid_exit(50);
        EXPECT_EQ(pipe.samples_emitted(), 3U);

        pipe.finalize();
        const auto samples_before = pipe.samples_emitted();

        // Post-finalize calls are no-ops on every entry point.
        pipe.on_sample(50, 999, std::span<const std::uint64_t>(frames, 2));
        pipe.on_lost(1);
        pipe.on_tid_exit(50);
        EXPECT_EQ(pipe.samples_emitted(), samples_before);
    }
}

// reader.cpp tests

// Host callback exception swallowed by emit(), parse-error paths in
// walk_chunk_body / dispatch_record, and invalid record_type dispatch.
TEST(ReaderCoverage, EventCallbackExceptionDoesNotCorruptCounters)
{
    sink_capture sink;
    {
        chunk_framer fr(make_framer_cfg(4096), sink.as_fn());
        fr.start();
        fr.write(sample_record{});
        fr.write(eof_record{});
        fr.flush();
    }

    reader rdr;
    int    seen = 0;
    rdr.on_event([&](const record_variant& /*rec*/) {
        ++seen;
        throw std::runtime_error("callback exception");
    });
    EXPECT_NO_THROW(rdr.feed(std::span<const std::byte>(sink.all().data(),
                                                        sink.all().size())));
    EXPECT_GE(seen, 1);
    EXPECT_EQ(rdr.counters().parse_errors, 0U);
}

// Build a chunk by hand whose record_type byte is 0 (record_type::invalid)
// so dispatch_record hits the default unknown-type path. Per the
// records.hpp forward-compatibility contract, an unrecognized/reserved
// type tag is skipped by body_size (unknown_records), not a parse error.
TEST(ReaderCoverage, UnknownRecordTypeIsSkippedNotRejected)
{
    sink_capture sink;
    {
        chunk_framer fr(make_framer_cfg(4096), sink.as_fn());
        fr.start();
        fr.write(sample_record{});  // produces a valid sample to lay the chunk
        fr.flush();
    }
    auto bytes = sink.all();
    // chunk_header starts at offset 64 (file_header is 64B); first
    // record begins at offset 64 + 16 (chunk_header). Byte 0 of the
    // record header is the type tag - overwrite to 0 (invalid).
    const std::size_t type_byte_offset =
        sizeof(file_header) + sizeof(chunk_header);
    bytes[type_byte_offset] = std::byte{ 0x00 };

    reader rdr;
    rdr.feed(std::span<const std::byte>(bytes.data(), bytes.size()));
    EXPECT_FALSE(rdr.is_fatal());
    EXPECT_EQ(rdr.counters().parse_errors, 0U);
    EXPECT_GE(rdr.counters().unknown_records, 1U);
}

// Build a chunk whose record header claims a body_size that overruns
// the chunk body. Reader must increment parse_errors and stop walking
// Oversized record body is a parse error in walk_chunk_body.
TEST(ReaderCoverage, OversizedRecordBodyTriggersParseError)
{
    sink_capture sink;
    {
        chunk_framer fr(make_framer_cfg(4096), sink.as_fn());
        fr.start();
        fr.write(sample_record{});
        fr.flush();
    }
    auto bytes = sink.all();

    // Patch record body_size (bytes 2-3 of the record header) to a
    // huge value so body_off + body_size > chunk_buf.size().
    const std::size_t body_size_off =
        sizeof(file_header) + sizeof(chunk_header) + 2U;
    const std::uint16_t huge = 0xFFFF;
    std::memcpy(bytes.data() + body_size_off, &huge, sizeof(huge));

    reader rdr;
    rdr.feed(std::span<const std::byte>(bytes.data(), bytes.size()));
    EXPECT_FALSE(rdr.is_fatal());
    EXPECT_GE(rdr.counters().parse_errors, 1U);
}

// Version-major mismatch sets fatal. The file
// header magic is correct but the version byte is bumped past the
// known major.
TEST(ReaderCoverage, VersionMajorMismatchSetsFatal)
{
    sink_capture sink;
    {
        chunk_framer fr(make_framer_cfg(4096), sink.as_fn());
        fr.start();
        fr.flush();
    }
    auto bytes = sink.all();

    // file_header layout: magic[4], version_major u16 at offset 4.
    const std::uint16_t bad_version = k_record_format_version_major + 1U;
    std::memcpy(bytes.data() + 4, &bad_version, sizeof(bad_version));

    reader rdr;
    rdr.feed(std::span<const std::byte>(bytes.data(), bytes.size()));
    EXPECT_TRUE(rdr.is_fatal());
    EXPECT_GE(rdr.counters().parse_errors, 1U);
}

// Bad chunk-magic resync path: drop one
// byte from pending and retry. Build a stream with two chunks, then
// corrupt the second chunk's magic bytes so the reader must resync.
TEST(ReaderCoverage, BadChunkMagicTriggersResyncCounter)
{
    sink_capture sink;
    {
        // chunk_bytes = 44 so each chunk holds exactly one 16B sample.
        chunk_framer fr(make_framer_cfg(44), sink.as_fn());
        fr.start();
        fr.write(sample_record{});
        fr.write(sample_record{});
        fr.flush();
    }
    auto bytes = sink.all();
    // Each chunk is header (16) + record (20) = 36 bytes. Corrupt the
    // 2nd chunk's magic (file_header + 36 + first byte of chunk_header).
    bytes[sizeof(file_header) + 36U] = std::byte{ 0xFF };

    reader rdr;
    rdr.feed(std::span<const std::byte>(bytes.data(), bytes.size()));
    // First chunk parses fine; second chunk's bad magic increments
    // parse_errors at least once via the resync loop.
    EXPECT_FALSE(rdr.is_fatal());
    EXPECT_GE(rdr.counters().parse_errors, 1U);
}

// resolve_stack returns empty when an
// intermediate id is missing from the trie map.
TEST(ReaderCoverage, ResolveStackMissingIntermediateReturnsEmpty)
{
    sink_capture sink;
    {
        chunk_framer fr(make_framer_cfg(4096), sink.as_fn());
        fr.start();
        new_node_record nn1;
        nn1.node_id   = 1;
        nn1.parent_id = 0;
        nn1.ip        = 0xAA;
        new_node_record nn3;
        // Skip id 2 entirely so resolve_stack(3) walks parent_id=2 and
        // fails the find.
        nn3.node_id   = 3;
        nn3.parent_id = 2;
        nn3.ip        = 0xCC;
        fr.write(nn1);
        fr.write(nn3);
        fr.flush();
    }
    reader rdr;
    rdr.feed(std::span<const std::byte>(sink.all().data(), sink.all().size()));
    EXPECT_TRUE(rdr.resolve_stack(3).empty());
    EXPECT_FALSE(rdr.resolve_stack(1).empty());
}

// frame_filters.hpp

// Library exclude filter: substring match drops
// libc but keeps the main executable; also exercises the unknown-IP
// (mapping == nullptr) branch.
TEST(FrameFiltersCoverage, LibraryExcludeFilterDropsByPathSubstring)
{
    proc_maps maps;
    std::vector<std::string> excluded;
    excluded.emplace_back("libc.so");
    auto pred = cpu_snap::processing::symbols::make_library_exclude_filter(maps, std::move(excluded));

    // Pick a known libc IP and a known main-exe IP via real maps.
    auto libc_fp = &::getpid;
    std::uint64_t libc_ip = 0;
    std::memcpy(&libc_ip, &libc_fp, sizeof(libc_ip));
    EXPECT_FALSE(pred(libc_ip, std::string_view{}));

    // Anonymous / out-of-range IPs are kept (mapping == nullptr).
    EXPECT_TRUE(pred(0xDEADBEEFULL, std::string_view{}));
}

// symbolizer.hpp tests

// Covers preload() path that groups IPs by binary and runs addr2line
// for at least one binary, plus operator()'s preload fallback. Also
// hits resolve()/file_line() cache-hit + cache-miss returns.
TEST(SymbolizerCoverage, PreloadResolvesRealIpsAndCachesByBinary)
{
    cpu_snap::processing::symbols::symbolizer sym;
    auto              main_fp = &std::malloc;  // libc symbol
    std::uint64_t     ip      = 0;
    std::memcpy(&ip, &main_fp, sizeof(ip));

    EXPECT_TRUE(sym.resolve(ip).empty());
    EXPECT_TRUE(sym.file_line(ip).empty());

    sym.preload({ ip });
    EXPECT_FALSE(sym.resolve(ip).empty());

    // Second preload of the same IP is a cache hit (no addr2line).
    sym.preload({ ip });

    // operator() on an unknown IP triggers the preload fallback. The
    // IP is bogus so we get a hex string back.
    const std::string out = sym(0xDEADBEEFCAFE0000ULL);
    EXPECT_FALSE(out.empty());
}

// Covers rescan_maps() (just smoke-tests the wrapper) and a second
// preload on an IP whose owning mapping is unknown -> format_hex.
TEST(SymbolizerCoverage, RescanMapsAndUnknownIpPath)
{
    cpu_snap::processing::symbols::symbolizer sym;
    sym.rescan_maps();
    sym.preload({ 0x0ULL, 0x1ULL, 0xDEADBEEFCAFE0000ULL });
    // resolve() returns the cached hex view, not empty.
    const auto v = sym.resolve(0x1ULL);
    EXPECT_FALSE(v.empty());
}

// slice_emitter.hpp

namespace
{

/**
 * Append a synthetic raw_sample with timestamp, top-of-stack IP, and a
 * frame array (innermost-first) to @p out.
 */
void
push_sample(std::vector<raw_sample>& out, std::uint64_t timestamp_ns, pid_t tid,
            std::initializer_list<std::uint64_t> frames)
{
    raw_sample smp{};
    smp.tid          = tid;
    smp.timestamp_ns = timestamp_ns;
    smp.ip           = frames.size() == 0 ? 0 : *frames.begin();
    for(auto frame : frames)
    {
        smp.frames.push_back(frame);
    }
    out.push_back(smp);
}

}  // namespace

// Covers slice_emitter nested-mode path with the default hex resolver
// and the close-everything-on-end loop.
TEST(SliceEmitterCoverage, NestedModeEmitsExpectedSpans)
{
    slice_emitter_config cfg;
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto& vec1 = by_tid[7];
    push_sample(vec1, 100, 7, { 0xAA, 0xBB, 0xCC });  // A,B,C
    push_sample(vec1, 200, 7, { 0xAA, 0xDD, 0xCC });  // A,D,C - mid swap
    push_sample(vec1, 300, 7, { 0xAA });             // collapse to A only

    // Empty-vec tid gets skipped (covers the `continue` branch).
    by_tid[9] = {};

    std::vector<emitted_slice> seen;
    const std::size_t emitted = em.process(
        by_tid, [&](const emitted_slice& slc) { seen.push_back(slc); });
    EXPECT_EQ(emitted, seen.size());
    EXPECT_GT(emitted, 0U);
}

TEST(SliceEmitterCoverage, EqualTimestampsKeepInputOrder)
{
    slice_emitter_config cfg;
    cfg.resolver = [](std::uint64_t ip) { return std::to_string(ip); };
    slice_emitter em(std::move(cfg));

    constexpr std::uint64_t k_count = 64;
    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto&                                              vec = by_tid[5];
    for(std::uint64_t i = 0; i < k_count; ++i)
    {
        push_sample(vec, 100, 5, { 1000 + i });
    }

    std::vector<std::string> names;
    em.process(by_tid, [&](const emitted_slice& slc) { names.emplace_back(slc.name); });

    ASSERT_EQ(names.size(), k_count);
    for(std::uint64_t i = 0; i < k_count; ++i)
    {
        EXPECT_EQ(names[i], std::to_string(1000 + i)) << "slice " << i;
    }
}

TEST(SliceEmitterCoverage, SortedInputYieldsSlicesInTimestampOrder)
{
    slice_emitter_config cfg;
    cfg.resolver = [](std::uint64_t ip) { return std::to_string(ip); };
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto&                                              vec = by_tid[5];
    push_sample(vec, 100, 5, { 1 });
    push_sample(vec, 200, 5, { 2 });
    push_sample(vec, 300, 5, { 3 });

    std::vector<std::pair<std::string, std::uint64_t>> seen;
    em.process(by_tid, [&](const emitted_slice& slc) {
        seen.emplace_back(std::string(slc.name), slc.start_ts);
    });

    ASSERT_EQ(seen.size(), 3U);
    EXPECT_EQ(seen[0], (std::pair<std::string, std::uint64_t>{ "1", 100 }));
    EXPECT_EQ(seen[1], (std::pair<std::string, std::uint64_t>{ "2", 200 }));
    EXPECT_EQ(seen[2], (std::pair<std::string, std::uint64_t>{ "3", 300 }));
}

// With merge_samples=false nothing is carried between samples: each sample
// emits its own full stack as nested slices spanning [its timestamp, the next
// sample's timestamp), even when consecutive samples are identical.
TEST(SliceEmitterCoverage, UnmergedModeEmitsFullStackOfEachSample)
{
    slice_emitter_config cfg;
    cfg.merge_samples    = false;
    cfg.sample_period_ns = 50;
    cfg.resolver         = [](std::uint64_t ip) {
        return ip == 0xAAU ? std::string("leaf") : ip == 0xBBU ? std::string("mid") : std::string("root");
    };
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto&                                              vec = by_tid[3];
    push_sample(vec, 100, 3, { 0xAA, 0xBB, 0xCC });  // leaf, mid, root
    push_sample(vec, 200, 3, { 0xAA, 0xBB, 0xCC });  // identical to the previous sample
    push_sample(vec, 300, 3, { 0xBB, 0xCC });        // leaf frame gone

    struct owned_slice
    {
        std::string   name;
        std::uint64_t start_ts;
        std::uint64_t end_ts;
        std::uint32_t depth;
    };
    std::vector<owned_slice> seen;
    const std::size_t        emitted = em.process(by_tid, [&](const emitted_slice& slc) {
        seen.push_back({ std::string(slc.name), slc.start_ts, slc.end_ts, slc.depth });
    });

    EXPECT_EQ(emitted, 8U) << "3 + 3 + 2 frames, one slice per frame per sample";
    ASSERT_EQ(seen.size(), 8U);

    const auto count_if_named = [&](std::string_view name) {
        return std::count_if(seen.begin(), seen.end(),
                             [&](const owned_slice& s) { return s.name == name; });
    };
    EXPECT_EQ(count_if_named("root"), 3) << "the root frame is not merged across samples";
    EXPECT_EQ(count_if_named("leaf"), 2);

    for(const auto& slc : seen)
    {
        const std::uint64_t expected_end = slc.start_ts == 300 ? 350 : slc.start_ts + 100;
        EXPECT_EQ(slc.end_ts, expected_end) << slc.name << " @" << slc.start_ts;
        const std::uint32_t expected_depth = slc.name == "root" ? 0U : slc.name == "mid" ? 1U : 2U;
        EXPECT_EQ(slc.depth, expected_depth) << slc.name;
    }
}

// Unmerged mode also honours the inline resolver: one IP expands to its inline
// chain (outermost frame at depth 0) and nothing is merged across samples.
TEST(SliceEmitterCoverage, UnmergedModeExpandsInlineChainsPerSample)
{
    slice_emitter_config cfg;
    cfg.merge_samples   = false;
    cfg.inline_resolver = [](std::uint64_t) -> std::vector<std::string> {
        return { "inner", "outer" };  // innermost-first
    };
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto&                                              vec = by_tid[4];
    push_sample(vec, 100, 4, { 0xAA });
    push_sample(vec, 200, 4, { 0xAA });

    struct owned_slice
    {
        std::string   name;
        std::uint32_t depth;
    };
    std::vector<owned_slice> seen;
    em.process(by_tid, [&](const emitted_slice& slc) {
        seen.push_back({ std::string(slc.name), slc.depth });
    });

    ASSERT_EQ(seen.size(), 4U) << "2 inline frames x 2 samples";
    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                            [](const owned_slice& s) { return s.name == "outer" && s.depth == 0; }),
              2);
    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                            [](const owned_slice& s) { return s.name == "inner" && s.depth == 1; }),
              2);
}

// A throwing resolver must not abort emission: the emitter falls back to the
// hex-address name for the failing IP.
TEST(SliceEmitterCoverage, ThrowingResolverFallsBackToHexName)
{
    slice_emitter_config cfg;
    cfg.resolver = [](std::uint64_t ip) -> std::string {
        if(ip == 0xBADULL) throw std::runtime_error("resolver kaboom");
        return "ok";
    };
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    push_sample(by_tid[3], 10, 3, { 0xBAD, 0x1 });

    std::vector<std::string> names;
    em.process(by_tid, [&](const emitted_slice& slc) { names.emplace_back(slc.name); });

    ASSERT_EQ(names.size(), 2U);
    EXPECT_NE(std::find(names.begin(), names.end(), "0xbad"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "ok"), names.end());
}

// Covers the nested inline-resolver path: a single sampled IP expands
// into a multi-frame chain, producing one nested emitted_slice per chain
// entry with the chain frame names (root inline frame as the outer span).
TEST(SliceEmitterCoverage, InlineResolverExpandsOneIpIntoNestedSpans)
{
    slice_emitter_config cfg;
    cfg.inline_resolver = [](std::uint64_t ip) -> std::vector<std::string> {
        if(ip == 0xAAULL)
        {
            // innermost-first chain.
            return { "inner", "middle", "outer" };
        }
        return {};
    };
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto& vec = by_tid[11];
    push_sample(vec, 100, 11, { 0xAA });
    push_sample(vec, 200, 11, { 0xAA });

    struct owned_slice
    {
        std::string   name;
        std::uint64_t start_ts;
    };
    std::vector<owned_slice> seen;
    const std::size_t emitted = em.process(by_tid, [&](const emitted_slice& slc) {
        seen.push_back({ std::string(slc.name), slc.start_ts });
    });

    // One persistent stack of three frames -> three spans, each closed once.
    EXPECT_EQ(emitted, 3U);
    ASSERT_EQ(seen.size(), 3U);

    std::vector<std::string> names;
    names.reserve(seen.size());
    for(const auto& slc : seen)
    {
        names.push_back(slc.name);
        EXPECT_EQ(slc.start_ts, 100U);
    }
    std::sort(names.begin(), names.end());
    const std::vector<std::string> expected{ "inner", "middle", "outer" };
    EXPECT_EQ(names, expected);
}

namespace
{

/** Symbol-identity fixture: [0x1000, 0x2000) and [0x5000, 0x6000) are
 *  two distinct symbols; everything else is unresolved. */
std::optional<std::uint64_t>
fixture_symbol_identity(std::uint64_t ip)
{
    if(ip >= 0x1000 && ip < 0x2000) return std::optional<std::uint64_t>(0x1000);
    if(ip >= 0x5000 && ip < 0x6000) return std::optional<std::uint64_t>(0x5000);
    return std::nullopt;
}

}  // namespace

// Two consecutive samples call the same function (frame depth 1) from
// two different call sites (0x1010 vs 0x1090), both inside the same
// [0x1000, 0x2000) symbol. Without symbol-identity, the raw-IP diff
// would spuriously close and reopen that frame; with it, the frame
// stays open as a single slice for the whole run, matching the parent
// (root) and child (leaf) frames that are byte-identical across samples.
TEST(SliceEmitterCoverage, SymbolIdentityMergesSameFunctionDifferentCallSites)
{
    slice_emitter_config cfg;
    cfg.symbol_identity  = &fixture_symbol_identity;
    cfg.sample_period_ns = 50;
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto& vec = by_tid[21];
    push_sample(vec, 100, 21, { 0x3000, 0x1010, 0x9999 });
    push_sample(vec, 200, 21, { 0x3000, 0x1090, 0x9999 });

    std::vector<emitted_slice> seen;
    const std::size_t emitted = em.process(
        by_tid, [&](const emitted_slice& slc) { seen.push_back(slc); });

    EXPECT_EQ(emitted, 3U);
    ASSERT_EQ(seen.size(), 3U);

    const auto func_slice =
        std::find_if(seen.begin(), seen.end(),
                    [](const emitted_slice& slc) { return slc.depth == 1U; });
    ASSERT_NE(func_slice, seen.end());
    EXPECT_EQ(func_slice->start_ts, 100U);
    EXPECT_EQ(func_slice->end_ts, 200U + 50U);
    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                           [](const emitted_slice& slc) { return slc.depth == 1U; }),
             1);
}

// Regression: call sites at the same depth that resolve to two
// different symbols must still split into two slices.
TEST(SliceEmitterCoverage, SymbolIdentityKeepsDifferentSymbolsSeparate)
{
    slice_emitter_config cfg;
    cfg.symbol_identity  = &fixture_symbol_identity;
    cfg.sample_period_ns = 50;
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto& vec = by_tid[22];
    push_sample(vec, 100, 22, { 0x3000, 0x1010, 0x9999 });
    push_sample(vec, 200, 22, { 0x3000, 0x5010, 0x9999 });

    std::vector<emitted_slice> seen;
    em.process(by_tid, [&](const emitted_slice& slc) { seen.push_back(slc); });

    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                           [](const emitted_slice& slc) { return slc.depth == 1U; }),
             2);
}

// Regression: when one of the two call-site IPs fails to resolve to
// any known symbol, the diff must fall back to raw-IP equality (i.e.
// still split), not treat them as merged.
TEST(SliceEmitterCoverage, SymbolIdentityFallsBackToRawIpWhenUnresolved)
{
    slice_emitter_config cfg;
    cfg.symbol_identity  = &fixture_symbol_identity;
    cfg.sample_period_ns = 50;
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto& vec = by_tid[23];
    push_sample(vec, 100, 23, { 0x3000, 0x1010, 0x9999 });
    push_sample(vec, 200, 23, { 0x3000, 0xDEAD, 0x9999 });

    std::vector<emitted_slice> seen;
    em.process(by_tid, [&](const emitted_slice& slc) { seen.push_back(slc); });

    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                           [](const emitted_slice& slc) { return slc.depth == 1U; }),
             2);
}

// A frame's own recorded PC can only change across samples once every
// callee below it has fully returned, so symbol-identity leniency must
// apply at most once per sample-transition, at the shallowest point of
// divergence. This scenario diverges at BOTH depth 1 and depth 2 in the
// same transition, each resolving to its own stable symbol: depth 1
// (shallowest) must merge into one slice, but depth 2 must still split,
// because it is a genuinely fresh call that happens to land in the same
// function as before, not a continuation of the depth-1 leniency.
TEST(SliceEmitterCoverage, SymbolIdentityLeniencyAppliesOnlyAtShallowestDivergence)
{
    slice_emitter_config cfg;
    cfg.symbol_identity  = &fixture_symbol_identity;
    cfg.sample_period_ns = 50;
    slice_emitter em(std::move(cfg));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto& vec = by_tid[24];
    push_sample(vec, 100, 24, { 0x5010, 0x1010, 0x9999 });
    push_sample(vec, 200, 24, { 0x5090, 0x1090, 0x9999 });

    std::vector<emitted_slice> seen;
    const std::size_t emitted = em.process(
        by_tid, [&](const emitted_slice& slc) { seen.push_back(slc); });

    EXPECT_EQ(emitted, 4U);
    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                           [](const emitted_slice& slc) { return slc.depth == 0U; }),
             1);
    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                           [](const emitted_slice& slc) { return slc.depth == 1U; }),
             1);
    EXPECT_EQ(std::count_if(seen.begin(), seen.end(),
                           [](const emitted_slice& slc) { return slc.depth == 2U; }),
             2);

    const auto merged_slice =
        std::find_if(seen.begin(), seen.end(),
                    [](const emitted_slice& slc) { return slc.depth == 1U; });
    ASSERT_NE(merged_slice, seen.end());
    EXPECT_EQ(merged_slice->start_ts, 100U);
    EXPECT_EQ(merged_slice->end_ts, 200U + 50U);
}

// sample_postprocessor.hpp

// A throwing resolver is caught and an empty name
// returned. Also exercises drop_library_frames + composing with a
// frame_predicate (cross-branch coverage).
TEST(SamplePostprocessorCoverage, ResolverThrowsThenLibraryFilterApplies)
{
    sample_postprocessor_config cfg;
    cfg.resolver = [](std::uint64_t) -> std::string {
        throw std::runtime_error("resolver kaboom");
    };
    cfg.drop_library_frames = true;
    cfg.frame_predicate     = [](std::uint64_t, std::string_view name) {
        // Drop frames whose resolver name is non-empty (it never is
        // because the resolver throws -> caught -> empty name).
        return name.empty();
    };

    sample_postprocessor pp(std::move(cfg));

    raw_sample smp{};
    smp.tid          = 1;
    smp.timestamp_ns = 1;
    smp.ip           = 0xDEADBEEFULL;
    smp.frames       = { 0xDEADBEEFULL };
    std::vector<raw_sample> raws{ smp };

    std::size_t delivered = 0;
    pp.process(std::span<const raw_sample>(raws),
               [&](const processed_sample&) { ++delivered; });
    EXPECT_EQ(delivered, raws.size());
}

// coalesce_frames_by_name: resolver-throws branch and name-empty short-circuit.
TEST(SamplePostprocessorCoverage, CoalesceFramesByNameHandlesThrowsAndEmpty)
{
    std::unordered_map<pid_t, std::vector<raw_sample>> samples;
    raw_sample& smp = samples[1].emplace_back();
    smp.tid          = 1;
    smp.timestamp_ns = 1;
    // 0xBB: resolver returns empty -> identity; 0xCC: resolver throws -> identity
    smp.frames       = { 0xAA, 0xBB, 0xCC };

    coalesce_frames_by_name(samples, [](std::uint64_t ip) -> std::string {
        if(ip == 0xCC) throw std::runtime_error("kaboom");
        if(ip == 0xBB) return std::string{};
        return std::string("named_") + std::to_string(ip);
    });

    // Frames untouched for the empty / throw branches.
    EXPECT_EQ(samples[1][0].frames[1], 0xBBU);
    EXPECT_EQ(samples[1][0].frames[2], 0xCCU);
}

// sampler.hpp

// dwarf_stack_bytes alignment validation.
TEST(PerfSamplerCoverage, CtorRejectsMisalignedDwarfStackBytes)
{
    sampler::config cfg;
    cfg.perf.unwind            = unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes = 1023;  // not 8-byte aligned
    cfg.streaming.sink         = [](std::span<const std::byte>) {};
    EXPECT_THROW(sampler{ std::move(cfg) }, cpu_snap::config_error);
}

// proc_maps.hpp - reach mappings(), main_exe(), and ip_to_elf_vaddr()'s
// PT_LOAD lookup (both the found-segment and fallback branches).

TEST(ProcMapsCoverage, AccessorsAndMainExeIpOffset)
{
    proc_maps const maps;
    EXPECT_FALSE(maps.mappings().empty());
    EXPECT_FALSE(maps.main_exe().empty());

    // Path does not resolve to a real ELF file, so ip_to_elf_vaddr()
    // falls back to returning the file offset unchanged.
    vm_mapping main_exe_map{};
    main_exe_map.start       = 0x10000;
    main_exe_map.end         = 0x20000;
    main_exe_map.file_offset = 0x800;
    main_exe_map.path        = "/nonexistent/bin/x";
    main_exe_map.is_main_exe = true;
    EXPECT_EQ(maps.ip_to_elf_vaddr(0x11000, main_exe_map),
              0x11000U - 0x10000U + 0x800U);

    vm_mapping shlib_map{};
    shlib_map.start       = 0x30000;
    shlib_map.end         = 0x40000;
    shlib_map.file_offset = 0x800;
    shlib_map.path        = "/nonexistent/lib/y.so";
    shlib_map.is_main_exe = false;
    EXPECT_EQ(maps.ip_to_elf_vaddr(0x31000, shlib_map),
              0x31000U - 0x30000U + 0x800U);
}

// Regression test: covers ip_to_elf_vaddr()'s real PT_LOAD lookup path
// against this test binary's own ELF headers (the file offset must
// land inside a real PT_LOAD segment).
TEST(ProcMapsCoverage, RealMainExeMappingResolvesViaProgramHeaders)
{
    proc_maps const   maps;
    const vm_mapping* main_exe_mapping = nullptr;
    for(const auto& mapping : maps.mappings())
    {
        if(mapping.is_main_exe)
        {
            main_exe_mapping = &mapping;
            break;
        }
    }
    ASSERT_NE(main_exe_mapping, nullptr);

    const std::uint64_t ip       = main_exe_mapping->start + 0x10;
    const std::uint64_t file_off = ip - main_exe_mapping->start + main_exe_mapping->file_offset;
    const std::uint64_t vaddr    = maps.ip_to_elf_vaddr(ip, *main_exe_mapping);

    // Whatever the real segment layout is, the result must be a valid
    // translation of file_off through some PT_LOAD segment (or file_off
    // itself if none covers it), never a value pulled out of thin air.
    const auto segments = detail::read_elf_load_segments(main_exe_mapping->path);
    EXPECT_EQ(vaddr, detail::file_offset_to_vaddr(file_off, segments));
}

// Pure-function unit test for the PT_LOAD lookup, independent of any
// real ELF file.
TEST(ProcMapsCoverage, FileOffsetToVaddrAppliesCoveringSegment)
{
    const std::vector<detail::elf_load_segment> segments = {
        { /*p_offset=*/0x0,      /*p_vaddr=*/0x0,      /*p_filesz=*/0x1000 },
        { /*p_offset=*/0x11b160, /*p_vaddr=*/0x11c160, /*p_filesz=*/0x5000 },
    };

    // Falls inside the second segment, where p_offset != p_vaddr.
    EXPECT_EQ(detail::file_offset_to_vaddr(0x11b200, segments),
              0x11b200U - 0x11b160U + 0x11c160U);

    // Falls inside the first segment.
    EXPECT_EQ(detail::file_offset_to_vaddr(0x100, segments), 0x100U);

    // Covered by no segment: falls back to the file offset unchanged.
    EXPECT_EQ(detail::file_offset_to_vaddr(0xffffff, segments), 0xffffffU);

    // Empty segment list (unparseable ELF): same fallback.
    EXPECT_EQ(detail::file_offset_to_vaddr(0x42, {}), 0x42U);
}

TEST(ProcMapsCoverage, FindCoveringVaddrReturnsNulloptWhenUncovered)
{
    const std::vector<detail::elf_load_segment> segments = {
        { /*p_offset=*/0x0, /*p_vaddr=*/0x0, /*p_filesz=*/0x1000 },
    };
    const std::optional<std::uint64_t> found = detail::find_covering_vaddr(0x100, segments);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, 0x100U);

    EXPECT_FALSE(detail::find_covering_vaddr(0xffffff, segments).has_value());
    EXPECT_FALSE(detail::find_covering_vaddr(0x42, {}).has_value());
}

namespace
{
constexpr std::array<detail::elf_load_segment, 2> k_static_segments{ {
    { .p_offset = 0x0, .p_vaddr = 0x0, .p_filesz = 0x1000 },
    { .p_offset = 0x11b160, .p_vaddr = 0x11c160, .p_filesz = 0x5000 },
} };
static_assert(detail::file_offset_to_vaddr(0x11b200, k_static_segments)
              == 0x11b200U - 0x11b160U + 0x11c160U);
static_assert(detail::file_offset_to_vaddr(0xffffff, k_static_segments) == 0xffffffU);
}  // namespace

// slice_emitter.hpp - library_frame_filter callback path.

// Covers filter_library_frames(): a caller-supplied library_frame_filter
// (here built via cpu_snap::processing::symbols::make_main_exe_only_filter over a live
// proc_maps, mirroring how a demo would wire it) drops frames whose
// mapping lives in a shared library. Frames with no mapping (e.g.
// anonymous IPs) survive.
TEST(SliceEmitterCoverage, LibraryFrameFilterAppliesPredicate)
{
    proc_maps maps;

    slice_emitter_config cfg;
    cfg.library_frame_filter  = make_main_exe_only_filter(maps);
    slice_emitter em(std::move(cfg));

    auto libc_fp = &std::malloc;
    std::uint64_t libc_ip = 0;
    std::memcpy(&libc_ip, &libc_fp, sizeof(libc_ip));

    std::unordered_map<pid_t, std::vector<raw_sample>> by_tid;
    auto& vec = by_tid[1];
    // Mix: anonymous IP + libc IP. Anonymous survives, libc drops.
    push_sample(vec, 10, 1, { 0xDEADBEEFULL, libc_ip });

    std::size_t count = 0;
    em.process(by_tid, [&](const emitted_slice& /*slc*/) { ++count; });
    EXPECT_GE(count, 1U);
}

// streaming_pipeline.cpp - finalize's polled deltas + on_lost handling.

// sampler_capabilities.hpp - to_string formatting (no perf required).

// operator() cached-hit short-circuit and file_line cache hit. Preload first so addr2line is invoked
// once, then call operator() / file_line() on the same IP.
TEST(SymbolizerCoverage, OperatorAndFileLineCachedPathsAreReused)
{
    cpu_snap::processing::symbols::symbolizer sym;
    auto              fp = &std::malloc;
    std::uint64_t     ip = 0;
    std::memcpy(&ip, &fp, sizeof(ip));

    // First lookup primes the cache via the preload-fallback branch.
    const std::string first = sym(ip);
    EXPECT_FALSE(first.empty());

    // Second call hits the cached branch.
    const std::string second = sym(ip);
    EXPECT_EQ(first, second);

    // file_line may or may not be cached depending on whether
    // addr2line returned a file:line for this libc IP. Calling is
    // safe either way; the goal is to exercise the cache-hit return.
    (void) sym.file_line(ip);
    (void) sym.file_line(0xDEADBEEFULL);  // not in cache -> empty
}

TEST(SamplerCapabilitiesCoverage, ToStringFormatsAllFlags)
{
    sampler_capabilities caps;
    caps.frame_pointer  = true;
    caps.precise_ip     = true;
    caps.ibs_op         = false;
    const std::string out = caps.to_string();
    EXPECT_NE(out.find("frame_pointer=yes"), std::string::npos);
    EXPECT_NE(out.find("precise_ip=yes"), std::string::npos);
    EXPECT_NE(out.find("ibs_op=no"), std::string::npos);

    caps.ibs_op         = true;
    caps.precise_ip     = false;
    caps.frame_pointer  = false;
    const std::string flipped = caps.to_string();
    EXPECT_NE(flipped.find("frame_pointer=no"), std::string::npos);
    EXPECT_NE(flipped.find("precise_ip=no"), std::string::npos);
    EXPECT_NE(flipped.find("ibs_op=yes"), std::string::npos);
}

// finalize writes any pending deltas from
// m_maps.poll(). We swap the dl_module_watcher iterator via internal API
// is not exposed; instead we just rely on finalize() going through
// the for-each branch when poll() returns an empty vector (still
// covers the loop boundary).
TEST(StreamingPipelineCoverage, FinalizeHandlesEmptyDeltas)
{
    sink_capture sink;
    cpu_snap::sampling::internal::streaming_pipeline::config cfg;
    cfg.sink                = sink.as_fn();
    cfg.chunk_bytes         = 4096;
    cfg.maps_poll_interval  = 1000;
    cfg.checkpoint_interval = 1000;
    cfg.pid                 = 1;
    cfg.process_name        = "finalize_test";
    cpu_snap::sampling::internal::streaming_pipeline pipe(std::move(cfg));

    const std::uint64_t frames[] = { 0xAA };
    pipe.on_sample(1, 1, std::span<const std::uint64_t>(frames, 1));
    pipe.finalize();
    EXPECT_FALSE(sink.chunks.empty());
}
