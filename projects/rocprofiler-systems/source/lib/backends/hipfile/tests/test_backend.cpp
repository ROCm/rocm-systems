// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock_wrapper.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace rocprofsys::backends::hipfile::testing
{
namespace
{
class HipFileBackendTest : public ::testing::Test
{
protected:
    void SetUp() override { mock_wrapper::reset(); }
    void TearDown() override { mock_wrapper::reset(); }

    static mock_per_gpu_stats_t& gpu(std::size_t ordinal)
    {
        return mock_wrapper::next_stats.per_gpu_stats[ordinal];
    }
};

constexpr std::uint64_t TS_1 = 1'000'000'000;
constexpr std::uint64_t TS_2 = 2'000'000'000;

TEST_F(HipFileBackendTest, field_mapping_is_exhaustive)
{
    auto& src              = gpu(0);
    src.read_bytes         = 4096;
    src.write_bytes        = 8192;
    src.n_total_reads      = 10;
    src.n_total_writes     = 20;
    src.n_nvfs_reads       = 7;
    src.n_nvfs_writes      = 13;
    src.n_posix_reads      = 3;
    src.n_posix_writes     = 7;
    src.n_unaligned_reads  = 1;
    src.n_unaligned_writes = 2;
    src.n_reads_err        = 5;
    src.n_writes_err       = 6;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    // Every field distinct so a transposed or duplicated mapping cannot pass. This is
    // the guard for the writes_bytes/write_bytes class of bug.
    EXPECT_EQ(out.read_bytes, 4096U);
    EXPECT_EQ(out.write_bytes, 8192U);
    EXPECT_EQ(out.read_ops, 10U);
    EXPECT_EQ(out.write_ops, 20U);
    EXPECT_EQ(out.fastpath_reads, 7U);
    EXPECT_EQ(out.fastpath_writes, 13U);
    EXPECT_EQ(out.fallback_reads, 3U);
    EXPECT_EQ(out.fallback_writes, 7U);
    EXPECT_EQ(out.unaligned_reads, 1U);
    EXPECT_EQ(out.unaligned_writes, 2U);
    EXPECT_EQ(out.read_errors, 5U);
    EXPECT_EQ(out.write_errors, 6U);
}

TEST_F(HipFileBackendTest, fastpath_only_reads)
{
    gpu(0).n_total_reads = 100;
    gpu(0).n_nvfs_reads  = 100;
    gpu(0).n_posix_reads = 0;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_reads, 100U);
    EXPECT_EQ(out.fallback_reads, 0U);
}

TEST_F(HipFileBackendTest, fallback_only_reads)
{
    // The unsupported-filesystem case: every read goes through POSIX.
    gpu(0).n_total_reads = 100;
    gpu(0).n_nvfs_reads  = 0;
    gpu(0).n_posix_reads = 100;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_reads, 0U);
    EXPECT_EQ(out.fallback_reads, 100U);
}

TEST_F(HipFileBackendTest, fastpath_only_writes)
{
    gpu(0).n_total_writes = 50;
    gpu(0).n_nvfs_writes  = 50;
    gpu(0).n_posix_writes = 0;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_writes, 50U);
    EXPECT_EQ(out.fallback_writes, 0U);
}

TEST_F(HipFileBackendTest, fallback_only_writes)
{
    gpu(0).n_total_writes = 50;
    gpu(0).n_nvfs_writes  = 0;
    gpu(0).n_posix_writes = 50;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_writes, 0U);
    EXPECT_EQ(out.fallback_writes, 50U);
}

TEST_F(HipFileBackendTest, mixed_fastpath_fallback_totals_match)
{
    // hipFileGetStatsL3 sums n_total_* across both backends, so this invariant holds by
    // construction upstream. Asserting it here catches a mapping that crosses the two.
    gpu(0).n_total_reads = 100;
    gpu(0).n_nvfs_reads  = 60;
    gpu(0).n_posix_reads = 40;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_reads + out.fallback_reads, out.read_ops);
}

TEST_F(HipFileBackendTest, snapshot_is_memoized_per_timestamp)
{
    gpu(0).read_bytes = 1024;

    mock_backend backend{};
    backend.get_stats(TS_1);
    backend.get_stats(TS_1);
    backend.get_stats(TS_1);

    // One sampling interval must cost exactly one hipFile query no matter how many
    // GPU devices read from the shared backend.
    EXPECT_EQ(mock_wrapper::call_count, 1U);
}

TEST_F(HipFileBackendTest, new_timestamp_triggers_new_query)
{
    mock_backend backend{};

    gpu(0).read_bytes = 1024;
    EXPECT_EQ(backend.get_stats(TS_1).per_gpu[0].read_bytes, 1024U);

    gpu(0).read_bytes = 4096;
    EXPECT_EQ(backend.get_stats(TS_2).per_gpu[0].read_bytes, 4096U);

    EXPECT_EQ(mock_wrapper::call_count, 2U);
}

TEST_F(HipFileBackendTest, snapshot_is_coherent_across_devices)
{
    gpu(0).read_bytes = 1000;
    gpu(1).read_bytes = 2000;

    mock_backend backend{};
    const auto&  first = backend.get_stats(TS_1);

    // Mutating the source between reads must not leak into the memoized snapshot,
    // otherwise per-GPU values within one interval could come from different queries.
    gpu(0).read_bytes = 999'999;

    const auto& second = backend.get_stats(TS_1);

    EXPECT_EQ(first.per_gpu[0].read_bytes, 1000U);
    EXPECT_EQ(second.per_gpu[0].read_bytes, 1000U);
    EXPECT_EQ(second.per_gpu[1].read_bytes, 2000U);
}

TEST_F(HipFileBackendTest, failed_query_reports_unavailable_and_zeros)
{
    gpu(0).read_bytes            = 4096;
    mock_wrapper::query_succeeds = false;

    mock_backend backend{};
    const auto&  snapshot = backend.get_stats(TS_1);

    EXPECT_FALSE(backend.is_available());
    EXPECT_EQ(snapshot.per_gpu[0].read_bytes, 0U);
}

TEST_F(HipFileBackendTest, successful_query_reports_available)
{
    mock_backend backend{};
    backend.get_stats(TS_1);

    EXPECT_TRUE(backend.is_available());
}

TEST_F(HipFileBackendTest, availability_recovers_after_failure)
{
    mock_backend backend{};

    mock_wrapper::query_succeeds = false;
    backend.get_stats(TS_1);
    EXPECT_FALSE(backend.is_available());

    // hipFile stats become readable once the target initializes them, which happens
    // after the collector is already running.
    mock_wrapper::query_succeeds = true;
    gpu(0).read_bytes            = 512;
    EXPECT_EQ(backend.get_stats(TS_2).per_gpu[0].read_bytes, 512U);
    EXPECT_TRUE(backend.is_available());
}

TEST_F(HipFileBackendTest, inactive_gpu_slots_are_zero_filled)
{
    gpu(3).read_bytes = 4096;

    mock_backend backend{};
    const auto&  snapshot = backend.get_stats(TS_1);

    // per_gpu_stats is indexed by ordinal, not packed, so the populated slot must stay
    // at its ordinal and every other slot must read zero rather than garbage.
    EXPECT_EQ(snapshot.per_gpu[3].read_bytes, 4096U);
    EXPECT_EQ(snapshot.per_gpu[0].read_bytes, 0U);
    EXPECT_EQ(snapshot.per_gpu[2].read_bytes, 0U);
    EXPECT_EQ(snapshot.per_gpu[4].read_bytes, 0U);
}

TEST_F(HipFileBackendTest, all_gpu_slots_are_readable)
{
    for(std::size_t i = 0; i < MAX_GPUS; ++i)
        gpu(i).read_bytes = static_cast<std::uint64_t>(i) + 1;

    mock_backend backend{};
    const auto&  snapshot = backend.get_stats(TS_1);

    for(std::size_t i = 0; i < MAX_GPUS; ++i)
        EXPECT_EQ(snapshot.per_gpu[i].read_bytes, static_cast<std::uint64_t>(i) + 1)
            << "GPU ordinal " << i;
}

TEST_F(HipFileBackendTest, factory_produces_usable_backend)
{
    auto backend = backend_factory<mock_wrapper>::create_backend();
    ASSERT_NE(backend, nullptr);

    gpu(0).read_bytes = 256;
    EXPECT_EQ(backend->get_stats(TS_1).per_gpu[0].read_bytes, 256U);
}

}  // namespace
}  // namespace rocprofsys::backends::hipfile::testing
