// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock_wrapper.hpp"

#include "backends/hipfile/backend.hpp"
#include "backends/hipfile/types.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace rocprofsys::backends::hipfile::testing
{
namespace
{
namespace field_mapping
{
constexpr std::uint64_t read_bytes       = 4096;
constexpr std::uint64_t write_bytes      = 8192;
constexpr std::uint64_t total_reads      = 10;
constexpr std::uint64_t total_writes     = 20;
constexpr std::uint64_t nvfs_reads       = 7;
constexpr std::uint64_t nvfs_writes      = 13;
constexpr std::uint64_t posix_reads      = 3;
constexpr std::uint64_t posix_writes     = 7;
constexpr std::uint64_t unaligned_reads  = 1;
constexpr std::uint64_t unaligned_writes = 2;
constexpr std::uint64_t read_errors      = 5;
constexpr std::uint64_t write_errors     = 6;
}  // namespace field_mapping

namespace counts
{
constexpr std::uint64_t hundred = 100;
constexpr std::uint64_t fifty   = 50;
constexpr std::uint64_t sixty   = 60;
constexpr std::uint64_t forty   = 40;
}  // namespace counts

namespace bytes
{
constexpr std::uint64_t b256    = 256;
constexpr std::uint64_t b512    = 512;
constexpr std::uint64_t kb1     = 1024;
constexpr std::uint64_t kb2     = 2048;
constexpr std::uint64_t kb4     = 4096;
constexpr std::uint64_t b1000   = 1000;
constexpr std::uint64_t b2000   = 2000;
constexpr std::uint64_t b999999 = 999'999;
}  // namespace bytes

constexpr std::size_t k_inactive_gpu_slot = 3;
constexpr std::size_t k_zero_gpu_slot     = 0;
constexpr std::size_t k_adjacent_gpu_slot = 2;
constexpr std::size_t k_far_gpu_slot      = 4;

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
    src.read_bytes         = field_mapping::read_bytes;
    src.write_bytes        = field_mapping::write_bytes;
    src.n_total_reads      = field_mapping::total_reads;
    src.n_total_writes     = field_mapping::total_writes;
    src.n_nvfs_reads       = field_mapping::nvfs_reads;
    src.n_nvfs_writes      = field_mapping::nvfs_writes;
    src.n_posix_reads      = field_mapping::posix_reads;
    src.n_posix_writes     = field_mapping::posix_writes;
    src.n_unaligned_reads  = field_mapping::unaligned_reads;
    src.n_unaligned_writes = field_mapping::unaligned_writes;
    src.n_reads_err        = field_mapping::read_errors;
    src.n_writes_err       = field_mapping::write_errors;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    // Every field distinct so a transposed or duplicated mapping cannot pass.
    EXPECT_EQ(out.read_bytes, field_mapping::read_bytes);
    EXPECT_EQ(out.write_bytes, field_mapping::write_bytes);
    EXPECT_EQ(out.read_ops, field_mapping::total_reads);
    EXPECT_EQ(out.write_ops, field_mapping::total_writes);
    EXPECT_EQ(out.fastpath_reads, field_mapping::nvfs_reads);
    EXPECT_EQ(out.fastpath_writes, field_mapping::nvfs_writes);
    EXPECT_EQ(out.fallback_reads, field_mapping::posix_reads);
    EXPECT_EQ(out.fallback_writes, field_mapping::posix_writes);
    EXPECT_EQ(out.unaligned_reads, field_mapping::unaligned_reads);
    EXPECT_EQ(out.unaligned_writes, field_mapping::unaligned_writes);
    EXPECT_EQ(out.read_errors, field_mapping::read_errors);
    EXPECT_EQ(out.write_errors, field_mapping::write_errors);
}

TEST_F(HipFileBackendTest, fastpath_only_reads)
{
    gpu(0).n_total_reads = counts::hundred;
    gpu(0).n_nvfs_reads  = counts::hundred;
    gpu(0).n_posix_reads = 0;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_reads, counts::hundred);
    EXPECT_EQ(out.fallback_reads, 0U);
}

TEST_F(HipFileBackendTest, fallback_only_reads)
{
    // The unsupported-filesystem case: every read goes through POSIX.
    gpu(0).n_total_reads = counts::hundred;
    gpu(0).n_nvfs_reads  = 0;
    gpu(0).n_posix_reads = counts::hundred;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_reads, 0U);
    EXPECT_EQ(out.fallback_reads, counts::hundred);
}

TEST_F(HipFileBackendTest, fastpath_only_writes)
{
    gpu(0).n_total_writes = counts::fifty;
    gpu(0).n_nvfs_writes  = counts::fifty;
    gpu(0).n_posix_writes = 0;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_writes, counts::fifty);
    EXPECT_EQ(out.fallback_writes, 0U);
}

TEST_F(HipFileBackendTest, fallback_only_writes)
{
    gpu(0).n_total_writes = counts::fifty;
    gpu(0).n_nvfs_writes  = 0;
    gpu(0).n_posix_writes = counts::fifty;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_writes, 0U);
    EXPECT_EQ(out.fallback_writes, counts::fifty);
}

TEST_F(HipFileBackendTest, mixed_fastpath_fallback_totals_match)
{
    // hipFileGetStatsL3 sums n_total_* across both backends, so this invariant holds by
    // construction upstream. Asserting it here catches a mapping that crosses the two.
    gpu(0).n_total_reads = counts::hundred;
    gpu(0).n_nvfs_reads  = counts::sixty;
    gpu(0).n_posix_reads = counts::forty;

    mock_backend backend{};
    const auto&  out = backend.get_stats(TS_1).per_gpu[0];

    EXPECT_EQ(out.fastpath_reads + out.fallback_reads, out.read_ops);
}

TEST_F(HipFileBackendTest, snapshot_is_memoized_per_timestamp)
{
    gpu(0).read_bytes = bytes::kb1;

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

    gpu(0).read_bytes = bytes::kb1;
    EXPECT_EQ(backend.get_stats(TS_1).per_gpu[0].read_bytes, bytes::kb1);

    gpu(0).read_bytes = bytes::kb4;
    EXPECT_EQ(backend.get_stats(TS_2).per_gpu[0].read_bytes, bytes::kb4);

    EXPECT_EQ(mock_wrapper::call_count, 2U);
}

TEST_F(HipFileBackendTest, snapshot_is_coherent_across_devices)
{
    gpu(0).read_bytes = bytes::b1000;
    gpu(1).read_bytes = bytes::b2000;

    mock_backend backend{};
    const auto&  first = backend.get_stats(TS_1);

    // Mutating the source between reads must not leak into the memoized snapshot,
    // otherwise per-GPU values within one interval could come from different queries.
    gpu(0).read_bytes = bytes::b999999;

    const auto& second = backend.get_stats(TS_1);

    EXPECT_EQ(first.per_gpu[0].read_bytes, bytes::b1000);
    EXPECT_EQ(second.per_gpu[0].read_bytes, bytes::b1000);
    EXPECT_EQ(second.per_gpu[1].read_bytes, bytes::b2000);
}

TEST_F(HipFileBackendTest, failed_query_reports_unavailable_and_zeros)
{
    gpu(0).read_bytes            = bytes::kb4;
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
    gpu(0).read_bytes            = bytes::b512;
    EXPECT_EQ(backend.get_stats(TS_2).per_gpu[0].read_bytes, bytes::b512);
    EXPECT_TRUE(backend.is_available());
}

// ?? Runtime version guard ???????????????????????????????????????????????????

TEST_F(HipFileBackendTest, old_runtime_reports_unavailable_and_zeros)
{
    gpu(0).read_bytes               = bytes::kb4;
    mock_wrapper::version_supported = false;

    mock_backend backend{};
    const auto&  snapshot = backend.get_stats(TS_1);

    EXPECT_FALSE(backend.is_available());
    EXPECT_EQ(snapshot.per_gpu[0].read_bytes, 0U);
}

TEST_F(HipFileBackendTest, old_runtime_never_calls_the_stats_api)
{
    // The point of the guard: a libhipfile predating the stats API does not export
    // hipFileGetStatsL3, so the call must not be attempted at all.
    mock_wrapper::version_supported = false;

    mock_backend backend{};
    const auto&  first  = backend.get_stats(TS_1);
    const auto&  second = backend.get_stats(TS_2);

    EXPECT_EQ(mock_wrapper::call_count, 0U);
    EXPECT_EQ(first.per_gpu[0].read_bytes, 0U);
    EXPECT_EQ(second.per_gpu[0].read_bytes, 0U);
}

TEST_F(HipFileBackendTest, supported_runtime_queries_normally)
{
    mock_wrapper::version_supported = true;
    gpu(0).read_bytes               = bytes::kb2;

    mock_backend backend{};

    EXPECT_EQ(backend.get_stats(TS_1).per_gpu[0].read_bytes, bytes::kb2);
    EXPECT_TRUE(backend.is_available());
    EXPECT_EQ(mock_wrapper::call_count, 1U);
}

TEST_F(HipFileBackendTest, inactive_gpu_slots_are_zero_filled)
{
    gpu(k_inactive_gpu_slot).read_bytes = bytes::kb4;

    mock_backend backend{};
    const auto&  snapshot = backend.get_stats(TS_1);

    // per_gpu_stats is indexed by ordinal, not packed, so the populated slot must stay
    // at its ordinal and every other slot must read zero rather than garbage.
    EXPECT_EQ(snapshot.per_gpu[k_inactive_gpu_slot].read_bytes, bytes::kb4);
    EXPECT_EQ(snapshot.per_gpu[k_zero_gpu_slot].read_bytes, 0U);
    EXPECT_EQ(snapshot.per_gpu[k_adjacent_gpu_slot].read_bytes, 0U);
    EXPECT_EQ(snapshot.per_gpu[k_far_gpu_slot].read_bytes, 0U);
}

TEST_F(HipFileBackendTest, all_gpu_slots_are_readable)
{
    for(std::size_t i = 0; i < MAX_GPUS; ++i)
    {
        gpu(i).read_bytes = static_cast<std::uint64_t>(i) + 1;
    }

    mock_backend backend{};
    const auto&  snapshot = backend.get_stats(TS_1);

    for(std::size_t i = 0; i < MAX_GPUS; ++i)
    {
        EXPECT_EQ(snapshot.per_gpu[i].read_bytes, static_cast<std::uint64_t>(i) + 1)
            << "GPU ordinal " << i;
    }
}

TEST_F(HipFileBackendTest, factory_produces_usable_backend)
{
    auto backend = backend_factory<mock_wrapper>::create_backend();
    ASSERT_NE(backend, nullptr);

    gpu(0).read_bytes = bytes::b256;
    EXPECT_EQ(backend->get_stats(TS_1).per_gpu[0].read_bytes, bytes::b256);
}

}  // namespace
}  // namespace rocprofsys::backends::hipfile::testing
