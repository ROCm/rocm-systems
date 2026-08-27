// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock_backend.hpp"

#include "library/pmc/collectors/hipfile/device.hpp"
#include "library/pmc/collectors/hipfile/hipfile_traits.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::hipfile::testing
{
namespace
{
using device_t = device<mock_backend>;
using traits_t = hipfile_traits<mock_provider, device_t>;

namespace test_values
{
constexpr std::size_t   out_of_range_ordinal = 99;
constexpr std::size_t   extra_visible_gpus   = 8;
constexpr std::size_t   gpu_index_four       = 4;
constexpr std::uint64_t read_bytes_gpu0      = 10;
constexpr std::uint64_t read_bytes_gpu1      = 20;
constexpr std::uint64_t read_bytes_slot0     = 111;
constexpr std::uint64_t read_bytes_slot1     = 222;
constexpr std::uint64_t read_bytes_filtered  = 333;
constexpr std::uint64_t read_bytes_excluded  = 444;
constexpr std::uint64_t read_bytes_single    = 2048;
constexpr std::size_t   filter_index         = 5;
}  // namespace test_values

/**
 * @brief Settings stand-in, so enumeration is testable without env vars or a ROCm
 *        runtime. Static state because the traits call these as static functions.
 */
struct stub_settings
{
    inline static device_filter            gpu_filter{};
    inline static std::vector<std::size_t> visible_type_indices{};
    inline static enabled_metrics          hipfile_metrics{};

    static void set_visible_identity(std::size_t n)
    {
        visible_type_indices.resize(n);
        for(std::size_t idx = 0; idx < n; ++idx)
        {
            visible_type_indices[idx] = idx;
        }
    }

    static void reset()
    {
        gpu_filter      = device_filter{};
        gpu_filter.mode = device_selection_mode::ALL;
        visible_type_indices.clear();
        hipfile_metrics.value = ALL_HIPFILE_METRICS;
    }

    static device_filter            get_gpu_device_filter() { return gpu_filter; }
    static std::vector<std::size_t> get_visible_gpu_type_indices()
    {
        return visible_type_indices;
    }
    static enabled_metrics get_hipfile_enabled_metrics() { return hipfile_metrics; }
};

class HipFileTraitsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        stub_settings::reset();
        m_provider = std::make_shared<mock_provider>();
    }

    [[nodiscard]] auto enumerate()
    {
        return traits_t::enumerate_devices<stub_settings>(m_provider);
    }

    std::shared_ptr<mock_provider> m_provider;
};

TEST_F(HipFileTraitsTest, enumerates_one_device_per_visible_gpu)
{
    stub_settings::set_visible_identity(4);

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 4U);
    for(std::size_t i = 0; i < entries.size(); ++i)
    {
        EXPECT_EQ(entries[i].device->get_index(), i);
        EXPECT_EQ(entries[i].device->get_hipfile_slot(), i);
    }
}

TEST_F(HipFileTraitsTest, gpu_zero_is_enumerated_like_any_other)
{
    stub_settings::set_visible_identity(2);

    const auto entries = enumerate();

    // GPU 0 previously carried a special case that kept it in the device set even with
    // no activity, so that process-scoped counters had somewhere to live. Those counters
    // are gone and GPU 0 is now ordinary.
    ASSERT_FALSE(entries.empty());
    EXPECT_EQ(entries.front().device->get_index(), 0U);
    EXPECT_EQ(entries.front().supported_metrics.value, ALL_HIPFILE_METRICS);
}

TEST_F(HipFileTraitsTest, no_visible_gpus_enumerates_nothing)
{
    stub_settings::visible_type_indices.clear();

    EXPECT_TRUE(enumerate().empty());
}

TEST_F(HipFileTraitsTest, disabled_filter_enumerates_nothing)
{
    stub_settings::set_visible_identity(4);
    stub_settings::gpu_filter.mode = device_selection_mode::NONE;

    EXPECT_TRUE(enumerate().empty());
}

TEST_F(HipFileTraitsTest, no_metrics_enabled_enumerates_nothing)
{
    stub_settings::set_visible_identity(4);
    stub_settings::hipfile_metrics.value = 0U;

    EXPECT_TRUE(enumerate().empty());
}

TEST_F(HipFileTraitsTest, specific_filter_selects_requested_ordinals)
{
    stub_settings::set_visible_identity(4);
    stub_settings::gpu_filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::gpu_filter.indices = { 1, 3 };

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entries[0].device->get_index(), 1U);
    EXPECT_EQ(entries[1].device->get_index(), 3U);
}

TEST_F(HipFileTraitsTest, specific_filter_ignores_out_of_range_ordinals)
{
    stub_settings::set_visible_identity(2);
    stub_settings::gpu_filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::gpu_filter.indices = { 0, test_values::out_of_range_ordinal };

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].device->get_index(), 0U);
}

TEST_F(HipFileTraitsTest, visible_gpus_clamped_to_snapshot_capacity)
{
    stub_settings::set_visible_identity(MAX_GPUS + test_values::extra_visible_gpus);

    const auto entries = enumerate();

    // hipFile's snapshot has a fixed number of slots; enumerating past it would index
    // out of bounds rather than report more GPUs.
    EXPECT_EQ(entries.size(), MAX_GPUS);
}

TEST_F(HipFileTraitsTest, enabled_metrics_come_from_settings)
{
    constexpr auto read_bytes  = metric_bit_mask("Read Bytes");
    constexpr auto write_bytes = metric_bit_mask("Write Bytes");

    stub_settings::hipfile_metrics.value = read_bytes;

    const auto enabled = traits_t::get_enabled_metrics<stub_settings>();

    EXPECT_EQ(enabled.value & read_bytes, read_bytes);
    EXPECT_EQ(enabled.value & write_bytes, 0U);
}

TEST_F(HipFileTraitsTest, get_metrics_delegates_to_device)
{
    stub_settings::set_visible_identity(1);
    m_provider->backend->gpu(0).read_bytes = test_values::read_bytes_single;

    const auto entries = enumerate();
    ASSERT_EQ(entries.size(), 1U);

    enabled_metrics enabled;
    enabled.value = ALL_HIPFILE_METRICS;

    EXPECT_EQ(traits_t::get_metrics(entries[0].device, enabled, 1'000'000'000).read_bytes,
              test_values::read_bytes_single);
}

TEST_F(HipFileTraitsTest, identity_mapping_is_unchanged_without_a_visibility_mask)
{
    stub_settings::set_visible_identity(2);
    m_provider->backend->gpu(0).read_bytes = test_values::read_bytes_gpu0;
    m_provider->backend->gpu(1).read_bytes = test_values::read_bytes_gpu1;

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entries[0].device->get_index(), 0U);
    EXPECT_EQ(entries[0].device->get_hipfile_slot(), 0U);
    EXPECT_EQ(entries[1].device->get_index(), 1U);
    EXPECT_EQ(entries[1].device->get_hipfile_slot(), 1U);

    enabled_metrics enabled;
    enabled.value = ALL_HIPFILE_METRICS;
    EXPECT_EQ(entries[0].device->get_metrics(enabled, 1'000'000'000).read_bytes,
              test_values::read_bytes_gpu0);
    EXPECT_EQ(entries[1].device->get_metrics(enabled, 1'000'000'000).read_bytes,
              test_values::read_bytes_gpu1);
}

TEST_F(HipFileTraitsTest, subset_mask_maps_hipfile_slots_onto_profiler_indices)
{
    // HIP_VISIBLE_DEVICES=4,5: hipFile ordinal 0 is physical GPU 4, ordinal 1 is GPU 5.
    stub_settings::visible_type_indices    = { test_values::gpu_index_four,
                                               test_values::filter_index };
    m_provider->backend->gpu(0).read_bytes = test_values::read_bytes_slot0;
    m_provider->backend->gpu(1).read_bytes = test_values::read_bytes_slot1;

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entries[0].device->get_index(), test_values::gpu_index_four);
    EXPECT_EQ(entries[0].device->get_hipfile_slot(), 0U);
    EXPECT_EQ(entries[0].device->get_name(), "GPU 4");
    EXPECT_EQ(track_name(entries[0].device->get_index(), "Read Bytes"),
              "GPU [4] Storage Read Bytes (S)");
    EXPECT_EQ(entries[1].device->get_index(), test_values::filter_index);
    EXPECT_EQ(entries[1].device->get_hipfile_slot(), 1U);
    EXPECT_EQ(entries[1].device->get_name(), "GPU 5");
    EXPECT_EQ(track_name(entries[1].device->get_index(), "Read Bytes"),
              "GPU [5] Storage Read Bytes (S)");

    enabled_metrics enabled;
    enabled.value = ALL_HIPFILE_METRICS;
    EXPECT_EQ(entries[0].device->get_metrics(enabled, 1'000'000'000).read_bytes,
              test_values::read_bytes_slot0);
    EXPECT_EQ(entries[1].device->get_metrics(enabled, 1'000'000'000).read_bytes,
              test_values::read_bytes_slot1);
}

TEST_F(HipFileTraitsTest, sampling_gpus_filter_uses_profiler_index_not_hipfile_slot)
{
    stub_settings::visible_type_indices    = { test_values::gpu_index_four,
                                               test_values::filter_index };
    stub_settings::gpu_filter.mode         = device_selection_mode::SPECIFIC;
    stub_settings::gpu_filter.indices      = { test_values::gpu_index_four };
    m_provider->backend->gpu(0).read_bytes = test_values::read_bytes_filtered;
    m_provider->backend->gpu(1).read_bytes = test_values::read_bytes_excluded;

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].device->get_index(), test_values::gpu_index_four);
    EXPECT_EQ(entries[0].device->get_hipfile_slot(), 0U);

    enabled_metrics enabled;
    enabled.value = ALL_HIPFILE_METRICS;
    EXPECT_EQ(entries[0].device->get_metrics(enabled, 1'000'000'000).read_bytes,
              test_values::read_bytes_filtered);
}

}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
