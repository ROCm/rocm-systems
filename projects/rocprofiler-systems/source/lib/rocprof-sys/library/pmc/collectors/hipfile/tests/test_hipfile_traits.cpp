// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock_backend.hpp"

#include "library/pmc/collectors/hipfile/device.hpp"
#include "library/pmc/collectors/hipfile/hipfile_traits.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

namespace rocprofsys::pmc::collectors::hipfile::testing
{
namespace
{
using device_t = device<mock_backend>;
using traits_t = hipfile_traits<mock_provider, device_t>;

/**
 * @brief Settings stand-in, so enumeration is testable without env vars or a ROCm
 *        runtime. Static state because the traits call these as static functions.
 */
struct stub_settings
{
    inline static device_filter   gpu_filter{};
    inline static std::size_t     visible_gpus = 0;
    inline static enabled_metrics hipfile_metrics{};

    static void reset()
    {
        gpu_filter            = device_filter{};
        gpu_filter.mode       = device_selection_mode::ALL;
        visible_gpus          = 0;
        hipfile_metrics.value = ALL_HIPFILE_METRICS;
    }

    static device_filter   get_gpu_device_filter() { return gpu_filter; }
    static std::size_t     get_visible_gpu_count() { return visible_gpus; }
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
    stub_settings::visible_gpus = 4;

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 4U);
    for(std::size_t i = 0; i < entries.size(); ++i)
        EXPECT_EQ(entries[i].device->get_index(), i);
}

TEST_F(HipFileTraitsTest, gpu_zero_is_enumerated_like_any_other)
{
    stub_settings::visible_gpus = 2;

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
    stub_settings::visible_gpus = 0;

    EXPECT_TRUE(enumerate().empty());
}

TEST_F(HipFileTraitsTest, disabled_filter_enumerates_nothing)
{
    stub_settings::visible_gpus    = 4;
    stub_settings::gpu_filter.mode = device_selection_mode::NONE;

    EXPECT_TRUE(enumerate().empty());
}

TEST_F(HipFileTraitsTest, specific_filter_selects_requested_ordinals)
{
    stub_settings::visible_gpus       = 4;
    stub_settings::gpu_filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::gpu_filter.indices = { 1, 3 };

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entries[0].device->get_index(), 1U);
    EXPECT_EQ(entries[1].device->get_index(), 3U);
}

TEST_F(HipFileTraitsTest, specific_filter_ignores_out_of_range_ordinals)
{
    stub_settings::visible_gpus       = 2;
    stub_settings::gpu_filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::gpu_filter.indices = { 0, 99 };

    const auto entries = enumerate();

    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].device->get_index(), 0U);
}

TEST_F(HipFileTraitsTest, visible_gpus_clamped_to_snapshot_capacity)
{
    stub_settings::visible_gpus = MAX_GPUS + 8;

    const auto entries = enumerate();

    // hipFile's snapshot has a fixed number of slots; enumerating past it would index
    // out of bounds rather than report more GPUs.
    EXPECT_EQ(entries.size(), MAX_GPUS);
}

TEST_F(HipFileTraitsTest, enabled_metrics_come_from_settings)
{
    stub_settings::hipfile_metrics.value           = 0U;
    stub_settings::hipfile_metrics.bits.read_bytes = 1;

    const auto enabled = traits_t::get_enabled_metrics<stub_settings>();

    EXPECT_EQ(enabled.bits.read_bytes, 1U);
    EXPECT_EQ(enabled.bits.write_bytes, 0U);
}

TEST_F(HipFileTraitsTest, get_metrics_delegates_to_device)
{
    stub_settings::visible_gpus            = 1;
    m_provider->backend->gpu(0).read_bytes = 2048;

    const auto entries = enumerate();
    ASSERT_EQ(entries.size(), 1U);

    enabled_metrics enabled;
    enabled.value = ALL_HIPFILE_METRICS;

    EXPECT_EQ(traits_t::get_metrics(entries[0].device, enabled, 1'000'000'000).read_bytes,
              2048U);
}

}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
