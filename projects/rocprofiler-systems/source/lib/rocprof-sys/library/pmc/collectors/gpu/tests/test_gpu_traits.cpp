// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/collectors/gpu/gpu_traits.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "mock_gpu_backend.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Throw;

using MockBackend =
    ::testing::NiceMock<rocprofsys::backends::amd_smi::testing::mock_gpu_backend>;

namespace rocprofsys::pmc::collectors::gpu::testing
{

using gpu_device_t = device<MockBackend>;

// Settings stub standing in for collectors::settings_policy. Keeping the device filter
// and the runtime-visible BDF set behind the settings policy is what lets these tests
// drive enumerate_devices without linking the ROCm/AMD SMI runtime.
struct stub_settings
{
    inline static device_filter                        filter{};
    inline static std::optional<std::set<std::string>> visible_bdfs{};
    inline static int                                  visible_query_count = 0;

    static void reset()
    {
        filter              = device_filter{};
        filter.mode         = device_selection_mode::ALL;
        visible_bdfs        = std::set<std::string>{};
        visible_query_count = 0;
    }

    static device_filter get_gpu_device_filter() { return filter; }

    static std::optional<std::set<std::string>> get_visible_gpu_bdfs()
    {
        ++visible_query_count;
        return visible_bdfs;
    }
};

struct stub_provider
{
    std::vector<std::shared_ptr<gpu_device_t>> devices;

    template <typename Device>
    std::vector<std::shared_ptr<Device>> get_gpu_devices()
    {
        static_assert(std::is_same_v<Device, gpu_device_t>,
                      "stub_provider only serves the device type under test");
        return devices;
    }
};

using traits_t = gpu_traits<stub_provider, gpu_device_t>;

class GpuTraitsEnumerateTest : public ::testing::Test
{
protected:
    void SetUp() override { stub_settings::reset(); }

    // Builds a device whose backend reports `bdf`, or throws when `bdf` is empty, which
    // is how AMD SMI signals that the BDF could not be determined.
    static std::shared_ptr<gpu_device_t> make_device(size_t index, const std::string& bdf)
    {
        auto backend = std::make_shared<MockBackend>();

        ON_CALL(*backend, get_gpu_asic_info())
            .WillByDefault(Return(asic_info{ "Test GPU", "AMD" }));
        ON_CALL(*backend, probe_sdma_gpu_support()).WillByDefault(Return(true));

        if(bdf.empty())
        {
            ON_CALL(*backend, get_bdf())
                .WillByDefault(Throw(std::runtime_error("BDF unavailable")));
        }
        else
        {
            ON_CALL(*backend, get_bdf()).WillByDefault(Return(bdf));
        }

        return std::make_shared<gpu_device_t>(backend, index);
    }

    static std::shared_ptr<stub_provider> make_provider(
        std::vector<std::shared_ptr<gpu_device_t>> devices)
    {
        auto provider     = std::make_shared<stub_provider>();
        provider->devices = std::move(devices);
        return provider;
    }

    static std::set<size_t> enumerated_indices(
        const std::vector<traits_t::device_entry>& entries)
    {
        std::set<size_t> indices;
        for(const auto& entry : entries)
        {
            indices.insert(entry.device->get_index());
        }
        return indices;
    }
};

TEST_F(GpuTraitsEnumerateTest, keeps_only_devices_the_runtime_exposes)
{
    stub_settings::filter.mode  = device_selection_mode::ALL;
    stub_settings::visible_bdfs = std::set<std::string>{ "0000:26:00.0" };

    auto provider =
        make_provider({ make_device(0, "0000:05:00.0"), make_device(1, "0000:26:00.0") });

    const auto entries = traits_t::enumerate_devices<stub_settings>(provider);

    EXPECT_EQ(enumerated_indices(entries), (std::set<size_t>{ 1 }));
}

TEST_F(GpuTraitsEnumerateTest, all_devices_kept_when_all_are_visible)
{
    stub_settings::filter.mode  = device_selection_mode::ALL;
    stub_settings::visible_bdfs = std::set<std::string>{ "0000:05:00.0", "0000:26:00.0" };

    auto provider =
        make_provider({ make_device(0, "0000:05:00.0"), make_device(1, "0000:26:00.0") });

    const auto entries = traits_t::enumerate_devices<stub_settings>(provider);

    EXPECT_EQ(enumerated_indices(entries), (std::set<size_t>{ 0, 1 }));
}

// Agents exist but every one is masked off: an empty (not absent) visible set, so the
// filter applies and nothing is sampled.
TEST_F(GpuTraitsEnumerateTest, empty_visible_set_excludes_every_device)
{
    stub_settings::filter.mode  = device_selection_mode::ALL;
    stub_settings::visible_bdfs = std::set<std::string>{};

    auto provider =
        make_provider({ make_device(0, "0000:05:00.0"), make_device(1, "0000:26:00.0") });

    EXPECT_TRUE(traits_t::enumerate_devices<stub_settings>(provider).empty());
}

// Runtime visibility could not be determined at all (no ROCm agents reported). That is
// an absence of information, not evidence that nothing is visible, so the filter is
// skipped rather than silently dropping every device.
TEST_F(GpuTraitsEnumerateTest, unknown_visibility_skips_the_filter_entirely)
{
    stub_settings::filter.mode  = device_selection_mode::ALL;
    stub_settings::visible_bdfs = std::nullopt;

    auto provider =
        make_provider({ make_device(0, "0000:05:00.0"), make_device(1, "0000:26:00.0") });

    const auto entries = traits_t::enumerate_devices<stub_settings>(provider);

    EXPECT_EQ(enumerated_indices(entries), (std::set<size_t>{ 0, 1 }));
}

// With visibility unknown, a device that cannot report a BDF is still sampled: there is
// nothing to correlate it against, so there is no basis for excluding it.
TEST_F(GpuTraitsEnumerateTest, unknown_visibility_keeps_device_without_bdf)
{
    stub_settings::filter.mode  = device_selection_mode::ALL;
    stub_settings::visible_bdfs = std::nullopt;

    auto provider = make_provider({ make_device(0, "") });

    EXPECT_EQ(traits_t::enumerate_devices<stub_settings>(provider).size(), 1U);
}

// A machine with no AMD SMI GPUs and no ROCm agents is not an error state; enumeration
// simply yields nothing.
TEST_F(GpuTraitsEnumerateTest, no_devices_and_unknown_visibility_is_not_an_error)
{
    stub_settings::filter.mode  = device_selection_mode::ALL;
    stub_settings::visible_bdfs = std::nullopt;

    EXPECT_TRUE(traits_t::enumerate_devices<stub_settings>(make_provider({})).empty());
}

// Explicit index selection is still honored when visibility is unknown.
TEST_F(GpuTraitsEnumerateTest, unknown_visibility_still_applies_index_filter)
{
    stub_settings::filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::filter.indices = { 1 };
    stub_settings::visible_bdfs   = std::nullopt;

    auto provider =
        make_provider({ make_device(0, "0000:05:00.0"), make_device(1, "0000:26:00.0") });

    const auto entries = traits_t::enumerate_devices<stub_settings>(provider);

    EXPECT_EQ(enumerated_indices(entries), (std::set<size_t>{ 1 }));
}

// A backend that cannot report a BDF surfaces as an empty string through
// device::get_bdf(); such a device cannot be correlated and must not be sampled.
TEST_F(GpuTraitsEnumerateTest, device_with_unknown_bdf_is_excluded)
{
    stub_settings::filter.mode  = device_selection_mode::ALL;
    stub_settings::visible_bdfs = std::set<std::string>{ "0000:05:00.0" };

    auto provider = make_provider({ make_device(0, ""), make_device(1, "0000:05:00.0") });

    const auto entries = traits_t::enumerate_devices<stub_settings>(provider);

    EXPECT_EQ(enumerated_indices(entries), (std::set<size_t>{ 1 }));
}

TEST_F(GpuTraitsEnumerateTest, explicit_index_selection_still_honors_visibility)
{
    stub_settings::filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::filter.indices = { 0, 1 };
    stub_settings::visible_bdfs   = std::set<std::string>{ "0000:26:00.0" };

    auto provider =
        make_provider({ make_device(0, "0000:05:00.0"), make_device(1, "0000:26:00.0") });

    const auto entries = traits_t::enumerate_devices<stub_settings>(provider);

    EXPECT_EQ(enumerated_indices(entries), (std::set<size_t>{ 1 }));
}

// ROCPROFSYS_SAMPLING_GPUS names a device the runtime has masked off: the request
// cannot be honored, and the device is not sampled.
TEST_F(GpuTraitsEnumerateTest, explicitly_requested_masked_device_yields_nothing)
{
    stub_settings::filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::filter.indices = { 0 };
    stub_settings::visible_bdfs   = std::set<std::string>{ "0000:26:00.0" };

    auto provider =
        make_provider({ make_device(0, "0000:05:00.0"), make_device(1, "0000:26:00.0") });

    EXPECT_TRUE(traits_t::enumerate_devices<stub_settings>(provider).empty());
}

TEST_F(GpuTraitsEnumerateTest, sampling_disabled_skips_the_visibility_query)
{
    stub_settings::filter.mode = device_selection_mode::NONE;

    auto provider = make_provider({ make_device(0, "0000:05:00.0") });

    EXPECT_TRUE(traits_t::enumerate_devices<stub_settings>(provider).empty());
    EXPECT_EQ(stub_settings::visible_query_count, 0);
}

// Devices the index filter already rejected must not be probed for a BDF; the
// visibility check is deliberately short-circuited behind the index filter.
TEST_F(GpuTraitsEnumerateTest, device_rejected_by_index_filter_is_not_probed_for_bdf)
{
    stub_settings::filter.mode    = device_selection_mode::SPECIFIC;
    stub_settings::filter.indices = { 1 };
    stub_settings::visible_bdfs = std::set<std::string>{ "0000:05:00.0", "0000:26:00.0" };

    auto excluded_backend = std::make_shared<MockBackend>();
    ON_CALL(*excluded_backend, get_gpu_asic_info())
        .WillByDefault(Return(asic_info{ "Test GPU", "AMD" }));
    ON_CALL(*excluded_backend, probe_sdma_gpu_support()).WillByDefault(Return(true));
    EXPECT_CALL(*excluded_backend, get_bdf()).Times(0);

    auto provider = make_provider({ std::make_shared<gpu_device_t>(excluded_backend, 0),
                                    make_device(1, "0000:26:00.0") });

    const auto entries = traits_t::enumerate_devices<stub_settings>(provider);

    EXPECT_EQ(enumerated_indices(entries), (std::set<size_t>{ 1 }));
}

// The runtime-visibility predicate itself. These pin the three behaviors that decide
// whether a physical GPU enumerated by AMD SMI is sampled: visible device kept,
// non-visible device excluded, and unknown/empty BDF excluded (fail-closed).

TEST(gpu_runtime_visibility, visible_device_is_included)
{
    const std::set<std::string> visible{ "0000:05:00.0", "0000:26:00.0" };
    EXPECT_TRUE(is_runtime_visible("0000:26:00.0", visible));
}

TEST(gpu_runtime_visibility, non_visible_device_is_excluded)
{
    const std::set<std::string> visible{ "0000:26:00.0" };
    // A different physical GPU (masked by ROCR/HIP_VISIBLE_DEVICES) must not match.
    EXPECT_FALSE(is_runtime_visible("0000:05:00.0", visible));
}

TEST(gpu_runtime_visibility, empty_bdf_is_excluded_fail_closed)
{
    const std::set<std::string> visible{ "0000:05:00.0" };
    // Device whose BDF could not be determined: never sample it.
    EXPECT_FALSE(is_runtime_visible("", visible));
}

TEST(gpu_runtime_visibility, empty_visible_set_excludes_everything)
{
    // No runtime-visible GPUs (e.g. all masked): nothing is sampled.
    const std::set<std::string> visible{};
    EXPECT_FALSE(is_runtime_visible("0000:05:00.0", visible));
    EXPECT_FALSE(is_runtime_visible("", visible));
}

TEST(gpu_runtime_visibility, match_is_exact_not_substring)
{
    const std::set<std::string> visible{ "0000:05:00.0" };
    // Correlation must be an exact BDF match, not a prefix/substring.
    EXPECT_FALSE(is_runtime_visible("0000:05:00.1", visible));
    EXPECT_FALSE(is_runtime_visible("0000:05:00", visible));
}

}  // namespace rocprofsys::pmc::collectors::gpu::testing
