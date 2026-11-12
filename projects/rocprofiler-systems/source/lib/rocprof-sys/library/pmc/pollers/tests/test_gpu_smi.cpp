// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/collection/metric_row.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/collection/tests/recording_sink.hpp"
#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/collectors/gpu/tests/mock_gpu_backend.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/common/types.hpp"
#include "library/pmc/pollers/catalog/gpu_smi.hpp"
#include "library/pmc/pollers/gpu_smi.hpp"
#include "rocprofiler-systems/categories.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::pollers
{
namespace
{
namespace gpu = pmc::collectors::gpu;

using mock_backend = ::testing::StrictMock<backends::amd_smi::testing::mock_gpu_backend>;
using device_t     = gpu::device<mock_backend>;
using collection::testing::log_capture;
using collection::testing::recording_sink;
using ::testing::AnyNumber;
using ::testing::Return;
using ::testing::Throw;

constexpr std::uint16_t k_unsupported_16 = std::numeric_limits<std::uint16_t>::max();
constexpr std::uint64_t k_unsupported_64 = gpu::METRIC_VALUE_NOT_SUPPORTED_64;
constexpr std::uint64_t k_timestamp      = 1000;
constexpr std::uint64_t k_next_timestamp = 2000;
constexpr std::uint32_t k_gfx_busy       = 85;
constexpr std::uint32_t k_current_power  = 250;
constexpr std::uint32_t k_average_power  = 200;
constexpr std::int64_t  k_hotspot_temp   = 90;
constexpr std::int64_t  k_edge_temp      = 60;
constexpr std::uint16_t k_vcn_busy       = 40;
constexpr std::size_t   k_second_engine  = 2;
constexpr std::size_t   k_device_index   = 3;

struct stub_settings
{
    inline static gpu::enabled_metrics enabled{};

    static device_filter get_gpu_device_filter() { return {}; }
    // Unknown visibility: the BDF filter is skipped and get_bdf() is never called.
    static std::optional<std::set<std::string>> get_visible_gpu_bdfs()
    {
        return std::nullopt;
    }
    static gpu::enabled_metrics get_enabled_metrics() { return enabled; }
};

struct stub_provider
{
    std::vector<std::shared_ptr<device_t>> devices;

    template <typename Device>
    std::vector<std::shared_ptr<Device>> get_gpu_devices()
    {
        static_assert(std::is_same_v<Device, device_t>);
        return devices;
    }
};

using poller_t = gpu_smi<stub_provider, device_t, stub_settings>;

/** A raw AMD SMI reading where nothing is reported. */
gpu::metrics
unsupported_reading()
{
    gpu::metrics reading{};
    reading.current_socket_power = gpu::METRIC_VALUE_NOT_SUPPORTED_16;
    reading.average_socket_power = gpu::METRIC_VALUE_NOT_SUPPORTED_16;
    reading.gfx_activity         = gpu::METRIC_VALUE_NOT_SUPPORTED_16;
    reading.umc_activity         = gpu::METRIC_VALUE_NOT_SUPPORTED_16;
    reading.mm_activity          = gpu::METRIC_VALUE_NOT_SUPPORTED_16;
    reading.gfx_clock_mhz        = gpu::METRIC_VALUE_NOT_SUPPORTED_16;
    reading.mem_clock_mhz        = gpu::METRIC_VALUE_NOT_SUPPORTED_16;
    for(auto& xcp : reading.xcp_stats)
    {
        xcp.vcn_busy.fill(k_unsupported_16);
        xcp.jpeg_busy.fill(k_unsupported_16);
    }
    reading.vcn_activity.fill(k_unsupported_16);
    reading.jpeg_activity.fill(k_unsupported_16);
    reading.xgmi.link.width = k_unsupported_16;
    reading.xgmi.link.speed = k_unsupported_16;
    reading.xgmi.data_acc.read.fill(k_unsupported_64);
    reading.xgmi.data_acc.write.fill(k_unsupported_64);
    reading.pcie.link.width     = k_unsupported_16;
    reading.pcie.link.speed     = k_unsupported_16;
    reading.pcie.bandwidth.acc  = k_unsupported_64;
    reading.pcie.bandwidth.inst = k_unsupported_64;
    return reading;
}

/** What a mocked GPU reports; nullopt temperature = sensor not available. */
struct gpu_spec
{
    gpu::metrics                init_reading = unsupported_reading();
    gpu::metrics                poll_reading = unsupported_reading();
    std::optional<std::int64_t> hotspot;
    std::optional<std::int64_t> edge;
};

void
expect_temperature(const std::shared_ptr<mock_backend>& backend, bool hotspot,
                   std::optional<std::int64_t> value)
{
    auto& call = hotspot ? EXPECT_CALL(*backend, get_hotspot_temperature())
                         : EXPECT_CALL(*backend, get_edge_temperature());
    if(value.has_value())
    {
        call.Times(AnyNumber()).WillRepeatedly(Return(*value));
        return;
    }
    call.Times(AnyNumber()).WillRepeatedly(Throw(std::runtime_error("no sensor")));
}

/** Backend with the device-info calls allowed; get_metrics() is left to the caller. */
std::shared_ptr<mock_backend>
make_backend(const gpu_spec& spec)
{
    const auto backend = std::make_shared<mock_backend>();
    EXPECT_CALL(*backend, get_gpu_asic_info())
        .WillOnce(Return(gpu::asic_info{ .product_name = "GPU", .vendor_name = "AMD" }));
    EXPECT_CALL(*backend, get_memory_usage())
        .Times(AnyNumber())
        .WillRepeatedly(Return(k_unsupported_64));
    EXPECT_CALL(*backend, probe_sdma_gpu_support())
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    expect_temperature(backend, true, spec.hotspot);
    expect_temperature(backend, false, spec.edge);
    return backend;
}

std::shared_ptr<device_t>
make_device(const gpu_spec& spec, std::size_t index = 0)
{
    const auto backend = make_backend(spec);
    EXPECT_CALL(*backend, get_metrics())
        .WillOnce(Return(spec.init_reading))
        .WillRepeatedly(Return(spec.poll_reading));
    return std::make_shared<device_t>(backend, index);
}

std::vector<std::string>
pmc_names(const collection::schema& description)
{
    std::vector<std::string> names;
    names.reserve(description.columns.size());
    for(const auto& column : description.columns)
    {
        names.push_back(column.pmc_name);
    }
    return names;
}

/** Every column name of one kind ({pmc, perfetto, rocpd}) of a schema. */
std::vector<std::string>
column_names(const collection::schema& description,
             std::string collection::column_spec::*name)
{
    std::vector<std::string> names;
    names.reserve(description.columns.size());
    for(const auto& column : description.columns)
    {
        names.push_back(column.*name);
    }
    return names;
}

[[nodiscard]] bool
all_distinct(const std::vector<std::string>& names)
{
    return std::set<std::string>{ names.begin(), names.end() }.size() == names.size();
}

/** One column per element of every array row of the catalog, on GPU 0. */
[[nodiscard]] collection::schema
every_array_element()
{
    collection::schema result;
    for(const auto& row : catalog::gpu_smi::k_array_metrics)
    {
        for(std::size_t index = 0; index < row.outer * row.inner; ++index)
        {
            result.columns.push_back(catalog::gpu_smi::make_column(
                row, 0, index / row.inner, index % row.inner));
        }
    }
    return result;
}

/** A catalog row named `base`, so element_names can be checked without a device. */
[[nodiscard]] catalog::gpu_smi::row_t
named_row(collection::element_naming naming)
{
    catalog::gpu_smi::row_t row;
    row.category_id = ROCPROFSYS_CATEGORY_AMD_SMI;
    row.label       = "Label";
    row.pmc_name    = "base";
    row.naming      = naming;
    return row;
}

class gpu_smi_poller_test_interface : public ::testing::Test
{
protected:
    void SetUp() override { stub_settings::enabled.value = 0; }

    [[nodiscard]] bool describe(std::vector<std::shared_ptr<device_t>> devices)
    {
        const auto provider = std::make_shared<stub_provider>();
        provider->devices   = std::move(devices);
        m_poller            = std::make_unique<poller_t>(provider);
        return m_poller->describe(m_sink);
    }

    [[nodiscard]] collection::schema only_schema() const
    {
        const auto schemas = m_sink.schemas();
        EXPECT_EQ(schemas.size(), 1U);
        return schemas.empty() ? collection::schema{} : schemas.front();
    }

    recording_sink            m_sink;
    std::unique_ptr<poller_t> m_poller;
};

TEST_F(gpu_smi_poller_test_interface, columns_are_the_enabled_and_supported_metrics)
{
    gpu_spec spec;
    spec.init_reading.gfx_activity           = k_gfx_busy;
    spec.init_reading.umc_activity           = k_gfx_busy;
    stub_settings::enabled.bits.gfx_activity = 1;
    stub_settings::enabled.bits.mm_activity  = 1;

    ASSERT_TRUE(describe({ make_device(spec, k_device_index) }));

    const auto description = only_schema();
    EXPECT_EQ(description.agent, agent_type::gpu);
    EXPECT_EQ(description.device, k_device_index);
    EXPECT_EQ(description.target_arch, "GPU");
    EXPECT_EQ(description.event_category, "amd_smi");
    ASSERT_EQ(description.columns.size(), 1U);
    const auto& column = description.columns.front();
    EXPECT_EQ(column.pmc_name, "device_busy_gfx");
    EXPECT_EQ(column.symbol, "GFX Busy");
    EXPECT_EQ(column.unit, "%");
    EXPECT_EQ(column.perfetto_track_name, "GPU [3] GFX Busy (S)");
    EXPECT_EQ(column.rocpd_track_name, "device_busy_gfx");
    EXPECT_EQ(column.category_id, ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_GFX);
}

TEST_F(gpu_smi_poller_test_interface, poll_writes_one_row_per_gpu_at_the_tick_time)
{
    gpu_spec spec;
    spec.init_reading.gfx_activity           = k_gfx_busy;
    spec.poll_reading.gfx_activity           = k_gfx_busy;
    stub_settings::enabled.bits.gfx_activity = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    EXPECT_EQ(batches.front().timestamp, k_timestamp);
    EXPECT_THAT(batches.front().values, ::testing::ElementsAre(k_gfx_busy));
}

TEST_F(gpu_smi_poller_test_interface, current_power_wins_over_average_power)
{
    gpu_spec spec;
    spec.init_reading.current_socket_power           = k_current_power;
    spec.init_reading.average_socket_power           = k_average_power;
    spec.poll_reading                                = spec.init_reading;
    stub_settings::enabled.bits.current_socket_power = 1;
    stub_settings::enabled.bits.average_socket_power = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto description = only_schema();
    ASSERT_EQ(description.columns.size(), 1U);
    EXPECT_EQ(description.columns.front().pmc_name, "device_power");
    EXPECT_EQ(description.columns.front().perfetto_track_name,
              "GPU [0] Current Power (S)");
    const auto rows = m_sink.batches();
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_THAT(rows.front().values, ::testing::ElementsAre(k_current_power));
}

TEST_F(gpu_smi_poller_test_interface, average_power_when_current_power_is_not_reported)
{
    gpu_spec spec;
    spec.init_reading.average_socket_power           = k_average_power;
    spec.poll_reading                                = spec.init_reading;
    stub_settings::enabled.bits.current_socket_power = 1;
    stub_settings::enabled.bits.average_socket_power = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto description = only_schema();
    ASSERT_EQ(description.columns.size(), 1U);
    EXPECT_EQ(description.columns.front().pmc_name, "device_power");
    EXPECT_EQ(description.columns.front().perfetto_track_name, "GPU [0] Avg. Power (S)");
    const auto rows = m_sink.batches();
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_THAT(rows.front().values, ::testing::ElementsAre(k_average_power));
}

TEST_F(gpu_smi_poller_test_interface, hotspot_temperature_wins_over_edge_temperature)
{
    gpu_spec spec;
    spec.hotspot                                    = k_hotspot_temp;
    spec.edge                                       = k_edge_temp;
    stub_settings::enabled.bits.hotspot_temperature = 1;
    stub_settings::enabled.bits.edge_temperature    = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto description = only_schema();
    ASSERT_EQ(description.columns.size(), 1U);
    EXPECT_EQ(description.columns.front().pmc_name, "device_temp");
    const auto rows = m_sink.batches();
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_THAT(rows.front().values, ::testing::ElementsAre(k_hotspot_temp));
}

TEST_F(gpu_smi_poller_test_interface, edge_temperature_when_hotspot_is_not_available)
{
    gpu_spec spec;
    spec.edge                                       = k_edge_temp;
    stub_settings::enabled.bits.hotspot_temperature = 1;
    stub_settings::enabled.bits.edge_temperature    = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto description = only_schema();
    ASSERT_EQ(description.columns.size(), 1U);
    EXPECT_EQ(description.columns.front().pmc_name, "device_temp");
    const auto rows = m_sink.batches();
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_THAT(rows.front().values, ::testing::ElementsAre(k_edge_temp));
}

TEST_F(gpu_smi_poller_test_interface, vcn_busy_has_a_column_per_engine_reported_at_init)
{
    gpu_spec spec;
    spec.init_reading.xcp_stats[0].vcn_busy[0]               = 0;
    spec.init_reading.xcp_stats[0].vcn_busy[k_second_engine] = 0;
    spec.poll_reading                                        = spec.init_reading;
    spec.poll_reading.xcp_stats[0].vcn_busy[0]               = k_vcn_busy;
    // reported at init but not in this reading: no value
    spec.poll_reading.xcp_stats[0].vcn_busy[k_second_engine] = k_unsupported_16;
    stub_settings::enabled.bits.vcn_busy                     = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    m_poller->poll(k_timestamp, m_sink);

    const auto description = only_schema();
    EXPECT_THAT(
        pmc_names(description),
        ::testing::ElementsAre("device_vcn_activity_0_0", "device_vcn_activity_0_2"));
    ASSERT_EQ(description.columns.size(), 2U);
    EXPECT_EQ(description.columns[1].perfetto_track_name,
              "GPU [0] VCN Busy XCP_0: [02] (S)");
    EXPECT_EQ(description.columns[1].rocpd_track_name, "device_vcn_activity_0_2");
    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    const auto& values = batches.front().values;
    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(values[0], k_vcn_busy);
    EXPECT_TRUE(std::isnan(values[1]));
}

TEST_F(gpu_smi_poller_test_interface, jpeg_activity_has_a_column_per_reported_engine)
{
    gpu_spec spec;
    spec.init_reading.jpeg_activity[1]        = 0;
    stub_settings::enabled.bits.jpeg_activity = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    const auto description = only_schema();
    EXPECT_THAT(pmc_names(description), ::testing::ElementsAre("device_jpeg_activity_1"));
    ASSERT_EQ(description.columns.size(), 1U);
    EXPECT_EQ(description.columns.front().perfetto_track_name,
              "GPU [0] JPEG Activity [01] (S)");
}

TEST_F(gpu_smi_poller_test_interface,
       xgmi_has_width_speed_and_a_read_and_write_column_per_link)
{
    gpu_spec spec;
    spec.init_reading.xgmi.link.width = 0;
    stub_settings::enabled.bits.xgmi  = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    const auto names = pmc_names(only_schema());
    EXPECT_EQ(names.size(), 2 + (2 * gpu::MAX_NUM_XGMI_LINKS));
    EXPECT_THAT(names, ::testing::Contains("device_xgmi_link_width"));
    EXPECT_THAT(names, ::testing::Contains("device_xgmi_link_speed"));
    EXPECT_THAT(names, ::testing::Contains("device_xgmi_read_data_link0"));
    EXPECT_THAT(names, ::testing::Contains("device_xgmi_write_data_link7"));
    const auto description = only_schema();
    const auto read_link   = std::ranges::find(description.columns,
                                               std::string{ "device_xgmi_read_data_link3" },
                                               &collection::column_spec::pmc_name);
    ASSERT_NE(read_link, description.columns.end());
    EXPECT_EQ(read_link->rocpd_track_name, "device_xgmi_read_data [Link 3]");
    EXPECT_EQ(read_link->perfetto_track_name, "GPU [0] XGMI Read Data [03] (S)");
}

TEST_F(gpu_smi_poller_test_interface, failed_init_reading_gives_no_array_columns)
{
    const auto backend = make_backend({ .init_reading = unsupported_reading(),
                                        .poll_reading = unsupported_reading(),
                                        .hotspot      = k_hotspot_temp,
                                        .edge         = std::nullopt });
    EXPECT_CALL(*backend, get_metrics())
        .WillRepeatedly(Throw(std::runtime_error("metrics not available")));
    stub_settings::enabled.value = ~0U;

    ASSERT_TRUE(describe({ std::make_shared<device_t>(backend, 0) }));

    EXPECT_THAT(pmc_names(only_schema()), ::testing::ElementsAre("device_temp"));
}

TEST_F(gpu_smi_poller_test_interface, describe_is_false_without_devices)
{
    stub_settings::enabled.bits.gfx_activity = 1;

    EXPECT_FALSE(describe({}));
    EXPECT_TRUE(m_sink.schemas().empty());
}

TEST_F(gpu_smi_poller_test_interface, describe_is_false_when_no_gpu_has_a_column)
{
    gpu_spec spec;
    spec.init_reading.gfx_activity           = k_gfx_busy;
    stub_settings::enabled.bits.umc_activity = 1;

    EXPECT_FALSE(describe({ make_device(spec) }));
    EXPECT_TRUE(m_sink.schemas().empty());
}

TEST_F(gpu_smi_poller_test_interface, describe_is_false_without_provider)
{
    poller_t poller{ nullptr };

    EXPECT_FALSE(poller.describe(m_sink));
}

TEST_F(gpu_smi_poller_test_interface, failed_gpu_is_disabled_and_the_others_keep_writing)
{
    gpu_spec spec;
    spec.init_reading.gfx_activity           = k_gfx_busy;
    spec.poll_reading                        = spec.init_reading;
    stub_settings::enabled.bits.gfx_activity = 1;
    const auto failing                       = make_backend(spec);
    EXPECT_CALL(*failing, get_metrics())
        .WillOnce(Return(spec.init_reading))
        .WillOnce(Throw(std::logic_error("device lost")));
    ASSERT_TRUE(
        describe({ std::make_shared<device_t>(failing, 0), make_device(spec, 1) }));
    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 2U);

    const log_capture logs;
    m_poller->poll(k_timestamp, m_sink);
    m_poller->poll(k_next_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches_of(0).empty());
    EXPECT_EQ(m_sink.batches_of(1).size(), 2U);
    EXPECT_EQ(logs.count("error", "gpu_smi device"), 1U);
}

TEST_F(gpu_smi_poller_test_interface, release_drops_the_gpus)
{
    gpu_spec spec;
    spec.init_reading.gfx_activity           = k_gfx_busy;
    stub_settings::enabled.bits.gfx_activity = 1;
    const auto backend                       = make_backend(spec);
    EXPECT_CALL(*backend, get_metrics()).WillOnce(Return(spec.init_reading));
    ASSERT_TRUE(describe({ std::make_shared<device_t>(backend, 0) }));

    m_poller->release();
    m_poller->poll(k_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches().empty());
}
TEST_F(gpu_smi_poller_test_interface, xgmi_links_have_distinct_names)
{
    gpu_spec spec;
    spec.init_reading.xgmi.link.width = 0;
    stub_settings::enabled.bits.xgmi  = 1;
    ASSERT_TRUE(describe({ make_device(spec) }));

    const auto description = only_schema();
    EXPECT_TRUE(
        all_distinct(column_names(description, &collection::column_spec::pmc_name)));
    EXPECT_TRUE(all_distinct(
        column_names(description, &collection::column_spec::perfetto_track_name)));
    EXPECT_TRUE(all_distinct(
        column_names(description, &collection::column_spec::rocpd_track_name)));
}

// Device-level (Radeon) and per-XCP (MI300) VCN/JPEG rows share a base name; a device
// reports only one of each pair, but their element names must still not collide.
TEST(gpu_smi_catalog, array_element_names_are_distinct_across_rows)
{
    const auto all_elements = every_array_element();

    EXPECT_TRUE(
        all_distinct(column_names(all_elements, &collection::column_spec::pmc_name)));
    EXPECT_TRUE(all_distinct(
        column_names(all_elements, &collection::column_spec::perfetto_track_name)));
    EXPECT_TRUE(all_distinct(
        column_names(all_elements, &collection::column_spec::rocpd_track_name)));
}

TEST(gpu_smi_catalog, scalar_names)
{
    const auto names = catalog::gpu_smi::element_names(
        named_row(collection::element_naming::scalar), k_device_index, 0, 0);

    EXPECT_EQ(names.pmc, "base");
    EXPECT_EQ(names.perfetto, "GPU [3] Label (S)");
    EXPECT_EQ(names.rocpd, "base");
}

TEST(gpu_smi_catalog, xcp_engine_names)
{
    const auto names =
        catalog::gpu_smi::element_names(named_row(collection::element_naming::xcp_engine),
                                        k_device_index, 1, k_second_engine);

    EXPECT_EQ(names.pmc, "base_1_2");
    EXPECT_EQ(names.perfetto, "GPU [3] Label XCP_1: [02] (S)");
    EXPECT_EQ(names.rocpd, "base_1_2");
}

TEST(gpu_smi_catalog, engine_names)
{
    const auto names =
        catalog::gpu_smi::element_names(named_row(collection::element_naming::engine),
                                        k_device_index, 0, k_second_engine);

    EXPECT_EQ(names.pmc, "base_2");
    EXPECT_EQ(names.perfetto, "GPU [3] Label [02] (S)");
    EXPECT_EQ(names.rocpd, "base_2");
}

TEST(gpu_smi_catalog, link_names)
{
    const auto names = catalog::gpu_smi::element_names(
        named_row(collection::element_naming::link), k_device_index, 0, k_second_engine);

    EXPECT_EQ(names.pmc, "base_link2");
    EXPECT_EQ(names.perfetto, "GPU [3] Label [02] (S)");
    EXPECT_EQ(names.rocpd, "base [Link 2]");
}

TEST(gpu_smi_catalog, unknown_naming_throws)
{
    constexpr auto k_unknown =
        static_cast<collection::element_naming>(k_device_index + 1);

    EXPECT_THROW(static_cast<void>(catalog::gpu_smi::element_names(named_row(k_unknown),
                                                                   k_device_index, 0, 0)),
                 std::logic_error);
}
}  // namespace
}  // namespace rocprofsys::pmc::pollers
