// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/units/data_size.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "rocprofiler-systems/categories.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <string>

namespace rocprofsys::collection::sources::gpu_smi
{
using enabled_t = pmc::collectors::gpu::enabled_metrics;
using metrics_t = pmc::collectors::gpu::metrics;

/** Reads one value out of a metrics reading; `first` and `second` select the array
 * element (the XCP and the engine, or 0 and the link). */
using reader_fn = double (*)(const metrics_t&, std::size_t first, std::size_t second);

struct reader
{
    reader_fn   read   = nullptr;
    std::size_t first  = 0;
    std::size_t second = 0;
};

struct scalar_metric
{
    bool (*enabled)(const enabled_t&);
    reader_fn     read;
    std::uint32_t category_id;
    const char*   pmc_name;  ///< the rocpd event and track name
    const char*   perfetto_label;
    const char*   unit;
    const char*   symbol;
    const char*   description;
};

struct column_names
{
    std::string pmc;
    std::string perfetto;
    std::string rocpd;
};

struct array_metric
{
    bool (*enabled)(const enabled_t&);
    reader_fn read;  ///< NaN for an element the device does not report
    column_names (*names)(std::uint32_t device, const array_metric&, std::size_t first,
                          std::size_t second);
    std::uint32_t category_id;
    const char*   base_name;  ///< trait name of the category
    const char*   perfetto_label;
    const char*   unit;
    const char*   description;
    std::size_t   first_count;   ///< XCPs, or 1
    std::size_t   second_count;  ///< engines or links
};

namespace detail
{
constexpr double
nan_value() noexcept
{
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr double
unsupported_as_nan(std::uint16_t value) noexcept
{
    return value == pmc::collectors::gpu::METRIC_VALUE_NOT_SUPPORTED_16
               ? nan_value()
               : static_cast<double>(value);
}

inline double
to_megabytes(std::uint64_t bytes_value) noexcept
{
    using rocprofsys::common::units::bytes;
    using rocprofsys::common::units::data_size_cast;
    using rocprofsys::common::units::megabytes;
    return data_size_cast<megabytes>(bytes{ static_cast<double>(bytes_value) }).count();
}

inline column_names
xcp_engine_names(std::uint32_t device, const array_metric& metric, std::size_t first,
                 std::size_t second)
{
    const auto pmc = fmt::format("{}_{}_{}", metric.base_name, first, second);
    return { .pmc      = pmc,
             .perfetto = fmt::format("GPU [{}] {} XCP_{}: [{:02}] (S)", device,
                                     metric.perfetto_label, first, second),
             .rocpd    = pmc };
}

inline column_names
engine_names(std::uint32_t device, const array_metric& metric,
             [[maybe_unused]] std::size_t first, std::size_t second)
{
    const auto pmc = fmt::format("{}_{}", metric.base_name, second);
    return { .pmc      = pmc,
             .perfetto = fmt::format("GPU [{}] {} [{:02}] (S)", device,
                                     metric.perfetto_label, second),
             .rocpd    = pmc };
}

inline column_names
link_names(std::uint32_t device, const array_metric& metric,
           [[maybe_unused]] std::size_t first, std::size_t second)
{
    return { .pmc = trace_cache::info::format_link_pmc_name(metric.base_name, second),
             .perfetto = fmt::format("GPU [{}] {} [{:02}] (S)", device,
                                     metric.perfetto_label, second),
             .rocpd =
                 trace_cache::info::format_link_track_name(metric.base_name, second) };
}

template <typename Category>
constexpr const char*
name_of() noexcept
{
    return tim::trait::name<Category>::value;
}

template <typename Category>
constexpr const char*
description_of() noexcept
{
    return tim::trait::name<Category>::description;
}
// Socket power and temperature each have two readings; exactly one of the pair is
// enabled: the current power and the hotspot temperature win when both are available.
inline constexpr std::array<scalar_metric, 17> k_scalar_metrics{ {
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.gfx_activity != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.gfx_activity); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_GFX,
      .pmc_name       = name_of<tim::category::amd_smi_gfx_busy>(),
      .perfetto_label = "GFX Busy",
      .unit           = "%",
      .symbol         = "GFX Busy",
      .description    = description_of<tim::category::amd_smi_gfx_busy>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.umc_activity != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.umc_activity); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_UMC,
      .pmc_name       = name_of<tim::category::amd_smi_umc_busy>(),
      .perfetto_label = "UMC Avg. Busy",
      .unit           = "%",
      .symbol         = "UMC Avg. Busy",
      .description    = description_of<tim::category::amd_smi_umc_busy>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.mm_activity != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.mm_activity); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_MM,
      .pmc_name       = name_of<tim::category::amd_smi_mm_busy>(),
      .perfetto_label = "MM Busy",
      .unit           = "%",
      .symbol         = "MM Busy",
      .description    = description_of<tim::category::amd_smi_mm_busy>() },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.hotspot_temperature != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.hotspot_temperature);
          },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_TEMP,
      .pmc_name       = name_of<tim::category::amd_smi_temp>(),
      .perfetto_label = "Temperature",
      .unit           = "\u00B0C",
      .symbol         = "Temp",
      .description    = description_of<tim::category::amd_smi_temp>() },
    { .enabled =
          [](const enabled_t& enabled) {
              return enabled.bits.edge_temperature != 0 &&
                     enabled.bits.hotspot_temperature == 0;
          },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.edge_temperature); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_TEMP,
      .pmc_name       = name_of<tim::category::amd_smi_temp>(),
      .perfetto_label = "Temperature",
      .unit           = "\u00B0C",
      .symbol         = "Temp",
      .description    = description_of<tim::category::amd_smi_temp>() },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.current_socket_power != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.current_socket_power);
          },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_POWER,
      .pmc_name       = name_of<tim::category::amd_smi_power>(),
      .perfetto_label = "Current Power",
      .unit           = "W",
      .symbol         = "Pow",
      .description    = description_of<tim::category::amd_smi_power>() },
    { .enabled =
          [](const enabled_t& enabled) {
              return enabled.bits.average_socket_power != 0 &&
                     enabled.bits.current_socket_power == 0;
          },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.average_socket_power);
          },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_POWER,
      .pmc_name       = name_of<tim::category::amd_smi_power>(),
      .perfetto_label = "Avg. Power",
      .unit           = "W",
      .symbol         = "Pow",
      .description    = description_of<tim::category::amd_smi_power>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.memory_usage != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return to_megabytes(reading.memory_usage); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_MEMORY_USAGE,
      .pmc_name       = name_of<tim::category::amd_smi_memory_usage>(),
      .perfetto_label = "Memory Usage",
      .unit           = "MB",
      .symbol         = "MemUsg",
      .description    = description_of<tim::category::amd_smi_memory_usage>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.sdma_usage != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.sdma_usage); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_SDMA_USAGE,
      .pmc_name       = name_of<tim::category::amd_smi_sdma_usage>(),
      .perfetto_label = "SDMA Usage",
      .unit           = "%",
      .symbol         = "SDMA Usage",
      .description    = description_of<tim::category::amd_smi_sdma_usage>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.gfx_clock != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.gfx_clock_mhz); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_GFX_CLOCK,
      .pmc_name       = name_of<tim::category::amd_smi_gfx_clock>(),
      .perfetto_label = "GFX Clock",
      .unit           = "MHz",
      .symbol         = "GFX Clock",
      .description    = description_of<tim::category::amd_smi_gfx_clock>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.mem_clock != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.mem_clock_mhz); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_MEM_CLOCK,
      .pmc_name       = name_of<tim::category::amd_smi_mem_clock>(),
      .perfetto_label = "Memory Clock",
      .unit           = "MHz",
      .symbol         = "Mem Clock",
      .description    = description_of<tim::category::amd_smi_mem_clock>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.xgmi.link.width); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_LINK_WIDTH,
      .pmc_name       = name_of<tim::category::amd_smi_xgmi_link_width>(),
      .perfetto_label = "XGMI Link Width",
      .unit           = "lanes",
      .symbol         = "XGMI Width",
      .description    = description_of<tim::category::amd_smi_xgmi_link_width>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.xgmi.link.speed); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_LINK_SPEED,
      .pmc_name       = name_of<tim::category::amd_smi_xgmi_link_speed>(),
      .perfetto_label = "XGMI Link Speed",
      .unit           = "Mbps",
      .symbol         = "XGMI Speed",
      .description    = description_of<tim::category::amd_smi_xgmi_link_speed>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.pcie.link.width); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_LINK_WIDTH,
      .pmc_name       = name_of<tim::category::amd_smi_pcie_link_width>(),
      .perfetto_label = "PCIe Link Width",
      .unit           = "lanes",
      .symbol         = "PCIe Width",
      .description    = description_of<tim::category::amd_smi_pcie_link_width>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.pcie.link.speed); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_LINK_SPEED,
      .pmc_name       = name_of<tim::category::amd_smi_pcie_link_speed>(),
      .perfetto_label = "PCIe Link Speed",
      .unit           = "MT/s",
      .symbol         = "PCIe Speed",
      .description    = description_of<tim::category::amd_smi_pcie_link_speed>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.pcie.bandwidth.acc); },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_BANDWIDTH_ACC,
      .pmc_name       = name_of<tim::category::amd_smi_pcie_bandwidth_acc>(),
      .perfetto_label = "PCIe Bandwidth Acc",
      .unit           = "bytes",
      .symbol         = "PCIe BW Acc",
      .description    = description_of<tim::category::amd_smi_pcie_bandwidth_acc>() },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.pcie.bandwidth.inst);
          },
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_BANDWIDTH_INST,
      .pmc_name       = name_of<tim::category::amd_smi_pcie_bandwidth_inst>(),
      .perfetto_label = "PCIe Bandwidth Inst",
      .unit           = "bytes/s",
      .symbol         = "PCIe BW Inst",
      .description    = description_of<tim::category::amd_smi_pcie_bandwidth_inst>() },
} };

// The columns of an array metric exist for the elements the device reports when the
// source starts (NaN on that first reading means: no column). XGMI data of an
// unsupported link reads as zero, so every link gets a column, as before.
inline constexpr auto k_xcps    = pmc::collectors::gpu::MAX_NUM_XCP;
inline constexpr auto k_vcn     = pmc::collectors::gpu::MAX_NUM_VCN;
inline constexpr auto k_jpeg    = pmc::collectors::gpu::MAX_NUM_JPEG;
inline constexpr auto k_jpeg_v1 = pmc::collectors::gpu::MAX_NUM_JPEG_V1;
inline constexpr auto k_links   = pmc::collectors::gpu::MAX_NUM_XGMI_LINKS;

inline constexpr std::array<array_metric, 6> k_array_metrics{ {
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.vcn_busy != 0; },
      .read =
          [](const metrics_t& reading, std::size_t xcp, std::size_t engine) {
              return unsupported_as_nan(reading.xcp_stats[xcp].vcn_busy[engine]);
          },
      .names          = &xcp_engine_names,
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_VCN_ACTIVITY,
      .base_name      = name_of<tim::category::amd_smi_vcn_activity>(),
      .perfetto_label = "VCN Busy",
      .unit           = "%",
      .description    = "VCN (Video Decode) Engine Activity",
      .first_count    = k_xcps,
      .second_count   = k_vcn },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.jpeg_busy != 0; },
      .read =
          [](const metrics_t& reading, std::size_t xcp, std::size_t engine) {
              return unsupported_as_nan(reading.xcp_stats[xcp].jpeg_busy[engine]);
          },
      .names          = &xcp_engine_names,
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_JPEG_ACTIVITY,
      .base_name      = name_of<tim::category::amd_smi_jpeg_activity>(),
      .perfetto_label = "JPEG Busy",
      .unit           = "%",
      .description    = "JPEG (Image Decode) Engine Activity",
      .first_count    = k_xcps,
      .second_count   = k_jpeg_v1 },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.vcn_activity != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t engine) {
              return unsupported_as_nan(reading.vcn_activity[engine]);
          },
      .names          = &engine_names,
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_VCN_ACTIVITY,
      .base_name      = name_of<tim::category::amd_smi_vcn_activity>(),
      .perfetto_label = "VCN Activity",
      .unit           = "%",
      .description    = "VCN (Video Decode) Engine Activity",
      .first_count    = 1,
      .second_count   = k_vcn },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.jpeg_activity != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t engine) {
              return unsupported_as_nan(reading.jpeg_activity[engine]);
          },
      .names          = &engine_names,
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_JPEG_ACTIVITY,
      .base_name      = name_of<tim::category::amd_smi_jpeg_activity>(),
      .perfetto_label = "JPEG Activity",
      .unit           = "%",
      .description    = "JPEG (Image Decode) Engine Activity",
      .first_count    = 1,
      .second_count   = k_jpeg },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t link) {
              return static_cast<double>(reading.xgmi.data_acc.read[link]);
          },
      .names          = &link_names,
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_READ_DATA,
      .base_name      = name_of<tim::category::amd_smi_xgmi_read_data>(),
      .perfetto_label = "XGMI Read Data",
      .unit           = "KB",
      .description    = description_of<tim::category::amd_smi_xgmi_read_data>(),
      .first_count    = 1,
      .second_count   = k_links },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t link) {
              return static_cast<double>(reading.xgmi.data_acc.write[link]);
          },
      .names          = &link_names,
      .category_id    = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_WRITE_DATA,
      .base_name      = name_of<tim::category::amd_smi_xgmi_write_data>(),
      .perfetto_label = "XGMI Write Data",
      .unit           = "KB",
      .description    = description_of<tim::category::amd_smi_xgmi_write_data>(),
      .first_count    = 1,
      .second_count   = k_links },
} };
}  // namespace detail

inline std::span<const scalar_metric>
scalar_metrics()
{
    return detail::k_scalar_metrics;
}

inline std::span<const array_metric>
array_metrics()
{
    return detail::k_array_metrics;
}
}  // namespace rocprofsys::collection::sources::gpu_smi
