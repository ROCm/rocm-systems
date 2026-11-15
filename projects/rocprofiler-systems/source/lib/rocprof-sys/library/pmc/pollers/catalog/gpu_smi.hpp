// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/units/data_size.hpp"
#include "core/categories.hpp"
#include "core/collection/metric_row.hpp"
#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "rocprofiler-systems/categories.h"

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace rocprofsys::pmc::pollers::catalog::gpu_smi
{
using enabled_t = pmc::collectors::gpu::enabled_metrics;
using metrics_t = pmc::collectors::gpu::metrics;
using row_t     = collection::metric_row<enabled_t, metrics_t>;

/** One column of a device: the row's read function and the element it reads. */
struct reader
{
    /** The catalog row's read function. */
    double (*read)(const metrics_t&, std::size_t, std::size_t) = nullptr;
    /** XCP index, or 0. */
    std::size_t outer = 0;
    /** Engine or link index, or 0. */
    std::size_t inner = 0;
};

/** An element the device does not report reads as k_no_value. */
[[nodiscard]] constexpr double
unsupported_as_nan(std::uint16_t value) noexcept
{
    return value == pmc::collectors::gpu::METRIC_VALUE_NOT_SUPPORTED_16
               ? collection::k_no_value
               : static_cast<double>(value);
}

/** The scalar metrics. Socket power and temperature each have two readings; exactly one
 * of the pair is enabled: the current power and the hotspot temperature win when both are
 * available. */
inline constexpr auto k_scalar_metrics = std::to_array<row_t>({
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.gfx_activity != 0; },
      .read    = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.gfx_activity); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_GFX,
      .label       = "GFX Busy",
      .unit        = "%",
      .symbol      = "GFX Busy",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.umc_activity != 0; },
      .read    = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.umc_activity); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_UMC,
      .label       = "UMC Avg. Busy",
      .unit        = "%",
      .symbol      = "UMC Avg. Busy",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.mm_activity != 0; },
      .read    = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.mm_activity); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_MM,
      .label       = "MM Busy",
      .unit        = "%",
      .symbol      = "MM Busy",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.hotspot_temperature != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.hotspot_temperature);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_TEMP,
      .label       = "Temperature",
      .unit        = "\u00B0C",
      .symbol      = "Temp",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled =
          [](const enabled_t& enabled) {
              return enabled.bits.edge_temperature != 0 &&
                     enabled.bits.hotspot_temperature == 0;
          },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.edge_temperature); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_TEMP,
      .label       = "Temperature",
      .unit        = "\u00B0C",
      .symbol      = "Temp",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.current_socket_power != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.current_socket_power);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_POWER,
      .label       = "Current Power",
      .unit        = "W",
      .symbol      = "Pow",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled =
          [](const enabled_t& enabled) {
              return enabled.bits.average_socket_power != 0 &&
                     enabled.bits.current_socket_power == 0;
          },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.average_socket_power);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_POWER,
      .label       = "Avg. Power",
      .unit        = "W",
      .symbol      = "Pow",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.memory_usage != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              using common::units::bytes;
              using common::units::megabytes;
              return common::units::data_size_cast<megabytes>(
                         bytes{ static_cast<double>(reading.memory_usage) })
                  .count();
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_MEMORY_USAGE,
      .label       = "Memory Usage",
      .unit        = "MB",
      .symbol      = "MemUsg",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.sdma_usage != 0; },
      .read    = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.sdma_usage); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_SDMA_USAGE,
      .label       = "SDMA Usage",
      .unit        = "%",
      .symbol      = "SDMA Usage",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.gfx_clock != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.gfx_clock_mhz); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_GFX_CLOCK,
      .label       = "GFX Clock",
      .unit        = "MHz",
      .symbol      = "GFX Clock",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.mem_clock != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.mem_clock_mhz); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_MEM_CLOCK,
      .label       = "Memory Clock",
      .unit        = "MHz",
      .symbol      = "Mem Clock",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.xgmi.link.width); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_LINK_WIDTH,
      .label       = "XGMI Link Width",
      .unit        = "lanes",
      .symbol      = "XGMI Width",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.xgmi.link.speed); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_LINK_SPEED,
      .label       = "XGMI Link Speed",
      .unit        = "Mbps",
      .symbol      = "XGMI Speed",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.pcie.link.width); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_LINK_WIDTH,
      .label       = "PCIe Link Width",
      .unit        = "lanes",
      .symbol      = "PCIe Width",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.pcie.link.speed); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_LINK_SPEED,
      .label       = "PCIe Link Speed",
      .unit        = "MT/s",
      .symbol      = "PCIe Speed",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.pcie.bandwidth.acc); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_BANDWIDTH_ACC,
      .label       = "PCIe Bandwidth Acc",
      .unit        = "bytes",
      .symbol      = "PCIe BW Acc",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.pcie != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.pcie.bandwidth.inst);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_PCIE_BANDWIDTH_INST,
      .label       = "PCIe Bandwidth Inst",
      .unit        = "bytes/s",
      .symbol      = "PCIe BW Inst",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {} },
});

/** The array metrics. A device gets a column for each element its support_reading()
 * reports (non-NaN). XGMI data of an unsupported link reads as zero, so every link gets a
 * column. */
inline constexpr auto k_array_metrics = std::to_array<row_t>({
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.vcn_busy != 0; },
      .read =
          [](const metrics_t& reading, std::size_t xcp, std::size_t engine) {
              return unsupported_as_nan(reading.xcp_stats[xcp].vcn_busy[engine]);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_VCN_ACTIVITY,
      .label       = "VCN Busy",
      .unit        = "%",
      .symbol      = {},
      .pmc_name    = {},
      .description = "VCN (Video Decode) Engine Activity",
      .rocpd_track = {},
      .outer       = pmc::collectors::gpu::MAX_NUM_XCP,
      .inner       = pmc::collectors::gpu::MAX_NUM_VCN,
      .naming      = collection::element_naming::xcp_engine },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.jpeg_busy != 0; },
      .read =
          [](const metrics_t& reading, std::size_t xcp, std::size_t engine) {
              return unsupported_as_nan(reading.xcp_stats[xcp].jpeg_busy[engine]);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_JPEG_ACTIVITY,
      .label       = "JPEG Busy",
      .unit        = "%",
      .symbol      = {},
      .pmc_name    = {},
      .description = "JPEG (Image Decode) Engine Activity",
      .rocpd_track = {},
      .outer       = pmc::collectors::gpu::MAX_NUM_XCP,
      .inner       = pmc::collectors::gpu::MAX_NUM_JPEG_V1,
      .naming      = collection::element_naming::xcp_engine },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.vcn_activity != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t engine) {
              return unsupported_as_nan(reading.vcn_activity[engine]);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_VCN_ACTIVITY,
      .label       = "VCN Activity",
      .unit        = "%",
      .symbol      = {},
      .pmc_name    = {},
      .description = "VCN (Video Decode) Engine Activity",
      .rocpd_track = {},
      .outer       = 1,
      .inner       = pmc::collectors::gpu::MAX_NUM_VCN,
      .naming      = collection::element_naming::engine },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.jpeg_activity != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t engine) {
              return unsupported_as_nan(reading.jpeg_activity[engine]);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_JPEG_ACTIVITY,
      .label       = "JPEG Activity",
      .unit        = "%",
      .symbol      = {},
      .pmc_name    = {},
      .description = "JPEG (Image Decode) Engine Activity",
      .rocpd_track = {},
      .outer       = 1,
      .inner       = pmc::collectors::gpu::MAX_NUM_JPEG,
      .naming      = collection::element_naming::engine },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t link) {
              return static_cast<double>(reading.xgmi.data_acc.read[link]);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_READ_DATA,
      .label       = "XGMI Read Data",
      .unit        = "KB",
      .symbol      = {},
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {},
      .outer       = 1,
      .inner       = pmc::collectors::gpu::MAX_NUM_XGMI_LINKS,
      .naming      = collection::element_naming::link },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.xgmi != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t link) {
              return static_cast<double>(reading.xgmi.data_acc.write[link]);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_XGMI_WRITE_DATA,
      .label       = "XGMI Write Data",
      .unit        = "KB",
      .symbol      = {},
      .pmc_name    = {},
      .description = {},
      .rocpd_track = {},
      .outer       = 1,
      .inner       = pmc::collectors::gpu::MAX_NUM_XGMI_LINKS,
      .naming      = collection::element_naming::link },
});

/** The pmc, Perfetto and rocpd names of one column. */
struct column_names
{
    /** rocpd pmc_info name. */
    std::string pmc;
    /** Perfetto track name. */
    std::string perfetto;
    /** rocpd track name. */
    std::string rocpd;
};

/** The pmc name of `row`: its own, else its category name. */
[[nodiscard]] inline std::string
base_name(const row_t& row)
{
    return std::string{ row.pmc_name.empty() ? category_info(row.category_id).name
                                             : row.pmc_name };
}

/** Names element [outer][inner] of `row` on GPU `device`.
 * @throws std::logic_error for an unhandled element_naming */
[[nodiscard]] inline column_names
element_names(const row_t& row, std::uint32_t device, std::size_t outer,
              std::size_t inner)
{
    const auto base = base_name(row);
    switch(row.naming)
    {
        case collection::element_naming::scalar:
            return { .pmc      = base,
                     .perfetto = fmt::format("GPU [{}] {} (S)", device, row.label),
                     .rocpd    = base };
        case collection::element_naming::xcp_engine:
        {
            const auto pmc = fmt::format("{}_{}_{}", base, outer, inner);
            return { .pmc      = pmc,
                     .perfetto = fmt::format("GPU [{}] {} XCP_{}: [{:02}] (S)", device,
                                             row.label, outer, inner),
                     .rocpd    = pmc };
        }
        case collection::element_naming::engine:
        {
            const auto pmc = fmt::format("{}_{}", base, inner);
            return { .pmc = pmc,
                     .perfetto =
                         fmt::format("GPU [{}] {} [{:02}] (S)", device, row.label, inner),
                     .rocpd = pmc };
        }
        case collection::element_naming::link:
            return { .pmc = trace_cache::info::format_link_pmc_name(base, inner),
                     .perfetto =
                         fmt::format("GPU [{}] {} [{:02}] (S)", device, row.label, inner),
                     .rocpd = trace_cache::info::format_link_track_name(base, inner) };
        default:
            throw std::logic_error(fmt::format("element_naming {} not handled",
                                               static_cast<int>(row.naming)));
    }
}

/** The column of element [outer][inner] of `row` on GPU `device`. */
[[nodiscard]] inline collection::column_spec
make_column(const row_t& row, std::uint32_t device, std::size_t outer, std::size_t inner)
{
    auto       names       = element_names(row, device, outer, inner);
    const auto description = row.description.empty()
                                 ? category_info(row.category_id).description
                                 : row.description;
    return { .symbol      = row.symbol.empty() ? names.pmc : std::string{ row.symbol },
             .description = std::string{ description },
             .unit        = std::string{ row.unit },
             .pmc_name    = std::move(names.pmc),
             .perfetto_track_name = std::move(names.perfetto),
             .rocpd_track_name    = std::move(names.rocpd),
             .category_id         = row.category_id,
             .block               = {},
             .expression          = {} };
}
}  // namespace rocprofsys::pmc::pollers::catalog::gpu_smi
