// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/units/data_size.hpp"
#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu
{

/** Reads one counter value from a GPU sample; NaN when the device reported none. */
struct value_reader
{
    double (*scalar)(const metrics&)               = nullptr;
    double (*element)(const metrics&, std::size_t) = nullptr;  ///< array counters
    std::size_t index                              = 0;        ///< element of an array

    [[nodiscard]] double operator()(const metrics& values) const
    {
        return scalar != nullptr ? scalar(values) : element(values, index);
    }
};

/**
 * @brief One AMD SMI GPU counter: the names and units both outputs use, when it is
 * collected and how its value is read.
 *
 * The Perfetto track of GPU n is "GPU [n] <perfetto_label> (S)"; rocpd names do not
 * contain the device.
 */
struct counter
{
    std::size_t   category = 0;  ///< Perfetto category id (category_enum_id)
    std::string   perfetto_label;
    const char*   perfetto_units = "";
    std::string   pmc_name;  ///< rocpd pmc_info and event name
    std::string   rocpd_track;
    std::string   symbol;
    const char*   description = "";
    const char*   rocpd_units = "";
    std::uint32_t required    = 0;  ///< collected when any of these metric bits is set
    std::uint32_t excluded =
        0;  ///< and none of these; a fallback of the counter before it
    value_reader read;
};

/** One array-valued metric; it becomes one counter per element. */
struct array_family
{
    const char*   label       = "";  ///< Perfetto label, e.g. "VCN Busy"
    const char*   symbol      = "";  ///< rocpd symbol of link counters, e.g. "XGMI Read"
    const char*   description = "";
    const char*   units       = "%";
    std::uint32_t required    = 0;
    double (*read)(const metrics&, std::size_t) = nullptr;
};

[[nodiscard]] inline double
or_absent(std::uint16_t value) noexcept
{
    return value == METRIC_VALUE_NOT_SUPPORTED_16
               ? std::numeric_limits<double>::quiet_NaN()
               : static_cast<double>(value);
}

[[nodiscard]] inline double
or_absent(std::uint64_t value) noexcept
{
    return value == METRIC_VALUE_NOT_SUPPORTED_64
               ? std::numeric_limits<double>::quiet_NaN()
               : static_cast<double>(value);
}

/** Fill the category id, rocpd names and description of a scalar counter. */
template <typename Category>
[[nodiscard]] counter
with_category(counter item)
{
    item.category    = category_enum_id<Category>::value;
    item.pmc_name    = trait::name<Category>::value;
    item.rocpd_track = trace_cache::info::format_track_name<Category>();
    item.description = trait::name<Category>::description;
    return item;
}

/** Per-XCP engines (MI300): Perfetto "<label> XCP_x: [ii]", rocpd "<category>_x_i". */
template <typename Category>
void
append_xcp_counters(std::vector<counter>& out, const array_family& family,
                    std::size_t engines)
{
    for(std::size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(std::size_t engine = 0; engine < engines; ++engine)
        {
            const auto name = trace_cache::info::format_track_name<Category>(
                static_cast<int>(xcp), static_cast<int>(engine));
            out.push_back({ .category       = category_enum_id<Category>::value,
                            .perfetto_label = fmt::format("{} XCP_{}: [{:02}]",
                                                          family.label, xcp, engine),
                            .perfetto_units = family.units,
                            .pmc_name       = name,
                            .rocpd_track    = name,
                            .symbol         = name,
                            .description    = family.description,
                            .rocpd_units    = family.units,
                            .required       = family.required,
                            .excluded       = 0,
                            .read           = { .element = family.read,
                                                .index   = (xcp * engines) + engine } });
        }
    }
}

/** Device-level engines (Radeon): Perfetto "<label> [ii]", rocpd "<category>_i". */
template <typename Category>
void
append_device_counters(std::vector<counter>& out, const array_family& family,
                       std::size_t engines)
{
    for(std::size_t engine = 0; engine < engines; ++engine)
    {
        const auto name = trace_cache::info::format_track_name<Category>(
            std::nullopt, static_cast<int>(engine));
        out.push_back({ .category       = category_enum_id<Category>::value,
                        .perfetto_label = fmt::format("{} [{:02}]", family.label, engine),
                        .perfetto_units = family.units,
                        .pmc_name       = name,
                        .rocpd_track    = name,
                        .symbol         = name,
                        .description    = family.description,
                        .rocpd_units    = family.units,
                        .required       = family.required,
                        .excluded       = 0,
                        .read           = { .element = family.read, .index = engine } });
    }
}

/** XGMI links: Perfetto "<label> [ii]", rocpd "<category>_link<i>" on "<category> [Link
 * i]". */
template <typename Category>
void
append_link_counters(std::vector<counter>& out, const array_family& family)
{
    constexpr auto* k_base = trait::name<Category>::value;
    for(std::size_t link = 0; link < MAX_NUM_XGMI_LINKS; ++link)
    {
        out.push_back(
            { .category       = category_enum_id<Category>::value,
              .perfetto_label = fmt::format("{} [{:02}]", family.label, link),
              .perfetto_units = family.units,
              .pmc_name       = trace_cache::info::format_link_pmc_name(k_base, link),
              .rocpd_track    = trace_cache::info::format_link_track_name(k_base, link),
              .symbol         = fmt::format("{} [Link {}]", family.symbol, link),
              .description    = trait::name<Category>::description,
              .rocpd_units    = family.units,
              .required       = family.required,
              .excluded       = 0,
              .read           = { .element = family.read, .index = link } });
    }
}

/** The scalar counters; temperature and power each have a fallback row. */
[[nodiscard]] inline std::vector<counter>
make_scalar_counters()
{
    namespace bit = metric_bit;
    using common::units::bytes;
    using common::units::data_size_cast;
    using common::units::megabytes;

    // clang-format off
    return {
        with_category<category::amd_smi_gfx_busy>({ .perfetto_label = "GFX Busy", .perfetto_units = "%", .symbol = "GFX Busy", .rocpd_units = "%", .required = bit::k_gfx_activity,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.gfx_activity); } } }),
        with_category<category::amd_smi_umc_busy>({ .perfetto_label = "UMC Avg. Busy", .perfetto_units = "%", .symbol = "UMC Avg. Busy", .rocpd_units = "%", .required = bit::k_umc_activity,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.umc_activity); } } }),
        with_category<category::amd_smi_mm_busy>({ .perfetto_label = "MM Busy", .perfetto_units = "%", .symbol = "MM Busy", .rocpd_units = "%", .required = bit::k_mm_activity,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.mm_activity); } } }),
        with_category<category::amd_smi_temp>({ .perfetto_label = "Temperature", .perfetto_units = "deg C", .symbol = "Temp", .rocpd_units = "\u00B0C", .required = bit::k_hotspot_temperature,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.hotspot_temperature); } } }),
        with_category<category::amd_smi_temp>({ .perfetto_label = "Temperature", .perfetto_units = "deg C", .symbol = "Temp", .rocpd_units = "\u00B0C", .required = bit::k_edge_temperature, .excluded = bit::k_hotspot_temperature,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.edge_temperature); } } }),
        with_category<category::amd_smi_power>({ .perfetto_label = "Current Power", .perfetto_units = "watts", .symbol = "Pow", .rocpd_units = "W", .required = bit::k_current_socket_power,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.current_socket_power); } } }),
        with_category<category::amd_smi_power>({ .perfetto_label = "Avg. Power", .perfetto_units = "watts", .symbol = "Pow", .rocpd_units = "W", .required = bit::k_average_socket_power, .excluded = bit::k_current_socket_power,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.average_socket_power); } } }),
        with_category<category::amd_smi_memory_usage>({ .perfetto_label = "Memory Usage", .perfetto_units = "megabytes", .symbol = "MemUsg", .rocpd_units = "MB", .required = bit::k_memory_usage,
            .read = { .scalar = [](const metrics& values) { return data_size_cast<megabytes>(bytes{ static_cast<double>(values.memory_usage) }).count(); } } }),
        with_category<category::amd_smi_sdma_usage>({ .perfetto_label = "SDMA Usage", .perfetto_units = "%", .symbol = "SDMA Usage", .rocpd_units = "%", .required = bit::k_sdma_usage,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.sdma_usage); } } }),
        with_category<category::amd_smi_gfx_clock>({ .perfetto_label = "GFX Clock", .perfetto_units = "MHz", .symbol = "GFX Clock", .rocpd_units = "MHz", .required = bit::k_gfx_clock,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.gfx_clock_mhz); } } }),
        with_category<category::amd_smi_mem_clock>({ .perfetto_label = "Memory Clock", .perfetto_units = "MHz", .symbol = "Mem Clock", .rocpd_units = "MHz", .required = bit::k_mem_clock,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.mem_clock_mhz); } } }),
        with_category<category::amd_smi_xgmi_link_width>({ .perfetto_label = "XGMI Link Width", .perfetto_units = "lanes", .symbol = "XGMI Width", .rocpd_units = "lanes", .required = bit::k_xgmi,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.xgmi.link.width); } } }),
        with_category<category::amd_smi_xgmi_link_speed>({ .perfetto_label = "XGMI Link Speed", .perfetto_units = "Mbps", .symbol = "XGMI Speed", .rocpd_units = "Mbps", .required = bit::k_xgmi,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.xgmi.link.speed); } } }),
        with_category<category::amd_smi_pcie_link_width>({ .perfetto_label = "PCIe Link Width", .perfetto_units = "lanes", .symbol = "PCIe Width", .rocpd_units = "lanes", .required = bit::k_pcie,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.pcie.link.width); } } }),
        with_category<category::amd_smi_pcie_link_speed>({ .perfetto_label = "PCIe Link Speed", .perfetto_units = "MT/s", .symbol = "PCIe Speed", .rocpd_units = "MT/s", .required = bit::k_pcie,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.pcie.link.speed); } } }),
        with_category<category::amd_smi_pcie_bandwidth_acc>({ .perfetto_label = "PCIe Bandwidth Acc", .perfetto_units = "bytes", .symbol = "PCIe BW Acc", .rocpd_units = "bytes", .required = bit::k_pcie,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.pcie.bandwidth.acc); } } }),
        with_category<category::amd_smi_pcie_bandwidth_inst>({ .perfetto_label = "PCIe Bandwidth Inst", .perfetto_units = "bytes/s", .symbol = "PCIe BW Inst", .rocpd_units = "bytes/s", .required = bit::k_pcie,
            .read = { .scalar = [](const metrics& values) { return static_cast<double>(values.pcie.bandwidth.inst); } } }),
    };
    // clang-format on
}

/** Every AMD SMI GPU counter, in a fixed order. */
[[nodiscard]] inline std::vector<counter>
make_counters()
{
    constexpr auto* k_vcn_description  = "VCN (Video Decode) Engine Activity";
    constexpr auto* k_jpeg_description = "JPEG (Image Decode) Engine Activity";

    auto out = make_scalar_counters();
    append_xcp_counters<category::amd_smi_vcn_activity>(
        out,
        { .label       = "VCN Busy",
          .description = k_vcn_description,
          .required    = metric_bit::k_vcn_busy,
          .read =
              [](const metrics& values, std::size_t index) {
                  return or_absent(values.xcp_stats[index / MAX_NUM_VCN]
                                       .vcn_busy[index % MAX_NUM_VCN]);
              } },
        MAX_NUM_VCN);
    append_device_counters<category::amd_smi_vcn_activity>(
        out,
        { .label       = "VCN Activity",
          .description = k_vcn_description,
          .required    = metric_bit::k_vcn_activity,
          .read =
              [](const metrics& values, std::size_t index) {
                  return or_absent(values.vcn_activity[index]);
              } },
        MAX_NUM_VCN);
    append_xcp_counters<category::amd_smi_jpeg_activity>(
        out,
        { .label       = "JPEG Busy",
          .description = k_jpeg_description,
          .required    = metric_bit::k_jpeg_busy,
          .read =
              [](const metrics& values, std::size_t index) {
                  return or_absent(values.xcp_stats[index / MAX_NUM_JPEG_V1]
                                       .jpeg_busy[index % MAX_NUM_JPEG_V1]);
              } },
        MAX_NUM_JPEG_V1);
    append_device_counters<category::amd_smi_jpeg_activity>(
        out,
        { .label       = "JPEG Activity",
          .description = k_jpeg_description,
          .required    = metric_bit::k_jpeg_activity,
          .read =
              [](const metrics& values, std::size_t index) {
                  return or_absent(values.jpeg_activity[index]);
              } },
        MAX_NUM_JPEG);
    append_link_counters<category::amd_smi_xgmi_read_data>(
        out, { .label    = "XGMI Read Data",
               .symbol   = "XGMI Read",
               .units    = "KB",
               .required = metric_bit::k_xgmi,
               .read     = [](const metrics& values, std::size_t index) {
                   return or_absent(values.xgmi.data_acc.read[index]);
               } });
    append_link_counters<category::amd_smi_xgmi_write_data>(
        out, { .label    = "XGMI Write Data",
               .symbol   = "XGMI Write",
               .units    = "KB",
               .required = metric_bit::k_xgmi,
               .read     = [](const metrics& values, std::size_t index) {
                   return or_absent(values.xgmi.data_acc.write[index]);
               } });
    return out;
}

/** Every AMD SMI GPU counter, built once. */
[[nodiscard]] inline const std::vector<counter>&
counters()
{
    static const auto k_counters = make_counters();
    return k_counters;
}

/** A fallback counter shares the pmc_info and the rocpd track of the counter before it.
 */
[[nodiscard]] inline bool
is_fallback(const counter& item) noexcept
{
    return item.excluded != 0;
}

[[nodiscard]] inline bool
is_collected(const counter& item, std::uint32_t enabled_mask) noexcept
{
    return (enabled_mask & item.required) != 0 && (enabled_mask & item.excluded) == 0;
}

/** The counters a GPU collects for its enabled and supported metrics, in schema order. */
[[nodiscard]] inline std::vector<const counter*>
collected_counters(std::uint32_t enabled_mask)
{
    std::vector<const counter*> out;
    for(const auto& item : counters())
    {
        if(is_collected(item, enabled_mask))
        {
            out.push_back(&item);
        }
    }
    return out;
}

/** The PMC schema of one GPU for the given collected counters. */
[[nodiscard]] inline trace_cache::info::pmc_schema
make_schema(std::size_t device_id, const std::vector<const counter*>& collected)
{
    trace_cache::info::pmc_schema schema{ .type             = agent_type::gpu,
                                          .agent_type_index = device_id,
                                          .event_category =
                                              trait::name<category::amd_smi>::value,
                                          .series = {} };
    schema.series.reserve(collected.size());
    for(const auto* item : collected)
    {
        schema.series.push_back({ .perfetto_category = item->category,
                                  .perfetto_track    = fmt::format(
                                      "GPU [{}] {} (S)", device_id, item->perfetto_label),
                                  .pmc_name    = item->pmc_name,
                                  .rocpd_track = item->rocpd_track,
                                  .units       = item->perfetto_units });
    }
    return schema;
}

}  // namespace rocprofsys::pmc::collectors::gpu
