/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef ROCM_SMI_ROCM_SMI_DYN_GPU_METRICS_H_
#define ROCM_SMI_ROCM_SMI_DYN_GPU_METRICS_H_

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <variant>


namespace amd::smi
{


/*
 *  NOTES:
 *
 *  For the new dynamic metrics implementation, we need to define a `schema`.
 *  The `schema` defines the `types` of the `attributes` (or `properties`, much like a syntax) it
 *  defines the type of data that can be stored in an attribute. It acts as a blueprint.
 *
 *  If we think of the metrics system as a database, the schema is like the table structure.
 *  It defines the fields (attributes) that can be stored, their types, and any constraints on them.
 *  This allows for a flexible and extensible system where new metrics can be added without
 *  needing to change the underlying codebase significantly.
 *
 */

namespace details
{
/*
 *  NOTE:
 *  Namespace for internal details of 'dynamic gpu metrics'.
 *  This namespace contains implementation details that are not intended for public use.
 *  It is used to encapsulate the internal workings of the dynamic GPU metrics system.
 *  This allows for better organization of code and separation of concerns.
 *  The public API will interact with this namespace, but the details will be hidden from the user.
 *  This is a common practice in C++ to keep the public interface clean and maintainable.
 *
 *  Guidelines for using namespace details:
 *  -----------------------------------------
 *    * Use namespace details in headers for:
 *      - Implementation-specific types (used in templates or PImpl).
 *      - SFINAE helpers, type traits, or metaprogramming utilities.
 *      - Internal functions needed for templates/inline functions.
 *      - Internal constants or enums that are not part of the public API.
 *      - Internal classes or structs that are not meant for public use.
 *
 *    * Use namespace details in implementation files for:
 *      - Helper functions/constants not meant for public use.
 *      - Internal state management (e.g., PImpl details).
 *    * Avoid exposing namespace details in documentation or public API.
 *
 * This improves encapsulation and prevents users from relying on internal details that may change.
 *
 */

/*
 * Data types for the attributes
 */
enum class AMDGpuMetricAttributeType_t
{
    TYPE_UINT8,
    TYPE_INT8,
    TYPE_UINT16,
    TYPE_INT16,
    TYPE_UINT32,
    TYPE_INT32,
    TYPE_UINT64,
    TYPE_INT64
};

/*
 * Attribute IDs for the GPU metrics
 */
enum class AMDGpuMetricAttributeId_t
{
    TEMPERATURE_HOTSPOT,
    TEMPERATURE_MEM,
    TEMPERATURE_VRSOC,
    CURR_SOCKET_POWER,
    AVERAGE_GFX_ACTIVITY,
    AVERAGE_UMC_ACTIVITY,
    MEM_MAX_BANDWIDTH,
    ENERGY_ACCUMULATOR,
    SYSTEM_CLOCK_COUNTER,
    ACCUMULATION_COUNTER,
    PROCHOT_RESIDENCY_ACC,
    PPT_RESIDENCY_ACC,
    SOCKET_THM_RESIDENCY_ACC,
    VR_THM_RESIDENCY_ACC,
    HBM_THM_RESIDENCY_ACC,
    GFXCLK_LOCK_STATUS,
    PCIE_LINK_WIDTH,
    PCIE_LINK_SPEED,
    XGMI_LINK_WIDTH,
    XGMI_LINK_SPEED,
    GFX_ACTIVITY_ACC,
    MEM_ACTIVITY_ACC,
    PCIE_BANDWIDTH_ACC,
    PCIE_BANDWIDTH_INST,
    PCIE_L0_TO_RECOV_COUNT_ACC,
    PCIE_REPLAY_COUNT_ACC,
    PCIE_REPLAY_ROVER_COUNT_ACC,
    PCIE_NAK_SENT_COUNT_ACC,
    PCIE_NAK_RCVD_COUNT_ACC,
    XGMI_READ_DATA_ACC,
    XGMI_WRITE_DATA_ACC,
    XGMI_LINK_STATUS,
    FIRMWARE_TIMESTAMP,
    CURRENT_GFXCLK,
    CURRENT_SOCCLK,
    CURRENT_VCLK0,
    CURRENT_DCLK0,
    CURRENT_UCLK,
    NUM_PARTITION,
    PCIE_LC_PERF_OTHER_END_RECOVERY,
    GFX_BUSY_INST,
    JPEG_BUSY,
    VCN_BUSY,
    GFX_BUSY_ACC,
    GFX_BELOW_HOST_LIMIT_PPT_ACC,
    GFX_BELOW_HOST_LIMIT_THM_ACC,
    GFX_LOW_UTILIZATION_ACC,
    GFX_BELOW_HOST_LIMIT_TOTAL_ACC,
};

struct AMDGpuDynamicTranslationTextInfo_t
{
    public:
        std::string m_short_info;
        std::string m_long_info;

    private:

};
using AMDGpuMetricAttributeIdTranslationTable_t = std::unordered_map<AMDGpuMetricAttributeId_t, AMDGpuDynamicTranslationTextInfo_t>;

static const auto AMDGpuMetricAttributeIdToString = AMDGpuMetricAttributeIdTranslationTable_t {
    {AMDGpuMetricAttributeId_t::TEMPERATURE_HOTSPOT,              {"TEMPERATURE_HOTSPOT",               "Temperature of the GPU hotspot"}},
    {AMDGpuMetricAttributeId_t::TEMPERATURE_MEM,                  {"TEMPERATURE_MEM",                   "Temperature of the GPU memory"}},
    {AMDGpuMetricAttributeId_t::TEMPERATURE_VRSOC,                {"TEMPERATURE_VRSOC",                 "Temperature of the VR SOC"}},
    {AMDGpuMetricAttributeId_t::CURR_SOCKET_POWER,                {"CURR_SOCKET_POWER",                 "Current power consumption of the socket"}},
    {AMDGpuMetricAttributeId_t::AVERAGE_GFX_ACTIVITY,             {"AVERAGE_GFX_ACTIVITY",              "Average GPU activity percentage"}},
    {AMDGpuMetricAttributeId_t::AVERAGE_UMC_ACTIVITY,             {"AVERAGE_UMC_ACTIVITY",              "Average UMC activity percentage"}},
    {AMDGpuMetricAttributeId_t::MEM_MAX_BANDWIDTH,                {"MEM_MAX_BANDWIDTH",                 "Maximum memory bandwidth in GB/s"}},
    {AMDGpuMetricAttributeId_t::ENERGY_ACCUMULATOR,               {"ENERGY_ACCUMULATOR",                "Energy consumed in Joules"}},
    {AMDGpuMetricAttributeId_t::SYSTEM_CLOCK_COUNTER,             {"SYSTEM_CLOCK_COUNTER",              "System clock counter in nanoseconds"}},
    {AMDGpuMetricAttributeId_t::ACCUMULATION_COUNTER,             {"ACCUMULATION_COUNTER",              "Counter for accumulated metrics"}},
    {AMDGpuMetricAttributeId_t::PROCHOT_RESIDENCY_ACC,            {"PROCHOT_RESIDENCY_ACC",             "Accumulator for 'Processor Hot' residency time"}},
    {AMDGpuMetricAttributeId_t::PPT_RESIDENCY_ACC,                {"PPT_RESIDENCY_ACC",                 "Accumulator for 'Package Power Tracking' residency time"}},
    {AMDGpuMetricAttributeId_t::SOCKET_THM_RESIDENCY_ACC,         {"SOCKET_THM_RESIDENCY_ACC",          "Accumulator for socket thermal residency time"}},
    {AMDGpuMetricAttributeId_t::VR_THM_RESIDENCY_ACC,             {"VR_THM_RESIDENCY_ACC",              "Accumulator for 'Voltage Regulator' thermal residency time"}},
    {AMDGpuMetricAttributeId_t::HBM_THM_RESIDENCY_ACC,            {"HBM_THM_RESIDENCY_ACC",             "Accumulator for 'High Bandwidth Memory' thermal residency time"}},
    {AMDGpuMetricAttributeId_t::GFXCLK_LOCK_STATUS,               {"GFXCLK_LOCK_STATUS",                "Status of GFX clock lock"}},
    {AMDGpuMetricAttributeId_t::PCIE_LINK_WIDTH,                  {"PCIE_LINK_WIDTH",                   "Width of the PCIe link"}},
    {AMDGpuMetricAttributeId_t::PCIE_LINK_SPEED,                  {"PCIE_LINK_SPEED",                   "Speed of the PCIe link"}},
    {AMDGpuMetricAttributeId_t::XGMI_LINK_WIDTH,                  {"XGMI_LINK_WIDTH",                   "Width of the XGMI link"}},
    {AMDGpuMetricAttributeId_t::XGMI_LINK_SPEED,                  {"XGMI_LINK_SPEED",                   "Speed of the XGMI link"}},
    {AMDGpuMetricAttributeId_t::GFX_ACTIVITY_ACC,                 {"GFX_ACTIVITY_ACC",                  "Accumulator for GFX activity"}},
    {AMDGpuMetricAttributeId_t::MEM_ACTIVITY_ACC,                 {"MEM_ACTIVITY_ACC",                  "Accumulator for memory activity"}},
    {AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_ACC,               {"PCIE_BANDWIDTH_ACC",                "Accumulator for PCIe bandwidth"}},
    {AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_INST,              {"PCIE_BANDWIDTH_INST",               "Instantaneous PCIe bandwidth"}},
    {AMDGpuMetricAttributeId_t::PCIE_L0_TO_RECOV_COUNT_ACC,       {"PCIE_L0_TO_RECOV_COUNT_ACC",        "Accumulator for PCIe L0 to recovery count"}},
    {AMDGpuMetricAttributeId_t::PCIE_REPLAY_COUNT_ACC,            {"PCIE_REPLAY_COUNT_ACC",             "Accumulator for PCIe replay count"}},
    {AMDGpuMetricAttributeId_t::PCIE_REPLAY_ROVER_COUNT_ACC,      {"PCIE_REPLAY_ROVER_COUNT_ACC",       "Accumulator for PCIe replay rover count"}},
    {AMDGpuMetricAttributeId_t::PCIE_NAK_SENT_COUNT_ACC,          {"PCIE_NAK_SENT_COUNT_ACC",           "Accumulator for PCIe NAK sent count"}},
    {AMDGpuMetricAttributeId_t::PCIE_NAK_RCVD_COUNT_ACC,          {"PCIE_NAK_RCVD_COUNT_ACC",           "Accumulator for PCIe NAK received count"}},
    {AMDGpuMetricAttributeId_t::XGMI_READ_DATA_ACC,               {"XGMI_READ_DATA_ACC",                "Accumulator for XGMI read data"}},
    {AMDGpuMetricAttributeId_t::XGMI_WRITE_DATA_ACC,              {"XGMI_WRITE_DATA_ACC",               "Accumulator for XGMI write data"}},
    {AMDGpuMetricAttributeId_t::XGMI_LINK_STATUS,                 {"XGMI_LINK_STATUS",                  "Status of the XGMI link"}},
    {AMDGpuMetricAttributeId_t::FIRMWARE_TIMESTAMP,               {"Firmware Timestamp",                "Timestamp from the firmware"}},
    {AMDGpuMetricAttributeId_t::CURRENT_GFXCLK,                   {"CURRENT_GFXCLK",                    "Current GFX clock frequency in MHz"}},
    {AMDGpuMetricAttributeId_t::CURRENT_SOCCLK,                   {"CURRENT_SOCCLK",                    "Current SOC clock frequency in MHz"}},
    {AMDGpuMetricAttributeId_t::CURRENT_VCLK0,                    {"CURRENT_VCLK0",                     "Current VCLK0 frequency in MHz"}},
    {AMDGpuMetricAttributeId_t::CURRENT_DCLK0,                    {"CURRENT_DCLK0",                     "Current DCLK0 frequency in MHz"}},
    {AMDGpuMetricAttributeId_t::CURRENT_UCLK,                     {"CURRENT_UCLK",                      "Current UCLK frequency in MHz"}},
    {AMDGpuMetricAttributeId_t::NUM_PARTITION,                    {"NUM_PARTITION",                     "Number of GPU partitions"}},
    {AMDGpuMetricAttributeId_t::PCIE_LC_PERF_OTHER_END_RECOVERY,  {"PCIE_LC_PERF_OTHER_END_RECOVERY",   "PCIe link controller performance other end recovery"}},
    {AMDGpuMetricAttributeId_t::GFX_BUSY_INST,                    {"GFX_BUSY_INST",                     "Instantaneous GFX busy percentage"}},
    {AMDGpuMetricAttributeId_t::JPEG_BUSY,                        {"JPEG_BUSY",                         "JPEG engine busy percentage"}},
    {AMDGpuMetricAttributeId_t::VCN_BUSY,                         {"VCN_BUSY",                          "Video Core Next engine busy percentage"}},
    {AMDGpuMetricAttributeId_t::GFX_BUSY_ACC,                     {"GFX_BUSY_ACC",                      "Accumulator for GFX busy percentage"}},
    {AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_PPT_ACC,     {"GFX_BELOW_HOST_LIMIT_PPT_ACC",      "Accumulator for GFX below host limit due to PPT"}},
    {AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_THM_ACC,     {"GFX_BELOW_HOST_LIMIT_THM_ACC",      "Accumulator for GFX below host limit due to thermal"}},
    {AMDGpuMetricAttributeId_t::GFX_LOW_UTILIZATION_ACC,          {"GFX_LOW_UTILIZATION_ACC",           "Accumulator for GFX low utilization"}},
    {AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_TOTAL_ACC,   {"GFX_BELOW_HOST_LIMIT_TOTAL_ACC",    "Total accumulator for GFX below host limit"}},
};


/*
 * Unit types used by attribute instances
 */
enum class AMDGpuMetricUnitType_t
{
    NONE,

    /*
     * Temperature units
     */
    CELSIUS,
    CELSIUS_ACCUMULATOR,

    /*
     * Bandwidth/Data Rate units
     */
    BIT_PER_SECOND,
    BYTE_PER_SECOND,
    KILOBYTE_PER_SECOND,
    KILOBYTE_PER_SECOND_ACCUMULATOR,
    GIGABYTE_PER_SECOND,
    GIGABYTE_PER_SECOND_ACCUMULATOR,

    /*
     * Power/Energy units
     */
    WATT,
    JOULE,

    /*
     * Electrical units
     */
    VOLTAGE,

    /*
     * Time/Frequency units
     */
    TIMESTAMP_NANOSECONDS,
    CLOCK_MEGAHERTZ,

    /*
     * Unitless or generic units
     */
    PERCENT,
    COUNT_ACCUMULATOR,
    QUANTITY,
    STATUS_FLAG
};

using AMDGpuMetricUnitTypeTranslationTable_t = std::unordered_map<AMDGpuMetricUnitType_t, AMDGpuDynamicTranslationTextInfo_t>;

static const auto AMDGpuMetricUnitTypeToString = AMDGpuMetricUnitTypeTranslationTable_t {
    {AMDGpuMetricUnitType_t::NONE,                            {"NONE",                            "No unit"}},
    {AMDGpuMetricUnitType_t::CELSIUS,                         {"CELSIUS",                         "Temperature (°C)"}},
    {AMDGpuMetricUnitType_t::CELSIUS_ACCUMULATOR,             {"CELSIUS_ACCUMULATOR",             "Accumulated temperature counter (°C)"}},
    {AMDGpuMetricUnitType_t::BIT_PER_SECOND,                  {"BIT_PER_SECOND",                  "Throughput (bit/s)"}},
    {AMDGpuMetricUnitType_t::BYTE_PER_SECOND,                 {"BYTE_PER_SECOND",                 "Throughput (B/s)"}},
    {AMDGpuMetricUnitType_t::KILOBYTE_PER_SECOND,             {"KILOBYTE_PER_SECOND",             "Throughput (KB/s)"}},
    {AMDGpuMetricUnitType_t::KILOBYTE_PER_SECOND_ACCUMULATOR, {"KILOBYTE_PER_SECOND_ACCUMULATOR", "Accumulated KB/s counter"}},
    {AMDGpuMetricUnitType_t::GIGABYTE_PER_SECOND,             {"GIGABYTE_PER_SECOND",             "Throughput (GB/s)"}},
    {AMDGpuMetricUnitType_t::GIGABYTE_PER_SECOND_ACCUMULATOR, {"GIGABYTE_PER_SECOND_ACCUMULATOR", "Accumulated GB/s counter"}},
    {AMDGpuMetricUnitType_t::WATT,                            {"WATT",                            "Power (W)"}},
    {AMDGpuMetricUnitType_t::JOULE,                           {"JOULE",                           "Energy (J)"}},
    {AMDGpuMetricUnitType_t::VOLTAGE,                         {"VOLTAGE",                         "Voltage (V)"}},
    {AMDGpuMetricUnitType_t::TIMESTAMP_NANOSECONDS,           {"TIMESTAMP_NANOSECONDS",           "Timestamp / time (ns)"}},
    {AMDGpuMetricUnitType_t::CLOCK_MEGAHERTZ,                 {"CLOCK_MEGAHERTZ",                 "Frequency (MHz)"}},
    {AMDGpuMetricUnitType_t::PERCENT,                         {"PERCENT",                         "Percentage (%)"}},
    {AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR,               {"COUNT_ACCUMULATOR",               "Monotonic count"}},
    {AMDGpuMetricUnitType_t::QUANTITY,                        {"QUANTITY",                        "Unitless Quantity"}},
    {AMDGpuMetricUnitType_t::STATUS_FLAG,                     {"STATUS_FLAG",                     "Status bit/flag (bitmask)"}},
};

/*
 *  Header structure for dynamic GPU metrics
 */
struct AMDGpuDynamicMetricsHeader_v1_t
{
    public:
        uint16_t m_structure_size;
        uint8_t  m_format_revision;
        uint8_t  m_content_revision;

        static constexpr auto get_size() -> std::size_t
        {
            return sizeof(AMDGpuDynamicMetricsHeader_v1_t);
        }


    private:

};

using AMDGpuDynamicMetricsVersion_t = std::set<std::pair<std::uint8_t, std::uint8_t>>;


/*
 * Attribute IDs for the GPU metrics
 */
constexpr auto get_metric_data_type_size(AMDGpuMetricAttributeType_t attrib_type) -> std::size_t;

struct AMDGpuMetricAttributeInstance_t
{
    public:
        std::string m_name;
        std::string m_description;
        AMDGpuMetricAttributeId_t m_attribute_id;
        AMDGpuMetricAttributeType_t m_attribute_type;
        AMDGpuMetricUnitType_t m_unit_type;

        AMDGpuMetricAttributeInstance_t() = default;

        AMDGpuMetricAttributeInstance_t(const std::string& name,
                                        const std::string& description,
                                        AMDGpuMetricAttributeId_t attribute_id,
                                        AMDGpuMetricAttributeType_t attribute_type,
                                        AMDGpuMetricUnitType_t unit_type)
            : m_name(name),
              m_description(description),
              m_attribute_id(attribute_id),
              m_attribute_type(attribute_type),
              m_unit_type(unit_type)
        {
            m_unique_id = get_unique_attribute_id(attribute_id, attribute_type);

            /*
             *  The availability version is a set of pairs representing the major and minor version.
             *  This allows for tracking the availability of the metric across different versions.
             *  For now, we initialize it to an empty set, meaning the metric is available in all versions.
             */
            m_availability_version = {{0, 0}};
        }

        AMDGpuMetricAttributeInstance_t(const std::string& name,
                                        const std::string& description,
                                        AMDGpuMetricAttributeId_t attribute_id,
                                        AMDGpuMetricAttributeType_t attribute_type,
                                        AMDGpuMetricUnitType_t unit_type,
                                        const AMDGpuDynamicMetricsVersion_t& availability_version)
            : m_name(name),
              m_description(description),
              m_attribute_id(attribute_id),
              m_attribute_type(attribute_type),
              m_unit_type(unit_type),
              m_availability_version(availability_version)
        {
            m_unique_id = get_unique_attribute_id(attribute_id, attribute_type);
        }


        /*
         *  Get the unique ID of the metric instance.
         */
        constexpr auto get_unique_attribute_id(AMDGpuMetricAttributeId_t attribute_id, AMDGpuMetricAttributeType_t attribute_type) -> std::uint64_t
        {
            /*
             *  The unique ID is calculated based on the attribute ID and type.
             *  This allows for a unique identifier for each metric instance.
             *
             *  Example:
             *    If attribute_id is TEMPERATURE_MEM (1) and attribute_type is TYPE_INT32 (5),
             *    then m_unique_id will be 1 * 100 + 5 = 105.
             *
             *  We might need to revisit this, but for now, it serves as a unique identifier.
             */
            return (static_cast<std::uint64_t>(attribute_id) * 100 + static_cast<std::uint64_t>(attribute_type));
        }

        constexpr auto get_type_size() const -> std::size_t
        {
            return get_metric_data_type_size(m_attribute_type);
        }


    private:
        std::uint64_t m_unique_id;
        AMDGpuDynamicMetricsVersion_t m_availability_version;

};


/*
 *  Based on supported value types in `AMDGpuMetricAttributeType_t`
 */
using AMDGpuMetricAttributeValue_t = std::variant<std::uint8_t,  std::int8_t,
                                                  std::uint16_t, std::int16_t,
                                                  std::uint32_t, std::int32_t,
                                                  std::uint64_t, std::int64_t,
                                                  std::vector<std::uint8_t>, std::vector<std::int8_t>,
                                                  std::vector<std::uint16_t>, std::vector<std::int16_t>,
                                                  std::vector<std::uint32_t>, std::vector<std::int32_t>,
                                                  std::vector<std::uint64_t>, std::vector<std::int64_t>>;


struct AMDGpuMetricValueDataSizeVisitor_t
{
    public:
        /*
         *  Helper to check if Tp is a std::vector
         */
        template<typename Tp>
        struct is_std_vector : std::false_type {};

        template<typename U, typename Alloc>
        struct is_std_vector<std::vector<U, Alloc>> : std::true_type {};

        /*
         *  Scalar types only
         */
        template<typename Tp>
        constexpr auto operator()(const Tp& value) const -> std::size_t
        {
            if constexpr (!is_std_vector<Tp>::value) {
                return sizeof(Tp);
            } else {
                return value.size() * sizeof(typename Tp::value_type);
            }
        }

    private:

};


struct AMDGpuMetricAttributeData_t
{
    public:
        AMDGpuMetricAttributeInstance_t m_instance;
        AMDGpuMetricAttributeValue_t m_value;

        AMDGpuMetricAttributeData_t() = default;
        AMDGpuMetricAttributeData_t(const AMDGpuMetricAttributeInstance_t& metric_instance,
                                    const AMDGpuMetricAttributeValue_t& metric_value)
            : m_instance(metric_instance),
              m_value(metric_value)
        { }

        auto is_multivalued() const -> bool
        {
            return (std::holds_alternative<std::vector<std::uint8_t>>(m_value)  ||
                    std::holds_alternative<std::vector<std::int8_t>>(m_value)   ||
                    std::holds_alternative<std::vector<std::uint16_t>>(m_value) ||
                    std::holds_alternative<std::vector<std::int16_t>>(m_value)  ||
                    std::holds_alternative<std::vector<std::uint32_t>>(m_value) ||
                    std::holds_alternative<std::vector<std::int32_t>>(m_value)  ||
                    std::holds_alternative<std::vector<std::uint64_t>>(m_value) ||
                    std::holds_alternative<std::vector<std::int64_t>>(m_value));
        }

        constexpr auto get_metric_serialized_data_size() const -> std::size_t
        {
            return std::visit(AMDGpuMetricValueDataSizeVisitor_t{}, m_value);
        }


    private:

};

// Hash for enum-class keys
struct AttributeIdHash_t {
  size_t operator()(AMDGpuMetricAttributeId_t id) const noexcept {
    using U = std::underlying_type_t<AMDGpuMetricAttributeId_t>;
    return std::hash<U>{}(static_cast<U>(id));
  }
};

using AMDGpuMetricSchemaType_t = std::vector<AMDGpuMetricAttributeData_t>;

using AMDGpuMetricSchemaMapType_t =
  std::unordered_map<AMDGpuMetricAttributeId_t,
                     AMDGpuMetricAttributeData_t,
                     AttributeIdHash_t>;

// Check if type Tp has callable 0 arg member function named "is_multivalued()"
template<typename Tp, typename = void>
struct is_multivalued_attribute : std::false_type { };

template<typename Tp>
struct is_multivalued_attribute<
    Tp,
    std::void_t<decltype(std::declval<Tp>().is_multivalued())>
> : std::true_type { };

constexpr auto get_metric_data_type_size(AMDGpuMetricAttributeType_t attrib_type) -> std::size_t
{
    switch (attrib_type) {
        case (AMDGpuMetricAttributeType_t::TYPE_UINT8):
            return sizeof(std::uint8_t);

        case (AMDGpuMetricAttributeType_t::TYPE_INT8):
            return sizeof(std::int8_t);

        case (AMDGpuMetricAttributeType_t::TYPE_UINT16):
            return sizeof(std::uint16_t);

        case (AMDGpuMetricAttributeType_t::TYPE_INT16):
            return sizeof(std::int16_t);

        case (AMDGpuMetricAttributeType_t::TYPE_UINT32):
            return sizeof(std::uint32_t);

        case (AMDGpuMetricAttributeType_t::TYPE_INT32):
            return sizeof(std::int32_t);

        case (AMDGpuMetricAttributeType_t::TYPE_UINT64):
            return sizeof(std::uint64_t);

        case (AMDGpuMetricAttributeType_t::TYPE_INT64):
            return sizeof(std::int64_t);

        default:
            throw std::runtime_error("Error: Metric attribute type unknown... ");
    }
}

enum class AMDGpuMetricAttributeTypeFlag_t : std::uint32_t
{
    ATTRIBUTE_FLAG_TYPE_NONE = (0x0),
    ATTRIBUTE_FLAG_TYPE8  = (0x1 << 0),
    ATTRIBUTE_FLAG_TYPE16 = (0x1 << 1),
    ATTRIBUTE_FLAG_TYPE32 = (0x1 << 2),
    ATTRIBUTE_FLAG_TYPE64 = (0x1 << 3),
};

// Used to determine how far to skip when parsing gpu_metrics file
constexpr auto get_metric_bytes(AMDGpuMetricAttributeType_t attrib_type) -> std::size_t
{
    using Flag = AMDGpuMetricAttributeTypeFlag_t;

    switch (attrib_type) {
        case AMDGpuMetricAttributeType_t::TYPE_UINT8:
        case AMDGpuMetricAttributeType_t::TYPE_INT8:
            return static_cast<std::size_t>(Flag::ATTRIBUTE_FLAG_TYPE8);

        case AMDGpuMetricAttributeType_t::TYPE_UINT16:
        case AMDGpuMetricAttributeType_t::TYPE_INT16:
            return static_cast<std::size_t>(Flag::ATTRIBUTE_FLAG_TYPE16);

        case AMDGpuMetricAttributeType_t::TYPE_UINT32:
        case AMDGpuMetricAttributeType_t::TYPE_INT32:
            return static_cast<std::size_t>(Flag::ATTRIBUTE_FLAG_TYPE32);

        case AMDGpuMetricAttributeType_t::TYPE_UINT64:
        case AMDGpuMetricAttributeType_t::TYPE_INT64:
            return static_cast<std::size_t>(Flag::ATTRIBUTE_FLAG_TYPE64);
    }
    return 0; // Unreachable
}

constexpr auto ATTR_INST_BITS = std::uint8_t(10);       // ATTR_INST_MASK (0x000003FF)
constexpr auto ATTR_ID_BITS   = std::uint8_t(10);       // ATTR_ID_MASK   (0x000FFC00)
constexpr auto ATTR_TYPE_BITS = std::uint8_t(4);        // ATTR_TYPE_MASK (0x00F00000)
constexpr auto ATTR_UNIT_BITS = std::uint8_t(8);        // ATTR_UNIT_MASK (0xFF000000)

/*
 *  Bit shifting are constant, derived from bit sizes
 */
constexpr auto ATTR_ID_SHIFT   = (ATTR_INST_BITS);                      // 10
constexpr auto ATTR_TYPE_SHIFT = (ATTR_ID_SHIFT + ATTR_ID_BITS);        // 20
constexpr auto ATTR_UNIT_SHIFT = (ATTR_TYPE_SHIFT + ATTR_TYPE_BITS);    // 24

/*
 *  Masks are constant and used for decoding values safely
 *  - They are derived from bit sizes and shifts
 *  - They help in isolating specific fields when encoding/decoding
 */
constexpr auto ATTR_INST_MASK = static_cast<std::uint64_t>((1ULL  << ATTR_INST_BITS) - 1);
constexpr auto ATTR_ID_MASK   = static_cast<std::uint64_t>(((1ULL << ATTR_ID_BITS)   - 1) << ATTR_ID_SHIFT);
constexpr auto ATTR_TYPE_MASK = static_cast<std::uint64_t>(((1ULL << ATTR_TYPE_BITS) - 1) << ATTR_TYPE_SHIFT);
constexpr auto ATTR_UNIT_MASK = static_cast<std::uint64_t>(((1ULL << ATTR_UNIT_BITS) - 1) << ATTR_UNIT_SHIFT);


struct AMDGpuMetricAttributeDecode_t
{
    public:
        uint64_t m_attr_unit; // Unit type, currently unused
        uint64_t m_attr_type; // Type (e.g., U8, S16, U32, etc.)
        uint64_t m_attr_id; // Attribute ID (enumerated in `AMDGpuMetricAttributeId_t`)
        uint64_t m_attr_instance; // Instance count (number of values for this attribute)

        constexpr auto operator==(const AMDGpuMetricAttributeDecode_t& other) const noexcept -> bool
        {
            return ((m_attr_unit == other.m_attr_unit) &&
                    (m_attr_type == other.m_attr_type) &&
                    (m_attr_id == other.m_attr_id) &&
                    (m_attr_instance == other.m_attr_instance));
        }

    private:

};

/*
 *  Function to encode the attribute type, ID, and instance into a single uint32_t value.
 *  So we can do something like:
 *    auto attribute1 = amdgpu_metrics_enc_attr(AMDGpuMetricAttributeType_t::TYPE_UINT32,
 *                                              AMDGpuMetricAttributeId_t::GFX_BUSY_INST,
 *
*/
[[nodiscard]]
constexpr auto amdgpu_metrics_encode_attr(std::uint64_t attr_unit,
                                              std::uint64_t attr_type,
                                              std::uint64_t attr_id,
                                              std::uint64_t attr_instance) noexcept -> std::uint64_t
{
    return ((attr_unit << ATTR_UNIT_SHIFT) |
            (attr_type << ATTR_TYPE_SHIFT) |
            (attr_id   << ATTR_ID_SHIFT)   |
            (attr_instance));
}

[[nodiscard]]
constexpr auto amdgpu_metrics_decode_attr(std::uint64_t encoded_attr) noexcept -> AMDGpuMetricAttributeDecode_t
{
    return AMDGpuMetricAttributeDecode_t {
        .m_attr_unit = ((encoded_attr & ATTR_UNIT_MASK) >> ATTR_UNIT_SHIFT),
        .m_attr_type = ((encoded_attr & ATTR_TYPE_MASK) >> ATTR_TYPE_SHIFT),
        .m_attr_id =   ((encoded_attr & ATTR_ID_MASK)   >> ATTR_ID_SHIFT),
        .m_attr_instance = (encoded_attr & ATTR_INST_MASK)
    };
}

}   // namespace details


/*
 *  The AMDGpuMetricsBaseSchema is a predefined schema for GPU metrics.
 *  It contains a list of metric instances with their respective attributes and initial values.
 *  This schema is used to define the structure of the GPU metrics that can be collected.
 */
static const auto AMDGpuMetricsBaseSchema = details::AMDGpuMetricSchemaMapType_t{
    { details::AMDGpuMetricAttributeId_t::TEMPERATURE_HOTSPOT,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Temperature Hotspot",
                                                   "Temperature of the GPU hotspot",
                                                   details::AMDGpuMetricAttributeId_t::TEMPERATURE_HOTSPOT,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::TEMPERATURE_MEM,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Temperature Memory",
                                                   "Temperature of the GPU memory",
                                                   details::AMDGpuMetricAttributeId_t::TEMPERATURE_MEM,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::TEMPERATURE_VRSOC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Temperature VR SOC",
                                                   "Temperature of the VR SOC",
                                                   details::AMDGpuMetricAttributeId_t::TEMPERATURE_VRSOC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::CURR_SOCKET_POWER,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Current Socket Power",
                                                   "Current power consumption of the socket",
                                                   details::AMDGpuMetricAttributeId_t::CURR_SOCKET_POWER,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::WATT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::AVERAGE_GFX_ACTIVITY,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Average GFX Activity",
                                                   "Average GPU activity percentage",
                                                   details::AMDGpuMetricAttributeId_t::AVERAGE_GFX_ACTIVITY,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::AVERAGE_UMC_ACTIVITY,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Average UMC Activity",
                                                   "Average UMC activity percentage",
                                                   details::AMDGpuMetricAttributeId_t::AVERAGE_UMC_ACTIVITY,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::MEM_MAX_BANDWIDTH,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Memory Max Bandwidth",
                                                   "Maximum memory bandwidth in GB/s",
                                                   details::AMDGpuMetricAttributeId_t::MEM_MAX_BANDWIDTH,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::GIGABYTE_PER_SECOND),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::ENERGY_ACCUMULATOR,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Energy Accumulator",
                                                   "Energy consumed in Joules",
                                                   details::AMDGpuMetricAttributeId_t::ENERGY_ACCUMULATOR,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::JOULE),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::SYSTEM_CLOCK_COUNTER,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("System Clock Counter",
                                                   "System clock counter in nanoseconds",
                                                   details::AMDGpuMetricAttributeId_t::SYSTEM_CLOCK_COUNTER,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::TIMESTAMP_NANOSECONDS),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::ACCUMULATION_COUNTER,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Accumulation Counter",
                                                   "Counter for accumulated metrics",
                                                   details::AMDGpuMetricAttributeId_t::ACCUMULATION_COUNTER,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PROCHOT_RESIDENCY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("ProcHot Residency Accumulator",
                                                   "Accumulator for 'Processor Hot' residency time",
                                                   details::AMDGpuMetricAttributeId_t::PROCHOT_RESIDENCY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PPT_RESIDENCY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PPT Residency Accumulator",
                                                   "Accumulator for 'Package Power Tracking' residency time",
                                                   details::AMDGpuMetricAttributeId_t::PPT_RESIDENCY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::SOCKET_THM_RESIDENCY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Socket Thermal Residency Accumulator",
                                                   "Accumulator for socket thermal residency time",
                                                   details::AMDGpuMetricAttributeId_t::SOCKET_THM_RESIDENCY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::VR_THM_RESIDENCY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("VR Thermal Residency Accumulator",
                                                   "Accumulator for 'Voltage Regulator' thermal residency time",
                                                   details::AMDGpuMetricAttributeId_t::VR_THM_RESIDENCY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::HBM_THM_RESIDENCY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("HBM Thermal Residency Accumulator",
                                                   "Accumulator for 'High Bandwidth Memory' thermal residency time",
                                                   details::AMDGpuMetricAttributeId_t::HBM_THM_RESIDENCY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::CELSIUS_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFXCLK_LOCK_STATUS,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFXCLK Lock Status",
                                                   "Status of GFX clock lock",
                                                   details::AMDGpuMetricAttributeId_t::GFXCLK_LOCK_STATUS,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::STATUS_FLAG),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_LINK_WIDTH,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe Link Width",
                                                   "Current PCIe link width",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_LINK_WIDTH,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::QUANTITY),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_LINK_SPEED,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe Link Speed",
                                                   "Current PCIe link speed in GT/s",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_LINK_SPEED,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::GIGABYTE_PER_SECOND),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::XGMI_LINK_WIDTH,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("XGMI Link Width",
                                                   "Current XGMI link width",
                                                   details::AMDGpuMetricAttributeId_t::XGMI_LINK_WIDTH,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::QUANTITY),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::XGMI_LINK_SPEED,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("XGMI Link Speed",
                                                   "Current XGMI link speed in GT/s",
                                                   details::AMDGpuMetricAttributeId_t::XGMI_LINK_SPEED,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::GIGABYTE_PER_SECOND),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFX_ACTIVITY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFX Activity Accumulator",
                                                   "Accumulator for GFX activity percentage",
                                                   details::AMDGpuMetricAttributeId_t::GFX_ACTIVITY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::MEM_ACTIVITY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Memory Activity Accumulator",
                                                   "Accumulator for memory activity percentage",
                                                   details::AMDGpuMetricAttributeId_t::MEM_ACTIVITY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe Bandwidth Accumulator",
                                                   "Accumulator for PCIe bandwidth in GB/s",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::GIGABYTE_PER_SECOND_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_INST,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe Bandwidth Instantaneous",
                                                   "Instantaneous PCIe bandwidth in GB/s",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_INST,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::GIGABYTE_PER_SECOND),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_L0_TO_RECOV_COUNT_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe L0 to Recovery Count Accumulator",
                                                   "Accumulator for PCIe L0 to recovery count",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_L0_TO_RECOV_COUNT_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_COUNT_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe Replay Count Accumulator",
                                                   "Accumulator for PCIe replay count",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_COUNT_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_ROVER_COUNT_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe Replay Rover Count Accumulator",
                                                   "Accumulator for PCIe replay rover count",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_ROVER_COUNT_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_NAK_SENT_COUNT_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe NAK Sent Count Accumulator",
                                                   "Accumulator for PCIe NAK sent count",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_NAK_SENT_COUNT_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_NAK_RCVD_COUNT_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe NAK Received Count Accumulator",
                                                   "Accumulator for PCIe NAK received count",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_NAK_RCVD_COUNT_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::XGMI_READ_DATA_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("XGMI Read Data Accumulator",
                                                   "Accumulator for XGMI read data in bytes",
                                                   details::AMDGpuMetricAttributeId_t::XGMI_READ_DATA_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::KILOBYTE_PER_SECOND_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::XGMI_WRITE_DATA_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("XGMI Write Data Accumulator",
                                                   "Accumulator for XGMI write data in bytes",
                                                   details::AMDGpuMetricAttributeId_t::XGMI_WRITE_DATA_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::KILOBYTE_PER_SECOND_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::XGMI_LINK_STATUS,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("XGMI Link Status",
                                                   "Status of the XGMI link",
                                                   details::AMDGpuMetricAttributeId_t::XGMI_LINK_STATUS,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::STATUS_FLAG),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::FIRMWARE_TIMESTAMP,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Firmware Timestamp",
                                                   "Timestamp of the firmware in nanoseconds",
                                                   details::AMDGpuMetricAttributeId_t::FIRMWARE_TIMESTAMP,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::TIMESTAMP_NANOSECONDS),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::CURRENT_GFXCLK,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Current GFX Clock",
                                                   "Current GFX clock frequency in MHz",
                                                   details::AMDGpuMetricAttributeId_t::CURRENT_GFXCLK,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CLOCK_MEGAHERTZ),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::CURRENT_SOCCLK,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Current SOC Clock",
                                                   "Current SOC clock frequency in MHz",
                                                   details::AMDGpuMetricAttributeId_t::CURRENT_SOCCLK,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CLOCK_MEGAHERTZ),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::CURRENT_VCLK0,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Current VCLK0",
                                                   "Current VCLK0 frequency in MHz",
                                                   details::AMDGpuMetricAttributeId_t::CURRENT_VCLK0,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CLOCK_MEGAHERTZ),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::CURRENT_DCLK0,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Current DCLK0",
                                                   "Current DCLK0 frequency in MHz",
                                                   details::AMDGpuMetricAttributeId_t::CURRENT_DCLK0,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CLOCK_MEGAHERTZ),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::CURRENT_UCLK,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Current UCLK",
                                                   "Current UCLK frequency in MHz",
                                                   details::AMDGpuMetricAttributeId_t::CURRENT_UCLK,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::CLOCK_MEGAHERTZ),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::NUM_PARTITION,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("Number of Partitions",
                                                   "Number of partitions in the GPU",
                                                   details::AMDGpuMetricAttributeId_t::NUM_PARTITION,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::QUANTITY),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::PCIE_LC_PERF_OTHER_END_RECOVERY,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("PCIe LC Perf Other End Recovery",
                                                   "PCIe link control performance other end recovery",
                                                   details::AMDGpuMetricAttributeId_t::PCIE_LC_PERF_OTHER_END_RECOVERY,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::COUNT_ACCUMULATOR),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFX_BUSY_INST,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFX Busy Instantaneous",
                                                   "GFX Busy Instantaneous in percent",
                                                   details::AMDGpuMetricAttributeId_t::GFX_BUSY_INST,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT32,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::JPEG_BUSY,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("JPEG Busy Instantaneous",
                                                   "JPEG Busy Instantaneous in percent",
                                                   details::AMDGpuMetricAttributeId_t::JPEG_BUSY,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::VCN_BUSY,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("VCN Busy Instantaneous",
                                                   "VCN Busy Instantaneous in percent",
                                                   details::AMDGpuMetricAttributeId_t::VCN_BUSY,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT16,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFX_BUSY_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFX Busy Accumulator",
                                                   "GFX Busy Accumulator in percent",
                                                   details::AMDGpuMetricAttributeId_t::GFX_BUSY_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_PPT_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFX Below Host Limit PPT Accumulator",
                                                   "GFX Below Host Limit PPT Accumulator in percent",
                                                   details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_PPT_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_THM_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFX Below Host Limit THM Accumulator",
                                                   "GFX Below Host Limit THM Accumulator in percent",
                                                   details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_THM_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFX_LOW_UTILIZATION_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFX Low Utilization Accumulator",
                                                   "GFX Low Utilization Accumulator in percent",
                                                   details::AMDGpuMetricAttributeId_t::GFX_LOW_UTILIZATION_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }},

    { details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_TOTAL_ACC,
      details::AMDGpuMetricAttributeData_t{
          details::AMDGpuMetricAttributeInstance_t("GFX Below Host Limit Total Accumulator",
                                                   "GFX Below Host Limit Total Accumulator in percent",
                                                   details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_TOTAL_ACC,
                                                   details::AMDGpuMetricAttributeType_t::TYPE_UINT64,
                                                   details::AMDGpuMetricUnitType_t::PERCENT),
          static_cast<details::AMDGpuMetricAttributeValue_t>(0)
      }}
};


using AMDGpuDynamicMetricsOffsetMap_t = std::map<std::uint64_t, std::uint32_t>;
using AMDGpuDynamicMetricsOffsetIt_t  = AMDGpuDynamicMetricsOffsetMap_t::const_iterator;

/*
 *  This is the actual representation of the whole dynamic metrics data structure, for either:
 *    - 'gpu_metrics' file
 *    - 'xcp_metrics' file
 */
class AMDGpuDynamicMetrics_t
{
    public:
        AMDGpuDynamicMetrics_t() = default;
        AMDGpuDynamicMetrics_t(const AMDGpuDynamicMetrics_t&) = delete;
        AMDGpuDynamicMetrics_t(AMDGpuDynamicMetrics_t&& inst) noexcept
            : m_header(inst.m_header),
            m_dynamic_metrics_data(std::move(inst.m_dynamic_metrics_data)),
            m_dynamic_metrics_data_offsets(std::move(inst.m_dynamic_metrics_data_offsets)) {
        }
        AMDGpuDynamicMetrics_t& operator=(const AMDGpuDynamicMetrics_t&) = delete;
        AMDGpuDynamicMetrics_t& operator=(AMDGpuDynamicMetrics_t&& inst) {
            m_header = inst.m_header;
            m_dynamic_metrics_data = std::move(inst.m_dynamic_metrics_data);
            m_dynamic_metrics_data_offsets = std::move(inst.m_dynamic_metrics_data_offsets);
            return *this;
        }
        ~AMDGpuDynamicMetrics_t() {
            //{
            //    std::unique_lock<std::shared_mutex> lock(m_mutex);
            //    m_current_conditional_var.notify_all();
            //}
        }

        // Parsing helpers
        auto parse_from_buffer(const std::byte* data, std::size_t size) noexcept -> rsmi_status_t;
        auto parse_from_file(const std::string& metrics_file_path, std::size_t read_size = 0) -> rsmi_status_t;

        auto get_metric_rows() const noexcept
        -> const details::AMDGpuMetricSchemaType_t& { return m_dynamic_metrics_data; }

        auto get_header() const noexcept
        -> const details::AMDGpuDynamicMetricsHeader_v1_t& { return m_header; }

        /*
         *  The Cursor here, is a helper class to help with navigation within the dynamic metrics data
         *  based on the data offsets
         *
         */
        class AMDGpuDynamicMetricsCursor_t
        {
            public:
                AMDGpuDynamicMetricsCursor_t(const AMDGpuDynamicMetrics_t& metrics_data,
                                             std::uint64_t start_offset = 0)
                    : m_metrics(metrics_data),
                      m_current_offset(start_offset),
                      m_read_lock(metrics_data.m_mutex) {
                          m_current_metric_attribute =  m_metrics.m_dynamic_metrics_data_offsets.lower_bound(0);
                }

                ~AMDGpuDynamicMetricsCursor_t() = default;

            private:
                const AMDGpuDynamicMetrics_t&   m_metrics;
                std::uint64_t                   m_current_offset{0};
                AMDGpuDynamicMetricsOffsetIt_t  m_current_metric_attribute;
                mutable std::shared_lock<std::shared_mutex> m_read_lock;
        };

    private:
        std::string m_metric_source_file_path{};
        details::AMDGpuDynamicMetricsHeader_v1_t    m_header;
        uint32_t                                    m_attr_count;
        details::AMDGpuMetricSchemaType_t           m_dynamic_metrics_data{};
        AMDGpuDynamicMetricsOffsetMap_t             m_dynamic_metrics_data_offsets{};
        mutable std::shared_mutex                   m_mutex;

};

template<typename Tp>
constexpr auto is_multivalued_attribute_v = details::is_multivalued_attribute<Tp>::value;

using AMDGPUMetricsDynDataBuffer_t = std::vector<std::byte>;
rsmi_status_t read_dynamic_gpu_metrics_file(const std::string& metrics_file_path,
                                            const size_t read_size,
                                            AMDGPUMetricsDynDataBuffer_t& out);
}   // namespace amd::smi


#endif  // ROCM_SMI_ROCM_SMI_DYN_GPU_METRICS_H_
