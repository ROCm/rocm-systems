// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-pcie.hpp"

#include "../base64.hpp"
#include "addc/detail/format.hpp"

namespace addc::cper::sections
{

namespace
{

// --- PCIe section GUID -------------------------------------------------------
//   d995e954-bbc1-430f-ad91-b44dcb3c6f35
inline constexpr Guid kPcieGuid{
    0xd995e954U,
    0xbbc1U,
    0x430fU,
    {0xadU, 0x91U, 0xb4U, 0x4dU, 0xcbU, 0x3cU, 0x6fU, 0x35U}};

// --- Lookup tables -----------------------------------------------------------

constexpr std::string_view pciePortTypeName(uint32_t v) noexcept
{
    switch (v)
    {
        case 0:
            return "PCI Express End Point";
        case 1:
            return "Legacy PCI End Point Device";
        case 4:
            return "Root Port";
        case 5:
            return "Upstream Switch Port";
        case 6:
            return "Downstream Switch Port";
        case 7:
            return "PCI Express to PCI/PCI-X Bridge";
        case 8:
            return "PCI/PCI-X Bridge to PCI Express Bridge";
        case 9:
            return "Root Complex Integrated Endpoint Device";
        case 10:
            return "Root Complex Event Collector";
        default:
            return "Unknown";
    }
}

constexpr std::string_view devicePortTypeName(uint32_t v) noexcept
{
    switch (v)
    {
        case 0x0:
            return "PCIE";
        case 0x1:
            return "PCI";
        case 0x4:
            return "ROOT_PORT";
        case 0x5:
            return "UPSTREAM";
        case 0x6:
            return "DOWNSTREAM";
        case 0x7:
            return "PCIE_PCI_BRIDGE";
        case 0x8:
            return "PCI_PCIE_BRIDGE";
        case 0x9:
            return "RCiEP";
        case 0xA:
            return "RCEC";
        default:
            return "";
    }
}

constexpr std::string_view severityName(uint32_t v) noexcept
{
    return ((v & 0x1U) != 0U) ? "Fatal" : "NonFatal";
}

constexpr std::string_view supportedName(uint32_t v) noexcept
{
    return ((v & 0x1U) != 0U) ? "Supported" : "NotSupported";
}

constexpr std::string_view enabledName(uint32_t v) noexcept
{
    return ((v & 0x1U) != 0U) ? "Enabled" : "Disabled";
}

constexpr std::string_view passingName(uint32_t v) noexcept
{
    return ((v & 0x1U) != 0U) ? "Passing" : "Failed";
}

// ------------------------------------------------------------------------------
// LE readers
// ------------------------------------------------------------------------------

constexpr uint16_t readU16(std::span<const uint8_t> d, std::size_t o) noexcept
{
    if (o + 2U > d.size())
    {
        return 0U;
    }
    return static_cast<uint16_t>(d[o]) |
           (static_cast<uint16_t>(d[o + 1U]) << 8U);
}

constexpr uint32_t readU32(std::span<const uint8_t> d, std::size_t o) noexcept
{
    if (o + 4U > d.size())
    {
        return 0U;
    }
    return static_cast<uint32_t>(d[o]) |
           (static_cast<uint32_t>(d[o + 1U]) << 8U) |
           (static_cast<uint32_t>(d[o + 2U]) << 16U) |
           (static_cast<uint32_t>(d[o + 3U]) << 24U);
}

constexpr uint64_t readU64(std::span<const uint8_t> d, std::size_t o) noexcept
{
    if (o + 8U > d.size())
    {
        return 0U;
    }
    uint64_t v{};
    for (std::size_t i = 0U; i < 8U; ++i)
    {
        v |= static_cast<uint64_t>(d[o + i]) << (8U * i);
    }
    return v;
}

// ------------------------------------------------------------------------------
// DeviceID (16 bytes at offset 24)
// ------------------------------------------------------------------------------
nlohmann::json buildDeviceId(std::span<const uint8_t> body, std::size_t off)
{
    const uint16_t vendor_id = readU16(body, off + 0U);
    const uint16_t device_id = readU16(body, off + 2U);
    const uint32_t class_code =
        (off + 7U < body.size())
            ? (static_cast<uint32_t>(body[off + 4U]) |
               (static_cast<uint32_t>(body[off + 5U]) << 8U) |
               (static_cast<uint32_t>(body[off + 6U]) << 16U))
            : 0U;
    const uint8_t func_num = (off + 7U < body.size()) ? body[off + 7U] : 0U;
    const uint8_t dev_num = (off + 8U < body.size()) ? body[off + 8U] : 0U;
    const uint16_t seg_num = readU16(body, off + 9U);
    const uint8_t primary_bus =
        (off + 11U < body.size()) ? body[off + 11U] : 0U;
    const uint8_t second_bus = (off + 12U < body.size()) ? body[off + 12U] : 0U;
    const uint16_t slot_num = readU16(body, off + 13U);

    return nlohmann::json{
        {"deviceID_Hex", addc::format("0x{:04x}", device_id)},
        {"vendorID_Hex", addc::format("0x{:04x}", vendor_id)},
        {"classCode_Hex", addc::format("0x{:06x}", class_code)},
        {"functionNumber_Hex", addc::format("0x{:02x}", func_num)},
        {"deviceNumber_Hex", addc::format("0x{:02x}", dev_num)},
        {"segmentNumber_Hex", addc::format("0x{:02x}", seg_num)},
        {"primaryOrDeviceBusNumber_Hex", addc::format("0x{:02x}", primary_bus)},
        {"secondaryBusNumber_Hex", addc::format("0x{:02x}", second_bus)},
        {"slotNumber_Hex", addc::format("0x{:02x}", slot_num)},
        {"deviceID", static_cast<uint64_t>(device_id)},
        {"vendorID", static_cast<uint64_t>(vendor_id)},
        {"classCode", static_cast<uint64_t>(class_code)},
        {"functionNumber", static_cast<uint64_t>(func_num)},
        {"deviceNumber", static_cast<uint64_t>(dev_num)},
        {"segmentNumber", static_cast<uint64_t>(seg_num)},
        {"primaryOrDeviceBusNumber", static_cast<uint64_t>(primary_bus)},
        {"secondaryBusNumber", static_cast<uint64_t>(second_bus)},
        {"slotNumber", static_cast<uint64_t>(slot_num)},
    };
}

// ------------------------------------------------------------------------------
// Capability Structure (60 bytes at offset 52)
// ------------------------------------------------------------------------------
nlohmann::json buildCapabilityStruct(std::span<const uint8_t> d,
                                     const ParseContext& context)
{
    // Always include data: base64 of raw bytes
    const nlohmann::json encoded = detail::encode_binary(d, context);
    if (d.size() < 60U)
    {
        return nlohmann::json{{"data", encoded}};
    }

    const uint16_t cap_hdr = readU16(d, 0U);
    const uint16_t caps = readU16(d, 2U);
    const uint32_t dev_cap = readU32(d, 4U);
    const uint16_t dev_ctrl = readU16(d, 8U);
    const uint16_t dev_sts = readU16(d, 10U);
    const uint32_t lnk_cap = readU32(d, 12U);
    const uint16_t lnk_ctrl = readU16(d, 16U);
    const uint16_t lnk_sts = readU16(d, 18U);
    const uint32_t slt_cap = readU32(d, 20U);
    const uint16_t slt_ctrl = readU16(d, 24U);
    const uint16_t slt_sts = readU16(d, 26U);
    const uint32_t root_sts = readU32(d, 32U);
    const uint32_t dev_cap2 = readU32(d, 36U);
    const uint16_t dev_ctrl2 = readU16(d, 40U);
    const uint32_t lnk_cap2 = readU32(d, 44U);
    const uint16_t lnk_ctrl2 = readU16(d, 48U);
    const uint16_t lnk_sts2 = readU16(d, 50U);

    return nlohmann::json{
        {"data", encoded},
        {"pcie_capability_header",
         {
             {"capability_id", {{"raw", cap_hdr & 0xFFU}}},
             {"next_capability_pointer", (cap_hdr >> 8U) & 0xFFU},
         }},
        {"pcie_capabilities",
         {
             {"capability_version", caps & 0xFU},
             {"device_port_type",
              {
                  {"raw", (caps >> 4U) & 0xFU},
                  {"value",
                   std::string{devicePortTypeName((caps >> 4U) & 0xFU)}},
              }},
             {"slot_implemented", bool((caps >> 8U) & 0x1U)},
             {"interrupt_message_number", (caps >> 9U) & 0x1FU},
             {"flit_mode_supported",
              std::string{supportedName((caps >> 15U) & 0x1U)}},
         }},
        {"device_capabilities",
         {
             {"max_payload_size_supported", dev_cap & 0x7U},
             {"phantom_functions_supported",
              std::string{supportedName((dev_cap >> 3U) & 0x3U)}},
             {"extended_tag_field_supported",
              std::string{supportedName((dev_cap >> 5U) & 0x1U)}},
             {"endpoint_l0s_acceptable_latency",
              {{"raw", (dev_cap >> 6U) & 0x7U}}},
             {"endpoint_l1_acceptable_latency",
              {{"raw", (dev_cap >> 9U) & 0x7U}}},
             {"role_based_error_reporting", bool((dev_cap >> 15U) & 0x1U)},
             {"err_cor_subclass_capable", bool((dev_cap >> 16U) & 0x1U)},
             {"rx_mps_fixed", (dev_cap >> 17U) & 0x1U},
             {"captured_slot_power_limit_value", (dev_cap >> 18U) & 0xFFU},
             {"captured_slot_power_limit_scale", (dev_cap >> 26U) & 0x3U},
             {"function_level_reset_capability_supported",
              std::string{supportedName((dev_cap >> 28U) & 0x1U)}},
             {"mixed_mps_supported",
              std::string{supportedName((dev_cap >> 29U) & 0x1U)}},
             {"tee_io_supported",
              std::string{supportedName((dev_cap >> 30U) & 0x1U)}},
         }},
        {"device_control",
         {
             {"correctable_error_reporting_enable",
              std::string{enabledName(dev_ctrl & 0x1U)}},
             {"non_fatal_error_reporting_enable",
              std::string{enabledName((dev_ctrl >> 1U) & 0x1U)}},
             {"fatal_error_reporting_enable",
              std::string{enabledName((dev_ctrl >> 2U) & 0x1U)}},
             {"unsupported_request_reporting_enabled",
              std::string{enabledName((dev_ctrl >> 3U) & 0x1U)}},
             {"relaxed_ordering_enable",
              std::string{enabledName((dev_ctrl >> 4U) & 0x1U)}},
             {"max_payload_size", (dev_ctrl >> 5U) & 0x7U},
             {"extended_tag_field_enable",
              std::string{enabledName((dev_ctrl >> 8U) & 0x1U)}},
             {"phantom_functions_enable",
              std::string{enabledName((dev_ctrl >> 9U) & 0x1U)}},
             {"aux_power_pm_enable",
              std::string{enabledName((dev_ctrl >> 10U) & 0x1U)}},
             {"enable_no_snoop", (dev_ctrl >> 11U) & 0x1U},
             {"max_read_request_size", (dev_ctrl >> 12U) & 0x7U},
             {"function_level_reset", bool((dev_ctrl >> 15U) & 0x1U)},
         }},
        {"device_status",
         {
             {"correctable_error_detected", bool(dev_sts & 0x1U)},
             {"non_fatal_error_detected", bool((dev_sts >> 1U) & 0x1U)},
             {"fatal_error_detected", bool((dev_sts >> 2U) & 0x1U)},
             {"unsupported_request_detected", bool((dev_sts >> 3U) & 0x1U)},
             {"aux_power_detected", bool((dev_sts >> 4U) & 0x1U)},
             {"transactions_pending", bool((dev_sts >> 5U) & 0x1U)},
             {"emergency_power_reduction", (dev_sts >> 6U) & 0x3U},
         }},
        {"link_capabilities",
         {
             {"max_link_speed", lnk_cap & 0xFU},
             {"maximum_link_width", (lnk_cap >> 4U) & 0x3FU},
             {"aspm_support", (lnk_cap >> 10U) & 0x3U},
             {"l0s_exit_latency", (lnk_cap >> 12U) & 0x7U},
             {"l1_exit_latency", (lnk_cap >> 15U) & 0x7U},
             {"clock_power_management", bool((lnk_cap >> 18U) & 0x1U)},
             {"surprise_down_error_reporting_capable",
              bool((lnk_cap >> 19U) & 0x1U)},
             {"data_link_layer_link_active_reporting_capable",
              bool((lnk_cap >> 20U) & 0x1U)},
             {"link_bandwidth_notification_capability",
              bool((lnk_cap >> 21U) & 0x1U)},
             {"aspm_optionality_compliance", bool((lnk_cap >> 22U) & 0x1U)},
             {"port_number", (lnk_cap >> 24U) & 0xFFU},
         }},
        {"link_control",
         {
             {"aspm_control", lnk_ctrl & 0x3U},
             {"ptm_prop_delay_adaptation_interpretation",
              bool((lnk_ctrl >> 2U) & 0x1U)},
             {"link_disable", (lnk_ctrl >> 4U) & 0x1U},
             {"retrain_link", (lnk_ctrl >> 5U) & 0x1U},
             {"extended_synch", (lnk_ctrl >> 7U) & 0x1U},
             {"sris_clocking", (lnk_ctrl >> 12U) & 0x1U},
             {"flit_mode_disable", (lnk_ctrl >> 13U) & 0x1U},
         }},
        {"link_status",
         {
             {"current_link_speed", lnk_sts & 0xFU},
             {"negotiated_link_width", (lnk_sts >> 4U) & 0x3FU},
             {"link_training", (lnk_sts >> 11U) & 0x1U},
         }},
        {"slot_capabilities",
         {
             {"slot_power_limit_value", {{"raw", (slt_cap >> 7U) & 0xFFU}}},
             {"slot_power_limit_scale", (slt_cap >> 15U) & 0x3U},
             {"physical_slot_number", (slt_cap >> 19U) & 0x1FFFU},
         }},
        {"slot_control",
         {
             {"attention_indicator_control", (slt_ctrl >> 6U) & 0x3U},
             {"power_indicator_control", (slt_ctrl >> 8U) & 0x3U},
         }},
        {"slot_status",
         {
             {"mrl_sensor_changed", (slt_sts >> 2U) & 0x1U},
             {"command_completed", (slt_sts >> 4U) & 0x1U},
             {"mrl_sensor_state", (slt_sts >> 5U) & 0x1U},
         }},
        {"root_status",
         {
             {"pme_requester_id", root_sts & 0xFFFFU},
             {"pme_status", (root_sts >> 16U) & 0x1U},
             {"pme_pending", (root_sts >> 17U) & 0x1U},
         }},
        {"device_capabilities2",
         {
             {"completion_timeout_ranges_supported", dev_cap2 & 0xFU},
             {"completion_timeout_disable_supported",
              std::string{supportedName((dev_cap2 >> 4U) & 0x1U)}},
             {"ari_forwarding_supported",
              std::string{supportedName((dev_cap2 >> 5U) & 0x1U)}},
             {"atomic_op_routing_supported",
              std::string{supportedName((dev_cap2 >> 6U) & 0x1U)}},
             {"_32_bit_atomicop_completer_supported",
              std::string{supportedName((dev_cap2 >> 7U) & 0x1U)}},
             {"_64_bit_atomicop_completer_supported",
              std::string{supportedName((dev_cap2 >> 8U) & 0x1U)}},
             {"_128_bit_cas_completer_supported",
              std::string{supportedName((dev_cap2 >> 9U) & 0x1U)}},
             {"no_ro_enabled_pr_pr_passing",
              std::string{passingName((dev_cap2 >> 10U) & 0x1U)}},
             {"ltr_mechanism_supported",
              std::string{supportedName((dev_cap2 >> 11U) & 0x1U)}},
             {"tph_completer_supported",
              std::string{supportedName((dev_cap2 >> 12U) & 0x3U)}},
             {"obff_supported",
              std::string{supportedName((dev_cap2 >> 18U) & 0x3U)}},
             {"max_end_end_tlp_prefixes", {{"raw", (dev_cap2 >> 22U) & 0x3U}}},
             {"emergency_power_reduction_supported",
              std::string{supportedName((dev_cap2 >> 24U) & 0x3U)}},
             {"dmwr_lengths_supported", (dev_cap2 >> 29U) & 0x3U},
         }},
        {"device_control2",
         {
             {"completion_timeout_value", dev_ctrl2 & 0xFU},
             {"ari_forwarding_enable", bool((dev_ctrl2 >> 5U) & 0x1U)},
             {"atomicop_requester_enable", bool((dev_ctrl2 >> 6U) & 0x1U)},
             {"atomicop_egress_blocking", bool((dev_ctrl2 >> 7U) & 0x1U)},
             {"ido_request_enable", bool((dev_ctrl2 >> 8U) & 0x1U)},
             {"ido_completion_enable", bool((dev_ctrl2 >> 9U) & 0x1U)},
             {"ltr_mechanism_enable", bool((dev_ctrl2 >> 10U) & 0x1U)},
             {"emergency_power_reduction_request",
              bool((dev_ctrl2 >> 11U) & 0x1U)},
             {"bit_tag_requester_10_enable", bool((dev_ctrl2 >> 12U) & 0x1U)},
             {"obff_enable", (dev_ctrl2 >> 13U) & 0x3U},
         }},
        {"link_capabilities2",
         {
             {"supported_link_speeds", (lnk_cap2 >> 1U) & 0x7FU},
             {"crosslink_supported",
              std::string{supportedName((lnk_cap2 >> 8U) & 0x1U)}},
             {"lower_skp_os_generation_supported", (lnk_cap2 >> 9U) & 0x7FU},
             {"lower_skp_os_reception_supported", (lnk_cap2 >> 16U) & 0x7FU},
             {"retimer_presence_detect_supported",
              std::string{supportedName((lnk_cap2 >> 23U) & 0x1U)}},
             {"two_retimers_presence_detect_supported",
              std::string{supportedName((lnk_cap2 >> 24U) & 0x1U)}},
             {"drs_supported",
              std::string{supportedName((lnk_cap2 >> 31U) & 0x1U)}},
         }},
        {"link_control2",
         {
             {"target_link_speed", {{"raw", lnk_ctrl2 & 0xFU}}},
             {"enter_compliance",
              std::string{supportedName((lnk_ctrl2 >> 4U) & 0x1U)}},
             {"hardware_autonomous_speed_disable",
              {{"raw", (lnk_ctrl2 >> 5U) & 0x1U}}},
             {"selectable_de_emphasis", bool((lnk_ctrl2 >> 6U) & 0x1U)},
             {"transmit_margin", (lnk_ctrl2 >> 7U) & 0x7U},
             {"enter_modified_compliance", bool((lnk_ctrl2 >> 10U) & 0x1U)},
             {"compliance_sos", bool((lnk_ctrl2 >> 11U) & 0x1U)},
             {"compliance_preset_de_emphasis", (lnk_ctrl2 >> 12U) & 0xFU},
         }},
        {"link_status2",
         {
             {"current_de_emphasis_level", lnk_sts2 & 0x1U},
             {"equalization_8gts_complete", bool((lnk_sts2 >> 1U) & 0x1U)},
             {"equalization_8gts_phase1_successful",
              bool((lnk_sts2 >> 2U) & 0x1U)},
             {"equalization_8gts_phase2_successful",
              bool((lnk_sts2 >> 3U) & 0x1U)},
             {"equalization_8gts_phase3_successful",
              bool((lnk_sts2 >> 4U) & 0x1U)},
             {"link_equalization_request_8gts", bool((lnk_sts2 >> 5U) & 0x1U)},
             {"retimer_presence_detected", bool((lnk_sts2 >> 6U) & 0x1U)},
             {"two_retimers_presence_detected", bool((lnk_sts2 >> 7U) & 0x1U)},
             {"crosslink_resolution", (lnk_sts2 >> 8U) & 0x3U},
             {"flit_mode_status", (lnk_sts2 >> 10U) & 0x1U},
             {"downstream_component_presence", (lnk_sts2 >> 12U) & 0x7U},
             {"drs_message_received", bool((lnk_sts2 >> 15U) & 0x1U)},
         }},
    };
}

// ------------------------------------------------------------------------------
// AER Info (96 bytes at offset 112)
// ------------------------------------------------------------------------------
nlohmann::json buildAerInfo(std::span<const uint8_t> d,
                            std::size_t deviated_offset,
                            const ParseContext& context)
{
    const nlohmann::json encoded = detail::encode_binary(d, context);
    if (d.size() < deviated_offset + 56U)
    {
        return nlohmann::json{{"data", encoded}};
    }

    nlohmann::json result;
    result["data"] = encoded;

    if (deviated_offset > 0U && d.size() >= deviated_offset)
    {
        result["amd_custom_data"] =
            detail::encode_binary(d.subspan(0U, deviated_offset), context);
    }

    const auto o = deviated_offset;
    const uint32_t cap_hdr = readU32(d, o + 0U);
    const uint32_t unc_sts = readU32(d, o + 4U);
    const uint32_t unc_mask = readU32(d, o + 8U);
    const uint32_t unc_sev = readU32(d, o + 12U);
    const uint32_t cor_sts = readU32(d, o + 16U);
    const uint32_t cor_mask = readU32(d, o + 20U);
    const uint32_t adv_ctrl = readU32(d, o + 24U);
    const uint32_t hdr_dw1 = readU32(d, o + 28U);
    const uint32_t hdr_dw2 = readU32(d, o + 32U);
    const uint32_t hdr_dw3 = readU32(d, o + 36U);
    const uint32_t hdr_dw4 = readU32(d, o + 40U);
    const uint32_t root_cmd = readU32(d, o + 44U);
    const uint32_t root_sts = readU32(d, o + 48U);
    const uint32_t err_src = readU32(d, o + 52U);

    result["aer_capability_header"] = nlohmann::json{
        {"capability_id", cap_hdr & 0xFFFFU},
        {"capability_version", (cap_hdr >> 16U) & 0xFU},
        {"next_capability_offset", (cap_hdr >> 20U) & 0xFFFU},
    };
    result["uncorrectable_error_status"] = nlohmann::json{
        {"value", unc_sts},
        {"data_link_protocol_error_status", bool((unc_sts >> 4U) & 0x1U)},
        {"surprise_down_error_status", bool((unc_sts >> 5U) & 0x1U)},
        {"poisoned_tlp_received", bool((unc_sts >> 12U) & 0x1U)},
        {"flow_control_protocol_error_status", bool((unc_sts >> 13U) & 0x1U)},
        {"completion_timeout_status", bool((unc_sts >> 14U) & 0x1U)},
        {"completer_abort_status", bool((unc_sts >> 15U) & 0x1U)},
        {"unexpected_completion_status", bool((unc_sts >> 16U) & 0x1U)},
        {"receiver_overflow_status", bool((unc_sts >> 17U) & 0x1U)},
        {"malformed_tlp_status", bool((unc_sts >> 18U) & 0x1U)},
        {"ecrc_error_status", bool((unc_sts >> 19U) & 0x1U)},
        {"unsupported_request_error_status", bool((unc_sts >> 20U) & 0x1U)},
        {"acs_violation_status", bool((unc_sts >> 21U) & 0x1U)},
        {"uncorrectable_internal_error_status", bool((unc_sts >> 22U) & 0x1U)},
        {"mc_blocked_tlp_status", bool((unc_sts >> 23U) & 0x1U)},
        {"atomicop_egress_blocked_status", bool((unc_sts >> 24U) & 0x1U)},
        {"tlp_prefix_blocked_error_status", bool((unc_sts >> 25U) & 0x1U)},
        {"poisoned_tlp_egress_blocked_status", bool((unc_sts >> 26U) & 0x1U)},
        {"dmwr_request_egress_blocked_status", bool((unc_sts >> 27U) & 0x1U)},
        {"ide_check_failed_status", bool((unc_sts >> 28U) & 0x1U)},
        {"misrouted_ide_tlp_status", bool((unc_sts >> 29U) & 0x1U)},
        {"pcrc_check_failed_status", bool((unc_sts >> 30U) & 0x1U)},
        {"tlp_translation_egress_blocked_status",
         bool((unc_sts >> 31U) & 0x1U)},
    };
    result["uncorrectable_error_mask"] = nlohmann::json{
        {"value", unc_mask},
        {"data_link_protocol_error_mask", (unc_mask >> 4U) & 0x1U},
        {"surprise_down_error_mask", (unc_mask >> 5U) & 0x1U},
        {"poisoned_tlp_received_mask", (unc_mask >> 12U) & 0x1U},
        {"flow_control_protocol_error_mask", (unc_mask >> 13U) & 0x1U},
        {"completion_timeout_mask", (unc_mask >> 14U) & 0x1U},
        {"completer_abort_mask", (unc_mask >> 15U) & 0x1U},
        {"unexpected_completion_mask", (unc_mask >> 16U) & 0x1U},
        {"receiver_overflow_mask", (unc_mask >> 17U) & 0x1U},
        {"malformed_tlp_mask", (unc_mask >> 18U) & 0x1U},
        {"ecrc_error_mask", (unc_mask >> 19U) & 0x1U},
        {"unsupported_request_error_mask", (unc_mask >> 20U) & 0x1U},
        {"acs_violation_mask", (unc_mask >> 21U) & 0x1U},
        {"uncorrectable_internal_error_mask", (unc_mask >> 22U) & 0x1U},
        {"mc_blocked_tlp_mask", (unc_mask >> 23U) & 0x1U},
        {"atomicop_egress_blocked_mask", (unc_mask >> 24U) & 0x1U},
        {"tlp_prefix_blocked_error_mask", (unc_mask >> 25U) & 0x1U},
        {"poisoned_tlp_egress_blocked_mask", (unc_mask >> 26U) & 0x1U},
        {"dmwr_request_egress_blocked_mask", (unc_mask >> 27U) & 0x1U},
        {"ide_check_failed_mask", (unc_mask >> 28U) & 0x1U},
        {"misrouted_ide_tlp_mask", (unc_mask >> 29U) & 0x1U},
        {"pcrc_check_failed_mask", (unc_mask >> 30U) & 0x1U},
        {"tlp_translation_egress_blocked_mask", (unc_mask >> 31U) & 0x1U},
    };
    result["uncorrectable_error_severity"] = nlohmann::json{
        {"value", unc_sev},
        {"data_link_protocol_error_severity",
         std::string{severityName((unc_sev >> 4U) & 0x1U)}},
        {"surprise_down_error_severity",
         std::string{severityName((unc_sev >> 5U) & 0x1U)}},
        {"poisoned_tlp_received_severity",
         std::string{severityName((unc_sev >> 12U) & 0x1U)}},
        {"flow_control_protocol_error_severity",
         std::string{severityName((unc_sev >> 13U) & 0x1U)}},
        {"completion_timeout_severity",
         std::string{severityName((unc_sev >> 14U) & 0x1U)}},
        {"completer_abort_severity",
         std::string{severityName((unc_sev >> 15U) & 0x1U)}},
        {"unexpected_completion_severity",
         std::string{severityName((unc_sev >> 16U) & 0x1U)}},
        {"receiver_overflow_severity",
         std::string{severityName((unc_sev >> 17U) & 0x1U)}},
        {"malformed_tlp_severity",
         std::string{severityName((unc_sev >> 18U) & 0x1U)}},
        {"ecrc_error_severity",
         std::string{severityName((unc_sev >> 19U) & 0x1U)}},
        {"unsupported_request_error_severity",
         std::string{severityName((unc_sev >> 20U) & 0x1U)}},
        {"acs_violation_severity",
         std::string{severityName((unc_sev >> 21U) & 0x1U)}},
        {"uncorrectable_internal_error_severity",
         std::string{severityName((unc_sev >> 22U) & 0x1U)}},
    };
    result["correctable_error_status"] = nlohmann::json{
        {"value", cor_sts},
        {"receiver_error_status", bool(cor_sts & 0x1U)},
        {"bad_tlp_status", bool((cor_sts >> 6U) & 0x1U)},
        {"bad_dllp_status", bool((cor_sts >> 7U) & 0x1U)},
        {"replay_num_rollover_status", bool((cor_sts >> 8U) & 0x1U)},
        {"replay_timer_timeout_status", bool((cor_sts >> 12U) & 0x1U)},
        {"advisory_non_fatal_error_status", bool((cor_sts >> 13U) & 0x1U)},
        {"corrected_internal_error_status", bool((cor_sts >> 14U) & 0x1U)},
        {"header_log_overflow_status", bool((cor_sts >> 15U) & 0x1U)},
    };
    result["correctable_error_mask"] = nlohmann::json{
        {"value", cor_mask},
        {"receiver_error_mask", (cor_mask) & 0x1U},
        {"bad_tlp_mask", (cor_mask >> 6U) & 0x1U},
        {"bad_dllp_mask", (cor_mask >> 7U) & 0x1U},
        {"replay_num_rollover_mask", (cor_mask >> 8U) & 0x1U},
        {"replay_timer_timeout_mask", (cor_mask >> 12U) & 0x1U},
        {"advisory_non_fatal_error_mask", (cor_mask >> 13U) & 0x1U},
        {"corrected_internal_error_mask", (cor_mask >> 14U) & 0x1U},
        {"header_log_overflow_mask", (cor_mask >> 15U) & 0x1U},
    };
    result["advanced_error_cap_ctrl"] = nlohmann::json{
        {"value", adv_ctrl},
        {"first_error_pointer", adv_ctrl & 0x1FU},
        {"ecrc_generation_capable", bool((adv_ctrl >> 5U) & 0x1U)},
        {"ecrc_generation_enable", bool((adv_ctrl >> 6U) & 0x1U)},
        {"ecrc_check_capable", bool((adv_ctrl >> 7U) & 0x1U)},
        {"ecrc_check_enable", bool((adv_ctrl >> 8U) & 0x1U)},
        {"multiple_header_recording_capable", bool((adv_ctrl >> 9U) & 0x1U)},
        {"multiple_header_recording_enable", bool((adv_ctrl >> 10U) & 0x1U)},
        {"tlp_prefix_log_present", bool((adv_ctrl >> 11U) & 0x1U)},
    };
    result["header_log_register"] = nlohmann::json{
        {"dw1", hdr_dw1},
        {"dw2", hdr_dw2},
        {"dw3", hdr_dw3},
        {"dw4", hdr_dw4},
    };
    result["root_error_command"] = nlohmann::json{
        {"value", root_cmd},
        {"correctable_error_reporting_enable", bool(root_cmd & 0x1U)},
        {"non_fatal_error_reporting_enable", bool((root_cmd >> 1U) & 0x1U)},
        {"fatal_error_reporting_enable", bool((root_cmd >> 2U) & 0x1U)},
    };
    result["root_error_status"] = nlohmann::json{
        {"value", root_sts},
        {"error_correctable", bool(root_sts & 0x1U)},
        {"multiple_error_correctable", bool((root_sts >> 1U) & 0x1U)},
        {"error_fatal_non_fatal", bool((root_sts >> 2U) & 0x1U)},
        {"multiple_error_fatal_non_fatal", bool((root_sts >> 3U) & 0x1U)},
        {"first_uncorrectable_fatal", bool((root_sts >> 4U) & 0x1U)},
        {"non_fatal_error_messages_received", bool((root_sts >> 5U) & 0x1U)},
        {"fatal_error_messages_received", bool((root_sts >> 6U) & 0x1U)},
        {"pme_requester_id", root_sts & 0xFFFFU},
        {"pme_status", (root_sts >> 16U) & 0x1U},
        {"pme_pending", (root_sts >> 17U) & 0x1U},
    };
    result["error_source_id"] = nlohmann::json{
        {"correctable_source_id", err_src & 0xFFFFU},
        {"fatal_non_fatal_source_id", (err_src >> 16U) & 0xFFFFU},
    };

    return result;
}

} // anonymous namespace

// --- PCIe classification ----------------------------------------------------

bool matches_pcie(const RecordHeader& /*header*/,
                  const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kPcieGuid;
}

// --- PCIe parsing -----------------------------------------------------------
// Body layout:
//   0:   8-byte valid bits
//   8:   4-byte port type
//   12:  4-byte version (minor@0, major@8)
//   16:  4-byte command status (command@0, status@16)
//   20:  4-byte RCRB high address
//   24: 16-byte device ID
//   40:  8-byte device serial number
//   48:  4-byte bridge control status
//   52: 60-byte capability struct
//   112: 96-byte AER info
//   208: 92-byte DPC info (stored as base64)
//   300:  2-byte flit logging depth
//   302: 60*depth bytes flit logging info

nlohmann::json parse_pcie(std::span<const uint8_t> body,
                          const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_pcie(body, desc, context);
}

nlohmann::json parse_pcie(std::span<const uint8_t> body,
                          const SectionDescriptor& desc,
                          ParseContext& context)
{
    if (body.size() < 8U)
    {
        return nullptr;
    }

    const uint64_t valid_bits = readU64(body, 0U);
    const uint32_t port_type_raw = readU32(body, 8U);
    const uint32_t version_raw = readU32(body, 12U);
    const uint32_t cmd_sts_raw = readU32(body, 16U);
    const uint64_t rcrb_hi = readU32(body, 20U);
    const uint64_t device_serial = readU64(body, 40U);
    const uint32_t bridge_sts = readU32(body, 48U);

    auto safe_span =
        [&](std::size_t off, std::size_t len) -> std::span<const uint8_t> {
        if (off >= body.size())
        {
            return {};
        }
        const std::size_t avail = body.size() - off;
        return body.subspan(off, std::min(len, avail));
    };

    nlohmann::json j;
    j["sectionValidBits"] = addc::format("0x{:x}", valid_bits);

    j["portType"] = {
        {"value", port_type_raw},
        {"name", std::string{pciePortTypeName(port_type_raw)}},
    };
    j["version"] = {
        {"minor", version_raw & 0xFFU},
        {"major", (version_raw >> 8U) & 0xFFU},
    };
    j["commandStatus"] = {
        {"commandRegister", cmd_sts_raw & 0xFFFFU},
        {"statusRegister", (cmd_sts_raw >> 16U) & 0xFFFFU},
    };
    j["rcrbHighAddress"] = rcrb_hi;
    j["deviceID"] =
        (body.size() >= 40U) ? buildDeviceId(body, 24U) : nlohmann::json{};
    j["deviceSerialNumber"] = device_serial;
    j["bridgeControlStatus"] = {
        {"secondaryStatusRegister", bridge_sts & 0xFFFFU},
        {"controlRegister", (bridge_sts >> 16U) & 0xFFFFU},
    };
    j["capabilityStructure"] =
        buildCapabilityStruct(safe_span(52U, 60U), context);

    auto program_rev = static_cast<uint8_t>(desc.revision_major & 0xFU);
    const auto gen_rev =
        static_cast<uint8_t>((desc.revision_major >> 4U) & 0xFU);
    const uint8_t minor_version = static_cast<uint8_t>(version_raw & 0xFFU);
    if (program_rev == 0U && gen_rev != 0U)
    {
        program_rev = 1U;
    }

    // Venice 0xE or later will have UEFI compliance so no need to deviate from
    // the spec. Venice 0xD and earlier will have a deviated AER layout.
    const bool is_venice_deviated =
        (program_rev == 1U && gen_rev == 3U && minor_version < 0xEU);
    const bool is_turin = (program_rev == 1U && gen_rev == 2U);
    const std::size_t aer_offset = (is_venice_deviated || is_turin) ? 4U : 0U;

    j["aerInfo"] = buildAerInfo(safe_span(112U, 96U), aer_offset, context);

    // DPC info (92 bytes at 208) - stored as raw base64
    j["dpcInfo"] = nlohmann::json{
        {"data", detail::encode_binary(safe_span(208U, 92U), context)}};

    // Flit logging
    // TODO: Decode flit logging info into structured fields (flit mode, flit
    // log entries,
    //       etc.) per PCIe 6.0+ spec. Currently raw base64 only - no decoding
    //       in either software decoder. See PCIe Gen 6 spec section on Flit
    //       Mode Logging.
    const uint16_t flit_depth = (body.size() > 300U) ? readU16(body, 300U) : 0U;
    j["flitLoggingDepth"] = static_cast<uint64_t>(flit_depth);
    const std::size_t flit_size = static_cast<std::size_t>(flit_depth) * 60U;
    j["flitLoggingInfo"] = nlohmann::json{
        {"data", detail::encode_binary(safe_span(302U, flit_size), context)}};

    return j;
}

} // namespace addc::cper::sections
