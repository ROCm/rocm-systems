// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/decoder/pcie/decoder.hpp"

#include "addc-common/base64.hpp"
#include "addc/detail/format.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace addc::decoder
{

namespace
{

[[nodiscard]] uint32_t readU32Le(const uint8_t* p) noexcept
{
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8U) |
           (static_cast<uint32_t>(p[2]) << 16U) |
           (static_cast<uint32_t>(p[3]) << 24U);
}

[[nodiscard]] std::optional<std::vector<uint8_t>> decodeJsonBytes(
    const nlohmann::json& value)
{
    if (value.is_binary())
    {
        const auto& bytes = value.get_binary();
        return std::vector<uint8_t>{bytes.begin(), bytes.end()};
    }
    if (value.is_string())
    {
        return addc::detail::base64_decode(value.get_ref<const std::string&>());
    }
    return std::nullopt;
}

struct PlatformInfo
{
    std::vector<int> valid_bits;
    std::string_view amd_custom_reg_name;
};

[[nodiscard]] PlatformInfo getPlatformInfo(uint8_t revision_major)
{
    auto program_rev = static_cast<uint8_t>(revision_major & 0xFU);
    const auto gen_rev = static_cast<uint8_t>((revision_major >> 4U) & 0xFU);
    if (program_rev == 0U && gen_rev != 0U)
    {
        program_rev = 1U;
    }

    const bool is_venice = (program_rev == 1U && gen_rev == 3U);
    const bool is_turin = (program_rev == 1U && gen_rev == 2U);

    PlatformInfo info;
    if (is_venice)
    {
        info.valid_bits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        info.amd_custom_reg_name = "pcie_dump_component";
    }
    else if (is_turin)
    {
        info.valid_bits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        info.amd_custom_reg_name = "root_port_instance_id";
    }
    else
    {
        info.valid_bits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        info.amd_custom_reg_name = {};
    }
    return info;
}

struct ValidBitsResult
{
    uint64_t raw{};
    bool port_type_valid = false;
    bool version_valid = false;
    bool command_status_valid = false;
    bool device_id_valid = false;
    bool device_serial_valid = false;
    bool bridge_control_sts_valid = false;
    bool capability_struct_valid = false;
    bool aer_info_valid = false;
    bool device_id_rcrb_valid = false;
    bool rcrb_high_address_valid = false;
    bool flit_logging_info_valid = false;
    bool dpc_info_valid = false;
};

[[nodiscard]] ValidBitsResult parseValidBits(const std::string& hex_str,
                                             const std::vector<int>& supported)
{
    ValidBitsResult r;
    if (hex_str.empty())
    {
        return r;
    }
    try
    {
        r.raw = std::stoull(hex_str, nullptr, 16);
    }
    catch (...)
    {
        return r;
    }

    auto is_supported = [&](int bit) {
        return std::find(supported.begin(), supported.end(), bit) !=
               supported.end();
    };

    if (is_supported(0))
    {
        r.port_type_valid = (((r.raw >> 0U) & 1U) != 0U);
    }
    if (is_supported(1))
    {
        r.version_valid = (((r.raw >> 1U) & 1U) != 0U);
    }
    if (is_supported(2))
    {
        r.command_status_valid = (((r.raw >> 2U) & 1U) != 0U);
    }
    if (is_supported(3))
    {
        r.device_id_valid = (((r.raw >> 3U) & 1U) != 0U);
    }
    if (is_supported(4))
    {
        r.device_serial_valid = (((r.raw >> 4U) & 1U) != 0U);
    }
    if (is_supported(5))
    {
        r.bridge_control_sts_valid = (((r.raw >> 5U) & 1U) != 0U);
    }
    if (is_supported(6))
    {
        r.capability_struct_valid = (((r.raw >> 6U) & 1U) != 0U);
    }
    if (is_supported(7))
    {
        r.aer_info_valid = (((r.raw >> 7U) & 1U) != 0U);
    }
    if (is_supported(8))
    {
        r.device_id_rcrb_valid = (((r.raw >> 8U) & 1U) != 0U);
    }
    if (is_supported(9))
    {
        r.rcrb_high_address_valid = (((r.raw >> 9U) & 1U) != 0U);
    }
    if (is_supported(10))
    {
        r.flit_logging_info_valid = (((r.raw >> 10U) & 1U) != 0U);
    }
    if (is_supported(11))
    {
        r.dpc_info_valid = (((r.raw >> 11U) & 1U) != 0U);
    }
    return r;
}

struct UncorrErrBit
{
    uint32_t bit;
    std::string_view name;
};
inline constexpr UncorrErrBit kUncorrBits[] = {
    {4, "Data Link Protocol Error"},
    {5, "Surprise Down Error"},
    {12, "Poisoned TLP Received"},
    {13, "Flow Control Protocol Error"},
    {14, "Completion Timeout"},
    {15, "Completer Abort"},
    {16, "Unexpected Completion"},
    {17, "Receiver Overflow"},
    {18, "Malformed TLP"},
    {19, "ECRC Error"},
    {20, "Unsupported Request Error"},
    {21, "ACS Violation"},
    {22, "Uncorrectable Internal Error"},
    {23, "MC Blocked TLP"},
    {24, "AtomicOp Egress Blocked"},
    {25, "TLP Prefix Blocked Error"},
    {26, "Poisoned TLP Egress Blocked"},
    {27, "DMWr Request Egress Blocked"},
    {28, "IDE Check Failed"},
    {29, "Misrouted IDE TLP"},
    {30, "PCRC Check Failed"},
    {31, "TLP Translation Egress Blocked"},
};

struct CorrErrBit
{
    uint32_t bit;
    std::string_view name;
};
inline constexpr CorrErrBit kCorrBits[] = {
    {0, "Receiver Error"},
    {6, "Bad TLP"},
    {7, "Bad DLLP"},
    {8, "REPLAY_NUM Rollover"},
    {12, "Replay Timer Timeout"},
    {13, "Advisory Non-Fatal Error"},
    {14, "Corrected Internal Error"},
    {15, "Header Log Overflow"},
};

struct RootErrBit
{
    uint32_t bit;
    std::string_view name;
};
inline constexpr RootErrBit kRootBits[] = {
    {0, "ERR_COR Received"},
    {1, "Multiple ERR_COR Received"},
    {2, "ERR_FATAL/NONFATAL Received"},
    {3, "Multiple ERR_FATAL/NONFATAL Received"},
    {4, "First Uncorrectable Fatal"},
    {5, "Non-Fatal Error Messages Received"},
    {6, "Fatal Error Messages Received"},
};

[[nodiscard]] std::vector<std::string> decodeSetBits(
    uint32_t sts, uint32_t mask, const auto& bit_table)
{
    std::vector<std::string> result;
    const uint32_t active = sts & ~mask;
    for (const auto& [bit, name] : bit_table)
    {
        if (active & (1U << bit))
        {
            result.emplace_back(name);
        }
    }
    return result;
}

void appendAfidsForSetBits(std::vector<int>& afids, uint32_t bits,
                           int range_start)
{
    for (uint32_t bit = 0; bit < 32U; ++bit)
    {
        if ((bits & (uint32_t{1} << bit)) != 0U)
        {
            afids.push_back(range_start + static_cast<int>(bit));
        }
    }
}

[[nodiscard]] std::string jsonStr(const nlohmann::json& obj,
                                  std::string_view key)
{
    if (!obj.is_object())
    {
        return {};
    }
    auto it = obj.find(std::string{key});
    if (it == obj.end() || !it->is_string())
    {
        return {};
    }
    return it->get<std::string>();
}

[[nodiscard]] uint64_t jsonU64(const nlohmann::json& obj, std::string_view key)
{
    if (!obj.is_object())
    {
        return 0U;
    }
    auto it = obj.find(std::string{key});
    if (it == obj.end() || !it->is_number())
    {
        return 0U;
    }
    return it->get<uint64_t>();
}

[[nodiscard]] uint32_t aerValue(const nlohmann::json& aerInfo,
                                std::string_view field)
{
    if (!aerInfo.contains(field))
    {
        return 0U;
    }
    const auto& sub = aerInfo[field];
    if (!sub.is_object() || !sub.contains("value"))
    {
        return 0U;
    }
    return sub["value"].get<uint32_t>();
}

} // namespace

DecodedPcie decode_pcie(const nlohmann::json& pcie_body, uint8_t revision_major)
{
    DecodedPcie result;

    const auto platform = getPlatformInfo(revision_major);
    const auto valid_bits_str = jsonStr(pcie_body, "sectionValidBits");
    const auto vb = parseValidBits(valid_bits_str, platform.valid_bits);

    // -- Validation ---------------------------------------------------------
    result.validation["validation_bits"] = addc::format("0x{:016x}", vb.raw);
    result.validation["port_type_valid"] = vb.port_type_valid;
    result.validation["version_valid"] = vb.version_valid;
    result.validation["command_status_valid"] = vb.command_status_valid;
    result.validation["device_id_valid"] = vb.device_id_valid;
    result.validation["device_serial_valid"] = vb.device_serial_valid;
    result.validation["bridge_control_sts_valid"] = vb.bridge_control_sts_valid;
    result.validation["capability_struct_valid"] = vb.capability_struct_valid;
    result.validation["aer_info_valid"] = vb.aer_info_valid;
    result.validation["device_id_rcrb_valid"] = vb.device_id_rcrb_valid;
    result.validation["rcrb_high_address_valid"] = vb.rcrb_high_address_valid;

    // -- Read AER values from IR -------------------------------------------
    const auto& aerInfo =
        pcie_body.contains("aerInfo") ? pcie_body["aerInfo"] : nlohmann::json{};

    const uint32_t unc_sts = aerValue(aerInfo, "uncorrectable_error_status");
    const uint32_t unc_mask = aerValue(aerInfo, "uncorrectable_error_mask");
    const uint32_t cor_sts = aerValue(aerInfo, "correctable_error_status");
    const uint32_t cor_mask = aerValue(aerInfo, "correctable_error_mask");
    const uint32_t root_sts = aerValue(aerInfo, "root_error_status");
    const uint32_t unc_sev = aerValue(aerInfo, "uncorrectable_error_severity");

    const uint32_t active_cor = cor_sts & ~cor_mask;
    const uint32_t active_unc = unc_sts & ~unc_mask;
    result.afid.clear();
    appendAfidsForSetBits(result.afid, cor_sts, 25380);
    appendAfidsForSetBits(result.afid, root_sts, 25412);
    appendAfidsForSetBits(result.afid, unc_sts & unc_sev, 25444);
    appendAfidsForSetBits(result.afid, unc_sts & ~unc_sev, 25476);
    if (result.afid.empty())
    {
        result.afid.push_back(addc::kAfidSentinel);
    }

    uint32_t cor_src_id = 0U;
    uint32_t fatal_nonfatal_src_id = 0U;
    if (aerInfo.contains("error_source_id") &&
        aerInfo["error_source_id"].is_object())
    {
        const auto& esi = aerInfo["error_source_id"];
        cor_src_id =
            static_cast<uint32_t>(jsonU64(esi, "correctable_source_id"));
        fatal_nonfatal_src_id =
            static_cast<uint32_t>(jsonU64(esi, "fatal_non_fatal_source_id"));
    }

    // -- event_report -------------------------------------------------------
    auto& er = result.event_report;

    if (pcie_body.contains("portType") && pcie_body["portType"].is_object())
    {
        er["port_type"] = nlohmann::ordered_json{
            {"value",
             addc::format("0x{:08x}", jsonU64(pcie_body["portType"], "value"))},
            {"name", jsonStr(pcie_body["portType"], "name")},
        };
    }

    // Device ID
    nlohmann::ordered_json device_id_report = nlohmann::ordered_json::object();
    if (pcie_body.contains("deviceID") && pcie_body["deviceID"].is_object())
    {
        const auto& did = pcie_body["deviceID"];
        device_id_report["vendor_id"] = jsonStr(did, "vendorID_Hex");
        device_id_report["device_id"] = jsonStr(did, "deviceID_Hex");
        device_id_report["class_code"] = jsonStr(did, "classCode_Hex");

        const auto seg = jsonU64(did, "segmentNumber");
        const auto bus = jsonU64(did, "primaryOrDeviceBusNumber");
        const auto dev = jsonU64(did, "deviceNumber");
        const auto func = jsonU64(did, "functionNumber");
        device_id_report["seg_bus_device_function"] =
            addc::format("{:04x}:{:02x}:{:02x}.{:x}", seg, bus, dev, func);
        device_id_report["secondary_bus_number"] =
            jsonStr(did, "secondaryBusNumber_Hex");
    }
    er["device_id"] = device_id_report;

    // AER event report - error lists only, no AMD custom decode
    nlohmann::ordered_json aer_report = nlohmann::ordered_json::object();

    aer_report["root_error_status"] = decodeSetBits(root_sts, 0U, kRootBits);
    aer_report["uncorrectable_error_status"] =
        decodeSetBits(unc_sts, unc_mask, kUncorrBits);
    aer_report["correctable_error_status"] =
        decodeSetBits(cor_sts, cor_mask, kCorrBits);

    // error_severity per PCIe AER spec:
    //   corrected     → any unmasked bit set in Correctable Error Status
    //   fatal         → any unmasked bit set in Uncorrectable Error Status
    //                   AND the corresponding Severity bit = 1 (Fatal)
    //   non-fatal     → any unmasked bit set in Uncorrectable Error Status
    //                   AND the corresponding Severity bit = 0 (Non-Fatal)
    //   informational → all three status registers are 0x0
    // Note: Root Error Status indicates error receipt at root port only;
    //       it does not alter error severity.
    std::vector<std::string> sev;
    if (active_cor != 0U)
    {
        sev.emplace_back("corrected");
    }
    if ((active_unc & unc_sev) != 0U)
    {
        sev.emplace_back("fatal");
    }
    if ((active_unc & ~unc_sev) != 0U)
    {
        sev.emplace_back("non-fatal");
    }
    if (sev.empty())
    {
        sev.emplace_back("informational");
    }
    aer_report["error_severity"] = sev;

    // error_source
    int custom_error_source_id = -1;
    if (active_unc != 0U)
    {
        custom_error_source_id = static_cast<int>(fatal_nonfatal_src_id);
    }
    else if (active_cor != 0U)
    {
        custom_error_source_id = static_cast<int>(cor_src_id);
    }

    if (custom_error_source_id >= 0)
    {
        const auto src = static_cast<uint32_t>(custom_error_source_id);
        aer_report["error_source"] = nlohmann::ordered_json{
            {"bus", addc::format("0x{:02x}", (src >> 8U) & 0xFFU)},
            {"device", addc::format("0x{:02x}", (src >> 3U) & 0x1FU)},
            {"function", addc::format("0x{:x}", src & 0x7U)},
        };
    }

    er["aer_info"] = aer_report;

    // -- data_array ---------------------------------------------------------
    auto& da = result.data_array;
    da["validation_bits"] = addc::format("0x{:016x}", vb.raw);

    if (pcie_body.contains("portType"))
    {
        da["port_type"] =
            addc::format("0x{:08x}", jsonU64(pcie_body["portType"], "value"));
    }

    if (pcie_body.contains("version") && pcie_body["version"].is_object())
    {
        const auto minor = jsonU64(pcie_body["version"], "minor");
        const auto major = jsonU64(pcie_body["version"], "major");
        da["version"] = addc::format(
            "0x{:08x}", static_cast<uint32_t>(minor | (major << 8U)));
    }

    if (pcie_body.contains("commandStatus") &&
        pcie_body["commandStatus"].is_object())
    {
        const auto cmd = jsonU64(pcie_body["commandStatus"], "commandRegister");
        const auto sts = jsonU64(pcie_body["commandStatus"], "statusRegister");
        da["command_status"] =
            addc::format("0x{:08x}", static_cast<uint32_t>(cmd | (sts << 16U)));
    }

    da["rcrb_high_address"] = addc::format(
        "0x{:08x}",
        static_cast<uint32_t>(jsonU64(pcie_body, "rcrbHighAddress")));

    if (pcie_body.contains("deviceID") && pcie_body["deviceID"].is_object())
    {
        const auto& did = pcie_body["deviceID"];
        const auto vid = jsonU64(did, "vendorID");
        const auto devid = jsonU64(did, "deviceID");
        const auto cls = jsonU64(did, "classCode");
        const auto func = jsonU64(did, "functionNumber");
        const auto devn = jsonU64(did, "deviceNumber");
        const auto seg = jsonU64(did, "segmentNumber");
        const auto pbus = jsonU64(did, "primaryOrDeviceBusNumber");
        const auto sbus = jsonU64(did, "secondaryBusNumber");
        const auto slot = jsonU64(did, "slotNumber");
        da["device_id"] = addc::format(
            "0x{:04x}{:04x}{:06x}{:02x}{:02x}{:04x}{:02x}{:02x}{:04x}", vid,
            devid, cls, func, devn, seg, pbus, sbus, slot);
    }

    da["device_serial"] = addc::format(
        "0x{:08x}",
        static_cast<uint32_t>(jsonU64(pcie_body, "deviceSerialNumber")));

    if (pcie_body.contains("bridgeControlStatus") &&
        pcie_body["bridgeControlStatus"].is_object())
    {
        const auto sec_sts = jsonU64(pcie_body["bridgeControlStatus"],
                                     "secondaryStatusRegister");
        const auto ctrl =
            jsonU64(pcie_body["bridgeControlStatus"], "controlRegister");
        da["bridge_control_status"] = addc::format(
            "0x{:08x}", static_cast<uint32_t>(sec_sts | (ctrl << 16U)));
    }

    if (pcie_body.contains("capabilityStructure") &&
        pcie_body["capabilityStructure"].contains("data"))
    {
        auto cap_bytes =
            decodeJsonBytes(pcie_body["capabilityStructure"]["data"]);
        if (cap_bytes)
        {
            std::string hex;
            for (unsigned char& it : std::views::reverse(*cap_bytes))
            {
                hex += addc::format("{:02x}", it);
            }
            da["capability_structure"] = "0x" + hex;
        }
    }

    // The common schema nests the AER data array under "aer_info".
    nlohmann::ordered_json aer_da = nlohmann::ordered_json::object();

    uint32_t cap_hdr = 0U;
    if (aerInfo.contains("aer_capability_header") &&
        aerInfo["aer_capability_header"].is_object())
    {
        const auto& ch = aerInfo["aer_capability_header"];
        const auto cap_id = static_cast<uint32_t>(jsonU64(ch, "capability_id"));
        const auto cap_ver =
            static_cast<uint32_t>(jsonU64(ch, "capability_version"));
        const auto next_off =
            static_cast<uint32_t>(jsonU64(ch, "next_capability_offset"));
        cap_hdr = (cap_id & 0xFFFFU) | ((cap_ver & 0xFU) << 16U) |
                  ((next_off & 0xFFFU) << 20U);
    }

    const uint32_t adv_ctrl = aerValue(aerInfo, "advanced_error_cap_ctrl");
    const uint32_t root_cmd = aerValue(aerInfo, "root_error_command");

    uint32_t hdr_dw1 = 0U;
    uint32_t hdr_dw2 = 0U;
    uint32_t hdr_dw3 = 0U;
    uint32_t hdr_dw4 = 0U;
    if (aerInfo.contains("header_log_register") &&
        aerInfo["header_log_register"].is_object())
    {
        const auto& hlr = aerInfo["header_log_register"];
        hdr_dw1 = static_cast<uint32_t>(jsonU64(hlr, "dw1"));
        hdr_dw2 = static_cast<uint32_t>(jsonU64(hlr, "dw2"));
        hdr_dw3 = static_cast<uint32_t>(jsonU64(hlr, "dw3"));
        hdr_dw4 = static_cast<uint32_t>(jsonU64(hlr, "dw4"));
    }

    aer_da["capability_header"] = addc::format("0x{:08x}", cap_hdr);
    aer_da["uncorrectable_error_status"] = addc::format("0x{:08x}", unc_sts);
    aer_da["uncorrectable_error_mask"] = addc::format("0x{:08x}", unc_mask);
    aer_da["uncorrectable_error_severity"] = addc::format("0x{:08x}", unc_sev);
    aer_da["correctable_error_status"] = addc::format("0x{:08x}", cor_sts);
    aer_da["correctable_error_mask"] = addc::format("0x{:08x}", cor_mask);
    aer_da["advanced_error_capabilities_and_control"] =
        addc::format("0x{:08x}", adv_ctrl);
    aer_da["header_log_register_dw1"] = addc::format("0x{:08x}", hdr_dw1);
    aer_da["header_log_register_dw2"] = addc::format("0x{:08x}", hdr_dw2);
    aer_da["header_log_register_dw3"] = addc::format("0x{:08x}", hdr_dw3);
    aer_da["header_log_register_dw4"] = addc::format("0x{:08x}", hdr_dw4);
    aer_da["root_error_command"] = addc::format("0x{:08x}", root_cmd);
    aer_da["root_error_status"] = addc::format("0x{:08x}", root_sts);

    aer_da["error_source_id"] =
        addc::format("0x{:08x}", cor_src_id | (fatal_nonfatal_src_id << 16U));

    // AMD custom register as raw hex
    if (!platform.amd_custom_reg_name.empty() &&
        aerInfo.contains("amd_custom_data"))
    {
        auto custom_bytes = decodeJsonBytes(aerInfo["amd_custom_data"]);
        if (custom_bytes && custom_bytes->size() >= 4U)
        {
            const uint32_t amd_custom_reg = readU32Le(custom_bytes->data());
            aer_da[std::string{platform.amd_custom_reg_name}] =
                addc::format("0x{:x}", amd_custom_reg);
        }
    }

    da["aer_info"] = aer_da;

    if (pcie_body.contains("dpcInfo") && pcie_body["dpcInfo"].contains("data"))
    {
        auto dpc_bytes = decodeJsonBytes(pcie_body["dpcInfo"]["data"]);
        if (dpc_bytes && !dpc_bytes->empty())
        {
            std::string hex;
            for (unsigned char& it : std::views::reverse(*dpc_bytes))
            {
                hex += addc::format("{:02x}", it);
            }
            da["dpc_info"] = "0x" + hex;
        }
    }

    if (pcie_body.contains("flitLoggingDepth"))
    {
        da["flit_logging_depth"] = addc::format(
            "0x{:04x}",
            static_cast<uint32_t>(jsonU64(pcie_body, "flitLoggingDepth")));

        std::string flit_hex;
        if (pcie_body.contains("flitLoggingInfo") &&
            pcie_body["flitLoggingInfo"].contains("data"))
        {
            auto flit_bytes =
                decodeJsonBytes(pcie_body["flitLoggingInfo"]["data"]);
            if (flit_bytes && !flit_bytes->empty())
            {
                for (unsigned char& it : std::views::reverse(*flit_bytes))
                {
                    flit_hex += addc::format("{:02x}", it);
                }
            }
        }
        da["flit_logging_info"] = flit_hex.empty() ? "0x0" : "0x" + flit_hex;
    }

    return result;
}

} // namespace addc::decoder
