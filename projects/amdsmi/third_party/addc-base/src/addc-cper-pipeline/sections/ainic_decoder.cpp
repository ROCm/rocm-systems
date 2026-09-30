// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/ainic_decoder.hpp"

#include "addc/common.hpp"
#include "addc/pipeline/section_data.hpp"

#include <array>
#include <string_view>

namespace addc::pipeline
{

namespace
{

struct AfidEntry
{
    int afid;
    std::string_view category;
    std::string_view type;
    std::string_view severity;
};

constexpr auto kAfidTable = std::to_array<AfidEntry>({
    {60001, "FW", "ASIC_BOOT", "Uncorrected"},
    {60002, "FW", "ASIC_BOOT", "Fatal"},
    {60003, "FW", "SUC_BOOT", "Uncorrected"},
    {60004, "FW", "SUC_BOOT", "Fatal"},
    {60005, "FW", "ASIC_CRASH", "Corrected"},
    {60006, "FW", "SUC_CRASH", "Corrected"},
    {60050, "POWER", "POWER_FAULT", "Fatal"},
    {60051, "POWER", "POWER_TIMEOUT", "Uncorrected"},
    {60100, "CPLD", "SPI_COMMS", "Fatal"},
    {60101, "CPLD", "BAD_DEVICE_ID", "Fatal"},
    {60102, "CPLD", "CONFIG_REFUSED", "Uncorrected"},
    {60103, "CPLD", "SECTOR_ERASE_FAIL", "Uncorrected"},
    {60104, "CPLD", "PAGE_PROGRAM_FAIL", "Uncorrected"},
    {60105, "CPLD", "WRONG_IMAGE", "Fatal"},
    {60106, "CPLD", "LOAD_FAILED", "Fatal"},
    {60107, "CPLD", "CMD_TIMEOUT", "Fatal"},
    {60108, "CPLD", "RELOAD_FAILED", "Fatal"},
    {60200, "ASIC_CPU", "ICACHE_ECC", "Corrected"},
    {60201, "ASIC_CPU", "DCACHE_ECC", "Corrected"},
    {60202, "ASIC_CPU", "DCACHE_ECC", "Fatal"},
    {60203, "ASIC_CPU", "TLB_ECC", "Corrected"},
    {60300, "ASIC_LCC", "S0_DATA_ECC", "Corrected"},
    {60301, "ASIC_LCC", "S0_DATA_ECC", "Fatal"},
    {60302, "ASIC_LCC", "S0_TAG_ECC", "Corrected"},
    {60303, "ASIC_LCC", "S0_TAG_ECC", "Fatal"},
    {60304, "ASIC_LCC", "S1_DATA_ECC", "Corrected"},
    {60305, "ASIC_LCC", "S1_DATA_ECC", "Fatal"},
    {60306, "ASIC_LCC", "S1_TAG_ECC", "Corrected"},
    {60307, "ASIC_LCC", "S1_TAG_ECC", "Fatal"},
    {60500, "SUC_MCRAM", "MCRAM_ECC", "Corrected"},
    {60501, "SUC_MCRAM", "MCRAM_ECC", "Fatal"},
    {60502, "SUC_DRMCTM", "DRMTCM_ITCM_ECC", "Corrected"},
    {60503, "SUC_DRMCTM", "DRMTCM_ITCM_ECC", "Fatal"},
    {60504, "SUC_DRMCTM", "DRMTCM_D0TCM_ECC", "Corrected"},
    {60505, "SUC_DRMCTM", "DRMTCM_D0TCM_ECC", "Fatal"},
    {60506, "SUC_DRMCTM", "DRMTCM_D1TCM_ECC", "Corrected"},
    {60507, "SUC_DRMCTM", "DRMTCM_D1TCM_ECC", "Fatal"},
    {61000, "PCIE_PL", "LINK_TRAIN", "Uncorrected"},
    {61001, "PCIE_PL", "LINK_WIDTH", "Uncorrected"},
    {61002, "PCIE_PL", "LINK_SPEED", "Uncorrected"},
    {61005, "PCIE_PL", "SRAM_ECC", "Corrected"},
    {61006, "PCIE_PL", "SRAM_ECC", "Fatal"},
    {61100, "PCIE_TL", "AER", "Corrected"},
    {61101, "PCIE_TL", "AER", "Uncorrected"},
    {61102, "PCIE_TL", "AER", "Fatal"},
    {61500, "UALINK_PL", "LINK_TRAIN", "Uncorrected"},
    {61501, "UALINK_PL", "LINK_WIDTH", "Uncorrected"},
    {61502, "UALINK_PL", "LINK_SPEED", "Uncorrected"},
    {62000, "ETHPORT", "FEC", "Corrected"},
    {62001, "ETHPORT", "FEC", "Uncorrected"},
    {62002, "ETHPORT", "PCS", "Uncorrected"},
    {62003, "ETHPORT", "MAC", "Uncorrected"},
    {62004, "ETHPORT", "OSFP", "Corrected"},
    {62005, "ETHPORT", "OSFP", "Uncorrected"},
    {62006, "ETHPORT", "PMA", "Uncorrected"},
    {62007, "ETHPORT", "ANTL", "Uncorrected"},
    {62100, "XRMAC", "FEC_RAM_ECC", "Corrected"},
    {62101, "XRMAC", "FEC_RAM_ECC", "Fatal"},
    {62102, "XRMAC", "RBP_RAM_ECC", "Corrected"},
    {62103, "XRMAC", "RBP_RAM_ECC", "Fatal"},
    {62104, "XRMAC", "RPTR_RAM_ECC", "Corrected"},
    {62105, "XRMAC", "RPTR_RAM_ECC", "Fatal"},
    {62106, "XRMAC", "FCS", "Uncorrected"},
    {65000, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_DIE",
     "Warning"},
    {65001, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_VREG",
     "Warning"},
    {65002, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_BOARD",
     "Warning"},
    {65003, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_OSFP",
     "Warning"},
    {65004, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_DIE",
     "Critical"},
    {65005, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_VREG",
     "Critical"},
    {65006, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_BOARD",
     "Critical"},
    {65007, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_OSFP",
     "Critical"},
    {65008, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_DIE", "Fatal"},
    {65009, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_VREG", "Fatal"},
    {65010, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_BOARD", "Fatal"},
    {65011, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_OSFP", "Fatal"},
    {65500, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_BOARD_IN",
     "Warning"},
    {65501, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_BOARD_OUT",
     "Warning"},
    {65502, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_VREG_1",
     "Warning"},
    {65503, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_VREG_2",
     "Warning"},
    {65504, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_VREG_3",
     "Warning"},
    {65505, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_DIE_1",
     "Warning"},
    {65506, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_DIE_2",
     "Warning"},
    {65507, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_DIE_3",
     "Warning"},
    {65508, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_DIE_4",
     "Warning"},
    {65509, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_DIE_5",
     "Warning"},
    {65510, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_DIE_6",
     "Warning"},
    {65511, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_OSFP_1",
     "Warning"},
    {65512, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_OSFP_2",
     "Warning"},
    {65513, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_OSFP_3",
     "Warning"},
    {65514, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_OSFP_4",
     "Warning"},
    {65515, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_OSFP_5",
     "Warning"},
    {65516, "TemperatureAboveUpperWarningThreshold", "T_VULCANO_OSFP_6",
     "Warning"},
    {65517, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_BOARD_IN",
     "Critical"},
    {65518, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_BOARD_OUT",
     "Critical"},
    {65519, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_VREG_1",
     "Critical"},
    {65520, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_VREG_2",
     "Critical"},
    {65521, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_VREG_3",
     "Critical"},
    {65522, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_DIE_1",
     "Critical"},
    {65523, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_DIE_2",
     "Critical"},
    {65524, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_DIE_3",
     "Critical"},
    {65525, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_DIE_4",
     "Critical"},
    {65526, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_DIE_5",
     "Critical"},
    {65527, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_DIE_6",
     "Critical"},
    {65528, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_OSFP_1",
     "Critical"},
    {65529, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_OSFP_2",
     "Critical"},
    {65530, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_OSFP_3",
     "Critical"},
    {65531, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_OSFP_4",
     "Critical"},
    {65532, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_OSFP_5",
     "Critical"},
    {65533, "TemperatureAboveUpperCriticalThreshold", "T_VULCANO_OSFP_6",
     "Critical"},
    {65534, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_BOARD_IN",
     "Fatal"},
    {65535, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_BOARD_OUT",
     "Fatal"},
    {65536, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_VREG_1", "Fatal"},
    {65537, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_VREG_2", "Fatal"},
    {65538, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_VREG_3", "Fatal"},
    {65539, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_DIE_1", "Fatal"},
    {65540, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_DIE_2", "Fatal"},
    {65541, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_DIE_3", "Fatal"},
    {65542, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_DIE_4", "Fatal"},
    {65543, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_DIE_5", "Fatal"},
    {65544, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_DIE_6", "Fatal"},
    {65545, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_OSFP_1", "Fatal"},
    {65546, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_OSFP_2", "Fatal"},
    {65547, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_OSFP_3", "Fatal"},
    {65548, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_OSFP_4", "Fatal"},
    {65549, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_OSFP_5", "Fatal"},
    {65550, "TemperatureAboveUpperFatalThreshold", "T_VULCANO_OSFP_6", "Fatal"},
});

[[nodiscard]] std::string_view mapCperSeverity(
    std::string_view cper_name) noexcept
{
    if (cper_name == "Fatal")
    {
        return "Fatal";
    }
    if (cper_name == "Corrected")
    {
        return "Corrected";
    }
    if (cper_name == "Recoverable")
    {
        return "Uncorrected";
    }
    if (cper_name == "Informational")
    {
        return "Informational";
    }
    return cper_name;
}

[[nodiscard]] int lookupAfid(std::string_view category, std::string_view type,
                             std::string_view severity) noexcept
{
    for (const auto& entry : kAfidTable)
    {
        if (entry.category == category && entry.type == type &&
            entry.severity == severity)
        {
            return entry.afid;
        }
    }
    return addc::kAfidSentinel;
}

} // namespace

std::vector<PipelineEvent> decode_ainic(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AinicPlatformContextSection& s, const DecodeContext& context,
    [[maybe_unused]] std::string* error_out)
{

    const std::string_view afid_severity =
        mapCperSeverity(context.header_severity);
    const int afid = lookupAfid(s.category, s.type, afid_severity);

    PipelineEvent ev;

    ev.metadata["section_index"] = context.section_index;
    ev.metadata["context_structure_index"] = 0;
    ev.metadata["event_class"] = "ainic-platform-context";
    ev.metadata["event_index"] = 0;
    ev.metadata["timestamp"] = context.timestamp;
    ev.metadata["addc_version"] = {{"name", "AINIC"}, {"value", "1.0.0"}};

    ev.platform["fru"] = s.fru;
    ev.platform["fru_id"] = s.fru_id;

    ev.event_report["error_category"] = s.category;
    ev.event_report["error_type"] = s.type;
    ev.event_report["error_severity"] = afid_severity;
    ev.event_report["board_serial"] = s.board_serial;
    ev.event_report["fw_version"] = s.fw_version;

    ev.data_array = nlohmann::ordered_json::object();

    ev.validation["struct"] = true;

    ev.analysis["afid"] = addc::afid_list(afid);

    std::vector<PipelineEvent> events;
    events.push_back(std::move(ev));
    return events;
}

} // namespace addc::pipeline
