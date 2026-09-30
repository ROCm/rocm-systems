// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/detail/format.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace addc::pipeline
{

// --- Descriptor flags
// ---------------------------------------------------------

/// CPER section descriptor flags (from sectionDescriptors[i].flags).
struct SectionDescriptorFlags
{
    bool primary = false;
    bool containment_warning = false;
    bool reset = false;
    bool error_threshold_exceeded = false;
    bool resource_not_accessible = false;
    bool latent_error = false;
    bool propagated = false;
    bool overflow = false;
};

// --- Parsed section descriptor -----------------------------------------------

/// Typed representation of a CPER section descriptor.
/// Passed alongside each typed section payload to domain functions.
struct SectionDescriptor
{
    std::string guid;             ///< Lower-case hyphen-separated GUID
    uint8_t revision_major{};     ///< Raw major revision byte
    uint8_t revision_minor{};     ///< Raw minor revision byte
    SectionDescriptorFlags flags;
    std::string section_severity; ///< Per-section severity name (e.g. "Fatal")
    std::string fru;
    std::string fru_id;
    std::string platform; ///< From detect_platform()
    std::string platform_name;
    std::string addc_version;
};

// --- Support types
// ------------------------------------------------------------

struct CpuidInfo
{
    uint64_t eax{};
    uint64_t ebx{};
    uint64_t ecx{};
    uint64_t edx{};
};

// --- AMD EPYC Crashdump section
// ----------------------------------------------- GUID:
// 32ac0c78-2623-48f6-b0d0-7365725fd6ae  (program_rev != 2 or ras_gen != 2)

// --- MCA bank types
// -----------------------------------------------------------

/// One MCA bank slot decoded from a 512-byte binary slot at fixed offsets.
struct McaBankDump
{
    uint64_t ctl{};       ///< 0x00
    uint64_t status{};    ///< 0x08
    uint64_t addr{};      ///< 0x10
    uint64_t misc0{};     ///< 0x18
    uint64_t config{};    ///< 0x20
    uint64_t ipid{};      ///< 0x28
    uint64_t synd{};      ///< 0x30
    uint64_t destat{};    ///< 0x38
    uint64_t deaddr{};    ///< 0x40
    uint64_t misc1{};     ///< 0x48
    uint64_t synd1{};     ///< 0x50
    uint64_t synd2{};     ///< 0x58
    uint64_t ctl_mask{};  ///< 0x60
    uint64_t transsynd{}; ///< 0x68
    uint64_t transaddr{}; ///< 0x70
    uint64_t transstat{}; ///< 0x78
};

/// Serialize the 16-register bank window into the canonical `data_array` JSON
/// shape. This is Base's single source of truth for that shape; inputs that
/// lack some registers leave the corresponding McaBankDump fields 0. Key order
/// is fixed.
[[nodiscard]] inline nlohmann::ordered_json to_data_array(const McaBankDump& b)
{
    auto hx = [](uint64_t v) { return addc::format("0x{:016x}", v); };
    return nlohmann::ordered_json{
        {"ctl", hx(b.ctl)},
        {"status", hx(b.status)},
        {"addr", hx(b.addr)},
        {"misc0", hx(b.misc0)},
        {"config", hx(b.config)},
        {"ipid", hx(b.ipid)},
        {"synd", hx(b.synd)},
        {"destat", hx(b.destat)},
        {"deaddr", hx(b.deaddr)},
        {"misc1", hx(b.misc1)},
        {"synd1", hx(b.synd1)},
        {"synd2", hx(b.synd2)},
        {"ctl_mask", hx(b.ctl_mask)},
        {"transsynd", hx(b.transsynd)},
        {"transaddr", hx(b.transaddr)},
        {"transstat", hx(b.transstat)},
    };
}

/// MCA crashdump: up to 32 banks, each a 512-byte slot.
struct CrashdumpData
{
    static constexpr std::size_t kNumBanks = 32u;
    static constexpr std::size_t kBankStride = 512u;
    std::vector<McaBankDump> banks;
};

// --- DBG_LOG types
// ------------------------------------------------------------

/// One instance within a DBG_LOG frame. Bytes remain native inside analysis;
/// encoders convert them only when a consumer-facing JSON field requires it.
struct DbgLogInstance
{
    std::vector<uint8_t> data;
};

/// One DBG_LOG frame: an ID with one or more instances.
struct DbgLogGroup
{
    uint8_t block_id{};    ///< High byte of the 4-byte frame header
    uint16_t block_size{}; ///< Per-instance byte count
    std::vector<DbgLogInstance> instances;
};

/// One ProcessorContextInfo entry parsed from the binary blob.
///
/// context_type (binary header offset 0):
///   1 = Crashdump
///   2 = Crashdump on MCA-initiated core shutdown
///   3 = Break Event
struct EpycProcessorContext
{
    uint16_t context_type{}; ///< Binary header offset 0
    uint16_t array_size{};   ///< Register array size in bytes (header offset 2)
    uint32_t ucode{};        ///< Microcode version (header offset 4)
    uint64_t ppin{};         ///< PPIN (header offset 8)

    /// types 1, 2: 16KB buffer parsed as up to 32 MCA banks (blob offset 16).
    std::optional<CrashdumpData> crashdump;

    /// type 3: 8 outbound message registers (blob offset 16, 8 Ã— 4B LE).
    ///   msg[0] = error code type (0xA1/0xB0/0xBA/0xBF)
    ///   msg[1] = sub_type_id (bits[4:0]) + format_version (bits[6:5])
    ///   msg[2..7] = payload (varies by code type + sub type)
    std::optional<std::array<uint32_t, 8>> outbound_msg;

    /// types 1, 2, 3: DfOrigWdtAddr dump (blob offset 16400, 128B).
    std::optional<std::array<uint8_t, 128>> df_wdt_addr;

    /// types 1, 2, 3: DBG_LOG frames (blob offset 16912, variable).
    std::vector<DbgLogGroup> dbg_logs;
};

struct AmdEpycCrashdumpSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "32ac0c78-2623-48f6-b0d0-7365725fd6ae";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    bool apic_id_valid{};
    uint64_t apic_id{};
    bool cpuid_valid{};
    CpuidInfo cpuid{};

    std::vector<EpycProcessorContext> contexts;
};

// --- IA32/x64 Processor Error section ----------------------------------------
// GUID: dc3ea0b0-a144-4797-b95b-53fa242b6e1d

/// MS Check error info extracted from ProcessorErrorInfo.checkInfo.
/// Bits per UEFI spec N.2.4.2.5, derived from MCA_STATUS by firmware.
struct MsCheckInfo
{
    uint8_t
        error_type{}; ///< checkInfo.errorType.value - bits[18:16] of check info
    bool processor_context_corrupt{}; ///< bit 19 - MCA_STATUS[PCC]
    bool uncorrected{};               ///< bit 20 - MCA_STATUS[UC]
    bool precise_ip{};                ///< bit 21
    bool restartable_ip{};            ///< bit 22
    bool overflow{};                  ///< bit 23 - MCA_STATUS[OVER]
};

/// One ProcessorContextInfo entry from an IA32/x64 section.
/// register_context_type == 1 means MSR Registers (Machine Check).
/// registerArray.data is the RegisterArray[] payload - one bank, 128B,
/// 16 registers Ã— 8B at the same fixed offsets as the EPYC section.
struct Ia32ProcessorContext
{
    uint16_t register_context_type{}; ///< 1 = MSR Registers (Machine Check)
    uint16_t register_array_size{};   ///< Byte size of the register array
    uint32_t msr_address{};           ///< Unused for OOB
    uint64_t mm_register_address{};   ///< Unused for OOB
    std::optional<McaBankDump>
        bank; ///< Parsed MCA bank (register_context_type == 1)
};

/// One paired (ProcessorErrorInfo + ProcessorContextInfo) entry.
struct Ia32x64Entry
{
    std::string error_type_name;
    std::optional<MsCheckInfo>
        ms_check; ///< Present for "MS Check Error" entries
    Ia32ProcessorContext context;
};

struct Ia32x64Section
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "dc3ea0b0-a144-4797-b95b-53fa242b6e1d";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    bool apic_id_valid{};
    uint64_t apic_id{};
    bool cpuid_valid{};
    CpuidInfo cpuid{};

    std::vector<Ia32x64Entry> entries;
};

// --- PCIe section
// ------------------------------------------------------------- GUID:
// d995e954-bbc1-430f-ad91-b44dcb3c6f35 PCIe data is fully structured JSON - no
// binary payload.

struct PcieSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "d995e954-bbc1-430f-ad91-b44dcb3c6f35";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    /// Full "PCIe" body JSON from the section IR.
    nlohmann::json raw_body = nlohmann::json::object();
};

// --- AMD GPU sections
// ---------------------------------------------------------

enum class GpuContextType : uint32_t
{
    Mca = 0u,
    Crashdump = 1u,
    BootMessage = 9u,
    Unknown = 0xFFFFFFFFu,
};

struct GpuPldmBundle
{
    uint8_t minor = 0u;
    uint8_t major = 0u;
    uint8_t year = 0u;
    uint8_t production_debug = 0u;
};

/// One GPU context structure.
/// raw_bytes carries the register array verbatim.
struct GpuContextStructure
{
    GpuContextType context_type{GpuContextType::Unknown};
    std::vector<uint8_t> raw_bytes;
};

/// GPU error info structure parsed from structured JSON (no binary payload).
struct GpuErrorInfoStructure
{
    std::string error_structure_type;
    bool check_info_valid = false;
    bool target_addr_id_valid = false;
    bool requester_id_valid = false;
    bool responder_id_valid = false;
    bool instruction_pointer_valid = false;
    uint8_t error_type = 0u;
    bool pcc = false;
    bool uncorrected = false;
    bool precise_ip = false;
    bool restartable_ip = false;
    bool overflow = false;
    std::string target_identifier;
    std::string requester_identifier;
    std::string responder_identifier;
    std::string instruction_identifier;
    bool fru_mpn_valid = false;
    std::string fru_manufacturer_part_number;
    bool redfish_event_log_id_valid = false;
    std::string redfish_event_log_id;
};

/// AMD GPU Non-Standard Crashdump section.
/// Same GUID as AMD EPYC; differentiated by program_gen==2 AND ras_gen==2.
/// GUID: 32ac0c78-2623-48f6-b0d0-7365725fd6ae
struct AmdGpuCrashdumpSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "32ac0c78-2623-48f6-b0d0-7365725fd6ae";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    bool fw_id_valid = false;
    bool pcie_devid_valid = false;
    bool pldm_bundle_valid = false;
    uint8_t gpu_context_info_count = 0u;
    std::string pcie_dev_id;
    GpuPldmBundle pldm_bundle;
    std::string fw_id;

    std::vector<GpuErrorInfoStructure> error_info_structures;
    std::vector<GpuContextStructure> contexts;
};

/// Boot status buffer from a type-9 context (8 bytes at blob offset 16).
struct GpuMi450BootStatus
{
    uint32_t boot_msg_lo{}; ///< EAM0_BootMsg[3:0]
    uint32_t boot_msg_hi{}; ///< EAM0_BootMsg[7:4]
};

/// GPU context with EPYC-style register array.
/// context_type == 1 (Crashdump): MCA 32x512B banks + DF/WDT (offset 16400) +
/// DBG_LOG (offset 16912). context_type == 5 (DBG Log): compact one-frame
/// DBG_LOG at blob offset 16. context_type == 9 (Boot Status): 8-byte boot
/// status buffer only — no WDT or DBG_LOG. context_type == 3 (Break Event): no
/// decoded payload.
struct GpuMi450Context
{
    uint16_t context_type{};
    uint16_t array_size{};

    std::optional<CrashdumpData> crashdump;              ///< type 1 only
    std::optional<std::array<uint8_t, 128>> df_wdt_addr; ///< type 1 only
    std::vector<DbgLogGroup> dbg_logs;                   ///< type 1 or type 5
    std::optional<GpuMi450BootStatus> boot_status;       ///< type 9 only
};

/// AMD GPU MI450 Crashdump section.
/// Same GUID as EPYC/MI300; differentiated by program_gen==2 AND ras_gen==3.
/// GPU-style outer header with EPYC-style register array layout.
struct AmdGpuMi450CrashdumpSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "32ac0c78-2623-48f6-b0d0-7365725fd6ae";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    bool fw_id_valid = false;
    bool pcie_devid_valid = false;
    bool pldm_bundle_valid = false;
    uint8_t gpu_context_info_count = 0u;
    std::string pcie_dev_id;
    GpuPldmBundle pldm_bundle;
    std::string fw_id;

    std::vector<GpuErrorInfoStructure> error_info_structures;
    std::vector<GpuMi450Context> contexts;
};

/// One GPU context structure for MI450 runtime (context_type==1, 128-byte
/// register array).
struct GpuMi450RuntimeContext
{
    uint16_t context_type{};
    uint16_t array_size{};
    std::optional<McaBankDump> bank; ///< Decoded if array_size >= 128
};

/// AMD GPU MI450 Non-Standard Runtime section.
/// Same GUID as MI300 runtime; differentiated by program_gen==2 AND ras_gen==3.
/// GUID: 32ac0c78-2623-48f6-81a2-ac691780551d
struct AmdGpuMi450RuntimeSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "32ac0c78-2623-48f6-81a2-ac691780551d";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    bool fw_id_valid = false;
    bool pcie_devid_valid = false;
    bool pldm_bundle_valid = false;
    uint8_t gpu_error_info_count = 0u;
    uint8_t gpu_context_info_count = 0u;
    std::string pcie_dev_id;
    GpuPldmBundle pldm_bundle;
    std::string fw_id;

    std::vector<GpuErrorInfoStructure> error_info_structures;
    std::vector<GpuMi450RuntimeContext> contexts;
};

/// AMD GPU Non-Standard Runtime section.
/// GUID: 32ac0c78-2623-48f6-81a2-ac691780551d
struct AmdGpuRuntimeSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "32ac0c78-2623-48f6-81a2-ac691780551d";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    bool fw_id_valid = false;
    bool pcie_devid_valid = false;
    bool pldm_bundle_valid = false;
    uint8_t gpu_error_info_count = 0u;
    uint8_t gpu_context_info_count = 0u;
    std::string pcie_dev_id;
    GpuPldmBundle pldm_bundle;
    std::string fw_id;

    std::vector<GpuErrorInfoStructure> error_info_structures;
    std::vector<GpuContextStructure> contexts;
};

// --- AMD EPYC MTL (MPx Trace Log) section -----------------------------------
// GUID: xxxx-1022-xxxx-... (d2==0x1022), program_gen=1, ras_gen>=3

struct AmdEpycMtlSection
{
    SectionDescriptorFlags descriptor_flags;
    // MTL sections share d2==0x1022 across all variants; match on substring.
    static constexpr std::string_view kGuidSubstr = "-1022-";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    uint32_t logging_enabled{};
    uint16_t tail_offset{};
    uint16_t entries{};
    uint32_t log_version{};
    uint32_t usec_timestamp{};
    std::vector<uint8_t> raw_data;
};

// --- AMD EPYC CDD (Core Debug Dump) section ---------------------------------
// GUID: fdbe4adf-86ec-47e3-89be-693061a0f468, program_gen=1, ras_gen>=3

struct AmdEpycCddSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "fdbe4adf-86ec-47e3-89be-693061a0f468";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    uint16_t valid_bits{};
    uint64_t apic_id{};
    uint64_t cpuid_eax{};
    uint64_t cpuid_ebx{};
    uint64_t cpuid_ecx{};
    uint64_t cpuid_edx{};
    std::vector<uint8_t> payload;
};

// --- AMD EPYC PMIC section -------------------------------------------------
// GUID: a7c9d4f2-6e3b-4b91-9a2f-c8715de4136b, program_gen=1, ras_gen>=3

struct AmdEpycPmicSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "a7c9d4f2-6e3b-4b91-9a2f-c8715de4136b";

    std::string platform;
    std::string platform_name;
    std::string fru;
    std::string fru_id;
    std::string addc_version;

    uint32_t section_version{};
    uint32_t section_length{};
    uint32_t valid_bits{};
    uint8_t socket_id{};
    uint8_t channel_id{};
    uint8_t dimm_slot{};
    uint8_t pmic_index{};
    uint8_t log_04{};
    uint8_t log_05{};
    uint8_t log_06{};
    uint8_t reg_08{};
    uint8_t reg_09{};
    uint8_t reg_0A{};
    uint8_t reg_0B{};
    uint8_t reg_33{};
    std::vector<std::pair<std::string, std::string>> persistent_logs;
};

// --- AINIC Platform Context section ------------------------------------------
// GUID: 3a8f1d2e-7c4b-5e96-af01-23b456c78d9e
// Fully structured JSON - no binary payload.

struct AinicPlatformContextSection
{
    SectionDescriptorFlags descriptor_flags;
    static constexpr std::string_view kGuid =
        "3a8f1d2e-7c4b-5e96-af01-23b456c78d9e";

    std::string category;
    std::string type;
    std::string board_serial;
    std::string fw_version;
    std::string fru;
    std::string fru_id;
};

[[nodiscard]] SectionDescriptor parse_descriptor(
    const nlohmann::json& descriptor);

} // namespace addc::pipeline
