// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-gpu-common.hpp"

#include "../endian.hpp"
#include "addc/detail/ascii.hpp"
#include "addc/detail/format.hpp"

#include <string>

namespace addc::cper::sections
{

using addc::cper::read_u64_le;

namespace
{

std::string guidStringFromBytes(std::span<const uint8_t> d, std::size_t off)
{
    if (off + 16U > d.size())
    {
        return "";
    }
    const Guid g =
        Guid::parse(std::span<const uint8_t, 16>{d.data() + off, 16U});
    return g.to_string();
}

// MS Check structure (Error Info offset 24): bits 15:0 valid bits, bits 18:16
// error type, bits 19-23 PCC / uncorrected / precise IP / restartable IP /
// overflow.
nlohmann::json parseMsCheckStruct(uint64_t v)
{
    return {
        {"validBits",
         {
             {"errorTypeValid", bool(v & 0x1U)},
             {"pccValid", bool((v >> 1U) & 0x1U)},
             {"uncorrectedValid", bool((v >> 2U) & 0x1U)},
             {"preciseIpValid", bool((v >> 3U) & 0x1U)},
             {"restartableIpValid", bool((v >> 4U) & 0x1U)},
             {"overflowValid", bool((v >> 5U) & 0x1U)},
         }},
        {"errorType", (v >> 16U) & 0x7U},
        {"pcc", (v >> 19U) & 0x1U},
        {"uncorrected", (v >> 20U) & 0x1U},
        {"preciseIp", (v >> 21U) & 0x1U},
        {"restartableIp", (v >> 22U) & 0x1U},
        {"overflow", (v >> 23U) & 0x1U},
    };
}

// MI300: bits 0-4 MS Check / identifier valid bits, bit 5 FRU MPN valid,
// bit 6 Redfish ID valid. Bytes 32-63 carry four 64-bit identifiers; the MPN
// and the 8-byte Redfish ID overlay them.
nlohmann::json parseMi300ErrorInfo(std::span<const uint8_t> d)
{
    const uint64_t vb = read_u64_le(d, 16U);
    nlohmann::json j = nlohmann::json::object();
    j["errorStructureType"] = guidStringFromBytes(d, 0U);
    j["validBits"] = {
        {"checkInfoValid", bool(vb & 0x1U)},
        {"targetAddressIdentifierValid", bool((vb >> 1U) & 0x1U)},
        {"requesterIdentifierValid", bool((vb >> 2U) & 0x1U)},
        {"responderIdentifierValid", bool((vb >> 3U) & 0x1U)},
        {"instructionPointerValid", bool((vb >> 4U) & 0x1U)},
        {"fruManufacturerPartNumberValid", bool((vb >> 5U) & 0x1U)},
        {"redfishEventLogIdValid", bool((vb >> 6U) & 0x1U)},
    };
    j["msCheckStructure"] = parseMsCheckStruct(read_u64_le(d, 24U));
    j["targetIdentifier"] = addc::format("0x{:016x}", read_u64_le(d, 32U));
    j["requesterIdentifier"] = addc::format("0x{:016x}", read_u64_le(d, 40U));
    j["responderIdentifier"] = addc::format("0x{:016x}", read_u64_le(d, 48U));
    j["instructionIdentifier"] =
        addc::format("0x{:016x}", read_u64_le(d, 56U));
    j["fruManufacturerPartNumber"] =
        addc::detail::read_ascii_field(d, 32U, 16U);
    j["redfishEventLogId"] = addc::detail::read_ascii_field(d, 56U, 8U);
    return j;
}

// MI450: bit 0 Check Info valid, bit 5 FRU MPN valid, bit 7 Redfish ID valid.
// Bits 1-4 are reserved for EPYC; bit 6 and bits 8-63 are reserved. Bytes
// 32-47 are the MPN and bytes 48-63 are the Redfish ID.
nlohmann::json parseMi450ErrorInfo(std::span<const uint8_t> d)
{
    const uint64_t vb = read_u64_le(d, 16U);
    nlohmann::json j = nlohmann::json::object();
    j["errorStructureType"] = guidStringFromBytes(d, 0U);
    j["validBits"] = {
        {"checkInfoValid", bool(vb & 0x1U)},
        {"fruManufacturerPartNumberValid", bool((vb >> 5U) & 0x1U)},
        {"redfishEventLogIdValid", bool((vb >> 7U) & 0x1U)},
        {"reservedBits", addc::format("0x{:016x}", vb & ~UINT64_C(0xA1))},
    };
    j["msCheckStructure"] = parseMsCheckStruct(read_u64_le(d, 24U));
    j["fruManufacturerPartNumber"] =
        addc::detail::read_ascii_field(d, 32U, 16U);
    j["redfishEventLogId"] = addc::detail::read_ascii_field(d, 48U, 16U);
    return j;
}

} // anonymous namespace

nlohmann::json parse_amd_gpu_pldm_bundle(uint32_t raw, AmdGpuLayout layout)
{
    if (layout == AmdGpuLayout::Mi450)
    {
        // Byte 0 Patch, byte 1 Minor, byte 2 Major/Year, byte 3 bits 0-2
        // buildType and bits 3-7 productId.
        return {
            {"patch", raw & 0xFFU},
            {"minor", (raw >> 8U) & 0xFFU},
            {"majorYear", (raw >> 16U) & 0xFFU},
            {"buildType", (raw >> 24U) & 0x7U},
            {"productId", (raw >> 27U) & 0x1FU},
        };
    }
    return {
        {"minor", raw & 0xFFU},
        {"major", (raw >> 8U) & 0xFFU},
        {"year", (raw >> 16U) & 0xFFU},
        {"productionDebug", (raw >> 24U) & 0xFFU},
    };
}

nlohmann::json parse_amd_gpu_error_info(std::span<const uint8_t> body,
                                        std::size_t base, AmdGpuLayout layout)
{
    constexpr std::size_t kSize = 64U;
    if (base + kSize > body.size())
    {
        return nullptr;
    }
    const auto d = body.subspan(base, kSize);
    return layout == AmdGpuLayout::Mi450 ? parseMi450ErrorInfo(d)
                                         : parseMi300ErrorInfo(d);
}

} // namespace addc::cper::sections
