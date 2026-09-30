// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-mi300-runtime.hpp"

#include "../base64.hpp"
#include "../endian.hpp"
#include "addc/detail/ascii.hpp"
#include "addc/detail/format.hpp"

#include <string>

namespace addc::cper::sections
{

using addc::cper::read_u16_le;
using addc::cper::read_u32_le;
using addc::cper::read_u64_le;

namespace
{

inline constexpr Guid kMi300RuntimeGuid{
    0x32ac0c78U,
    0x2623U,
    0x48f6U,
    {0x81U, 0xa2U, 0xacU, 0x69U, 0x17U, 0x80U, 0x55U, 0x1dU}};

// --- Null-terminated ASCII from raw bytes
// ------------------------------------- Reads len bytes from d at off as a
// null-terminated ASCII string.
// --- GUID from 16 raw bytes (EFI byte order) ---------------------------------
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

// --- ValidBits ---------------------------------------------------------------
nlohmann::json parseValidBitsRuntime(uint64_t v)
{
    return {
        {"fwIdValid", bool((v >> 1U) & 0x1U)},
        {"gpuErrorInfoCount", (v >> 2U) & 0x3FU},
        {"gpuContextInfoCount", (v >> 8U) & 0x3FU},
        {"pcieDevidValid", bool((v >> 14U) & 0x1U)},
        {"pldmBndlValid", bool((v >> 15U) & 0x1U)},
    };
}

// --- PldmBundle --------------------------------------------------------------
nlohmann::json parsePldmBundle(uint32_t v)
{
    return {
        {"minor", v & 0xFFU},
        {"major", (v >> 8U) & 0xFFU},
        {"year", (v >> 16U) & 0xFFU},
        {"productionDebug", (v >> 24U) & 0xFFU},
    };
}

// --- MsCheckStructure ValidBits ----------------------------------------------
nlohmann::json parseMsCheckValidBits(uint64_t v)
{
    return {
        {"checkInfoValid", bool(v & 0x1U)},
        {"targetAddressIdentifierValid", bool((v >> 1U) & 0x1U)},
        {"requesterIdentifierValid", bool((v >> 2U) & 0x1U)},
        {"responderIdentifierValid", bool((v >> 3U) & 0x1U)},
        {"instructionPointerValid", bool((v >> 4U) & 0x1U)},
    };
}

// --- GpuContextInfoStructureMsCheckStructureValidBits ------------------------
nlohmann::json parseMsCheckStructVbits(uint64_t v)
{
    return {
        {"errorTypeValid", bool(v & 0x1U)},
        {"pccValid", bool((v >> 1U) & 0x1U)},
        {"uncorrectedValid", bool((v >> 2U) & 0x1U)},
        {"preciseIpValid", bool((v >> 3U) & 0x1U)},
        {"restartableIpValid", bool((v >> 4U) & 0x1U)},
        {"overflowValid", bool((v >> 5U) & 0x1U)},
    };
}

// --- GpuContextInfoStructureMsCheckStructure ---------------------------------
nlohmann::json parseMsCheckStruct(uint64_t v)
{
    return {
        {"validBits", parseMsCheckStructVbits(v & 0xFFFFU)},
        {"errorType", (v >> 16U) & 0x7U},
        {"pcc", (v >> 19U) & 0x1U},
        {"uncorrected", (v >> 20U) & 0x1U},
        {"preciseIp", (v >> 21U) & 0x1U},
        {"restartableIp", (v >> 22U) & 0x1U},
        {"overflow", (v >> 23U) & 0x1U},
    };
}

// --- MI300 Error Info Structure (64 bytes) -----------------------------------
nlohmann::json parseMi300ErrorInfo(std::span<const uint8_t> body,
                                   std::size_t base)
{
    if (base + 64U > body.size())
    {
        return nullptr;
    }
    const auto d = body.subspan(base, 64U);
    const uint64_t vb = read_u64_le(d, 16U);
    const uint64_t ms_raw = read_u64_le(d, 24U);
    const uint64_t target = read_u64_le(d, 32U);
    const uint64_t requester = read_u64_le(d, 40U);
    const uint64_t responder = read_u64_le(d, 48U);
    const uint64_t instr = read_u64_le(d, 56U);
    nlohmann::json j = nlohmann::json::object();
    j["errorStructureType"] = guidStringFromBytes(d, 0U);

    nlohmann::json vb_j = parseMsCheckValidBits(vb);
    vb_j["fruManufacturerPartNumberValid"] = bool((vb >> 5U) & 0x1U);
    vb_j["redfishEventLogIdValid"] = bool((vb >> 6U) & 0x1U);
    j["validBits"] = std::move(vb_j);

    j["msCheckStructure"] = parseMsCheckStruct(ms_raw);
    j["targetIdentifier"] = addc::format("0x{:016x}", target);
    j["requesterIdentifier"] = addc::format("0x{:016x}", requester);
    j["responderIdentifier"] = addc::format("0x{:016x}", responder);
    j["instructionIdentifier"] = addc::format("0x{:016x}", instr);
    j["fruManufacturerPartNumber"] =
        addc::detail::read_ascii_field(d, 32U, 16U);
    j["redfishEventLogId"] =
        addc::detail::read_ascii_field(d, 56U, 8U);
    return j;
}

// --- MI300 Context Structure (runtime - with msrAddress + mmRegisterAddress) -
nlohmann::json parseMi300ContextRuntime(std::span<const uint8_t> body,
                                        std::size_t base,
                                        const ParseContext& context)
{
    if (base + 4U > body.size())
    {
        return nullptr;
    }
    const uint16_t ctx_type = read_u16_le(body, base);
    const uint16_t arr_size = read_u16_le(body, base + 2U);
    const uint32_t msr_addr = read_u32_le(body, base + 4U);
    const uint64_t mm_addr = read_u64_le(body, base + 8U);
    const std::size_t data_off = base + 16U;
    const std::size_t avail =
        (data_off < body.size()) ? body.size() - data_off : 0U;
    const std::size_t take = std::min<std::size_t>(arr_size, avail);
    return {
        {"registerContextType", static_cast<uint64_t>(ctx_type)},
        {"registerArraySize", static_cast<uint64_t>(arr_size)},
        {"msrAddress", addc::format("0x{:08x}", msr_addr)},
        {"mmRegisterAddress", addc::format("0x{:016x}", mm_addr)},
        {"registerArray",
         detail::encode_binary(body.subspan(data_off, take), context)},
    };
}

} // anonymous namespace

// --- MI300 runtime classification -------------------------------------------

bool matches_amd_mi300_runtime(const RecordHeader& /*header*/,
                               const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kMi300RuntimeGuid;
}

// --- MI300 runtime parsing --------------------------------------------------
// Body layout:
//   0:  u64 validBits
//   8:  4-byte pcieDevId  -> null-terminated ASCII
//  12:  u32 pldmBundle
//  16: 48-byte fwId       -> null-terminated ASCII
//  64 + i*64: MI300 ErrorInfoStructure[gpuErrorInfoCount]  (each 64 bytes)
//  after: MI300 ContextStructure[] (variable), gpuContextInfoCount of them

nlohmann::json parse_amd_mi300_runtime(std::span<const uint8_t> body,
                                       const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_mi300_runtime(body, desc, context);
}

nlohmann::json parse_amd_mi300_runtime(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/,
    ParseContext& context)
{
    if (body.size() < 8U)
    {
        return nullptr;
    }

    const uint64_t valid_raw = read_u64_le(body, 0U);
    const auto valid_bits_j = parseValidBitsRuntime(valid_raw);
    const uint64_t err_count = (valid_raw >> 2U) & 0x3FU;
    const uint64_t ctx_count = (valid_raw >> 8U) & 0x3FU;

    const std::string pcie_dev_id =
        addc::detail::read_ascii_field(body, 8U, 4U);
    const uint32_t pldm_raw = read_u32_le(body, 12U);
    const std::string fw_id =
        addc::detail::read_ascii_field(body, 16U, 48U);

    // Per spec (Table 2.4): ErrorInfoStructures at offset 64, each 64 bytes,
    // followed by ContextStructures. Counts are per-section.
    // Note: some CPER generators populate the count fields with the total
    // across all sections in the record rather than per-section. When the
    // declared counts exceed the section body size, fall back to 1.
    uint64_t actual_err = err_count;
    uint64_t actual_ctx = ctx_count;
    if (64U + actual_err * 64U + actual_ctx * 16U > body.size())
    {
        actual_err = 1U;
        actual_ctx = 1U;
        if (err_count != actual_err &&
            !context.repair(
                "gpu_error_count_substituted", "gpu_error_info_count",
                err_count, actual_err,
                "declared GPU entry counts exceed the section body"))
        {
            return nullptr;
        }
        if (ctx_count != actual_ctx &&
            !context.repair(
                "gpu_context_count_substituted", "gpu_context_info_count",
                ctx_count, actual_ctx,
                "declared GPU entry counts exceed the section body"))
        {
            return nullptr;
        }
    }

    nlohmann::json error_infos = nlohmann::json::array();
    std::size_t cur = 64U;
    for (uint64_t i = 0U; i < actual_err; ++i)
    {
        if (cur + 64U > body.size())
        {
            break;
        }
        auto entry = parseMi300ErrorInfo(body, cur);
        if (!entry.is_null())
        {
            error_infos.push_back(std::move(entry));
        }
        cur += 64U;
    }
    if (error_infos.size() != actual_err &&
        !context.repair("gpu_error_count_truncated", "gpu_error_info_count",
                        actual_err, error_infos.size(),
                        "GPU error entries do not fit in the section body"))
    {
        return nullptr;
    }

    nlohmann::json ctx_structs = nlohmann::json::array();
    for (uint64_t i = 0U; i < actual_ctx; ++i)
    {
        if (cur + 4U > body.size())
        {
            break;
        }
        const uint16_t arr_size = read_u16_le(body, cur + 2U);
        const std::size_t struct_sz = 16U + arr_size;
        if (cur + struct_sz > body.size())
        {
            break;
        }
        auto entry = parseMi300ContextRuntime(body, cur, context);
        if (!entry.is_null())
        {
            ctx_structs.push_back(std::move(entry));
        }
        cur += struct_sz;
    }
    if (ctx_structs.size() != actual_ctx &&
        !context.repair("gpu_context_count_truncated", "gpu_context_info_count",
                        actual_ctx, ctx_structs.size(),
                        "GPU context entries do not fit in the section body"))
    {
        return nullptr;
    }

    return {
        {"validBits", valid_bits_j},
        {"pcieDevId", pcie_dev_id},
        {"pldmBundle", parsePldmBundle(pldm_raw)},
        {"fwId", fw_id},
        {"gpuErrorInfoStructures", std::move(error_infos)},
        {"gpuContextStructures", std::move(ctx_structs)},
    };
}

} // namespace addc::cper::sections
