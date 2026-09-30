// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-mi300-crashdump.hpp"

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

inline constexpr Guid kMi300CrashdumpGuid{
    0x32ac0c78U,
    0x2623U,
    0x48f6U,
    {0xb0U, 0xd0U, 0x73U, 0x65U, 0x72U, 0x5fU, 0xd6U, 0xaeU}};

nlohmann::json parsePldmBundleCd(uint32_t v)
{
    nlohmann::json j = nlohmann::json::object();
    j["minor"] = v & 0xFFU;
    j["major"] = (v >> 8U) & 0xFFU;
    j["year"] = (v >> 16U) & 0xFFU;
    j["productionDebug"] = (v >> 24U) & 0xFFU;
    return j;
}

nlohmann::json parseValidBitsCrashdump(uint64_t v)
{
    nlohmann::json j = nlohmann::json::object();
    j["fwIdValid"] = bool((v >> 1U) & 0x1U);
    j["gpuErrorInfoCount"] = (v >> 2U) & 0x3FU;
    j["gpuContextInfoCount"] = (v >> 8U) & 0x3FU;
    j["pcieDevidValid"] = bool((v >> 14U) & 0x1U);
    j["pldmBndlValid"] = bool((v >> 15U) & 0x1U);
    return j;
}

std::string guidStringFromBytesCd(std::span<const uint8_t> d, std::size_t off)
{
    if (off + 16U > d.size())
    {
        return "";
    }
    const Guid g =
        Guid::parse(std::span<const uint8_t, 16>{d.data() + off, 16U});
    return g.to_string();
}

nlohmann::json parseMi300ErrorInfoCd(std::span<const uint8_t> body,
                                     std::size_t base)
{
    if (base + 64U > body.size())
    {
        return nullptr;
    }
    const auto d = body.subspan(base, 64U);
    nlohmann::json j = nlohmann::json::object();
    const uint64_t vb = read_u64_le(d, 16U);
    j["errorStructureType"] = guidStringFromBytesCd(d, 0U);
    // Spec shows bits 6/7 for crashdump, but firmware uses 5/6 (same as
    // runtime).
    j["validBits"] = nlohmann::json{
        {"fruManufacturerPartNumberValid", bool((vb >> 5U) & 0x1U)},
        {"redfishEventLogIdValid", bool((vb >> 6U) & 0x1U)},
    };
    j["msCheckStructure"] = read_u64_le(d, 24U);
    j["fruManufacturerPartNumber"] =
        addc::detail::read_ascii_field(d, 32U, 16U);
    j["reserved"] = addc::format("0x{:016x}", read_u64_le(d, 48U));
    j["redfishEventLogId"] =
        addc::detail::read_ascii_field(d, 56U, 8U);
    return j;
}

// --- MI300 Context Structure (crashdump -- RSVD msrAddress/mmRegisterAddress)
// ---
nlohmann::json parseMi300ContextCd(std::span<const uint8_t> body,
                                   std::size_t base, ParseContext& context)
{
    if (base + 16U > body.size())
    {
        return nullptr;
    }
    const uint16_t ctx_type = read_u16_le(body, base);
    const uint16_t declared_arr_size = read_u16_le(body, base + 2U);
    const std::size_t data_off = base + 16U;
    const std::size_t avail = body.size() - data_off;
    const auto effective_arr_size =
        static_cast<uint16_t>(std::min<std::size_t>(declared_arr_size, avail));
    if (effective_arr_size != declared_arr_size &&
        !context.repair("register_array_size_substituted",
                        "register_array_size", declared_arr_size,
                        effective_arr_size,
                        "declared register array exceeds the GPU context body"))
    {
        return nullptr;
    }
    nlohmann::json j = nlohmann::json::object();
    j["registerContextType"] = static_cast<uint64_t>(ctx_type);
    j["registerArraySize"] = static_cast<uint64_t>(effective_arr_size);
    j["registerArray"] = detail::encode_binary(
        body.subspan(data_off, effective_arr_size), context);
    return j;
}

} // anonymous namespace

// --- MI300 crashdump classification ---

bool matches_amd_mi300_crashdump(
    const RecordHeader& /*header*/,
    const SectionDescriptor& desc) noexcept
{
    if (!(desc.section_type == kMi300CrashdumpGuid))
    {
        return false;
    }
    const uint8_t program_gen = desc.revision_major & 0xFU;
    const uint8_t ras_gen = (desc.revision_major >> 4U) & 0xFU;
    return program_gen == 0x2U && ras_gen == 0x2U;
}

// Per spec Table 2.5:
//   0:  u64  validBits
//   8:  4-byte pcieDevId
//  12:  u32  pldmBundle
//  16: 48-byte fwId
//  64: ErrorInfoStructure[err_count] (each 64 bytes)
//  64 + err_count*64: ContextStructure[ctx_count] (16-byte header +
//  registerArray)

nlohmann::json parse_amd_mi300_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_mi300_crashdump(body, desc, context);
}

nlohmann::json parse_amd_mi300_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/,
    ParseContext& context)
{
    if (body.size() < 8U)
    {
        return nullptr;
    }

    const uint64_t valid_raw = read_u64_le(body, 0U);
    const auto valid_bits_j = parseValidBitsCrashdump(valid_raw);
    const uint64_t err_count = (valid_raw >> 2U) & 0x3FU;
    const uint64_t ctx_count = (valid_raw >> 8U) & 0x3FU;

    const std::string pcie_dev_id =
        addc::detail::read_ascii_field(body, 8U, 4U);
    const uint32_t pldm_raw = read_u32_le(body, 12U);
    const std::string fw_id =
        addc::detail::read_ascii_field(body, 16U, 48U);

    // Per spec: ErrorInfoStructures at offset 64, each 64 bytes.
    // The 64-byte error info slot is always reserved even when err_count=0.
    // Note: cap iteration by body size since some CPER generators populate
    // count fields with the total across all sections in the record.
    nlohmann::json error_infos = nlohmann::json::array();
    std::size_t cur = 64U;
    for (uint64_t i = 0U; i < err_count; ++i)
    {
        if (cur + 64U > body.size())
        {
            break;
        }
        auto entry = parseMi300ErrorInfoCd(body, cur);
        if (!entry.is_null())
        {
            error_infos.push_back(std::move(entry));
        }
        cur += 64U;
    }
    if (error_infos.size() != err_count &&
        !context.repair(
            "gpu_error_count_truncated", "gpu_error_info_count", err_count,
            error_infos.size(),
            "declared GPU error entries do not fit in the section body"))
    {
        return nullptr;
    }
    // Ensure context structures start after the error info region.
    // Minimum offset is 128 (64 header + at least one 64-byte error info slot).
    if (cur < 128U)
    {
        cur = 128U;
    }

    // Per spec: ContextStructures immediately after ErrorInfoStructures.
    // Note: some CPER generators set ctx_count=0 despite a valid context
    // structure being present. Attempt at least one read and skip if
    // arr_size == 0.
    nlohmann::json ctx_structs = nlohmann::json::array();
    const uint64_t iters = std::max<uint64_t>(1U, ctx_count);
    for (uint64_t i = 0U; i < iters; ++i)
    {
        if (cur + 16U > body.size())
        {
            break;
        }
        const uint16_t arr_size = read_u16_le(body, cur + 2U);
        if (arr_size == 0U && ctx_count == 0U)
        {
            break;
        }
        auto entry = parseMi300ContextCd(body, cur, context);
        if (entry.is_null())
        {
            break;
        }
        const auto effective_arr_size =
            entry["registerArraySize"].get<std::size_t>();
        ctx_structs.push_back(std::move(entry));
        cur += 16U + effective_arr_size;
    }
    const uint64_t expected_context_count =
        ctx_count == 0U && !ctx_structs.empty() ? 1U : ctx_count;
    if (ctx_count == 0U && expected_context_count == 1U &&
        !context.repair(
            "gpu_context_count_substituted", "gpu_context_info_count", 0U, 1U,
            "a bounded GPU context is present despite a zero declared count"))
    {
        return nullptr;
    }
    if (ctx_structs.size() != expected_context_count &&
        !context.repair(
            "gpu_context_count_truncated", "gpu_context_info_count",
            expected_context_count, ctx_structs.size(),
            "declared GPU context entries do not fit in the section body"))
    {
        return nullptr;
    }

    nlohmann::json result = nlohmann::json::object();
    result["validBits"] = valid_bits_j;
    result["pcieDevId"] = pcie_dev_id;
    result["pldmBundle"] = parsePldmBundleCd(pldm_raw);
    result["fwId"] = fw_id;
    result["gpuErrorInfoStructures"] = std::move(error_infos);
    result["gpuContextStructures"] = std::move(ctx_structs);
    return result;
}

} // namespace addc::cper::sections
