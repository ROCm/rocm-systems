// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-epyc-crashdump.hpp"

#include "../base64.hpp"
#include "../endian.hpp"
#include "addc/cper/amd_section_ids.hpp"
#include "addc/detail/format.hpp"

#include <algorithm>

namespace addc::cper::sections
{

using addc::cper::read_u16_le;
using addc::cper::read_u32_le;
using addc::cper::read_u64_le;

namespace
{

// --- ValidBits field constants -----------------------------------------------
constexpr uint64_t kValidApicId = 0x01U;
constexpr uint64_t kValidCpuid = 0x02U;
constexpr int kProcErrInfoShift = 2;
constexpr uint64_t kProcErrInfoMask = 0x3FU;
constexpr int kProcContextInfoShift = 8;
constexpr uint64_t kProcContextInfoMask = 0x3FU;

// --- Context type values -----------------------------------------------------
constexpr uint16_t kCtxTypeCrashdump = 1U;
constexpr uint16_t kCtxTypeMcaShutdown = 2U;
constexpr uint16_t kCtxTypeBreakEvent = 3U;
constexpr uint16_t kCtxTypeBootStatus = 9U;

std::string_view contextTypeName(uint16_t t) noexcept
{
    switch (t)
    {
        case kCtxTypeCrashdump:
            return "Crashdump";
        case kCtxTypeMcaShutdown:
            return "Crashdump on MCA initiated Shutdown";
        case kCtxTypeBreakEvent:
            return "Break Event";
        case kCtxTypeBootStatus:
            return "Boot";
        default:
            return "Unknown";
    }
}

// --- EFI_AMD_CRASHDUMP_ERROR_DATA section body offsets -----------------------
// Struct layout (all packed):
//   0:   ValidBits (8)
//   8:   ApicId    (8)
//   16:  CpuidInfo.Eax (8)
//   24:  CpuidInfo.Ebx (8)
//   32:  CpuidInfo.Ecx (8)
//   40:  CpuidInfo.Edx (8)
//   48:  CpuidInfo.Reserved (16)
//   64:  ProcessorErrorInfoReserved (64)
//   128: ProcessorContext {
//          RegisterContextType (2), RegisterArraySize (2),
//          UcodeVersion (4), Ppin (8), ...
//        }
constexpr std::size_t kOffValidBits = 0U;
constexpr std::size_t kOffApicId = 8U;
constexpr std::size_t kOffCpuidEax = 16U;
constexpr std::size_t kOffCpuidEbx = 24U;
constexpr std::size_t kOffCpuidEcx = 32U;
constexpr std::size_t kOffCpuidEdx = 40U;
constexpr std::size_t kOffCtxBase = 128U; // start of ProcessorContext
constexpr std::size_t kOffCtxType = kOffCtxBase + 0U;
constexpr std::size_t kOffCtxArrSz = kOffCtxBase + 2U;
constexpr std::size_t kOffCtxUcode = kOffCtxBase + 4U;
constexpr std::size_t kOffCtxPpin = kOffCtxBase + 8U;
constexpr std::size_t kCtxMinSize = kOffCtxBase + 16U; // through Ppin
// sizeof(AMD_CRASHDUMP_PROCESSOR_CONTEXT) = 16 + 16384 + 128 + 384 + 48496 =
// 65408
constexpr std::size_t kProcessorContextSize = 65408U;

// --- Portable LE readers -----------------------------------------------------

} // anonymous namespace

// --- EPYC crashdump classification ------------------------------------------
// Match criteria:
//   program_gen = revision.major & 0xF
//   ras_gen     = (revision.major >> 4) & 0xF
//   valid if program_gen in {0x0, 0x1, 0x4} (EPYC / Embedded)
//         or program_gen == 0x2 and ras_gen == 0x1 (MI300A/MI300C)

bool matches_amd_epyc_crashdump(const RecordHeader& /*header*/,
                                const SectionDescriptor& desc) noexcept
{
    if (!(desc.section_type == kAmdCrashdumpGuid))
    {
        return false;
    }
    const uint8_t program = program_gen(desc.revision_major);
    const uint8_t ras = ras_gen(desc.revision_major);

    if (program == 0x0U || program == 0x1U || program == 0x4U)
    {
        return true; // EPYC / Embedded
    }
    if (program == 0x2U && ras == 0x1U)
    {
        return true; // MI300A / MI300C
    }
    return false;
}

// --- EPYC crashdump parsing -------------------------------------------------
// Independently implements the AMD EPYC Crashdump specification and emits the
// JSON field names and structure expected by libcper-compatible consumers.

nlohmann::json parse_amd_epyc_crashdump(std::span<const uint8_t> body,
                                        const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_epyc_crashdump(body, desc, context);
}

nlohmann::json parse_amd_epyc_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/,
    ParseContext& context)
{
    if (body.size() < kOffCtxBase)
    {
        return nullptr;
    }

    nlohmann::json j;

    // -- Valid-bits decomposition ------------------------------------------
    const uint64_t valid_bits = read_u64_le(body, kOffValidBits);

    j["sectionValidBits"] = addc::format("0x{:016X}", valid_bits);
    j["apicIdValid"] = (valid_bits & kValidApicId) != 0U;
    j["cpuidValid"] = (valid_bits & kValidCpuid) != 0U;
    j["processorErrorInfoNum"] = static_cast<int64_t>(
        (valid_bits >> kProcErrInfoShift) & kProcErrInfoMask);
    const uint64_t declared_context_count =
        (valid_bits >> kProcContextInfoShift) & kProcContextInfoMask;
    j["processorContextInfoNum"] = static_cast<int64_t>(declared_context_count);

    // -- APIC ID -----------------------------------------------------------
    j["apicId"] = read_u64_le(body, kOffApicId);

    // -- CPUID info --------------------------------------------------------
    j["cpuidInfo"] = {
        {"eax", read_u64_le(body, kOffCpuidEax)},
        {"ebx", read_u64_le(body, kOffCpuidEbx)},
        {"ecx", read_u64_le(body, kOffCpuidEcx)},
        {"edx", read_u64_le(body, kOffCpuidEdx)},
    };

    // -- ProcessorErrorInfo - always "Reserved" -------------------------
    j["processorErrorInfo"] = "Reserved";

    // -- ProcessorContext info array ------------------------------------
    nlohmann::json ctx_array = nlohmann::json::array();
    nlohmann::json ctx_entry;

    // This section format contains one fixed ProcessorContext structure. Older
    // firmware sometimes leaves the corresponding count at zero (or reports a
    // larger record-wide count), so tolerant mode keeps the historical single
    // entry while making the substitution observable.
    if (declared_context_count != 1U &&
        !context.repair("processor_context_count_substituted",
                        "processor_context_info_count", declared_context_count,
                        1U,
                        "the EPYC crashdump section contains one fixed "
                        "processor context"))
    {
        return nullptr;
    }

    if (body.size() >= kCtxMinSize)
    {
        const uint16_t ctx_type = read_u16_le(body, kOffCtxType);
        const uint16_t declared_ctx_arr_sz = read_u16_le(body, kOffCtxArrSz);
        const uint32_t ucode = read_u32_le(body, kOffCtxUcode);
        const uint64_t ppin = read_u64_le(body, kOffCtxPpin);

        // Raw ProcessorContext bytes -> base64 (entire struct, capped at known
        // size).
        const std::size_t ctx_avail =
            (body.size() > kOffCtxBase) ? body.size() - kOffCtxBase : 0U;
        const std::size_t ctx_len = std::min(ctx_avail, kProcessorContextSize);
        const auto ctx_span = body.subspan(kOffCtxBase, ctx_len);
        const auto effective_ctx_arr_sz = static_cast<uint16_t>(
            std::min<std::size_t>(declared_ctx_arr_sz,
                                  ctx_len > 16U ? ctx_len - 16U : 0U));
        if (effective_ctx_arr_sz != declared_ctx_arr_sz &&
            !context.repair(
                "register_array_size_substituted", "register_array_size",
                declared_ctx_arr_sz, effective_ctx_arr_sz,
                "declared register array exceeds the processor context body"))
        {
            return nullptr;
        }

        ctx_entry["registerContextType"] = {
            {"value", static_cast<uint64_t>(ctx_type)},
            {"name", contextTypeName(ctx_type)},
        };
        ctx_entry["registerArraySize"] =
            static_cast<uint64_t>(effective_ctx_arr_sz);
        if (ucode != 0U)
        {
            ctx_entry["ucode"] = addc::format("0x{:08X}", ucode);
        }
        if (ppin != 0U)
        {
            ctx_entry["ppin"] = addc::format("0x{:016X}", ppin);
        }
        ctx_entry["data"] = detail::encode_binary(ctx_span, context);
    }
    else
    {
        const uint64_t available = body.size() - kOffCtxBase;
        if (!context.repair("processor_context_truncated",
                            "processor_context_header_size", 16U, available,
                            "processor context header is truncated"))
        {
            return nullptr;
        }
    }

    ctx_array.push_back(std::move(ctx_entry));

    j["processorContextInfo"] = std::move(ctx_array);

    return j;
}

} // namespace addc::cper::sections
