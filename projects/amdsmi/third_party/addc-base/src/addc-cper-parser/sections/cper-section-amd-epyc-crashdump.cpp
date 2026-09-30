// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-epyc-crashdump.hpp"

#include "../base64.hpp"
#include "addc/detail/format.hpp"

#include <algorithm>

namespace addc::cper::sections
{

namespace
{

// --- AMD EPYC Crashdump GUID (b0d0 variant - the only registered one) --------
//   String: 32ac0c78-2623-48f6-b0d0-7365725fd6ae
inline constexpr Guid kAmdCrashdumpGuid{
    0x32ac0c78U,
    0x2623U,
    0x48f6U,
    {0xb0U, 0xd0U, 0x73U, 0x65U, 0x72U, 0x5fU, 0xd6U, 0xaeU}};

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

constexpr uint16_t readU16(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return static_cast<uint16_t>(d[o]) |
           static_cast<uint16_t>(static_cast<uint16_t>(d[o + 1U]) << 8U);
}

constexpr uint32_t readU32(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return static_cast<uint32_t>(d[o]) |
           (static_cast<uint32_t>(d[o + 1U]) << 8U) |
           (static_cast<uint32_t>(d[o + 2U]) << 16U) |
           (static_cast<uint32_t>(d[o + 3U]) << 24U);
}

constexpr uint64_t readU64(std::span<const uint8_t> d, std::size_t o) noexcept
{
    uint64_t v = 0U;
    for (std::size_t i = 0U; i < 8U; ++i)
    {
        v |= static_cast<uint64_t>(d[o + i]) << (8U * i);
    }
    return v;
}

/// Safe read: returns 0 if offset+width exceeds data.size().
template <typename Fn>
auto safeRead(std::span<const uint8_t> d, std::size_t offset, std::size_t width,
              Fn reader) noexcept -> decltype(reader(d, offset))
{
    if (offset + width > d.size())
    {
        return {};
    }
    return reader(d, offset);
}

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
    const uint8_t rev_major = desc.revision_major;
    const uint32_t program_gen = static_cast<uint32_t>(rev_major) & 0xFU;
    const uint32_t ras_gen = (static_cast<uint32_t>(rev_major) >> 4U) & 0xFU;

    if (program_gen == 0x0U || program_gen == 0x1U || program_gen == 0x4U)
    {
        return true; // EPYC / Embedded
    }
    if (program_gen == 0x2U && ras_gen == 0x1U)
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
    const uint64_t valid_bits = safeRead(body, kOffValidBits, 8U, readU64);

    j["sectionValidBits"] = addc::format("0x{:016X}", valid_bits);
    j["apicIdValid"] = (valid_bits & kValidApicId) != 0U;
    j["cpuidValid"] = (valid_bits & kValidCpuid) != 0U;
    j["processorErrorInfoNum"] = static_cast<int64_t>(
        (valid_bits >> kProcErrInfoShift) & kProcErrInfoMask);
    const uint64_t declared_context_count =
        (valid_bits >> kProcContextInfoShift) & kProcContextInfoMask;
    j["processorContextInfoNum"] = static_cast<int64_t>(declared_context_count);

    // -- APIC ID -----------------------------------------------------------
    j["apicId"] = safeRead(body, kOffApicId, 8U, readU64);

    // -- CPUID info --------------------------------------------------------
    j["cpuidInfo"] = {
        {"eax", safeRead(body, kOffCpuidEax, 8U, readU64)},
        {"ebx", safeRead(body, kOffCpuidEbx, 8U, readU64)},
        {"ecx", safeRead(body, kOffCpuidEcx, 8U, readU64)},
        {"edx", safeRead(body, kOffCpuidEdx, 8U, readU64)},
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
        const uint16_t ctx_type = readU16(body, kOffCtxType);
        const uint16_t declared_ctx_arr_sz = readU16(body, kOffCtxArrSz);
        const uint32_t ucode = readU32(body, kOffCtxUcode);
        const uint64_t ppin = readU64(body, kOffCtxPpin);

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
