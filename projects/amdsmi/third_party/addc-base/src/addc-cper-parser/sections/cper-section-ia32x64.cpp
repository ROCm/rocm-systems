// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-ia32x64.hpp"

#include "../base64.hpp"
#include "addc/detail/format.hpp"

#include <algorithm>
#include <string>

namespace addc::cper::sections
{

namespace
{

// --- IA32/x64 Processor section GUID -----------------------------------------
//   dc3ea0b0-a144-4797-b95b-53fa242b6e1d
inline constexpr Guid kIa32X64Guid{
    0xdc3ea0b0U,
    0xa144U,
    0x4797U,
    {0xb9U, 0x5bU, 0x53U, 0xfaU, 0x24U, 0x2bU, 0x6eU, 0x1dU}};

// --- Error type GUIDs --------------------------------------------------------
inline constexpr Guid kCacheCheckGuid{
    0xa55701f5U,
    0xe3efU,
    0x43deU,
    {0xacU, 0x72U, 0x24U, 0x9bU, 0x57U, 0x3fU, 0xadU, 0x2cU}};
inline constexpr Guid kTlbCheckGuid{
    0xfc06b535U,
    0x5e1fU,
    0x4562U,
    {0x9fU, 0x25U, 0x0aU, 0x3bU, 0x9aU, 0xdbU, 0x63U, 0xc3U}};
inline constexpr Guid kBusCheckGuid{
    0x1cf3f8b3U,
    0xc5b1U,
    0x49a2U,
    {0xaaU, 0x59U, 0x5eU, 0xefU, 0x92U, 0xffU, 0xa6U, 0x3cU}};
inline constexpr Guid kMsCheckGuid{
    0x48ab7f57U,
    0xdc34U,
    0x4f6cU,
    {0xa7U, 0xd3U, 0xb0U, 0xb5U, 0xb0U, 0xa7U, 0x43U, 0x14U}};

// --- Context type names
// -------------------------------------------------------
constexpr std::string_view ia32x64ContextTypeName(uint16_t t) noexcept
{
    switch (t)
    {
        case 0:
            return "Unclassified Data";
        case 1:
            return "MSR Registers";
        case 2:
            return "32-bit Mode Execution Context";
        case 3:
            return "64-bit Mode Execution Context";
        case 4:
            return "FXSave Context";
        case 5:
            return "32-bit Mode Debug Registers";
        case 6:
            return "64-bit Mode Debug Registers";
        case 7:
            return "Memory Mapper Registers";
        default:
            return "Unknown (Reserved)";
    }
}

// --- MS Check error type names
// ------------------------------------------------
constexpr std::string_view msCheckErrorTypeName(uint32_t v) noexcept
{
    switch (v)
    {
        case 0:
            return "No Error";
        case 1:
            return "Unclassified";
        case 2:
            return "Microcode ROM Parity Error";
        case 3:
            return "External Error";
        case 4:
            return "FRC Error";
        case 5:
            return "Internal Unclassified";
        default:
            return "Unknown (Processor Specific)";
    }
}

// --- Portable LE readers
// ------------------------------------------------------

constexpr uint16_t readU16(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return static_cast<uint16_t>(d[o]) |
           (static_cast<uint16_t>(d[o + 1U]) << 8U);
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
    uint64_t v{};
    for (std::size_t i = 0U; i < 8U; ++i)
    {
        v |= static_cast<uint64_t>(d[o + i]) << (8U * i);
    }
    return v;
}

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

// --- Parse a 16-byte GUID from body at position pos
// ---------------------------
Guid readGuid(std::span<const uint8_t> d, std::size_t pos) noexcept
{
    if (pos + 16U > d.size())
    {
        return Guid{};
    }
    std::span<const uint8_t, 16> bytes{d.data() + pos, 16U};
    return Guid::parse(bytes);
}

// --- Determine error type name from GUID
// --------------------------------------
std::string ia32x64ErrorTypeName(const Guid& g)
{
    if (g == kCacheCheckGuid)
    {
        return "Cache Check Error";
    }
    if (g == kTlbCheckGuid)
    {
        return "TLB Check Error";
    }
    if (g == kBusCheckGuid)
    {
        return "Bus Check Error";
    }
    if (g == kMsCheckGuid)
    {
        return "MS Check Error";
    }
    return "Unknown";
}

// --- Build ProcessorErrorInfo JSON entry -------------------------------------
// Each entry is 64 bytes starting at body[offset].
// Layout:
//   offset+ 0: 16-byte type GUID
//   offset+16: 8-byte validation bits
//   offset+24: 8-byte check info
//   offset+32: 8-byte target identifier
//   offset+40: 8-byte requestor identifier
//   offset+48: 8-byte responder identifier
//   offset+56: 8-byte instruction pointer
nlohmann::json parseProcErrorInfo(std::span<const uint8_t> body,
                                  std::size_t offset,
                                  const ParseContext& context)
{
    if (offset + 64U > body.size())
    {
        return nullptr;
    }

    const Guid type_guid = readGuid(body, offset);
    const std::string type_name = ia32x64ErrorTypeName(type_guid);
    const uint64_t validation_bits = readU64(body, offset + 16U);
    const uint64_t check_info = readU64(body, offset + 24U);
    const uint64_t target_id = readU64(body, offset + 32U);
    const uint64_t requestor_id = readU64(body, offset + 40U);
    const uint64_t responder_id = readU64(body, offset + 48U);
    const uint64_t instr_ptr = readU64(body, offset + 56U);

    nlohmann::json entry;
    entry["type"] = {{"guid", type_guid.to_string()}, {"name", type_name}};
    entry["validationBits"] = validation_bits;

    // Build checkInfo: MS Check gets structured expansion, others get base64
    if (type_name == "MS Check Error")
    {
        const uint32_t error_type_val = (check_info >> 16U) & 0x7U;
        nlohmann::json ci;
        ci["data"] = check_info;
        ci["errorType"] = {
            {"value", error_type_val},
            {"name", std::string{msCheckErrorTypeName(error_type_val)}},
        };
        if (((check_info >> 1U) & 0x1U) != 0U)
        {
            ci["processorContextCorrupt"] = bool((check_info >> 19U) & 0x1U);
        }
        if (((check_info >> 2U) & 0x1U) != 0U)
        {
            ci["uncorrected"] = bool((check_info >> 20U) & 0x1U);
        }
        if (((check_info >> 3U) & 0x1U) != 0U)
        {
            ci["preciseIP"] = bool((check_info >> 21U) & 0x1U);
        }
        if (((check_info >> 4U) & 0x1U) != 0U)
        {
            ci["restartableIP"] = bool((check_info >> 22U) & 0x1U);
        }
        if (((check_info >> 5U) & 0x1U) != 0U)
        {
            ci["overflow"] = bool((check_info >> 23U) & 0x1U);
        }
        entry["checkInfo"] = std::move(ci);
    }
    else
    {
        // For Cache/TLB/Bus check errors, encode check_info as 8-byte LE
        // base-64
        std::array<uint8_t, 8> ci_bytes{};
        for (std::size_t i = 0U; i < 8U; ++i)
        {
            ci_bytes[i] =
                static_cast<uint8_t>((check_info >> (8U * i)) & 0xFFU);
        }
        entry["checkInfo"] =
            detail::encode_binary(std::span<const uint8_t>{ci_bytes}, context);
    }

    // Conditional fields driven by validation bits
    if (((validation_bits >> 1U) & 0x1U) != 0U)
    {
        entry["targetAddressID"] = target_id;
    }
    if (((validation_bits >> 2U) & 0x1U) != 0U)
    {
        entry["requestorID"] = requestor_id;
    }
    if (((validation_bits >> 3U) & 0x1U) != 0U)
    {
        entry["responderID"] = responder_id;
    }
    if (((validation_bits >> 4U) & 0x1U) != 0U)
    {
        entry["instructionPointer"] = instr_ptr;
    }

    return entry;
}

// --- Build ProcessorContextInfo JSON entry
// ------------------------------------ Variable-size: 16-byte header +
// register_array_size bytes. Layout:
//   offset+ 0: 2-byte context type
//   offset+ 2: 2-byte register array size
//   offset+ 4: 4-byte MSR address
//   offset+ 8: 8-byte MM register address
//   offset+16: register_array_size bytes of register array data
nlohmann::json parseProcContextInfo(std::span<const uint8_t> body,
                                    std::size_t offset, ParseContext& context)
{
    if (offset + 16U > body.size())
    {
        return nullptr;
    }

    const uint16_t ctx_type = readU16(body, offset + 0U);
    uint16_t arr_size = readU16(body, offset + 2U);
    const uint32_t msr_addr = readU32(body, offset + 4U);
    const uint64_t mm_addr = readU64(body, offset + 8U);

    // Compute bytes available for register array data from this context's
    // header to end of body.
    const std::size_t body_remaining =
        (body.size() > offset + 16U) ? (body.size() - offset - 16U) : 0U;

    const uint16_t declared_arr_size = arr_size;
    const auto effective_arr_size = static_cast<uint16_t>(
        std::min(body_remaining, static_cast<std::size_t>(UINT16_MAX)));
    if (declared_arr_size != effective_arr_size)
    {
        if (!context.repair(
                "register_array_size_substituted", "register_array_size",
                declared_arr_size, effective_arr_size,
                declared_arr_size > effective_arr_size
                    ? "declared register array exceeds the section body"
                    : "declared register array omits bounded trailing "
                      "register data"))
        {
            return nullptr;
        }
        arr_size = effective_arr_size;
    }

    const auto reg_span = body.subspan(offset + 16U, arr_size);

    nlohmann::json entry;
    entry["registerContextType"] = {
        {"value", static_cast<int64_t>(ctx_type)},
        {"name", std::string{ia32x64ContextTypeName(ctx_type)}},
    };
    entry["registerArraySize"] = static_cast<uint64_t>(arr_size);
    entry["msrAddress"] = static_cast<uint64_t>(msr_addr);
    entry["mmRegisterAddress"] = mm_addr;
    entry["registerArray"] =
        nlohmann::json{{"data", detail::encode_binary(reg_span, context)}};

    return entry;
}

} // anonymous namespace

// --- IA32/x64 classification ------------------------------------------------

bool matches_ia32x64(const RecordHeader& /*header*/,
                     const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kIa32X64Guid;
}

// --- IA32/x64 parsing -------------------------------------------------------
// Body layout:
//   0:  8-byte ValidBits
//   8:  8-byte APIC ID
//   16: 48-byte CPUID info (eax, ebx, ecx, edx each 8-bytes, 16-byte reserved)
//   64: N Ã— 64-byte ProcessorErrorInfo  (N = bits[7:2] of ValidBits)
//   64+N*64: M Ã— variable ProcessorContextInfo (M = bits[13:8] of ValidBits)

nlohmann::json parse_ia32x64(std::span<const uint8_t> body,
                             const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_ia32x64(body, desc, context);
}

nlohmann::json parse_ia32x64(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/,
    ParseContext& context)
{
    if (body.size() < 64U)
    {
        return nullptr;
    }

    const uint64_t valid_bits = safeRead(body, 0U, 8U, readU64);
    const uint64_t apic_id = safeRead(body, 8U, 8U, readU64);

    const bool apic_id_valid = (valid_bits & 0x1U) != 0U;
    const bool cpuid_valid = (valid_bits & 0x2U) != 0U;
    const int err_info_num = static_cast<int>((valid_bits >> 2U) & 0x3FU);
    const int ctx_info_num = static_cast<int>((valid_bits >> 8U) & 0x3FU);

    // CPUID: eax@16, ebx@24, ecx@32, edx@40 (each 8 bytes LE)
    nlohmann::json cpuid_info;
    if (cpuid_valid)
    {
        cpuid_info = {
            {"eax", safeRead(body, 16U, 8U, readU64)},
            {"ebx", safeRead(body, 24U, 8U, readU64)},
            {"ecx", safeRead(body, 32U, 8U, readU64)},
            {"edx", safeRead(body, 40U, 8U, readU64)},
        };
    }
    else
    {
        cpuid_info = {{"eax", 0U}, {"ebx", 0U}, {"ecx", 0U}, {"edx", 0U}};
    }

    // If either count is invalid (< 1), reset both to 1. Some records encode
    // zero or garbage in the ValidBits error/context count fields; treating
    // both as 1 recovers the single MCA context entry.
    const bool fallback = (err_info_num < 1 || ctx_info_num < 1);
    const int eff_err_num = fallback ? 1 : err_info_num;
    const int eff_ctx_num = fallback ? 1 : ctx_info_num;
    if (fallback)
    {
        if (err_info_num != eff_err_num &&
            !context.repair(
                "processor_error_count_substituted",
                "processor_error_info_count",
                static_cast<uint64_t>(err_info_num),
                static_cast<uint64_t>(eff_err_num),
                "zero count conflicts with the bounded processor payload"))
        {
            return nullptr;
        }
        if (ctx_info_num != eff_ctx_num &&
            !context.repair(
                "processor_context_count_substituted",
                "processor_context_info_count",
                static_cast<uint64_t>(ctx_info_num),
                static_cast<uint64_t>(eff_ctx_num),
                "zero count conflicts with the bounded processor payload"))
        {
            return nullptr;
        }
    }

    nlohmann::json proc_err_array = nlohmann::json::array();
    for (int n = 1; n <= eff_err_num; ++n)
    {
        const std::size_t offset = static_cast<std::size_t>(n) * 64U;
        if (auto entry = parseProcErrorInfo(body, offset, context);
            !entry.is_null())
        {
            proc_err_array.push_back(std::move(entry));
        }
    }
    if (proc_err_array.size() != static_cast<std::size_t>(eff_err_num) &&
        !context.repair(
            "processor_error_count_truncated", "processor_error_info_count",
            static_cast<uint64_t>(eff_err_num), proc_err_array.size(),
            "declared processor error entries do not fit in the section "
            "body"))
    {
        return nullptr;
    }

    std::size_t ctx_offset = 64U + static_cast<std::size_t>(eff_err_num) * 64U;
    nlohmann::json proc_ctx_array = nlohmann::json::array();
    for (int n = 0; n < eff_ctx_num; ++n)
    {
        auto entry = parseProcContextInfo(body, ctx_offset, context);
        if (entry.is_null())
        {
            break;
        }

        // Advance offset by this context's total size (16 header + arr_size)
        const uint16_t arr_sz = entry["registerArraySize"].get<uint16_t>();
        ctx_offset += 16U + arr_sz;

        proc_ctx_array.push_back(std::move(entry));
    }
    if (proc_ctx_array.size() != static_cast<std::size_t>(eff_ctx_num) &&
        !context.repair(
            "processor_context_count_truncated", "processor_context_info_count",
            static_cast<uint64_t>(eff_ctx_num), proc_ctx_array.size(),
            "declared processor context entries do not fit in the section "
            "body"))
    {
        return nullptr;
    }

    nlohmann::json j;
    j["sectionValidBits"] = addc::format("0x{:x}", valid_bits);
    j["apicIdValid"] = apic_id_valid;
    j["cpuidValid"] = cpuid_valid;
    j["processorErrorInfoNum"] = err_info_num;
    j["processorContextInfoNum"] = ctx_info_num;
    j["localAPICID"] = apic_id;
    j["cpuidInfo"] = std::move(cpuid_info);
    j["processorErrorInfo"] = std::move(proc_err_array);
    j["processorContextInfo"] = std::move(proc_ctx_array);

    return j;
}

} // namespace addc::cper::sections
