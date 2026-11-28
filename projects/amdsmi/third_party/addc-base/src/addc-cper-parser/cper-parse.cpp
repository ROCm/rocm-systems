// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/cper/parser.hpp"
#include "addc/cper/record.hpp"
#include "base64.hpp"
#include "addc/cper/section_dispatch.hpp"

namespace addc::cper
{

namespace
{

constexpr uint32_t kSigStart = 0x52455043U; // "CPER"
constexpr uint32_t kSigEnd = 0xFFFFFFFFU;

constexpr uint16_t readU16Le(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return static_cast<uint16_t>(d[o]) |
           static_cast<uint16_t>(static_cast<uint16_t>(d[o + 1U]) << 8U);
}

constexpr uint32_t readU32Le(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return static_cast<uint32_t>(d[o]) |
           (static_cast<uint32_t>(d[o + 1U]) << 8U) |
           (static_cast<uint32_t>(d[o + 2U]) << 16U) |
           (static_cast<uint32_t>(d[o + 3U]) << 24U);
}

nlohmann::json makeUnknownSection(std::span<const uint8_t> body)
{
    nlohmann::json section_ir;
    section_ir["data"] = detail::base64_encode(body);

    nlohmann::json result;
    result["Unknown"] = std::move(section_ir);
    return result;
}

} // anonymous namespace

// --- cper_is_valid
// ------------------------------------------------------------

bool cper_is_valid(std::span<const uint8_t> buf) noexcept
{
    constexpr std::size_t kHeaderSize = 128U;
    if (buf.size() < kHeaderSize)
    {
        return false;
    }
    if (readU32Le(buf, 0U) != kSigStart)
    {
        return false;
    }
    if (readU32Le(buf, 6U) != kSigEnd)
    {
        return false;
    }
    if (readU16Le(buf, 10U) == 0U)
    {
        return false;
    }
    return true;
}

nlohmann::json normalize_parser_json(const nlohmann::json& producer_json)
{
    // The native producer already defines the frozen canonical representation.
    // Keeping this as an explicit identity seam lets a separately validated
    // backend normalize without changing callers or typed conversion.
    return producer_json;
}

// --- cper_to_ir
// ---------------------------------------------------------------

nlohmann::json cper_to_ir(std::span<const uint8_t> buf)
{
    return cper_to_ir(buf, addc::DecodeOptions{});
}

nlohmann::json cper_to_ir(std::span<const uint8_t> buf,
                          const addc::DecodeOptions& options)
{
    return cper_to_ir(buf, options, nullptr);
}

nlohmann::json cper_to_ir(std::span<const uint8_t> buf,
                          const addc::DecodeOptions& options,
                          ParseFailure* failure_out)
{
    return normalize_parser_json(
        parse_native_json(buf, options, failure_out));
}

nlohmann::json parse_native_json(std::span<const uint8_t> buf,
                                 const addc::DecodeOptions& options,
                                 ParseFailure* failure_out)
{
    auto maybe_record = CperRecord::parse_detailed(buf, options, failure_out);
    if (!maybe_record)
    {
        return nullptr;
    }

    const CperRecord& record = *maybe_record;
    nlohmann::json root;

    // -- header ------------------------------------------------------------
    root["header"] = record.header;

    // -- sectionDescriptors[] ----------------------------------------------
    nlohmann::json descriptors = nlohmann::json::array();
    for (std::size_t i = 0U; i < record.descriptors.size(); ++i)
    {
        const SectionDescriptor& desc = record.descriptors[i];

        // Start from the ADL to_json, then patch sectionType.type.
        nlohmann::json desc_j = desc;

        const auto classification = classify_section(record.header, desc);
        desc_j["sectionType"]["type"] =
            std::string{classification.type_name};

        descriptors.push_back(std::move(desc_j));
    }
    root["sectionDescriptors"] = std::move(descriptors);

    // -- sections[] --------------------------------------------------------
    nlohmann::json sections = nlohmann::json::array();
    for (std::size_t i = 0U; i < record.descriptors.size(); ++i)
    {
        const SectionDescriptor& desc = record.descriptors[i];
        const auto& body = record.section_bodies[i];

        const auto classification = classify_section(record.header, desc);
        if (classification.kind == SectionKind::Opaque)
        {
            sections.push_back(
                makeUnknownSection(std::span<const uint8_t>{body}));
            continue;
        }

        ParseContext parse_context{
            .mode = options.parse_mode,
            .section_index = i,
            .repairs = &maybe_record->repairs,
        };
        nlohmann::json parsed = parse_section_json(
            classification.kind, std::span<const uint8_t>{body}, desc,
            parse_context);
        if (parsed.is_null())
        {
            if (options.parse_mode == addc::ParseMode::Strict)
            {
                if (failure_out != nullptr)
                {
                    *failure_out = ParseFailure{
                        .code = ParseError::InvalidSectionBody,
                        .offset = desc.section_offset,
                        .section_index = i,
                    };
                }
                return nullptr;
            }
            sections.push_back(
                makeUnknownSection(std::span<const uint8_t>{body}));
            continue;
        }

        nlohmann::json section_entry;
        const auto desc_str = classification.description;
        if (!desc_str.empty())
        {
            section_entry["message"] = std::string{desc_str};
        }
        section_entry[std::string{classification.json_key}] =
            std::move(parsed);

        sections.push_back(std::move(section_entry));
    }
    root["sections"] = std::move(sections);

    if (!record.repairs.empty())
    {
        root["recovery"] = {
            {"recovered", true},
            {"repairs", record.repairs},
        };
    }

    return root;
}

} // namespace addc::cper
