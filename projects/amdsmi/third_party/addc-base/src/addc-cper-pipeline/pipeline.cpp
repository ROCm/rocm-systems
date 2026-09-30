// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/pipeline.hpp"

#include "addc/cper/section_dispatch.hpp"
#include "addc/cper/structural.hpp"
#include "addc/detail/format.hpp"
#include "addc/pipeline/section_data.hpp"
#include "addc/pipeline/section_dispatch.hpp"
#include "addc/pipeline/structural_pipeline.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace addc::pipeline
{

[[nodiscard]] const nlohmann::json* jsonObjectPtr(const nlohmann::json& obj,
                                                  std::string_view key)
{
    if (!obj.is_object())
    {
        return nullptr;
    }
    const auto it = obj.find(std::string{key});
    return it != obj.end() && it->is_object() ? &(*it) : nullptr;
}

[[nodiscard]] std::optional<std::string> jsonString(const nlohmann::json& obj,
                                                    std::string_view key)
{
    if (!obj.is_object())
    {
        return std::nullopt;
    }
    const auto it = obj.find(std::string{key});
    return it != obj.end() && it->is_string()
               ? std::optional<std::string>{it->get<std::string>()}
               : std::nullopt;
}

[[nodiscard]] std::optional<PipelineOutput> fail(std::string message,
                                                 std::string* error_out)
{
    if (error_out != nullptr)
    {
        *error_out = std::move(message);
    }
    return std::nullopt;
}

void append_parser_recovery(PipelineOutput& output,
                            const nlohmann::json& recovery)
{
    if (!recovery.is_object())
    {
        return;
    }
    output.recovery = recovery;
    const auto repairs = recovery.find("repairs");
    if (repairs == recovery.end() || !repairs->is_array())
    {
        return;
    }
    for (const auto& repair : *repairs)
    {
        output.diagnostics.debug("parser_recovery",
                                 "Parser repaired malformed input", repair);
    }
}

PipelineOutput make_pipeline_output(const nlohmann::json& header,
                                    std::string_view filename,
                                    std::string_view tool_version)
{
    std::optional<std::string> partition_id;
    if (const auto value = header.find("partitionID");
        value != header.end() && value->is_string())
    {
        partition_id = value->get<std::string>();
    }

    uint64_t record_id = 0U;
    if (const auto value = header.find("recordID");
        value != header.end() && value->is_number_unsigned())
    {
        record_id = value->get<uint64_t>();
    }

    return PipelineOutput{
        .tool_version = std::string{tool_version},
        .timestamp = jsonString(header, "timestamp").value_or(std::string{}),
        .filename = std::string{filename},
        .platform_id = jsonString(header, "platformID").value_or(std::string{}),
        .partition_id = std::move(partition_id),
        .record_id = record_id,
        .creator_id = jsonString(header, "creatorID").value_or(std::string{}),
        .events = {},
    };
}

DecodeContext make_decode_context(
    const nlohmann::json& header, PipelineOutput& output,
    addc::mca::McaDecodeFunction mca_decode,
    addc::decoder::DbgLogDecodeFunction dbglog_decode,
    addc::decoder::WdtDecodeFunction wdt_decode)
{
    std::string header_severity;
    if (const auto* severity = jsonObjectPtr(header, "severity"))
    {
        header_severity = jsonString(*severity, "name").value_or(std::string{});
    }
    return DecodeContext{
        .section_index = 0U,
        .timestamp = output.timestamp,
        .header_severity = std::move(header_severity),
        .diagnostics = &output.diagnostics,
        .decode_mca = mca_decode,
        .decode_dbglog = dbglog_decode,
        .decode_wdt = wdt_decode,
    };
}

[[nodiscard]] bool appendDecodedSection(
    std::size_t index, const cper::StructuralSection& section,
    const DecodeContext& base_context, PipelineOutput& output,
    std::string* error_out)
{
    DecodeContext context = base_context;
    context.section_index = index;
    const SectionDescriptor descriptor =
        parse_descriptor(section.canonical_descriptor);
    std::string decode_error;
    auto decoded = decode_base_section(descriptor, section.payload, context,
                                       &decode_error);
    if (!decode_error.empty())
    {
        if (error_out != nullptr)
        {
            *error_out = std::move(decode_error);
        }
        return false;
    }
    if (!decoded.handled)
    {
        output.diagnostics.warn(
            "section_undecoded", "No decoder matched section",
            nlohmann::ordered_json{{"section_type", descriptor.guid}});
    }
    for (auto& event : decoded.events)
    {
        output.events.push_back(std::move(event));
    }
    return true;
}

[[nodiscard]] std::optional<std::size_t> unsignedField(
    const nlohmann::json& object, std::string_view name)
{
    const auto it = object.find(name);
    if (it == object.end() ||
        !(it->is_number_unsigned() || it->is_number_integer()))
    {
        return std::nullopt;
    }
    uint64_t value = 0U;
    if (it->is_number_unsigned())
    {
        value = it->get<uint64_t>();
    }
    else
    {
        const auto signed_value = it->get<int64_t>();
        if (signed_value < 0)
        {
            return std::nullopt;
        }
        value = static_cast<uint64_t>(signed_value);
    }
    if (value > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::optional<std::vector<uint8_t>> syntheticBacking(
    const nlohmann::json& ir)
{
    const auto descriptors = ir.find("sectionDescriptors");
    if (descriptors == ir.end() || !descriptors->is_array())
    {
        return std::nullopt;
    }
    std::size_t size = 0U;
    for (const auto& descriptor : *descriptors)
    {
        const auto offset = unsignedField(descriptor, "sectionOffset");
        const auto length = unsignedField(descriptor, "sectionLength");
        if (!offset || !length || *offset >
                                    std::numeric_limits<std::size_t>::max() -
                                        *length)
        {
            return std::nullopt;
        }
        size = std::max(size, *offset + *length);
    }
    return std::vector<uint8_t>(size);
}

std::optional<cper::StructuralRecord> prepare_structural_record(
    cper::CperRecord& record, const addc::DecodeOptions& options,
    std::optional<cper::ParseFailure>* parse_failure_out,
    std::string* error_out)
{
    nlohmann::json ir;
    ir["header"] = record.header;
    ir["sectionDescriptors"] = nlohmann::json::array();
    ir["sections"] = nlohmann::json::array();

    std::size_t backing_size = 0U;
    for (std::size_t index = 0U; index < record.descriptors.size(); ++index)
    {
        const auto& descriptor = record.descriptors[index];
        if (descriptor.section_offset >
            std::numeric_limits<std::size_t>::max() -
                descriptor.section_length)
        {
            return std::nullopt;
        }
        backing_size = std::max(
            backing_size, static_cast<std::size_t>(descriptor.section_offset) +
                              descriptor.section_length);
    }
    std::vector<uint8_t> backing(backing_size);

    const std::size_t count =
        std::min(record.descriptors.size(), record.section_bodies.size());
    for (std::size_t index = 0U; index < count; ++index)
    {
        const auto& descriptor = record.descriptors[index];
        const auto& body = record.section_bodies[index];
        nlohmann::json descriptor_json = descriptor;
        const auto classification =
            cper::classify_section(record.header, descriptor);
        descriptor_json["sectionType"]["type"] =
            std::string{classification.type_name};
        ir["sectionDescriptors"].push_back(descriptor_json);

        cper::ParseContext parse_context{
            .mode = options.parse_mode,
            .binary_representation = cper::BinaryRepresentation::Native,
            .section_index = index,
            .repairs = &record.repairs,
        };
        nlohmann::json parsed = cper::parse_section_json(
            classification.kind, std::span<const uint8_t>{body}, descriptor,
            parse_context);
        if (parsed.is_null() && options.parse_mode == addc::ParseMode::Strict &&
            classification.kind != cper::SectionKind::Opaque)
        {
            if (parse_failure_out != nullptr)
            {
                *parse_failure_out = cper::ParseFailure{
                    .code = cper::ParseError::InvalidSectionBody,
                    .offset = descriptor.section_offset,
                    .section_index = index,
                };
            }
            if (error_out != nullptr)
            {
                *error_out =
                    addc::format("section {}: invalid section body", index);
            }
            return std::nullopt;
        }

        nlohmann::json section_json;
        if (!parsed.is_null())
        {
            if (!classification.description.empty())
            {
                section_json["message"] =
                    std::string{classification.description};
            }
            section_json[std::string{classification.json_key}] =
                std::move(parsed);
        }
        ir["sections"].push_back(std::move(section_json));

        const std::size_t copy_size = std::min<std::size_t>(
            body.size(), descriptor.section_length);
        std::copy_n(body.begin(), copy_size,
                    backing.begin() + descriptor.section_offset);
    }
    if (!record.repairs.empty())
    {
        ir["recovery"]["recovered"] = true;
        ir["recovery"]["repairs"] = record.repairs;
    }

    cper::TypedRecordFailure failure;
    auto typed = cper::to_typed_record(ir, backing, {}, &failure);
    if (!typed && error_out != nullptr && error_out->empty())
    {
        *error_out = "typed CPER conversion failed";
    }
    return typed;
}

[[nodiscard]] std::optional<PipelineOutput> analyzeTyped(
    cper::StructuralRecord record, const nlohmann::json* recovery,
    std::string_view filename, std::string_view tool_version,
    std::string* error_out, addc::mca::McaDecodeFunction mca_decode,
    addc::decoder::DbgLogDecodeFunction dbglog_decode,
    addc::decoder::WdtDecodeFunction wdt_decode)
{
    PipelineOutput output =
        make_pipeline_output(record.canonical_header, filename, tool_version);
    if (recovery != nullptr)
    {
        append_parser_recovery(output, *recovery);
    }
    const DecodeContext context = make_decode_context(
        record.canonical_header, output, mca_decode, dbglog_decode, wdt_decode);
    for (std::size_t index = 0U; index < record.sections.size(); ++index)
    {
        if (!appendDecodedSection(index, record.sections[index], context,
                                  output, error_out))
        {
            return std::nullopt;
        }
    }
    return output;
}

std::optional<PipelineOutput> run_pipeline(
    const nlohmann::json& ir,
    std::string_view filename, std::string_view tool_version,
    std::string* error_out, addc::mca::McaDecodeFunction mca_decode,
    addc::decoder::DbgLogDecodeFunction dbglog_decode,
    addc::decoder::WdtDecodeFunction wdt_decode)
{
    auto backing = syntheticBacking(ir);
    if (!backing)
    {
        return fail("IR has invalid section descriptors", error_out);
    }
    cper::TypedRecordFailure failure;
    auto record = cper::to_typed_record(ir, *backing, {}, &failure);
    if (!record)
    {
        return fail("IR cannot be converted to a typed CPER record", error_out);
    }
    const auto recovery = ir.find("recovery");
    return analyzeTyped(std::move(*record),
                        recovery != ir.end() ? &(*recovery) : nullptr,
                        filename, tool_version, error_out, mca_decode,
                        dbglog_decode, wdt_decode);
}

std::optional<PipelineOutput> run_pipeline(
    cper::CperRecord record, const addc::DecodeOptions& options,
    std::string_view filename, std::string_view tool_version,
    std::string* error_out,
    std::optional<addc::cper::ParseFailure>* parse_failure_out,
    addc::mca::McaDecodeFunction mca_decode,
    addc::decoder::DbgLogDecodeFunction dbglog_decode,
    addc::decoder::WdtDecodeFunction wdt_decode)
{
    if (parse_failure_out != nullptr)
    {
        parse_failure_out->reset();
    }
    auto typed = prepare_structural_record(record, options, parse_failure_out,
                                           error_out);
    if (!typed)
    {
        return std::nullopt;
    }
    nlohmann::json recovery;
    const nlohmann::json* recovery_ptr = nullptr;
    if (!record.repairs.empty())
    {
        recovery["recovered"] = true;
        recovery["repairs"] = record.repairs;
        recovery_ptr = &recovery;
    }
    return analyzeTyped(std::move(*typed), recovery_ptr, filename,
                        tool_version, error_out, mca_decode, dbglog_decode,
                        wdt_decode);
}

} // namespace addc::pipeline
