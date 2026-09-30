// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/cper/structural.hpp"
#include "addc/cper/section_dispatch.hpp"

#include <limits>

namespace addc::cper
{

namespace
{

void fail(TypedRecordFailure* out, TypedRecordError code,
          std::optional<std::size_t> section_index = std::nullopt)
{
    if (out != nullptr)
    {
        *out = TypedRecordFailure{code, section_index};
    }
}

std::optional<std::size_t> unsignedField(const nlohmann::json& object,
                                         std::string_view name)
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

std::vector<ParseRepair> parseRepairs(const nlohmann::json& root)
{
    std::vector<ParseRepair> result;
    const auto recovery = root.find("recovery");
    if (recovery == root.end() || !recovery->is_object())
    {
        return result;
    }
    const auto repairs = recovery->find("repairs");
    if (repairs == recovery->end() || !repairs->is_array())
    {
        return result;
    }
    for (const auto& item : *repairs)
    {
        if (!item.is_object())
        {
            continue;
        }
        ParseRepair repair;
        repair.code = item.value("code", std::string{});
        repair.field = item.value("field", std::string{});
        repair.original_value = item.value("original_value", uint64_t{});
        repair.substituted_value =
            item.value("substituted_value", uint64_t{});
        repair.reason = item.value("reason", std::string{});
        if (const auto index = unsignedField(item, "section_index"))
        {
            repair.section_index = *index;
        }
        result.push_back(std::move(repair));
    }
    return result;
}

} // namespace

std::optional<StructuralRecord> to_typed_record(
    const nlohmann::json& canonical_parser_json,
    std::span<const uint8_t> original_bytes,
    const Diagnostics& boundary_diagnostics, TypedRecordFailure* failure_out)
{
    if (!canonical_parser_json.is_object())
    {
        fail(failure_out, TypedRecordError::InvalidRoot);
        return std::nullopt;
    }

    const auto header = canonical_parser_json.find("header");
    if (header == canonical_parser_json.end() || !header->is_object())
    {
        fail(failure_out, TypedRecordError::MissingHeader);
        return std::nullopt;
    }
    const auto descriptors = canonical_parser_json.find("sectionDescriptors");
    const auto sections = canonical_parser_json.find("sections");
    if (descriptors == canonical_parser_json.end() ||
        sections == canonical_parser_json.end() || !descriptors->is_array() ||
        !sections->is_array())
    {
        fail(failure_out, TypedRecordError::MissingSectionArrays);
        return std::nullopt;
    }
    if (descriptors->size() != sections->size())
    {
        fail(failure_out, TypedRecordError::SectionCountMismatch);
        return std::nullopt;
    }

    StructuralRecord record{
        .canonical_header = *header,
        .sections = {},
        .repairs = parseRepairs(canonical_parser_json),
        .boundary_diagnostics = boundary_diagnostics,
    };
    record.sections.reserve(sections->size());

    for (std::size_t index = 0; index < sections->size(); ++index)
    {
        const auto& descriptor = (*descriptors)[index];
        if (!descriptor.is_object())
        {
            fail(failure_out, TypedRecordError::InvalidDescriptor, index);
            return std::nullopt;
        }
        const auto offset = unsignedField(descriptor, "sectionOffset");
        const auto length = unsignedField(descriptor, "sectionLength");
        const auto type = descriptor.find("sectionType");
        const auto revision = descriptor.find("revision");
        if (!offset || !length || type == descriptor.end() ||
            !type->is_object() || revision == descriptor.end() ||
            !revision->is_object())
        {
            fail(failure_out, TypedRecordError::InvalidDescriptor, index);
            return std::nullopt;
        }
        if (*offset > original_bytes.size() ||
            *length > original_bytes.size() - *offset)
        {
            fail(failure_out, TypedRecordError::InvalidSourceRange, index);
            return std::nullopt;
        }

        const SourceRange range{*offset, *length};
        auto kind = section_kind_from_type_name(
            type->value("type", std::string{}));
        if (kind == SectionKind::AmdMi300Runtime)
        {
            const auto major = unsignedField(*revision, "major");
            if (major && (*major & 0xFU) == 2U &&
                ((*major >> 4U) & 0xFU) == 3U)
            {
                kind = SectionKind::AmdMi450Runtime;
            }
        }
        const std::span<const uint8_t> raw{
            original_bytes.data() + *offset, *length};
        std::string conversion_error;
        auto payload = convert_section(kind, descriptor, (*sections)[index],
                                       range, raw, &conversion_error);
        if (!conversion_error.empty())
        {
            fail(failure_out, TypedRecordError::InvalidDescriptor, index);
            return std::nullopt;
        }
        record.sections.push_back(StructuralSection{
            .kind = kind,
            .source_range = range,
            .raw_data = std::vector<uint8_t>{raw.begin(), raw.end()},
            .canonical_descriptor = descriptor,
            .canonical_section = (*sections)[index],
            .payload = std::move(payload),
        });
    }
    return record;
}

} // namespace addc::cper
