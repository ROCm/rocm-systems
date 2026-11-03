// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "schema_validator.hpp"

#include "../cache/schema_cache.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace profiler_hub::schema::schema_validator
{
namespace
{

[[nodiscard]] bool
field_was_set(const std::vector<field_assignment>& fields, std::string_view name)
{
    for(const auto& field : fields)
    {
        if(field.set && field.name == name) return true;
    }
    return false;
}

void
add_error(validation_result& result, validation_error error)
{
    result.ok = false;
    result.errors.push_back(std::move(error));
}

[[nodiscard]] validation_result
unsupported_version(std::string_view object_name, profiler_hub::version_t version)
{
    validation_result result;
    result.ok             = false;
    result.schema_version = version.to_string();
    result.object         = std::string{ object_name };

    validation_error error;
    error.object = result.object;
    error.reason = "Schema version " + result.schema_version + " is not supported.";
    error.documentation = documentation_link(version);
    result.errors.push_back(std::move(error));
    return result;
}

[[nodiscard]] std::vector<field_assignment>
fields_of(const pmc_event_input_t& input)
{
    std::vector<field_assignment> fields;
    auto                          add = [&](const char* name, bool set) {
        if(!set) return;
        fields.push_back(field_assignment{ name, true, std::nullopt });
    };

    add("event_id", input.event_id.has_value());
    add("sample_id", input.sample_id.has_value());
    add("pmc_id", input.pmc_id.has_value());
    add("value", input.value.has_value());
    add("xcc", input.xcc.has_value());
    add("shader_engine", input.shader_engine.has_value());
    add("instance", input.instance.has_value());
    add("extdata", input.extdata.has_value());
    return fields;
}

}  // namespace

validation_result
validate_fields(std::string_view                     object_name,
                std::string_view                     table_name,
                profiler_hub::version_t              version,
                const std::vector<field_assignment>& fields)
{
    validation_result result;
    result.schema_version = version.to_string();
    result.object         = std::string{ object_name };

    const schema_model* model = nullptr;
    try
    {
        model = &shared_cache().get(version);
    } catch(const std::invalid_argument&)
    {
        return unsupported_version(object_name, version);
    }

    const auto  link  = documentation_link(version);
    const auto* table = model->find_table(table_name);
    if(table == nullptr)
    {
        validation_error error;
        error.object = result.object;
        error.reason =
            result.object + " is not supported\nby the selected schema version.";
        error.documentation = link;
        add_error(result, std::move(error));
        return result;
    }

    for(const auto& field : fields)
    {
        if(!field.set) continue;

        const auto* column = table->find(field.name);
        if(column == nullptr)
        {
            validation_error error;
            error.object = result.object;
            error.field  = field.name;
            error.reason =
                "Field is not supported\nby schema version " + result.schema_version;
            error.documentation = link;
            add_error(result, std::move(error));
            continue;
        }

        if(column->allowed_values.empty() || !field.enum_value.has_value()) continue;

        const auto& allowed = column->allowed_values;
        bool        matches = false;
        for(const auto& value : allowed)
        {
            if(value == *field.enum_value)
            {
                matches = true;
                break;
            }
        }
        if(matches) continue;

        validation_error error;
        error.object = result.object;
        error.field  = field.name;
        error.reason = *field.enum_value + " is not supported by schema version " +
                       result.schema_version + ".";
        if(!allowed.empty())
        {
            error.reason += "\nAllowed values:";
            for(const auto& value : allowed)
            {
                error.reason += "\n" + value;
            }
        }
        error.documentation = link;
        add_error(result, std::move(error));
    }

    for(const auto& entry : table->columns)
    {
        const column_model& column = entry.second;
        if(!column.required_from_user()) continue;
        if(field_was_set(fields, column.name)) continue;

        validation_error error;
        error.object = result.object;
        error.field  = column.name;
        error.reason = "Field is required\nby schema version " + result.schema_version;
        error.documentation = link;
        add_error(result, std::move(error));
    }

    return result;
}

validation_result
validate(const pmc_event_input_t& input, profiler_hub::version_t version)
{
    return validate_fields("PMCEvent", "rocpd_pmc_event", version, fields_of(input));
}

validation_result
write_pmc_event(profiler_hub::writer_t&                                 writer,
                const pmc_event_input_t&                                input,
                profiler_hub::version_t                                 version,
                const profiler_hub::writer_types::pmc_event_data_t&     event,
                const profiler_hub::writer_types::pmc_info_unique_id_t& pmc_unique_id)
{
    // Existing writer is unchanged. This only decides whether to call it.
    auto result = validate(input, version);
    if(!result.success())
    {
        return result;
    }
    writer.insert_pmc_event_data(event, pmc_unique_id);
    return result;
}

}  // namespace profiler_hub::schema::schema_validator
