// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema_loader.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <fstream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
using json = nlohmann::json;

template <typename EnumT>
using enum_entry_t = std::pair<std::string_view, EnumT>;

constexpr auto k_param_types = std::array{
    enum_entry_t<param_type>{ "bool", param_type::boolean },
    enum_entry_t<param_type>{ "integer", param_type::integer },
    enum_entry_t<param_type>{ "double", param_type::number },
    enum_entry_t<param_type>{ "string", param_type::string },
    enum_entry_t<param_type>{ "path", param_type::path },
    enum_entry_t<param_type>{ "enum", param_type::enumeration },
    enum_entry_t<param_type>{ "enum_list", param_type::enumeration_list },
};

constexpr auto k_param_levels = std::array{
    enum_entry_t<param_level>{ "basic", param_level::basic },
    enum_entry_t<param_level>{ "advanced", param_level::advanced },
    enum_entry_t<param_level>{ "expert", param_level::expert },
};

constexpr auto k_path_kinds = std::array{
    enum_entry_t<path_kind>{ "file", path_kind::file },
    enum_entry_t<path_kind>{ "directory", path_kind::directory },
};

constexpr auto k_arg_forms = std::array{
    enum_entry_t<arg_form>{ "value", arg_form::value },
    enum_entry_t<arg_form>{ "flag", arg_form::flag },
};

constexpr auto k_assign_styles = std::array{
    enum_entry_t<assign_style>{ "separate", assign_style::separate },
    enum_entry_t<assign_style>{ "equals", assign_style::equals },
};

constexpr auto k_severities = std::array{
    enum_entry_t<severity>{ "info", severity::info },
    enum_entry_t<severity>{ "warning", severity::warning },
    enum_entry_t<severity>{ "error", severity::error },
};

constexpr auto k_constraint_kinds = std::array{
    enum_entry_t<constraint_kind>{ "mutually_exclusive",
                                   constraint_kind::mutually_exclusive },
    enum_entry_t<constraint_kind>{ "forces", constraint_kind::forces },
    enum_entry_t<constraint_kind>{ "warn", constraint_kind::warn },
};

const json*
find_member(const json& object, const char* key)
{
    auto itr = object.find(key);
    return (itr != object.end() && !itr->is_null()) ? &(*itr) : nullptr;
}

std::string
as_string(const json& value, std::string_view key)
{
    if(!value.is_string())
    {
        throw std::runtime_error(fmt::format("field '{}' must be a string", key));
    }
    return value.get<std::string>();
}

std::string
require_string(const json& object, const char* key)
{
    const auto* member = find_member(object, key);
    if(member == nullptr)
    {
        throw std::runtime_error(fmt::format("missing required field '{}'", key));
    }
    return as_string(*member, key);
}

std::optional<std::string>
read_string(const json& object, const char* key)
{
    const auto* member = find_member(object, key);
    return member ? std::optional{ as_string(*member, key) } : std::nullopt;
}

std::optional<double>
read_number(const json& object, const char* key)
{
    const auto* member = find_member(object, key);
    return member ? std::optional{ member->get<double>() } : std::nullopt;
}

bool
read_bool(const json& object, const char* key)
{
    const auto* member = find_member(object, key);
    return member && member->get<bool>();
}

std::vector<std::string>
read_string_list(const json& object, const char* key)
{
    const auto* member = find_member(object, key);
    return member ? member->get<std::vector<std::string>>() : std::vector<std::string>{};
}

template <typename EnumT, std::size_t SizeV>
EnumT
to_enum(const std::array<enum_entry_t<EnumT>, SizeV>& table, std::string_view field,
        std::string_view text)
{
    for(const auto& [name, value] : table)
    {
        if(name == text)
        {
            return value;
        }
    }
    throw std::runtime_error(fmt::format("unknown {} '{}'", field, text));
}

template <typename EnumT, std::size_t SizeV>
EnumT
read_enum(const json& object, const char* key,
          const std::array<enum_entry_t<EnumT>, SizeV>& table, EnumT fallback)
{
    const auto text = read_string(object, key);
    return text ? to_enum(table, key, *text) : fallback;
}

std::string
scalar_to_string(const json& value)
{
    if(value.is_string())
    {
        return value.get<std::string>();
    }
    if(value.is_boolean() || value.is_number())
    {
        return value.dump();
    }
    throw std::runtime_error(
        fmt::format("expected a scalar value, got '{}'", value.dump()));
}

std::string
to_value_string(const json& value, std::string_view separator)
{
    if(!value.is_array())
    {
        return scalar_to_string(value);
    }

    std::vector<std::string> items;
    items.reserve(value.size());
    for(const auto& item : value)
    {
        items.push_back(scalar_to_string(item));
    }
    return fmt::format("{}", fmt::join(items, separator));
}

std::vector<choice>
read_choices(const json& object)
{
    std::vector<choice> result;
    if(const auto* member = find_member(object, "choices"))
    {
        for(const auto& item : *member)
        {
            result.push_back(choice{
                .value       = require_string(item, "value"),
                .name        = require_string(item, "name"),
                .description = read_string(item, "description").value_or(""),
            });
        }
    }
    return result;
}

std::optional<env_spec>
read_env(const json& object)
{
    const auto* member = find_member(object, "env");
    if(member == nullptr)
    {
        return std::nullopt;
    }
    return env_spec{
        .name      = require_string(*member, "name"),
        .separator = read_string(*member, "separator").value_or(","),
    };
}

std::optional<arg_spec>
read_arg(const json& object)
{
    const auto* member = find_member(object, "arg");
    if(member == nullptr)
    {
        return std::nullopt;
    }
    return arg_spec{
        .name = require_string(*member, "name"),
        .form = read_enum(*member, "form", k_arg_forms, arg_form::value),
        .style =
            read_enum(*member, "assign_style", k_assign_styles, assign_style::separate),
    };
}

param_condition
read_param_condition(const json& object)
{
    const auto* equals = find_member(object, "equals");
    if(equals == nullptr)
    {
        throw std::runtime_error("condition is missing 'equals'");
    }
    return param_condition{
        .param  = require_string(object, "param"),
        .equals = scalar_to_string(*equals),
    };
}

std::vector<param_condition>
read_terms(const json& terms)
{
    std::vector<param_condition> result;
    for(const auto& term : terms)
    {
        result.push_back(read_param_condition(term));
    }
    return result;
}

condition
read_condition(const json& object)
{
    if(const auto* terms = find_member(object, "all"))
    {
        return condition{ .mode = condition_mode::all, .terms = read_terms(*terms) };
    }
    if(const auto* terms = find_member(object, "any"))
    {
        return condition{ .mode = condition_mode::any, .terms = read_terms(*terms) };
    }
    return condition{ .mode  = condition_mode::all,
                      .terms = { read_param_condition(object) } };
}

schema_param
read_param(const json& object)
{
    schema_param param{};
    param.id           = require_string(object, "id");
    param.name         = require_string(object, "name");
    param.description  = require_string(object, "description");
    param.type         = to_enum(k_param_types, "type", require_string(object, "type"));
    param.category     = read_string(object, "category").value_or("");
    param.level        = read_enum(object, "level", k_param_levels, param_level::basic);
    param.min          = read_number(object, "min");
    param.max          = read_number(object, "max");
    param.units        = read_string(object, "units").value_or("");
    param.path         = read_enum(object, "path_kind", k_path_kinds, path_kind::any);
    param.choices      = read_choices(object);
    param.allow_custom = read_bool(object, "allow_custom");
    param.pattern      = read_string(object, "pattern");
    param.required     = read_bool(object, "required");
    param.env          = read_env(object);
    param.arg          = read_arg(object);
    param.mutually_exclusive_with = read_string_list(object, "mutually_exclusive_with");
    param.prerequisites           = read_string_list(object, "requires");

    if(const auto* depends_on = find_member(object, "depends_on"))
    {
        param.depends_on = read_param_condition(*depends_on);
    }
    if(const auto* value = find_member(object, "default_value"))
    {
        param.default_value = to_value_string(*value, list_separator(param));
    }
    return param;
}

void
verify_type_fields(const schema_param& param)
{
    const bool is_numeric =
        param.type == param_type::integer || param.type == param_type::number;
    const bool is_enumeration = param.type == param_type::enumeration ||
                                param.type == param_type::enumeration_list;

    if(is_enumeration && param.choices.empty() && !param.allow_custom)
    {
        throw std::runtime_error("enum parameters need 'choices' or 'allow_custom'");
    }
    if(!is_enumeration && (!param.choices.empty() || param.allow_custom))
    {
        throw std::runtime_error("'choices' and 'allow_custom' only apply to enum types");
    }
    if(!is_numeric && (param.min || param.max))
    {
        throw std::runtime_error("'min' and 'max' only apply to numeric types");
    }
    if(param.min && param.max && *param.min > *param.max)
    {
        throw std::runtime_error("'min' is greater than 'max'");
    }
    if(param.path != path_kind::any && param.type != param_type::path)
    {
        throw std::runtime_error("'path_kind' only applies to path parameters");
    }
    if(param.arg && param.arg->form == arg_form::flag &&
       param.type != param_type::boolean)
    {
        throw std::runtime_error("flag arguments require a bool parameter");
    }
}

void
verify_pattern(const schema_param& param)
{
    if(!param.pattern)
    {
        return;
    }
    if(param.type != param_type::string && param.type != param_type::path)
    {
        throw std::runtime_error("'pattern' only applies to string and path parameters");
    }

    try
    {
        static_cast<void>(std::regex{ *param.pattern });
    } catch(const std::regex_error& err)
    {
        throw std::runtime_error(
            fmt::format("invalid pattern '{}': {}", *param.pattern, err.what()));
    }
}

void
verify_value_fields(const schema_param& param)
{
    if(param.env && param.env->separator.empty())
    {
        throw std::runtime_error("'env.separator' must not be empty");
    }

    verify_pattern(param);

    if(param.default_value && !is_valid_value(param, *param.default_value))
    {
        throw std::runtime_error(
            fmt::format("invalid default value '{}'", *param.default_value));
    }
}

constraint
read_constraint(const json& object)
{
    constraint result{};
    result.kind =
        to_enum(k_constraint_kinds, "constraint kind", require_string(object, "kind"));
    result.level   = read_enum(object, "severity", k_severities, severity::error);
    result.message = read_string(object, "message").value_or("");
    result.params  = read_string_list(object, "params");

    if(const auto* when = find_member(object, "when"))
    {
        result.when = read_condition(*when);
    }
    if(const auto* forced = find_member(object, "set"))
    {
        for(const auto& [param_id, value] : forced->items())
        {
            result.forced_values.emplace(param_id, scalar_to_string(value));
        }
    }
    return result;
}

void
verify_constraint_fields(const constraint& item)
{
    const bool is_conditional = item.kind != constraint_kind::mutually_exclusive;

    if(!is_conditional && item.params.size() < 2)
    {
        throw std::runtime_error("mutually_exclusive needs at least two 'params'");
    }
    if(is_conditional && item.when.terms.empty())
    {
        throw std::runtime_error("constraint needs a 'when' condition");
    }
    if(is_conditional && item.message.empty())
    {
        throw std::runtime_error("constraint needs a 'message'");
    }
    if(item.kind == constraint_kind::forces && item.forced_values.empty())
    {
        throw std::runtime_error("forces constraint needs a 'set' object");
    }
}

const schema_param&
require_param(const schema& result, std::string_view param_id, std::string_view owner)
{
    const auto* param = result.find_param(param_id);
    if(param == nullptr)
    {
        throw std::runtime_error(
            fmt::format("{} references unknown parameter '{}'", owner, param_id));
    }
    return *param;
}

void
verify_value_for(const schema& result, std::string_view param_id, std::string_view value,
                 std::string_view owner)
{
    if(!is_valid_value(require_param(result, param_id, owner), value))
    {
        throw std::runtime_error(fmt::format(
            "{} uses invalid value '{}' for parameter '{}'", owner, value, param_id));
    }
}

void
verify_param_references(const schema& result, const schema_param& param)
{
    const auto owner = fmt::format("parameter '{}'", param.id);
    if(!param.category.empty() && result.find_category(param.category) == nullptr)
    {
        throw std::runtime_error(
            fmt::format("{} references unknown category '{}'", owner, param.category));
    }
    if(param.depends_on)
    {
        verify_value_for(result, param.depends_on->param, param.depends_on->equals,
                         owner);
    }

    auto related = param.mutually_exclusive_with;
    related.insert(related.end(), param.prerequisites.begin(), param.prerequisites.end());
    for(const auto& param_id : related)
    {
        if(&require_param(result, param_id, owner) == &param)
        {
            throw std::runtime_error(fmt::format("{} references itself", owner));
        }
    }
}

void
verify_constraint_references(const schema& result, const constraint& item)
{
    constexpr std::string_view k_owner = "constraint";
    for(const auto& param_id : item.params)
    {
        static_cast<void>(require_param(result, param_id, k_owner));
    }
    for(const auto& term : item.when.terms)
    {
        verify_value_for(result, term.param, term.equals, k_owner);
    }
    for(const auto& [param_id, value] : item.forced_values)
    {
        verify_value_for(result, param_id, value, k_owner);
    }
}

std::string
describe_entry(const json& entry, std::size_t position)
{
    if(const auto* entry_id = find_member(entry, "id");
       entry_id != nullptr && entry_id->is_string())
    {
        return fmt::format("'{}'", entry_id->get<std::string>());
    }
    return fmt::format("#{}", position);
}

template <typename FuncT>
void
for_each_entry(const json& document, const char* key, std::string_view kind,
               const FuncT& visit)
{
    const auto* entries = find_member(document, key);
    if(entries == nullptr)
    {
        return;
    }
    if(!entries->is_array())
    {
        throw std::runtime_error(fmt::format("'{}' must be an array", key));
    }

    std::size_t position = 0;
    for(const auto& entry : *entries)
    {
        try
        {
            visit(entry);
        } catch(const std::exception& err)
        {
            throw std::runtime_error(fmt::format(
                "{} {}: {}", kind, describe_entry(entry, position), err.what()));
        }
        ++position;
    }
}
}  // namespace

schema
load_schema(const json& document)
{
    if(!document.is_object())
    {
        throw std::runtime_error("schema document must be a JSON object");
    }
    if(find_member(document, "params") == nullptr)
    {
        throw std::runtime_error("schema is missing the 'params' array");
    }

    auto result = schema{};
    result.set_version(require_string(document, "schema_version"));

    for_each_entry(document, "categories", "category", [&result](const json& entry) {
        auto value = category{
            .id   = require_string(entry, "id"),
            .name = require_string(entry, "name"),
        };
        if(!result.add_category(std::move(value)))
        {
            throw std::runtime_error("duplicate category id");
        }
    });

    for_each_entry(document, "params", "parameter", [&result](const json& entry) {
        auto param = read_param(entry);
        verify_type_fields(param);
        verify_value_fields(param);
        if(!result.add_param(std::move(param)))
        {
            throw std::runtime_error(
                "reuses an id, environment variable, or argument name");
        }
    });

    for_each_entry(document, "constraints", "constraint", [&result](const json& entry) {
        auto item = read_constraint(entry);
        verify_constraint_fields(item);
        verify_constraint_references(result, item);
        result.add_constraint(std::move(item));
    });

    for(const auto& param : result.params())
    {
        verify_param_references(result, param);
    }
    return result;
}

schema
load_schema_file(const std::string& filepath)
{
    std::ifstream ifs{ filepath };
    if(!ifs.is_open())
    {
        throw std::runtime_error(
            fmt::format("failed to open schema file '{}'", filepath));
    }

    auto document = json{};
    try
    {
        document = json::parse(ifs);
    } catch(const json::parse_error& err)
    {
        throw std::runtime_error(
            fmt::format("failed to parse schema file '{}': {}", filepath, err.what()));
    }

    return load_schema(document);
}

}  // namespace rocprofsys::parser
