// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/configuration_parser.hpp"
#include "parser/cli_parser.hpp"
#include "parser/config_store.hpp"
#include "parser/default_parser.hpp"
#include "parser/default_schema.hpp"
#include "parser/env_parser.hpp"
#include "parser/file_parser.hpp"
#include "parser/help.hpp"
#include "parser/parse_result.hpp"
#include "parser/precedence.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_provider.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <stdlib.h>  // POSIX ::setenv
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
constexpr std::string_view k_config_file_param = "config_file";
constexpr std::string_view k_preset_param      = "preset";
constexpr int              k_json_indent       = 2;
constexpr int              k_overwrite         = 1;

using json = nlohmann::json;

unknown_option
to_unknown(parse_mode mode)
{
    return mode == parse_mode::stop_at_unknown ? unknown_option::stop
                                               : unknown_option::report;
}

void
append_errors(std::vector<std::string>& dest, const std::vector<std::string>& extra)
{
    dest.insert(dest.end(), extra.begin(), extra.end());
}

void
reject_provider_values(const schema&                  config_schema,
                       const value_provider_registry& providers, parse_result& parsed)
{
    for(auto itr = parsed.values.begin(); itr != parsed.values.end();)
    {
        const auto* param    = config_schema.find_param(itr->first);
        const auto* provider = (param != nullptr && param->value_provider)
                                   ? providers.find(*param->value_provider)
                                   : nullptr;
        const auto  rejected =
            param != nullptr && param->value_provider &&
            (provider == nullptr || !provider->accepts(*param, itr->second));
        if(rejected)
        {
            parsed.errors.push_back(fmt::format("invalid value '{}' for parameter '{}'",
                                                itr->second, itr->first));
            itr = parsed.values.erase(itr);
        }
        else
        {
            ++itr;
        }
    }
}

sourced_values
make_source(config_source source, const parse_result& parsed)
{
    return sourced_values{ .source = source, .values = parsed.values };
}

std::optional<std::string>
selected_value(std::vector<sourced_values> sources, std::string_view param_id)
{
    const auto resolved = resolve_precedence(std::move(sources));
    const auto itr      = resolved.find(param_id);
    if(itr == resolved.end() || itr->second.value.empty())
    {
        return std::nullopt;
    }
    return itr->second.value;
}

env_parser
make_env(const env_parser::lookup_t& lookup)
{
    if(lookup)
    {
        return env_parser{ lookup };
    }
    return env_parser::from_process();
}

parse_result
read_config_file(const schema& config_schema, const std::optional<std::string>& path)
{
    if(!path)
    {
        return {};
    }
    return file_parser{}.parse_file(config_schema, *path);
}

parse_result
read_preset(const schema& config_schema, const std::optional<std::string>& name,
            const configuration_parser::preset_lookup_t& lookup,
            std::vector<std::string>&                    errors)
{
    if(!name)
    {
        return {};
    }
    if(!lookup)
    {
        errors.push_back(fmt::format("preset '{}' was not applied", *name));
        return {};
    }
    const auto settings = lookup(*name);
    if(settings.empty())
    {
        errors.push_back(fmt::format("unknown preset '{}'", *name));
        return {};
    }
    return preset_parser{}.parse(config_schema, settings);
}

struct collected_input
{
    parse_result defaults;
    parse_result environment;
    parse_result command_line;
    parse_result file;
    parse_result preset;
};

struct parse_request
{
    const schema&                                config_schema;
    const std::vector<std::string>&              arguments;
    parse_mode                                   mode = parse_mode::full;
    const env_parser::lookup_t&                  environment;
    const configuration_parser::preset_lookup_t& presets;
    const value_provider_registry&               providers;
    std::vector<std::string>&                    errors;
};

collected_input
read_primary(const parse_request& request)
{
    collected_input collected;
    collected.defaults    = default_parser{}.parse(request.config_schema);
    collected.environment = make_env(request.environment).parse(request.config_schema);
    collected.command_line =
        cli_parser{ request.arguments, to_unknown(request.mode) }.parse(
            request.config_schema);
    reject_provider_values(request.config_schema, request.providers, collected.defaults);
    reject_provider_values(request.config_schema, request.providers,
                           collected.environment);
    reject_provider_values(request.config_schema, request.providers,
                           collected.command_line);
    return collected;
}

void
read_indirect(const parse_request& request, collected_input& collected)
{
    const auto file_path = selected_value(
        { make_source(config_source::default_value, collected.defaults),
          make_source(config_source::environment, collected.environment),
          make_source(config_source::command_line, collected.command_line) },
        k_config_file_param);
    collected.file = read_config_file(request.config_schema, file_path);
    reject_provider_values(request.config_schema, request.providers, collected.file);

    const auto preset_name = selected_value(
        { make_source(config_source::default_value, collected.defaults),
          make_source(config_source::file, collected.file),
          make_source(config_source::environment, collected.environment),
          make_source(config_source::command_line, collected.command_line) },
        k_preset_param);
    collected.preset =
        read_preset(request.config_schema, preset_name, request.presets, request.errors);
    reject_provider_values(request.config_schema, request.providers, collected.preset);
}

void
collect_errors(std::vector<std::string>& errors, const collected_input& collected)
{
    append_errors(errors, collected.defaults.errors);
    append_errors(errors, collected.environment.errors);
    append_errors(errors, collected.command_line.errors);
    append_errors(errors, collected.file.errors);
    append_errors(errors, collected.preset.errors);
}

resolved_config
resolve_collected(const collected_input& collected)
{
    return resolve_precedence({
        make_source(config_source::default_value, collected.defaults),
        make_source(config_source::preset, collected.preset),
        make_source(config_source::file, collected.file),
        make_source(config_source::environment, collected.environment),
        make_source(config_source::command_line, collected.command_line),
    });
}

void
add_provider_defaults(resolved_config& resolved, const schema& config_schema,
                      const value_provider_registry& providers)
{
    for(const auto& param : config_schema.params())
    {
        const auto already_set =
            param.default_value.has_value() || resolved.contains(param.id);
        if(!param.value_provider || already_set)
        {
            continue;
        }
        const auto* provider = providers.find(*param.value_provider);
        if(provider == nullptr)
        {
            continue;
        }
        const auto fallback = provider->default_value(param).value_or("");
        if(fallback.empty() || !provider->accepts(param, fallback))
        {
            continue;
        }
        resolved.emplace(
            param.id,
            resolved_value{ .value = fallback, .source = config_source::default_value });
    }
}

struct effect_request
{
    std::string_view target;
    std::string      value;
    config_source    source    = config_source::default_value;
    bool             fill_only = false;
};

bool
keeps_existing(const resolved_config& resolved, const effect_request& request)
{
    const auto found = resolved.find(request.target);
    if(found == resolved.end())
    {
        return false;
    }
    if(request.fill_only)
    {
        return found->second.source != config_source::default_value;
    }
    return static_cast<std::uint8_t>(found->second.source) >
           static_cast<std::uint8_t>(request.source);
}

void
write_effect(resolved_config& resolved, effect_request request)
{
    if(keeps_existing(resolved, request))
    {
        return;
    }
    resolved.insert_or_assign(
        std::string{ request.target },
        resolved_value{ .value = std::move(request.value), .source = request.source });
}

void
write_pairs(resolved_config& resolved, const value_map_t& pairs, config_source source)
{
    for(const auto& [target, value] : pairs)
    {
        write_effect(resolved, effect_request{
                                   .target = target,
                                   .value  = value,
                                   .source = source,
                               });
    }
}

void
apply_choice_sets(resolved_config& resolved, const schema_param& param,
                  const resolved_value& origin)
{
    for(const auto& item : param.choices)
    {
        if(item.value == origin.value)
        {
            write_pairs(resolved, item.sets, origin.source);
        }
    }
}

void
apply_origin_effects(resolved_config& resolved, const schema_param& param,
                     const resolved_value& origin)
{
    for(const auto& target : param.copy_to)
    {
        write_effect(resolved, effect_request{
                                   .target = target,
                                   .value  = origin.value,
                                   .source = origin.source,
                               });
    }
    write_pairs(resolved, param.assigns, origin.source);
    for(const auto& mapping : param.maps_to)
    {
        if(mapping.when == origin.value)
        {
            write_effect(resolved, effect_request{
                                       .target = mapping.target,
                                       .value  = mapping.value,
                                       .source = origin.source,
                                   });
        }
    }
    apply_choice_sets(resolved, param, origin);
}

void
apply_fills(resolved_config& resolved, const schema_param& param,
            const resolved_value& origin)
{
    for(const auto& target : param.fills)
    {
        write_effect(resolved, effect_request{
                                   .target    = target,
                                   .value     = origin.value,
                                   .source    = origin.source,
                                   .fill_only = true,
                               });
    }
}

// Effects run from values that won precedence, not from defaults. A later
// effect does not see values written by an earlier one. fills runs last.
void
apply_effects(resolved_config& resolved, const schema& config_schema)
{
    std::vector<std::pair<const schema_param*, resolved_value>> origins;
    for(const auto& param : config_schema.params())
    {
        const auto found = resolved.find(param.id);
        if(found == resolved.end() ||
           found->second.source == config_source::default_value)
        {
            continue;
        }
        origins.emplace_back(&param, found->second);
    }
    for(const auto& [param, origin] : origins)
    {
        apply_origin_effects(resolved, *param, origin);
    }
    for(const auto& [param, origin] : origins)
    {
        apply_fills(resolved, *param, origin);
    }
}

void
append_violations(std::vector<std::string>& errors, const schema& config_schema,
                  const resolved_config& resolved)
{
    value_map_t values;
    for(const auto& [param_id, item] : resolved)
    {
        values.emplace(param_id, item.value);
    }
    for(const auto& item : validate_values(config_schema, values))
    {
        if(item.level != severity::info)
        {
            errors.push_back(item.message);
        }
    }
}

}  // namespace

void
configuration_parser::set_parse_mode(parse_mode mode)
{
    m_mode = mode;
}

void
configuration_parser::set_environment(env_parser::lookup_t lookup)
{
    m_environment = std::move(lookup);
}

void
configuration_parser::set_preset_lookup(preset_lookup_t lookup)
{
    m_presets = std::move(lookup);
}

bool
configuration_parser::initialize()
{
    try
    {
        return initialize(get_default_schema());
    } catch(const std::exception& err)
    {
        m_failure = err.what();
        return false;
    }
}

bool
configuration_parser::initialize(schema catalog)
{
    try
    {
        auto providers = std::make_unique<value_provider_registry>();
        providers->add_builtins();
        m_store     = std::make_unique<config_store>(std::move(catalog));
        m_providers = std::move(providers);
        m_failure.clear();
        m_errors.clear();
        m_command.clear();
        m_unparsed.clear();
        return true;
    } catch(const std::exception& err)
    {
        m_failure = err.what();
        return false;
    }
}

value_map_t
configuration_parser::parse(const std::vector<std::string>& arguments)
{
    m_command.clear();
    m_unparsed.clear();
    m_errors.clear();
    if(m_store == nullptr || m_providers == nullptr)
    {
        m_errors.emplace_back("parser is not initialized");
        return {};
    }

    const auto schema_ptr = m_store->config_schema();
    const auto request    = parse_request{ .config_schema = *schema_ptr,
                                           .arguments     = arguments,
                                           .mode          = m_mode,
                                           .environment   = m_environment,
                                           .presets       = m_presets,
                                           .providers     = *m_providers,
                                           .errors        = m_errors };
    auto       collected  = read_primary(request);
    m_command             = std::move(collected.command_line.command);
    m_unparsed            = std::move(collected.command_line.unparsed);
    read_indirect(request, collected);
    collect_errors(m_errors, collected);

    auto resolved = resolve_collected(collected);
    add_provider_defaults(resolved, request.config_schema, *m_providers);
    apply_effects(resolved, request.config_schema);
    append_violations(m_errors, request.config_schema, resolved);
    try
    {
        m_store->clear();
        apply_resolved(*m_store, resolved);
    } catch(const std::exception& err)
    {
        m_errors.emplace_back(err.what());
    }
    return m_store->values();
}

value_map_t
configuration_parser::parse(int argc, char** argv)
{
    return parse(cli_parser::arguments_from(argc, argv));
}

value_map_t
configuration_parser::export_as_environment() const
{
    value_map_t exported;
    if(m_store == nullptr)
    {
        return exported;
    }
    const auto schema_ptr = m_store->config_schema();
    for(const auto& [param_id, value] : m_store->values())
    {
        const auto* param = schema_ptr->find_param(param_id);
        if(param != nullptr && param->env)
        {
            exported.emplace(param->env->name, value);
        }
    }
    return exported;
}

std::string
configuration_parser::export_as_json() const
{
    auto document = json::object();
    if(m_store != nullptr)
    {
        for(const auto& [param_id, value] : m_store->values())
        {
            document[param_id] = value;
        }
    }
    return document.dump(k_json_indent);
}

void
configuration_parser::apply_to_environment() const
{
    for(const auto& [name, value] : export_as_environment())
    {
        if(::setenv(name.c_str(), value.c_str(), k_overwrite) != 0)
        {
            throw std::runtime_error(
                fmt::format("failed to set environment variable '{}'", name));
        }
    }
}

std::string
configuration_parser::render_help(help_style style, std::string_view topic) const
{
    if(m_store == nullptr)
    {
        return "parser is not initialized\n";
    }
    return format_help(*m_store->config_schema(), style, topic);
}

std::shared_ptr<const schema>
configuration_parser::config_schema() const
{
    if(m_store == nullptr)
    {
        return {};
    }
    return m_store->config_schema();
}

}  // namespace rocprofsys::parser
