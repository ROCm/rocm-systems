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
#include "parser/value_text.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
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

std::string_view
source_label(config_source source)
{
    switch(source)
    {
        case config_source::default_value: return "default";
        case config_source::file: return "file";
        case config_source::environment: return "environment";
        case config_source::preset: return "preset";
        case config_source::command_line: return "command line";
    }
    return "value";
}

void
reject_backend_values(const schema& config_schema, const backend_registry& backends,
                      config_source source, parse_result& parsed)
{
    for(auto itr = parsed.values.begin(); itr != parsed.values.end();)
    {
        const auto*    param = config_schema.find_param(itr->first);
        const backend* found = nullptr;
        if(param != nullptr && param->backend && !param->backend->empty())
        {
            found = backends.find(*param->backend);
        }
        const auto text = format_value(itr->second);
        if(found == nullptr || found->accepts(*param, text))
        {
            ++itr;
            continue;
        }
        parsed.errors.push_back(fmt::format(
            "{} from {}", invalid_value_message(text, itr->first), source_label(source)));
        itr = parsed.values.erase(itr);
    }
}

sourced_values
make_source(config_source source, const parse_result& parsed)
{
    return sourced_values{ .source = source, .values = parsed.values };
}

std::optional<std::string>
selected_text(std::vector<sourced_values> sources, std::string_view param_id)
{
    const auto resolved = resolve_precedence(std::move(sources));
    const auto itr      = resolved.find(param_id);
    if(itr == resolved.end())
    {
        return std::nullopt;
    }
    const auto text = format_value(itr->second.value);
    if(text.empty())
    {
        return std::nullopt;
    }
    return text;
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
    if(!name || name->empty())
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
    const backend_registry&                      backends;
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
    reject_backend_values(request.config_schema, request.backends,
                          config_source::default_value, collected.defaults);
    reject_backend_values(request.config_schema, request.backends,
                          config_source::environment, collected.environment);
    reject_backend_values(request.config_schema, request.backends,
                          config_source::command_line, collected.command_line);
    return collected;
}

void
read_indirect(const parse_request& request, collected_input& collected)
{
    const auto file_path = selected_text(
        { make_source(config_source::default_value, collected.defaults),
          make_source(config_source::environment, collected.environment),
          make_source(config_source::command_line, collected.command_line) },
        k_config_file_param);
    collected.file = read_config_file(request.config_schema, file_path);
    reject_backend_values(request.config_schema, request.backends, config_source::file,
                          collected.file);

    const auto preset_name = selected_text(
        { make_source(config_source::default_value, collected.defaults),
          make_source(config_source::file, collected.file),
          make_source(config_source::environment, collected.environment),
          make_source(config_source::command_line, collected.command_line) },
        k_preset_param);
    collected.preset =
        read_preset(request.config_schema, preset_name, request.presets, request.errors);
    reject_backend_values(request.config_schema, request.backends, config_source::preset,
                          collected.preset);
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
        make_source(config_source::file, collected.file),
        make_source(config_source::environment, collected.environment),
        make_source(config_source::preset, collected.preset),
        make_source(config_source::command_line, collected.command_line),
    });
}

void
add_backend_defaults(resolved_config& resolved, const schema& config_schema,
                     const backend_registry& backends)
{
    for(const auto& param : config_schema.params())
    {
        const auto already_set =
            param.default_value.has_value() || resolved.find(param.id) != resolved.end();
        if(!param.backend || param.backend->empty() || already_set)
        {
            continue;
        }
        const auto* found = backends.find(*param.backend);
        if(found == nullptr)
        {
            continue;
        }
        const auto fallback = found->default_value(param).value_or("");
        if(fallback.empty() || !found->accepts(param, fallback))
        {
            continue;
        }
        const auto parsed = parse_value(param, canonical_choice(param, fallback));
        if(!parsed)
        {
            continue;
        }
        resolved.emplace(
            param.id,
            resolved_value{ .value = *parsed, .source = config_source::default_value });
    }
}

bool
keeps_existing(const resolved_config& resolved, std::string_view target,
               config_source source)
{
    const auto found = resolved.find(target);
    if(found == resolved.end())
    {
        return false;
    }
    return static_cast<std::uint8_t>(found->second.source) >
           static_cast<std::uint8_t>(source);
}

/// Assigns run from values that won precedence, not from defaults. A target with a
/// higher source is left alone. A constant that does not parse fails the whole update.
bool
apply_assigns(resolved_config& resolved, const schema& config_schema,
              std::vector<std::string>& errors)
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

    auto failed = false;
    for(const auto& [param, origin] : origins)
    {
        for(const auto& [target_id, text] : param->assigns)
        {
            const auto* target = config_schema.find_param(target_id);
            if(target == nullptr)
            {
                errors.push_back(
                    fmt::format("parameter '{}' assigns unknown parameter '{}'",
                                param->id, target_id));
                failed = true;
                continue;
            }
            if(keeps_existing(resolved, target_id, origin.source))
            {
                continue;
            }
            const auto parsed = parse_value(*target, canonical_choice(*target, text));
            if(!parsed)
            {
                errors.push_back(invalid_value_message(text, target_id));
                failed = true;
                continue;
            }
            resolved.insert_or_assign(
                target_id, resolved_value{ .value = *parsed, .source = origin.source });
        }
    }
    return !failed;
}

void
append_violations(std::vector<std::string>& errors, const schema& config_schema,
                  const resolved_config& resolved)
{
    typed_values_t values;
    for(const auto& [param_id, item] : resolved)
    {
        if(item.source == config_source::default_value)
        {
            continue;
        }
        values.emplace(param_id, item.value);
    }
    for(const auto& item : validate_values(config_schema, values))
    {
        errors.push_back(item.message);
    }
}

std::string
catalog_errors(const schema& catalog)
{
    std::string message;
    for(const auto& item : validate_schema(catalog))
    {
        if(!message.empty())
        {
            message.append("; ");
        }
        message.append(item.message);
    }
    return message;
}

void
install_resolved(config_store& live, const schema& config_schema,
                 const resolved_config& resolved)
{
    config_store staged{ config_schema };
    apply_resolved(staged, resolved);
    const auto slots = staged.slots();
    live.clear();
    for(const auto& slot : slots)
    {
        live.set_value(slot.id, slot.value, slot.source);
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
    const auto errors = catalog_errors(catalog);
    if(!errors.empty())
    {
        m_failure = errors;
        return false;
    }
    try
    {
        auto store    = std::make_unique<config_store>(std::move(catalog));
        auto backends = std::make_unique<backend_registry>();
        m_store       = std::move(store);
        m_backends    = std::move(backends);
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

bool
configuration_parser::add_backend(std::unique_ptr<backend> item)
{
    if(m_backends == nullptr)
    {
        m_failure = "parser is not initialized";
        return false;
    }
    if(!m_backends->add(std::move(item)))
    {
        m_failure = "backend was not registered";
        return false;
    }
    return true;
}

value_map_t
configuration_parser::parse(const std::vector<std::string>& arguments)
{
    m_command.clear();
    m_unparsed.clear();
    m_errors.clear();
    if(m_store == nullptr || m_backends == nullptr)
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
                                           .backends      = *m_backends,
                                           .errors        = m_errors };
    auto       collected  = read_primary(request);
    m_command             = std::move(collected.command_line.command);
    m_unparsed            = std::move(collected.command_line.unparsed);
    read_indirect(request, collected);
    collect_errors(m_errors, collected);

    auto resolved = resolve_collected(collected);
    add_backend_defaults(resolved, request.config_schema, *m_backends);
    if(!apply_assigns(resolved, request.config_schema, m_errors))
    {
        return {};
    }
    append_violations(m_errors, request.config_schema, resolved);
    try
    {
        install_resolved(*m_store, *schema_ptr, resolved);
    } catch(const std::exception& err)
    {
        m_errors.emplace_back(err.what());
        return {};
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
        const auto  origin =
            m_store->source_of(param_id).value_or(config_source::default_value);
        if(param != nullptr && param->env && origin != config_source::default_value)
        {
            exported.emplace(*param->env, value);
        }
    }
    return exported;
}

std::string
configuration_parser::export_as_json() const
{
    if(m_store == nullptr)
    {
        return json::object().dump(k_json_indent);
    }
    const auto      schema_ptr = m_store->config_schema();
    resolved_config resolved;
    for(const auto& slot : m_store->slots())
    {
        resolved.emplace(slot.id,
                         resolved_value{ .value = slot.value, .source = slot.source });
    }
    return catalog_document(*schema_ptr, resolved).dump(k_json_indent);
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
    return format_help(*m_store->config_schema(), style, topic, m_backends.get());
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
