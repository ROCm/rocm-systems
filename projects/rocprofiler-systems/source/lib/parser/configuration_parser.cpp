// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/configuration_parser.hpp"
#include "parser/backend.hpp"
#include "parser/cli_parser.hpp"
#include "parser/config_store.hpp"
#include "parser/default_parser.hpp"
#include "parser/env_parser.hpp"
#include "parser/file_parser.hpp"
#include "parser/help.hpp"
#include "parser/parse_result.hpp"
#include "parser/precedence.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <stdlib.h>  // POSIX ::setenv

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
        default:
            throw std::logic_error(
                fmt::format("config_source {} not handled", static_cast<int>(source)));
    }
}

/// The part of a refused @p value to name: the first element of a list that @p backend
/// refuses, as the command line names the element at fault, or else the whole value.
std::string
refused_text(const backend_interface& backend, const schema_param& param,
             const config_value_t& value)
{
    if(const auto* items = std::get_if<std::vector<std::string>>(&value))
    {
        const auto found =
            std::ranges::find_if(*items, [&backend, &param](const std::string& item) {
                return !backend.accepts(param, item);
            });
        if(found != items->end())
        {
            return *found;
        }
    }
    return format_value(value);
}

void
reject_backend_values(const schema& config_schema, const backend_registry& backends,
                      config_source source, parse_result& parsed)
{
    for(auto itr = parsed.values.begin(); itr != parsed.values.end();)
    {
        const auto* param = config_schema.find_param(itr->first);
        const auto* found =
            param != nullptr && param->backend ? backends.find(*param->backend) : nullptr;
        const auto text = format_value(itr->second);
        if(found == nullptr || found->accepts(*param, text))
        {
            ++itr;
            continue;
        }
        parsed.errors.push_back(fmt::format(
            "{} from {}",
            invalid_value_message(refused_text(*found, *param, itr->second), itr->first),
            source_label(source)));
        itr = parsed.values.erase(itr);
    }
}

sourced_values
make_source(config_source source, const parse_result& parsed)
{
    return sourced_values{ .source = source, .values = parsed.values };
}

/// What the command line gave @p param_id, or nothing when it is absent.
std::optional<std::string>
command_line_text(const parse_result& command_line, std::string_view param_id)
{
    const auto found = command_line.values.find(param_id);
    return found == command_line.values.end()
               ? std::nullopt
               : std::optional{ format_value(found->second) };
}

env_parser
make_env(const env_parser::lookup_t& lookup)
{
    return lookup ? env_parser{ lookup } : env_parser::from_process();
}

parse_result
read_config_file(const schema& config_schema, const std::optional<std::string>& path)
{
    return path ? file_parser{}.parse_file(config_schema, *path) : parse_result{};
}

parse_result
read_preset(const schema& config_schema, const std::optional<std::string>& name,
            const configuration_parser::preset_lookup_t& lookup)
{
    parse_result result;
    if(!name || name->empty())
    {
        return result;
    }
    const auto settings = lookup ? lookup(*name) : preset_parser::settings_t{};
    if(settings.empty())
    {
        result.errors.push_back(lookup
                                    ? fmt::format("unknown preset '{}'", *name)
                                    : fmt::format("preset '{}' was not applied", *name));
        return result;
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
    unknown_option                               unknown = unknown_option::report;
    const env_parser::lookup_t&                  environment;
    const configuration_parser::preset_lookup_t& presets;
    const backend_registry&                      backends;
    std::vector<std::string>&                    errors;
};

/// Every source in order: the defaults, the environment, the command line, then the
/// config file and the preset the command line names. Each source's values are checked
/// against the backends as it is read, so a refused value leaves a lower source to win.
collected_input
read_sources(const parse_request& request)
{
    const auto& catalog = request.config_schema;
    const auto  checked = [&request](config_source source, parse_result parsed) {
        reject_backend_values(request.config_schema, request.backends, source, parsed);
        request.errors.insert(request.errors.end(), parsed.errors.begin(),
                               parsed.errors.end());
        return parsed;
    };
    collected_input collected;
    collected.defaults =
        checked(config_source::default_value, default_parser{}.parse(catalog));
    collected.environment =
        checked(config_source::environment, make_env(request.environment).parse(catalog));
    collected.command_line =
        checked(config_source::command_line,
                cli_parser{ request.arguments, request.unknown }.parse(catalog));
    const auto& given = collected.command_line;
    collected.file =
        checked(config_source::file,
                read_config_file(catalog, command_line_text(given, k_config_file_param)));
    collected.preset = checked(
        config_source::preset,
        read_preset(catalog, command_line_text(given, k_preset_param), request.presets));
    return collected;
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

/// Write what @p param assigns, under @p source, to each target whose value comes from a
/// lower source: gpu set in a config file leaves --use-process-sampling=false alone. A
/// value from @p source that the assign contradicts, as in --debug --log-level quiet,
/// and a value the target's backend refuses are reported and not written. initialize()
/// refused a catalog that assigns an unknown parameter or a value its target refuses,
/// but it could not ask a backend registered after it.
void
assign_targets(resolved_config& resolved, const parse_request& request,
               const schema_param& param, config_source source)
{
    for(const auto& [target_id, text] : param.assigns)
    {
        const auto* target = request.config_schema.find_param(target_id);
        const auto parsed = target == nullptr ? std::nullopt : parse_value(*target, text);
        const auto existing = resolved.find(target_id);
        if(!parsed || (existing != resolved.end() && existing->second.source > source))
        {
            continue;
        }
        const bool same_source =
            existing != resolved.end() && existing->second.source == source;
        if(same_source && existing->second.value != *parsed)
        {
            request.errors.push_back(fmt::format(
                "parameter '{}' assigns '{}' to '{}', which the {} sets to '{}'",
                param.id, text, target_id, source_label(source),
                format_value(existing->second.value)));
        }
        if(same_source)
        {
            continue;
        }
        const auto* backend =
            target->backend ? request.backends.find(*target->backend) : nullptr;
        if(backend != nullptr && !backend->accepts(*target, text))
        {
            request.errors.push_back(
                fmt::format("parameter '{}' assigns invalid value '{}' to '{}'", param.id,
                            text, target_id));
            continue;
        }
        resolved.insert_or_assign(target_id,
                                  resolved_value{ .value = *parsed, .source = source });
    }
}

/// Ex, --gpu sets use_amd_smi and use_process_sampling to true through its assigns.
void
apply_assigns(resolved_config& resolved, const parse_request& request)
{
    std::vector<std::pair<const schema_param*, config_source>> origins;
    for(const auto& param : request.config_schema.params())
    {
        const auto found = resolved.find(param.id);
        if(found != resolved.end() &&
           found->second.source != config_source::default_value &&
           is_enabled(found->second.value))
        {
            origins.emplace_back(&param, found->second.source);
        }
    }
    for(const auto& [param, source] : origins)
    {
        assign_targets(resolved, request, *param, source);
    }
}

/// Report conflicts and required, such as --profile with --flat-profile, for what the
/// sources set.
void
append_violations(std::vector<std::string>& errors, const schema& config_schema,
                  const resolved_config& resolved)
{
    typed_values_t values;
    for(const auto& [param_id, item] : resolved)
    {
        if(item.source != config_source::default_value)
        {
            values.emplace(param_id, item.value);
        }
    }
    std::ranges::transform(validate_values(config_schema, values),
                           std::back_inserter(errors), &violation::message);
}

std::string
catalog_errors(const schema& catalog)
{
    const auto problems = validate_schema(catalog);
    return fmt::format(
        "{}", fmt::join(problems | std::views::transform(&violation::message), "; "));
}

/// Copies argv[1..argc). argv[0], the program name, is not an option.
std::vector<std::string>
arguments_from(int argc, char** argv)
{
    std::vector<std::string> args;
    if(argv == nullptr || argc < 2)
    {
        return args;
    }
    args.reserve(static_cast<std::size_t>(argc - 1));
    for(int index = 1; index < argc; ++index)
    {
        if(argv[index] != nullptr)
        {
            args.emplace_back(argv[index]);
        }
    }
    return args;
}
}  // namespace

void
configuration_parser::set_unknown_option(unknown_option policy)
{
    m_unknown = policy;
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
configuration_parser::add_backend(std::unique_ptr<backend_interface> item)
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
                                           .unknown       = m_unknown,
                                           .environment   = m_environment,
                                           .presets       = m_presets,
                                           .backends      = *m_backends,
                                           .errors        = m_errors };
    auto       collected  = read_sources(request);
    m_command             = std::move(collected.command_line.command);
    m_unparsed            = std::move(collected.command_line.unparsed);

    auto resolved = resolve_collected(collected);
    apply_assigns(resolved, request);
    append_violations(m_errors, request.config_schema, resolved);
    m_store->clear();
    apply_resolved(*m_store, resolved);
    return m_store->values();
}

value_map_t
configuration_parser::parse(int argc, char** argv)
{
    return parse(arguments_from(argc, argv));
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
    for(const auto& slot : m_store->slots())
    {
        const auto* param = schema_ptr->find_param(slot.id);
        if(param != nullptr && param->env && slot.source != config_source::default_value)
        {
            exported.emplace(*param->env, format_value(slot.value));
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
configuration_parser::render_help(param_level max_level, std::string_view topic) const
{
    if(m_store == nullptr)
    {
        return "parser is not initialized\n";
    }
    return format_help(*m_store->config_schema(), max_level, topic, m_backends.get());
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
