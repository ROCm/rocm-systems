// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/backend.hpp"
#include "parser/cli_parser.hpp"
#include "parser/config_store.hpp"
#include "parser/env_parser.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
/**
 * Loads a schema and resolves defaults, a config file, the environment, a preset,
 * and the command line.
 *
 * Precedence, from lowest to highest, is default_value, file, environment, preset,
 * then command_line. After that, assigns runs for a parameter that is on, as
 * is_enabled() decides, and whose winning source is not the default, so --debug=false
 * assigns nothing. A target with a higher source is left alone. The config file
 * path and the preset name are the "config_file" and "preset" values given on the
 * command line; a config file or a preset holds settings and cannot choose either. An
 * empty preset name is unset. An empty lookup for a named preset means that name is
 * unknown. initialize() builds an empty backend registry. A missing registration
 * accepts the value. The parser is single-threaded.
 */
class configuration_parser
{
public:
    using preset_lookup_t = std::function<preset_parser::settings_t(std::string_view)>;

    /// What the command line does with an option the catalog does not know.
    /// unknown_option::report, the default, records an error. unknown_option::forward
    /// keeps it, and the plain arguments that follow it, in unparsed() for the tool a
    /// wrapper runs. A plain argument right after an option the catalog knows is an
    /// error either way.
    void set_unknown_option(unknown_option policy);
    /// Replace process-environment lookup. An empty function reads the process.
    void set_environment(env_parser::lookup_t lookup);
    void set_preset_lookup(preset_lookup_t lookup);

    /**
     * Install this catalog and an empty backend registry.
     * @return false, leaving any previously loaded schema in place, when setup fails.
     */
    [[nodiscard]] bool initialize(schema catalog);

    /**
     * Register a backend. The parser does not install one itself.
     * @return false, leaving the registry unchanged, when the parser is not
     * initialized or the name is empty or already taken.
     */
    [[nodiscard]] bool add_backend(std::unique_ptr<backend_interface> item);

    [[nodiscard]] const std::string& failure() const noexcept { return m_failure; }

    /// Resolve every source. Accepted values are returned as canonical text.
    [[nodiscard]] value_map_t parse(const std::vector<std::string>& arguments);
    [[nodiscard]] value_map_t parse(int argc, char** argv);

    template <typename ValueT>
    [[nodiscard]] std::optional<ValueT> get(std::string_view param_id) const;

    [[nodiscard]] const std::vector<std::string>& command() const noexcept
    {
        return m_command;
    }
    /// With unknown_option::forward, every unknown option before "--" with the plain
    /// arguments that follow it, and any plain argument that comes first, in order.
    [[nodiscard]] const std::vector<std::string>& unparsed() const noexcept
    {
        return m_unparsed;
    }
    [[nodiscard]] const std::vector<std::string>& errors() const noexcept
    {
        return m_errors;
    }

    /// Environment name to canonical text. Defaults are omitted. The environment parser
    /// reads each value back the same, except a blank one, which it reads as unset.
    [[nodiscard]] value_map_t export_as_environment() const;
    /// The category document catalog_document() writes, which the file parser reads
    /// back the same. Defaults are omitted.
    [[nodiscard]] std::string export_as_json() const;

    /// Copy export_as_environment() into the process environment.
    /// @throws std::runtime_error if a variable cannot be set.
    void apply_to_environment() const;

    /// format_help() for the catalog and the registered backends: @p max_level basic
    /// omits advanced rows, and @p topic names a category or one parameter's spelling.
    [[nodiscard]] std::string render_help(param_level      max_level = param_level::basic,
                                          std::string_view topic     = {}) const;

    [[nodiscard]] std::shared_ptr<const schema> config_schema() const;

private:
    std::unique_ptr<config_store>     m_store;
    std::unique_ptr<backend_registry> m_backends;
    unknown_option                    m_unknown = unknown_option::report;
    env_parser::lookup_t              m_environment;
    preset_lookup_t                   m_presets;
    std::string                       m_failure;
    std::vector<std::string>          m_command;
    std::vector<std::string>          m_unparsed;
    std::vector<std::string>          m_errors;
};

template <typename ValueT>
std::optional<ValueT>
configuration_parser::get(std::string_view param_id) const
{
    if(m_store == nullptr)
    {
        return std::nullopt;
    }
    return m_store->get_as<ValueT>(param_id);
}

}  // namespace rocprofsys::parser
