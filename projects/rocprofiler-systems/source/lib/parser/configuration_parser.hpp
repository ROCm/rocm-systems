// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/config_store.hpp"
#include "parser/env_parser.hpp"
#include "parser/help.hpp"
#include "parser/preset_parser.hpp"
#include "parser/value_provider.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
/// What to do with an argument that is not in the schema.
enum class parse_mode : std::uint8_t
{
    /// Record an error and keep parsing.
    full = 0,
    /// Stop and return that argument plus everything after it as unparsed.
    stop_at_unknown,
};

/**
 * Loads a schema and resolves defaults, a preset, a config file, the environment,
 * and the command line.
 *
 * Precedence, from lowest to highest, is default_value, preset, file, environment,
 * then command_line. The config file path is the resolved "config_file" parameter
 * from the defaults, environment, and command line. The preset name is the resolved
 * "preset" parameter after the file is included. Preset settings are supplied by
 * set_preset_lookup(); this library does not open the preset registry itself.
 * An empty lookup result means the preset name is unknown.
 */
class configuration_parser
{
public:
    using preset_lookup_t = std::function<preset_parser::settings_t(std::string_view)>;

    void set_parse_mode(parse_mode mode);
    /// Replace process-environment lookup. An empty function reads the process.
    void set_environment(env_parser::lookup_t lookup);
    void set_preset_lookup(preset_lookup_t lookup);

    /**
     * Load a schema and install the built-in value providers.
     * An empty path uses the schema embedded at build time.
     * @return false, leaving any previously loaded schema in place, when loading fails.
     */
    [[nodiscard]] bool initialize(std::string_view schema_path = {});

    [[nodiscard]] const std::string& failure() const noexcept { return m_failure; }

    /// Resolve every source. Accepted values are stored and returned by parameter id.
    [[nodiscard]] value_map_t parse(const std::vector<std::string>& arguments);
    [[nodiscard]] value_map_t parse(int argc, char** argv);

    template <typename ValueT>
    [[nodiscard]] std::optional<ValueT> get(std::string_view param_id) const;

    [[nodiscard]] const std::vector<std::string>& command() const noexcept
    {
        return m_command;
    }
    [[nodiscard]] const std::vector<std::string>& unparsed() const noexcept
    {
        return m_unparsed;
    }
    [[nodiscard]] const std::vector<std::string>& errors() const noexcept
    {
        return m_errors;
    }

    /// Environment variable name to textual value, for parameters that declare one.
    [[nodiscard]] value_map_t export_as_environment() const;
    [[nodiscard]] std::string export_as_json() const;

    /// Copy export_as_environment() into the process environment.
    /// @throws std::runtime_error if a variable cannot be set.
    void apply_to_environment() const;

    [[nodiscard]] std::string render_help(help_style       style = help_style::compact,
                                          std::string_view topic = {}) const;

    [[nodiscard]] std::shared_ptr<const schema> config_schema() const;

private:
    std::unique_ptr<config_store>            m_store;
    std::unique_ptr<value_provider_registry> m_providers;
    parse_mode                               m_mode = parse_mode::full;
    env_parser::lookup_t                     m_environment;
    preset_lookup_t                          m_presets;
    std::string                              m_failure;
    std::vector<std::string>                 m_command;
    std::vector<std::string>                 m_unparsed;
    std::vector<std::string>                 m_errors;
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
