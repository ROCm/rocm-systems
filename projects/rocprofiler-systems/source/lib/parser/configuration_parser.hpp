// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/backend.hpp"
#include "parser/config_store.hpp"
#include "parser/env_parser.hpp"
#include "parser/help.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
 * Loads a schema and resolves defaults, a config file, the environment, a preset,
 * and the command line.
 *
 * Precedence, from lowest to highest, is default_value, file, environment, preset,
 * then command_line. After that, assigns runs for a value whose winning source is
 * not the default. A target with a higher source is left alone. The config file
 * path is the resolved "config_file" parameter. The preset name is the resolved
 * "preset" parameter. An empty preset name is unset. An empty lookup for a named
 * preset means that name is unknown. initialize() builds an empty backend registry.
 * A missing registration accepts the value. The parser is single-threaded.
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
     * Install the demo catalog and an empty backend registry.
     * @return false, leaving any previously loaded schema in place, when setup fails.
     */
    [[nodiscard]] bool initialize();

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
    [[nodiscard]] bool add_backend(std::unique_ptr<backend> item);

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
    [[nodiscard]] const std::vector<std::string>& unparsed() const noexcept
    {
        return m_unparsed;
    }
    [[nodiscard]] const std::vector<std::string>& errors() const noexcept
    {
        return m_errors;
    }

    /// Environment name to canonical text. Defaults are omitted.
    [[nodiscard]] value_map_t export_as_environment() const;
    [[nodiscard]] std::string export_as_json() const;

    /// Copy export_as_environment() into the process environment.
    /// @throws std::runtime_error if a variable cannot be set.
    void apply_to_environment() const;

    [[nodiscard]] std::string render_help(help_style       style = help_style::compact,
                                          std::string_view topic = {}) const;

    [[nodiscard]] std::shared_ptr<const schema> config_schema() const;

private:
    std::unique_ptr<config_store>     m_store;
    std::unique_ptr<backend_registry> m_backends;
    parse_mode                        m_mode = parse_mode::full;
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
