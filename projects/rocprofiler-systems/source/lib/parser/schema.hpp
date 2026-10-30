// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
enum class param_type : std::uint8_t
{
    boolean = 0,
    integer,
    number,
    string,
    path,
    enumeration,
    enumeration_list,
};

enum class param_level : std::uint8_t
{
    basic = 0,
    advanced,
    expert,
};

enum class path_kind : std::uint8_t
{
    any = 0,
    file,
    directory,
};

enum class arg_form : std::uint8_t
{
    value = 0,
    flag,
};

enum class assign_style : std::uint8_t
{
    separate = 0,
    equals,
};

enum class severity : std::uint8_t
{
    info = 0,
    warning,
    error,
};

enum class constraint_kind : std::uint8_t
{
    mutually_exclusive = 0,
    forces,
    warn,
};

enum class condition_mode : std::uint8_t
{
    all = 0,
    any,
};

using value_map_t = std::map<std::string, std::string, std::less<>>;

struct category
{
    std::string id;
    std::string name;
};

struct choice
{
    std::string value;
    std::string name;
    std::string description;
};

struct env_spec
{
    std::string name;
    std::string separator = ",";
};

struct arg_spec
{
    std::string  name;
    arg_form     form  = arg_form::value;
    assign_style style = assign_style::separate;
};

/// Holds when parameter @c param resolves to @c equals.
struct param_condition
{
    std::string param;
    std::string equals;
};

struct condition
{
    condition_mode               mode = condition_mode::all;
    std::vector<param_condition> terms;
};

struct schema_param
{
    std::string                    id;
    std::string                    name;
    std::string                    description;
    std::string                    category;
    param_level                    level = param_level::basic;
    param_type                     type  = param_type::string;
    std::optional<std::string>     default_value;
    std::optional<double>          min;
    std::optional<double>          max;
    std::string                    units;
    path_kind                      path = path_kind::any;
    std::vector<choice>            choices;
    bool                           allow_custom = false;
    std::optional<std::string>     pattern;
    bool                           required = false;
    std::optional<env_spec>        env;
    std::optional<arg_spec>        arg;
    std::optional<param_condition> depends_on;
    std::vector<std::string>       mutually_exclusive_with;
    std::vector<std::string>       prerequisites;
};

/// Separator between elements of an enumeration_list value; "," unless the
/// parameter's environment spec overrides it.
[[nodiscard]] std::string_view
list_separator(const schema_param& param) noexcept;

struct constraint
{
    constraint_kind          kind  = constraint_kind::mutually_exclusive;
    severity                 level = severity::error;
    std::string              message;
    std::vector<std::string> params;
    condition                when;
    value_map_t              forced_values;
};

/**
 * Parameters, categories, and constraints for one catalog.
 *
 * Categories and parameters keep insertion order. Parameters are additionally
 * indexed by id, environment variable, and CLI argument name.
 */
class schema
{
public:
    void set_version(std::string version);

    /// @return false, leaving the schema unchanged, if the id is already registered.
    [[nodiscard]] bool add_category(category value);

    /**
     * @return false, leaving the schema unchanged, if the id, environment variable, or
     *         CLI argument name is already registered.
     */
    [[nodiscard]] bool add_param(schema_param param);

    void add_constraint(constraint value);

    [[nodiscard]] const std::string& version() const noexcept { return m_version; }
    [[nodiscard]] const std::vector<category>& categories() const noexcept
    {
        return m_categories;
    }
    [[nodiscard]] const std::vector<schema_param>& params() const noexcept
    {
        return m_params;
    }
    [[nodiscard]] const std::vector<constraint>& constraints() const noexcept
    {
        return m_constraints;
    }

    [[nodiscard]] const category*     find_category(std::string_view category_id) const;
    [[nodiscard]] const schema_param* find_param(std::string_view param_id) const;
    [[nodiscard]] const schema_param* find_param_by_env(std::string_view name) const;
    [[nodiscard]] const schema_param* find_param_by_arg(std::string_view name) const;

    [[nodiscard]] std::vector<const schema_param*> params_in_category(
        std::string_view category_id) const;

private:
    using index_map_t = std::map<std::string, std::size_t, std::less<>>;

    [[nodiscard]] const schema_param* find_indexed(const index_map_t& index,
                                                   std::string_view   key) const;

    std::string               m_version;
    std::vector<category>     m_categories;
    std::vector<schema_param> m_params;
    std::vector<constraint>   m_constraints;
    index_map_t               m_category_index;
    index_map_t               m_param_index;
    index_map_t               m_env_index;
    index_map_t               m_arg_index;
};

}  // namespace rocprofsys::parser
