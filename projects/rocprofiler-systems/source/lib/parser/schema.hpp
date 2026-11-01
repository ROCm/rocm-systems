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
    /// Reports when the named values do not already hold. Does not write them.
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

/// One row of a parameter's maps_to table.
struct value_mapping
{
    std::string when;
    std::string target;
    std::string value;
};

struct choice
{
    std::string value;
    std::string name;
    std::string description;
    /// Constants written when this choice is selected.
    value_map_t sets;
    /// Extra tokens that select this choice.
    std::vector<std::string> aliases;
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
    /// Tokens this flag accepts. Unset means one value, or none for a flag.
    std::optional<std::size_t> min_count;
    std::optional<std::size_t> max_count;
    /// Parameter id that receives each token, in order.
    std::vector<std::string> binds;
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
    /// Other parameter ids that receive this parameter's value.
    std::vector<std::string> copy_to;
    /// Constants written when this parameter is set.
    value_map_t assigns;
    /// Parameter ids filled with this value only while they still have the default.
    std::vector<std::string> fills;
    /// Rows mapping this parameter's value to another parameter's value.
    std::vector<value_mapping> maps_to;
};

/// Separator between elements of an enumeration_list value; "," unless the
/// parameter's environment spec overrides it.
[[nodiscard]] std::string_view
list_separator(const schema_param& param) noexcept;

/// Split a list on @p separator, trimming each element.
/// @return nullopt when @p separator is empty. An empty @p value is an empty list.
[[nodiscard]] std::optional<std::vector<std::string>>
split_list(std::string_view value, std::string_view separator);

/// Choice alias rewritten as the choice value. A list is rewritten element by
/// element. Text that is not an alias is unchanged.
[[nodiscard]] std::string
canonical_choice(const schema_param& param, std::string_view value);

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

    /// @return false, leaving the schema unchanged, if the id is empty or already
    /// registered.
    [[nodiscard]] bool add_category(category value);

    /**
     * @return false, leaving the schema unchanged, if the id, environment name, or
     *         argument name is empty or already registered.
     */
    [[nodiscard]] bool add_param(schema_param param);

    /**
     * @return false, leaving the schema unchanged, when a referenced parameter is
     *         missing or empty, or when an equals or forced value is not valid for
     *         that parameter.
     */
    [[nodiscard]] bool add_constraint(constraint value);

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
