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

using value_map_t = std::map<std::string, std::string, std::less<>>;

struct choice
{
    std::string              value;
    std::vector<std::string> aliases;
};

struct schema_param
{
    std::string                id;
    std::string                description;
    std::string                category;
    param_level                level = param_level::basic;
    param_type                 type  = param_type::string;
    std::optional<std::string> default_value;
    std::optional<double>      min;
    std::optional<double>      max;
    std::vector<choice>        choices;
    bool                       allow_custom = false;
    bool                       flag         = false;
    std::optional<std::string> env;
    std::optional<std::string> arg;
    std::optional<std::size_t> arg_min_count;
    std::optional<std::size_t> arg_max_count;
    std::optional<std::string> backend;
    std::vector<std::string>   conflicts;
    std::vector<std::string>   required;
    value_map_t                assigns;
};

/// Separator between elements of an enumeration_list value.
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

/**
 * Parameters for one catalog.
 *
 * Parameters keep insertion order and are indexed by id, environment variable,
 * and command-line spelling.
 */
class schema
{
public:
    void set_version(std::string version);

    /**
     * @return false, leaving the schema unchanged, if the id is empty or already
     *         registered, the environment variable or command-line spelling is already
     *         registered, or backend is empty or set on a type other than string,
     *         enumeration, or enumeration_list.
     */
    [[nodiscard]] bool add_param(schema_param param);

    [[nodiscard]] const std::string& version() const noexcept { return m_version; }
    [[nodiscard]] const std::vector<schema_param>& params() const noexcept
    {
        return m_params;
    }

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
    std::vector<schema_param> m_params;
    index_map_t               m_param_index;
    index_map_t               m_env_index;
    index_map_t               m_arg_index;
};

}  // namespace rocprofsys::parser
