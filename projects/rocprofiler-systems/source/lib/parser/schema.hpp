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
};

using value_map_t = std::map<std::string, std::string, std::less<>>;

struct schema_param
{
    /// Unique key, used by get() and as the key in a configuration file.
    std::string id;
    /// One line of help text.
    std::string description;
    /// Help heading, and the configuration-file section that holds the parameter.
    std::string category;
    /// basic rows are shown in help by default; advanced rows only when asked for.
    param_level level = param_level::basic;
    /// How a value is parsed and checked.
    param_type type = param_type::string;
    /// Value when no source sets the parameter.
    std::optional<std::string> default_value;
    /// Inclusive bounds of an integer or number.
    std::optional<double> min;
    std::optional<double> max;
    /// Allowed values of an enumeration or enumeration_list.
    std::vector<std::string> choices;
    /// Accept a value that is not one of choices. For example, the common
    /// ROCPROFSYS_GPU_PERF_COUNTERS counters could be listed, but any counter the
    /// device supports must still be accepted.
    bool allow_custom = false;
    /// A command-line switch that takes no value, such as "--trace". Present means
    /// true, and "--trace=false" sets false.
    bool flag = false;
    /// Environment variable that sets the parameter.
    std::optional<std::string> env;
    /// Command-line spelling, such as "--output-format". Help and errors use it.
    std::optional<std::string> arg;
    /// Other command-line spellings of arg, such as "-f".
    std::vector<std::string> arg_aliases;
    /// How many tokens follow the option. Unset means one, or none for a flag.
    std::optional<std::size_t> arg_min_count;
    std::optional<std::size_t> arg_max_count;
    /// Name of the backend that lists the allowed values at run time.
    std::optional<std::string> backend;
    /// Parameters that cannot be set together with this one.
    std::vector<std::string> conflicts;
    /// Parameters that must also be set when this one is.
    std::vector<std::string> required;
    /// Parameter ids and the values they take when this one is set. For example,
    /// "--gpu" uses it to set use_amd_smi and use_process_sampling to "true".
    value_map_t assigns;
};

/// Separator between elements of an enumeration_list value.
[[nodiscard]] std::string_view
list_separator(const schema_param& param) noexcept;

/// Split a list on @p separator, trimming each element.
/// @return nullopt when @p separator is empty. An empty @p value is an empty list.
[[nodiscard]] std::optional<std::vector<std::string>>
split_list(std::string_view value, std::string_view separator);

/**
 * Parameters for one catalog.
 *
 * Parameters keep insertion order and are indexed by id, environment variable,
 * and every command-line spelling (arg and its aliases).
 */
class schema
{
public:
    void set_version(std::string version);

    /**
     * @return false, leaving the schema unchanged, if the id is empty or already
     *         registered, the environment variable is already registered, a
     *         command-line spelling (arg or an alias) is already registered, repeated,
     *         or empty, aliases are given without arg, or backend is empty or set on a
     *         type other than string, enumeration, or enumeration_list.
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
