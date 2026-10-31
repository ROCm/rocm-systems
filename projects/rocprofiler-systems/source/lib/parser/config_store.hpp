// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
/**
 * Where a stored value came from.
 *
 * Listed from lowest precedence to highest: command_line overrides environment,
 * which overrides file, preset, and default_value. This store records the source
 * but does not apply precedence; a later resolver decides whether to replace a value.
 */
enum class config_source : std::uint8_t
{
    default_value = 0,
    preset,
    file,
    environment,
    command_line,
};

/**
 * Values for the parameters of one schema.
 *
 * Construct a store with its schema. instance() is the process-wide store used in
 * place of tim::settings; call reset() before using it. set() accepts only parameter
 * ids and textual values the schema allows, and records which source supplied the
 * value. get_as<T>() converts a stored value when T matches the parameter type:
 * bool, int64_t, double, string, or vector<string> for an enumeration_list.
 */
class config_store
{
public:
    using change_callback_t = std::function<void(std::string_view, std::string_view)>;

    explicit config_store(schema config_schema);

    /// Process-wide store. Starts empty; call reset() with the tool schema.
    [[nodiscard]] static config_store& instance();

    /// Replace the schema and drop stored values and callbacks.
    void reset(schema config_schema);

    /**
     * Store a textual value for a known parameter.
     * @throws std::runtime_error if the id is unknown or the value is not valid.
     * Change callbacks run after the value is stored, and only when the text changes.
     * A callback may read the store.
     */
    void set(std::string_view param_id, std::string_view value, config_source source);

    [[nodiscard]] std::optional<std::string>   get(std::string_view param_id) const;
    [[nodiscard]] bool                         contains(std::string_view param_id) const;
    [[nodiscard]] std::optional<config_source> source_of(std::string_view param_id) const;

    template <typename ValueT>
    [[nodiscard]] std::optional<ValueT> get_as(std::string_view param_id) const;

    /// Stored parameter ids in that category, in schema order.
    [[nodiscard]] std::vector<std::string> ids_in_category(
        std::string_view category_id) const;

    [[nodiscard]] value_map_t values() const;

    /// @throws std::runtime_error if the id is unknown.
    void on_change(std::string_view param_id, change_callback_t callback);

    /// Drop stored values. The schema and callbacks stay.
    void clear();

    [[nodiscard]] std::shared_ptr<const schema> config_schema() const;

private:
    config_store(const config_store&)            = delete;
    config_store& operator=(const config_store&) = delete;

    struct stored_value
    {
        std::string   value;
        config_source source = config_source::default_value;
    };

    struct stored_text
    {
        param_type  type = param_type::string;
        std::string text;
        std::string separator;
    };

    [[nodiscard]] std::shared_ptr<const schema> locked_schema() const;
    [[nodiscard]] std::optional<stored_text>    lookup(std::string_view param_id) const;

    std::shared_ptr<const schema>                                      m_schema;
    mutable std::shared_mutex                                          m_mutex;
    std::map<std::string, stored_value, std::less<>>                   m_values;
    std::map<std::string, std::vector<change_callback_t>, std::less<>> m_callbacks;
};

}  // namespace rocprofsys::parser
