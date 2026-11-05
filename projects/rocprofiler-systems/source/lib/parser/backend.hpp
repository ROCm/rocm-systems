// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

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
 * Supplies the allowed values for a parameter whose schema names a backend.
 *
 * choices() and default_value() see the parameter, so one registered backend can
 * serve more than one catalog row. accepts() is the shared check: it compares a
 * value with choices() unless allow_custom is set.
 */
class backend
{
public:
    virtual ~backend() = default;

    backend(const backend&)            = delete;
    backend& operator=(const backend&) = delete;
    backend(backend&&)                 = delete;
    backend& operator=(backend&&)      = delete;

    [[nodiscard]] virtual std::string_view name() const = 0;

    [[nodiscard]] virtual std::vector<std::string> choices(
        const schema_param& param) const = 0;

    [[nodiscard]] virtual std::optional<std::string> default_value(
        const schema_param& param) const = 0;

    /**
     * @return true when @p value is allowed for @p param.
     * An enumeration must name one choice. An empty enumeration is refused.
     * An enumeration_list, and a string that contains a delimiter, is split on
     * space, comma, semicolon, colon, tab, and newline. Each element must be a
     * choice. An empty list is allowed. With allow_custom, an enumeration may be
     * any non-empty text and a list only needs one element.
     */
    [[nodiscard]] bool accepts(const schema_param& param, std::string_view value) const;

protected:
    backend() = default;
};

/// Named backends. add() rejects a missing or duplicate name.
class backend_registry
{
public:
    backend_registry() = default;

    backend_registry(const backend_registry&)            = delete;
    backend_registry& operator=(const backend_registry&) = delete;
    backend_registry(backend_registry&&)                 = delete;
    backend_registry& operator=(backend_registry&&)      = delete;

    /// @return false, leaving the registry unchanged, when @p item is null or its
    /// name is empty or already registered.
    bool add(std::unique_ptr<backend> item);

    [[nodiscard]] const backend* find(std::string_view name) const;

private:
    mutable std::shared_mutex                                    m_mutex;
    std::map<std::string, std::unique_ptr<backend>, std::less<>> m_backends;
};

}  // namespace rocprofsys::parser
