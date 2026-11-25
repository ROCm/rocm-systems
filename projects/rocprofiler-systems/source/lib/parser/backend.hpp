// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::parser
{
/**
 * Supplies the allowed values for a parameter whose schema names a backend.
 *
 * choices() sees the parameter, so one registered backend can serve more than one
 * parameter. A parameter names one backend, and a backend can combine several sources,
 * such as rocprofiler-sdk domains and the runtimes for --traces. accepts() is the
 * shared check.
 */
class backend_interface
{
public:
    virtual ~backend_interface() = default;

    backend_interface(const backend_interface&)            = delete;
    backend_interface& operator=(const backend_interface&) = delete;
    backend_interface(backend_interface&&)                 = delete;
    backend_interface& operator=(backend_interface&&)      = delete;

    /**
     * Registry key. add() copies it and refuses an empty name or one already
     * registered. The view remains valid for this backend's lifetime.
     */
    [[nodiscard]] virtual std::string_view name() const = 0;

    /**
     * Allowed value names for @p param. A backend may list different names for each
     * parameter that names it. accepts() checks a value against these names and
     * @p param's own choices.
     */
    [[nodiscard]] virtual std::vector<std::string> choices(
        const schema_param& param) const = 0;

    /**
     * @return true when @p value is allowed for @p param: the text is read as
     * parse_value() reads it, against @p param's own choices and this backend's. A
     * name neither lists is refused, and a list follows the rules of any other
     * enumeration_list, so "temp,,power" and "temp;power" are refused, and an empty
     * list is allowed.
     *
     * @p param must name this backend; it is then an enumeration or enumeration_list,
     * as schema::add_param() requires of a row that names a backend.
     * @throws std::logic_error if @p param names another backend, or none.
     */
    [[nodiscard]] bool accepts(const schema_param& param, std::string_view value) const;

protected:
    backend_interface() = default;
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
    [[nodiscard]] bool add(std::unique_ptr<backend_interface> item);

    /**
     * @return the backend registered under @p name, or nullptr when none is. The
     * pointer remains valid for this registry's lifetime. Lookups may run concurrently
     * with each other and with add().
     */
    [[nodiscard]] const backend_interface* find(std::string_view name) const;

private:
    mutable std::shared_mutex                                              m_mutex;
    std::map<std::string, std::unique_ptr<backend_interface>, std::less<>> m_backends;
};

}  // namespace rocprofsys::parser
