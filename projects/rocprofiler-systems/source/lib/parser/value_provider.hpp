// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

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
 * Supplies the allowed values for a parameter whose schema names a provider.
 *
 * choices() and default_value() describe the catalog. accepts() checks one
 * textual value. A list is split on space, comma, semicolon, colon, tab, and
 * newline, and each name is compared exactly. The built-in catalogs are the
 * ROCm domain, runtime, and GPU metric names the runtime accepts. A
 * hardware-counter name is any non-empty text.
 */
class value_provider_interface
{
public:
    virtual ~value_provider_interface() = default;

    value_provider_interface(const value_provider_interface&)            = delete;
    value_provider_interface& operator=(const value_provider_interface&) = delete;
    value_provider_interface(value_provider_interface&&)                 = delete;
    value_provider_interface& operator=(value_provider_interface&&)      = delete;

    [[nodiscard]] virtual std::string_view name() const = 0;

    [[nodiscard]] virtual std::vector<std::string> choices(
        const schema_param& param) const = 0;

    [[nodiscard]] virtual std::optional<std::string> default_value(
        const schema_param& param) const = 0;

    /// @return true when @p value is allowed for @p param.
    /// Empty text is allowed for a string or an enumeration_list and refused
    /// for an enumeration. A list is split on space, comma, semicolon, colon,
    /// tab, and newline. Consecutive delimiters do not create an empty element.
    /// Names are compared exactly. Without allow_custom, each element must
    /// appear in choices(). With allow_custom, a list only requires a real
    /// element, and one enumeration value may be any non-empty text.
    [[nodiscard]] virtual bool accepts(const schema_param& param,
                                       std::string_view    value) const;

protected:
    value_provider_interface() = default;
};

/// Named providers. add() rejects a missing or duplicate name.
class value_provider_registry
{
public:
    value_provider_registry() = default;

    value_provider_registry(const value_provider_registry&)            = delete;
    value_provider_registry& operator=(const value_provider_registry&) = delete;
    value_provider_registry(value_provider_registry&&)                 = delete;
    value_provider_registry& operator=(value_provider_registry&&)      = delete;

    /// @return false, leaving the registry unchanged, when the name is empty or taken.
    bool add(std::unique_ptr<value_provider_interface> provider);

    /// Install rocm_domains_sdk, parallel_runtimes, gpu_metrics, and hw_counters.
    /// A name that is already registered is left as it is.
    void add_builtins();

    [[nodiscard]] const value_provider_interface* find(std::string_view name) const;

private:
    mutable std::shared_mutex m_mutex;
    std::map<std::string, std::unique_ptr<value_provider_interface>, std::less<>>
        m_providers;
};

}  // namespace rocprofsys::parser
