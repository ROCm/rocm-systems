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
 * Supplies the allowed values for a parameter whose schema names a provider.
 *
 * choices() and default_value() describe the catalog. accepts() checks one
 * textual value, including a separator-joined list. The built-in catalogs are
 * the known ROCm domain, runtime, GPU metric, and hardware-counter names.
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

    /// @return true when @p value is empty for a string or enumeration_list, or
    ///         each element is one of choices(). allow_custom accepts any non-empty text.
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

    /// Process-wide registry. Starts empty; call add_builtins() to install the catalogs.
    [[nodiscard]] static value_provider_registry& instance();

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
