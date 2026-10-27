// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/context.hpp"
#include "result.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys::trace
{

/// @brief One host runtime written by `--traces`. Kokkos is not in this list.
struct runtime_choice
{
    std::string_view env_key = {};
    bool             enabled = {};
};

/// @brief Resolved `--traces` names. @ref rocm_domains is empty when only host
///        runtimes were selected. Every Phase 2 runtime is present, enabled or not.
struct trace_selection
{
    std::vector<std::string>    rocm_domains = {};
    std::vector<runtime_choice> runtimes     = {};
};

/// @brief One row of `rocsys avail --traces`.
struct catalog_entry
{
    std::string name       = {};
    std::string what       = {};
    std::string expands_to = {};
    bool        available  = {};
    std::string reason     = {};
};

/// @brief Tier-1 SDK domains, tier-2a friendly names, and tier-2b host runtimes.
///        Domain lists are read on the first query.
class trace_catalog
{
public:
    explicit trace_catalog(options::domain_query query);

    /// @brief Expand names into ROCm domains and runtime env keys.
    ///        An unknown or unavailable name is an error.
    [[nodiscard]] result<trace_selection> expand(
        std::span<const std::string> names) const;

    /// @brief Friendly names, then host runtimes, then the remaining SDK domains.
    [[nodiscard]] std::vector<catalog_entry> entries() const;

    /// @brief Canonical name within two edits. A tie or an exact match is empty.
    [[nodiscard]] std::optional<std::string> suggest(std::string_view name) const;

    /// @brief SDK domain names from the injected query, in sorted order.
    [[nodiscard]] std::span<const std::string> sdk_domains() const;

private:
    void load() const;

    options::domain_query            m_query     = {};
    mutable bool                     m_loaded    = {};
    mutable std::vector<std::string> m_choices   = {};
    mutable std::vector<std::string> m_supported = {};
};

}  // namespace rocprofsys::rocsys::trace
