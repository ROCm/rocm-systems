// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "options/avail_traces.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <string_view>

namespace rocprofsys::rocsys::options
{
namespace
{
constexpr std::string_view k_name      = "NAME";
constexpr std::string_view k_what      = "WHAT IT TRACES";
constexpr std::string_view k_expands   = "EXPANDS TO";
constexpr std::string_view k_available = "AVAILABLE";
constexpr std::string_view k_unified_note =
    "If ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING is enabled, the library also "
    "traces KFD page-fault and page-migrate domains.";

[[nodiscard]] std::string
available_cell(const trace::catalog_entry& entry)
{
    if(entry.available)
    {
        return "yes";
    }
    return fmt::format("no: {}", entry.reason);
}

}  // namespace

void
write_avail_traces(std::ostream& out, const trace::trace_catalog& catalog)
{
    const auto rows          = catalog.entries();
    auto       name_width    = k_name.size();
    auto       what_width    = k_what.size();
    auto       expands_width = k_expands.size();
    for(const auto& row : rows)
    {
        name_width    = std::max(name_width, row.name.size());
        what_width    = std::max(what_width, row.what.size());
        expands_width = std::max(expands_width, row.expands_to.size());
    }

    out << fmt::format("  {:<{}}  {:<{}}  {:<{}}  {}\n", k_name, name_width, k_what,
                       what_width, k_expands, expands_width, k_available);
    for(const auto& row : rows)
    {
        out << fmt::format("  {:<{}}  {:<{}}  {:<{}}  {}\n", row.name, name_width,
                           row.what, what_width, row.expands_to, expands_width,
                           available_cell(row));
    }

    out << '\n' << k_unified_note << "\n\nSDK domains for this install:\n";
    for(const auto& domain : catalog.sdk_domains())
    {
        out << "  " << domain << '\n';
    }
}

}  // namespace rocprofsys::rocsys::options
