// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "trace/catalog.hpp"

#include "common/runtime_table.hpp"
#include "options/normalize.hpp"
#include "options/suggest.hpp"
#include "trace/aliases.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <utility>

namespace rocprofsys::rocsys::trace
{
namespace
{
constexpr std::string_view k_missing_from_sdk =
    "not provided by the loaded rocprofiler-sdk";
constexpr std::string_view k_missing_from_headers =
    "not supported by the rocprofiler-sdk headers in this build";
constexpr std::string_view k_not_compiled = "not compiled in this build";
constexpr std::string_view k_kokkos       = "kokkos";
constexpr std::string_view k_sdk_what     = "SDK domain";

[[nodiscard]] bool
listed(std::span<const std::string> names, std::string_view name)
{
    return std::ranges::any_of(names, [name](const std::string& item) {
        return options::same_choice(item, name);
    });
}

[[nodiscard]] std::string
domain_reason(std::span<const std::string> choices,
              std::span<const std::string> supported, std::string_view domain)
{
    if(!listed(choices, domain))
    {
        return std::string{ k_missing_from_sdk };
    }
    if(!listed(supported, domain))
    {
        return std::string{ k_missing_from_headers };
    }
    return {};
}

[[nodiscard]] const runtime_table::row*
matching_runtime(std::string_view token)
{
    for(const auto& spec : runtime_table::rows())
    {
        if(spec.name == k_kokkos)
        {
            continue;
        }
        if(options::same_choice(spec.name, token))
        {
            return &spec;
        }
        for(const auto alias : spec.aliases)
        {
            if(options::same_choice(alias, token))
            {
                return &spec;
            }
        }
    }
    return nullptr;
}

[[nodiscard]] const runtime_table::row*
phase2_runtime(std::string_view token)
{
    for(const auto& item : runtime_table::trace_overrides())
    {
        if(options::same_choice(item.name, token))
        {
            return matching_runtime(item.canonical);
        }
    }
    return matching_runtime(token);
}

[[nodiscard]] std::string
friendly_reason(std::span<const std::string> choices,
                std::span<const std::string> supported, const friendly_name& alias)
{
    for(const auto domain : alias.domains)
    {
        if(domain.empty())
        {
            continue;
        }
        if(auto reason = domain_reason(choices, supported, domain); !reason.empty())
        {
            return reason;
        }
    }
    return {};
}

[[nodiscard]] std::string
friendly_expands_to(const friendly_name& alias)
{
    auto text = std::string{};
    for(const auto domain : alias.domains)
    {
        if(domain.empty())
        {
            continue;
        }
        if(!text.empty())
        {
            text.append(", ");
        }
        text.append(domain);
    }
    return text;
}

[[nodiscard]] const friendly_name*
friendly(std::string_view token)
{
    for(const auto& item : k_friendly_names)
    {
        if(options::same_choice(item.name, token))
        {
            return &item;
        }
    }
    return nullptr;
}

[[nodiscard]] bool
claimed_by_alias(std::string_view domain)
{
    if(friendly(domain) != nullptr || phase2_runtime(domain) != nullptr)
    {
        return true;
    }
    for(const auto& item : runtime_table::trace_overrides())
    {
        if(options::same_choice(item.name, domain))
        {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::string
display_name(const runtime_table::row& spec)
{
    if(spec.aliases.empty())
    {
        return std::string{ spec.name };
    }
    auto text = std::string{ spec.name };
    text.append(" (");
    for(std::size_t index = 0; index < spec.aliases.size(); ++index)
    {
        if(index != 0)
        {
            text.append(", ");
        }
        text.append(spec.aliases[index]);
    }
    text.push_back(')');
    return text;
}

[[nodiscard]] std::string
runtime_expands_to(const runtime_table::row& spec)
{
    if(spec.rocm_domain.empty())
    {
        return std::string{ spec.env_key };
    }
    return fmt::format("{}, {}", spec.env_key, spec.rocm_domain);
}

[[nodiscard]] std::string
matched_choice(std::span<const std::string> choices, std::string_view name)
{
    for(const auto& choice : choices)
    {
        if(options::same_choice(choice, name))
        {
            return choice;
        }
    }
    return std::string{ name };
}

void
add_domain(std::vector<std::string>& domains, std::string domain)
{
    if(std::ranges::find(domains, domain) == domains.end())
    {
        domains.push_back(std::move(domain));
    }
}

void
enable_runtime(std::vector<runtime_choice>& runtimes, std::string_view env_key)
{
    for(auto& choice : runtimes)
    {
        if(choice.env_key == env_key)
        {
            choice.enabled = true;
        }
    }
}

[[nodiscard]] std::vector<runtime_choice>
runtime_slots()
{
    std::vector<runtime_choice> slots;
    for(const auto& spec : runtime_table::rows())
    {
        if(spec.name == k_kokkos)
        {
            continue;
        }
        slots.push_back(runtime_choice{ .env_key = spec.env_key, .enabled = false });
    }
    return slots;
}

[[nodiscard]] std::string_view
runtime_what(const runtime_table::row& spec)
{
    if(spec.name == "mpi")
    {
        return "MPI functions";
    }
    if(spec.name == "openmp")
    {
        return "OpenMP tools";
    }
    if(spec.name == "rccl")
    {
        return "RCCL collectives";
    }
    if(spec.name == "ucx")
    {
        return "UCX functions";
    }
    if(spec.name == "openshmem")
    {
        return "OpenSHMEM functions";
    }
    return spec.name;
}

[[nodiscard]] std::string
runtime_reason(const options::domain_query& query, std::span<const std::string> choices,
               std::span<const std::string> supported, const runtime_table::row& spec)
{
    const bool compiled =
        query.runtime_compiled != nullptr && query.runtime_compiled(spec.name);
    if(!compiled)
    {
        return std::string{ k_not_compiled };
    }
    if(spec.rocm_domain.empty())
    {
        return {};
    }
    return domain_reason(choices, supported, spec.rocm_domain);
}

}  // namespace

trace_catalog::trace_catalog(options::domain_query query)
: m_query(query)
{}

void
trace_catalog::load() const
{
    if(m_loaded)
    {
        return;
    }
    if(m_query.choices != nullptr)
    {
        m_choices = m_query.choices();
    }
    if(m_query.supported != nullptr)
    {
        m_supported = m_query.supported();
    }
    std::ranges::sort(m_choices);
    std::ranges::sort(m_supported);
    m_loaded = true;
}

result<trace_selection>
trace_catalog::expand(std::span<const std::string> names) const
{
    load();
    auto selection = trace_selection{ .runtimes = runtime_slots() };
    for(const auto& token : names)
    {
        if(const auto* runtime = phase2_runtime(token))
        {
            const auto reason = runtime_reason(m_query, m_choices, m_supported, *runtime);
            if(!reason.empty())
            {
                return diagnostic{ .error = fmt::format(
                                       "trace name '{}' is not available", token),
                                   .hint = reason };
            }
            enable_runtime(selection.runtimes, runtime->env_key);
            continue;
        }
        if(const auto* alias = friendly(token))
        {
            const auto reason = friendly_reason(m_choices, m_supported, *alias);
            if(!reason.empty())
            {
                return diagnostic{ .error = fmt::format(
                                       "trace name '{}' is not available", token),
                                   .hint = reason };
            }
            for(const auto domain : alias->domains)
            {
                if(!domain.empty())
                {
                    add_domain(selection.rocm_domains, matched_choice(m_choices, domain));
                }
            }
            continue;
        }
        if(listed(m_choices, token))
        {
            const auto domain = matched_choice(m_choices, token);
            const auto reason = domain_reason(m_choices, m_supported, domain);
            if(!reason.empty())
            {
                return diagnostic{ .error = fmt::format(
                                       "trace name '{}' is not available", token),
                                   .hint = reason };
            }
            add_domain(selection.rocm_domains, domain);
            continue;
        }
        auto hint = std::string{ "see 'rocsys avail --traces'" };
        if(const auto guess = suggest(token))
        {
            hint = fmt::format("did you mean '{}'? {}", *guess, hint);
        }
        return diagnostic{ .error = fmt::format("unknown trace name '{}'", token),
                           .hint  = std::move(hint) };
    }
    std::ranges::sort(selection.rocm_domains);
    return selection;
}

bool
trace_catalog::recognizes(std::string_view name) const
{
    load();
    return phase2_runtime(name) != nullptr || friendly(name) != nullptr ||
           listed(m_choices, name);
}

std::vector<std::string>
trace_catalog::names() const
{
    load();
    std::vector<std::string> choices;
    for(const auto& alias : k_friendly_names)
    {
        choices.emplace_back(alias.name);
    }
    for(const auto& spec : runtime_table::rows())
    {
        if(spec.name != k_kokkos)
        {
            choices.emplace_back(spec.name);
        }
    }
    for(const auto& choice : m_choices)
    {
        if(!claimed_by_alias(choice))
        {
            choices.push_back(choice);
        }
    }
    return choices;
}

std::vector<catalog_entry>
trace_catalog::entries() const
{
    load();
    std::vector<catalog_entry> rows;
    for(const auto& alias : k_friendly_names)
    {
        const auto reason = friendly_reason(m_choices, m_supported, alias);
        rows.push_back(catalog_entry{
            .name       = std::string{ alias.name },
            .what       = std::string{ alias.what },
            .expands_to = friendly_expands_to(alias),
            .available  = reason.empty(),
            .reason     = reason,
        });
    }
    for(const auto& spec : runtime_table::rows())
    {
        if(spec.name == k_kokkos)
        {
            continue;
        }
        const auto reason = runtime_reason(m_query, m_choices, m_supported, spec);
        rows.push_back(catalog_entry{
            .name       = display_name(spec),
            .what       = std::string{ runtime_what(spec) },
            .expands_to = runtime_expands_to(spec),
            .available  = reason.empty(),
            .reason     = reason,
        });
    }
    for(const auto& choice : m_choices)
    {
        if(claimed_by_alias(choice))
        {
            continue;
        }
        const auto reason = domain_reason(m_choices, m_supported, choice);
        rows.push_back(catalog_entry{
            .name       = choice,
            .what       = std::string{ k_sdk_what },
            .expands_to = choice,
            .available  = reason.empty(),
            .reason     = reason,
        });
    }
    return rows;
}

std::optional<std::string>
trace_catalog::suggest(std::string_view name) const
{
    return options::suggest_choice(name, names());
}

std::span<const std::string>
trace_catalog::sdk_domains() const
{
    load();
    return m_choices;
}

}  // namespace rocprofsys::rocsys::trace
