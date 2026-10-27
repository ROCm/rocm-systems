// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "options/traces.hpp"

#include "common/env_vars.hpp"
#include "common/runtime_table.hpp"
#include "common/string_utility.hpp"
#include "trace/catalog.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <string_view>

namespace rocprofsys::rocsys::options
{
namespace
{
constexpr std::string_view k_origin            = "--traces";
constexpr std::string_view k_writes_key        = "writes a --traces setting";
constexpr std::string_view k_one_or_other      = "use one or the other";
constexpr std::string_view k_use_dry_run       = "use --dry-run";
constexpr std::string_view k_run_binary        = "rocprof-sys-run";
constexpr std::string_view k_instrument        = "rocprof-sys-instrument";
constexpr std::string_view k_kokkos            = "kokkos";
constexpr auto             k_trace_env_keys    = std::to_array<std::string_view>({
    env_vars::ROCM_DOMAINS,
    env_vars::USE_MPIP,
    env_vars::USE_OMPT,
    env_vars::USE_RCCLP,
    env_vars::USE_UCX,
    env_vars::USE_OPENSHMEM,
});
constexpr auto             k_include_values    = std::to_array<std::string_view>({
    "all",
    "mpip",
    "ompt",
    "rcclp",
});
constexpr auto             k_domain_spellings  = std::to_array<std::string_view>({
    "--rocm",
    "--rocm-domains",
    "--parallel",
});
constexpr auto             k_include_spellings = std::to_array<std::string_view>({
    "-I",
    "--include",
    "-E",
    "--exclude",
});
constexpr auto             k_use_spellings     = std::to_array<std::string_view>({
    "--use-mpip",
    "--use-rcclp",
    "--use-ucx",
    "--use-openshmem",
});
constexpr auto             k_preset_spellings  = std::to_array<std::string_view>({
    "--preset",
    "--balanced",
    "--profile-only",
    "--detailed",
    "--trace-hpc",
    "--workload-trace",
    "--sys-trace",
    "--runtime-trace",
    "--trace-gpu",
    "--trace-openmp",
    "--profile-mpi",
    "--trace-hw-counters",
});
constexpr auto             k_export_spellings =
    std::to_array<std::string_view>({ "--export-config" });
constexpr auto k_env_spellings = std::to_array<std::string_view>({ "--env" });

[[nodiscard]] std::string_view
trim_token(std::string_view token)
{
    while(!token.empty() && token.front() == ' ')
    {
        token.remove_prefix(1);
    }
    while(!token.empty() && token.back() == ' ')
    {
        token.remove_suffix(1);
    }
    return token;
}

[[nodiscard]] bool
is_include_value(std::string_view token)
{
    const auto piece = trim_token(token);
    return std::ranges::find(k_include_values, piece) != k_include_values.end();
}

[[nodiscard]] bool
include_conflicts(std::string_view value)
{
    while(true)
    {
        const auto comma = value.find(',');
        const auto piece =
            comma == std::string_view::npos ? value : value.substr(0, comma);
        if(is_include_value(piece))
        {
            return true;
        }
        if(comma == std::string_view::npos)
        {
            return false;
        }
        value.remove_prefix(comma + 1);
    }
}

[[nodiscard]] bool
writes_trace_key(std::string_view value)
{
    const auto assignment = trim_token(value);
    for(const auto key : k_trace_env_keys)
    {
        if(assignment == key)
        {
            return true;
        }
        if(assignment.size() > key.size() && assignment.starts_with(key) &&
           assignment[key.size()] == '=')
        {
            return true;
        }
    }
    return false;
}

const legacy_conflict k_conflicts[] = {
    {
        .binary    = k_run_binary,
        .spellings = k_domain_spellings,
        .reason    = k_writes_key,
    },
    {
        .binary        = k_run_binary,
        .spellings     = k_include_spellings,
        .matches_value = include_conflicts,
        .reason        = k_writes_key,
    },
    {
        .binary    = k_run_binary,
        .spellings = k_use_spellings,
        .reason    = k_writes_key,
    },
    {
        .binary    = k_run_binary,
        .spellings = k_preset_spellings,
        .reason    = k_one_or_other,
    },
    {
        .binary    = k_run_binary,
        .spellings = k_export_spellings,
        .reason    = k_use_dry_run,
    },
    {
        .binary        = k_instrument,
        .spellings     = k_env_spellings,
        .matches_value = writes_trace_key,
        .reason        = k_writes_key,
    },
};

[[nodiscard]] bool
accept_trace(std::string_view token, const context& ctx)
{
    return trace::trace_catalog{ ctx.domains }.recognizes(token);
}

[[nodiscard]] std::vector<std::string>
trace_choices(const context& ctx)
{
    return trace::trace_catalog{ ctx.domains }.names();
}

[[nodiscard]] std::string
join_domains(std::span<const std::string> domains)
{
    std::string text;
    for(const auto& domain : domains)
    {
        if(!text.empty())
        {
            text.push_back(',');
        }
        text.append(domain);
    }
    return text;
}

[[nodiscard]] env_op
make_op(std::string_view key, std::string value, bool embed)
{
    return { .key    = std::string{ key },
             .value  = std::move(value),
             .origin = std::string{ k_origin },
             .embed  = embed };
}

void
append_trace_ops(const trace::trace_selection& selection, const verb_spec& verb,
                 env_plan& out)
{
    const bool embed = has_capability(verb.caps, capability::embeds_env_in_binary);
    out.push_back(
        make_op(env_vars::ROCM_DOMAINS, join_domains(selection.rocm_domains), embed));
    for(const auto& runtime : selection.runtimes)
    {
        out.push_back(
            make_op(runtime.env_key, runtime.enabled ? "true" : "false", embed));
    }
}

[[nodiscard]] std::optional<diagnostic>
resolve_traces(std::span<const std::string> values, const verb_spec& verb,
               const context& ctx, env_plan& out)
{
    auto selection = trace::trace_catalog{ ctx.domains }.expand(values);
    if(!selection)
    {
        return selection.error();
    }
    append_trace_ops(selection.value(), verb, out);
    return std::nullopt;
}

[[nodiscard]] bool
runtime_is_on(const runtime_table::row& spec, char* (*get_env)(const char*) )
{
    if(get_env != nullptr)
    {
        const auto key = std::string{ spec.env_key };
        if(const char* value = get_env(key.c_str()))
        {
            return utility::string::to_bool(value, spec.library_default);
        }
    }
    return spec.library_default;
}

[[nodiscard]] std::string_view
note_label(std::string_view name)
{
    if(name == "mpi")
    {
        return "MPI (MPIP)";
    }
    if(name == "openmp")
    {
        return "OpenMP (OMPT)";
    }
    if(name == "rccl")
    {
        return "RCCL";
    }
    if(name == "ucx")
    {
        return "UCX";
    }
    if(name == "openshmem")
    {
        return "OpenSHMEM";
    }
    return name;
}

[[nodiscard]] const env_op*
find_op(const env_plan& ops, std::string_view key)
{
    const auto found = std::ranges::find(ops, key, &env_op::key);
    return found == ops.end() ? nullptr : &*found;
}

[[nodiscard]] std::string
disabled_note(const runtime_table::row& spec)
{
    return fmt::format(
        "{} profiling and tracing disabled because '{}' is not in --traces",
        note_label(spec.name), spec.name);
}

}  // namespace

const option_spec k_traces{
    .name          = "traces",
    .kind          = value_kind::token_list,
    .required_caps = static_cast<capability_set>(capability::honours_rocm_tracing),
    .summary       = "Select APIs to trace. Names that are not listed are turned off",
    .env_keys      = k_trace_env_keys,
    .conflicts     = k_conflicts,
    .min_values    = 1,
    .accepts       = accept_trace,
    .choices       = trace_choices,
    .resolve       = resolve_traces,
};

std::vector<std::string>
disabled_runtime_notes(const env_plan& ops, char* (*get_env)(const char*) )
{
    std::vector<std::string> notes;
    for(const auto& spec : runtime_table::rows())
    {
        if(spec.name == k_kokkos)
        {
            continue;
        }
        const auto* op = find_op(ops, spec.env_key);
        if(op == nullptr || utility::string::to_bool(op->value, false))
        {
            continue;
        }
        if(!runtime_is_on(spec, get_env))
        {
            continue;
        }
        notes.push_back(disabled_note(spec));
    }
    return notes;
}

}  // namespace rocprofsys::rocsys::options
