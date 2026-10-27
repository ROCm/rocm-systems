// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_vars.hpp"

#include <map>
#include <span>
#include <string>
#include <string_view>

namespace rocprofsys::runtime_table
{

/// @brief One host runtime. Legacy `--parallel` uses only rows with
///        @ref row::in_legacy_parallel set. `--traces` will use the whole table.
struct row
{
    std::string_view                  name               = {};
    std::span<const std::string_view> aliases            = {};
    std::string_view                  env_key            = {};
    bool                              library_default    = {};
    bool                              in_legacy_parallel = {};
    /// @brief Token accepted by `-I` and `-E`. Empty when those flags do not
    ///        mention this runtime.
    std::string_view include_token = {};
    /// @brief ROCm domain the library adds when this runtime is enabled. Empty
    ///        when enabling the runtime adds no domain.
    std::string_view rocm_domain = {};
};

/// @brief A raw name that selects a canonical row. `rccl_api` is not a legacy
///        `--parallel` alias; `ompt` is also an alias of `openmp`.
struct name_override
{
    std::string_view name      = {};
    std::string_view canonical = {};
};

inline constexpr std::string_view k_mpi_aliases[]       = { "mpip" };
inline constexpr std::string_view k_openmp_aliases[]    = { "ompt", "omp" };
inline constexpr std::string_view k_rccl_aliases[]      = { "rcclp" };
inline constexpr std::string_view k_kokkos_aliases[]    = { "kokkosp" };
inline constexpr std::string_view k_openshmem_aliases[] = { "shmem" };

inline constexpr row k_rows[] = {
    {
        .name               = "mpi",
        .aliases            = k_mpi_aliases,
        .env_key            = env_vars::USE_MPIP,
        .library_default    = true,
        .in_legacy_parallel = true,
        .include_token      = "mpip",
    },
    {
        .name               = "openmp",
        .aliases            = k_openmp_aliases,
        .env_key            = env_vars::USE_OMPT,
        .library_default    = false,
        .in_legacy_parallel = true,
        .include_token      = "ompt",
        .rocm_domain        = "ompt",
    },
    {
        .name               = "rccl",
        .aliases            = k_rccl_aliases,
        .env_key            = env_vars::USE_RCCLP,
        .library_default    = false,
        .in_legacy_parallel = true,
        .include_token      = "rcclp",
        .rocm_domain        = "rccl_api",
    },
    {
        .name               = "kokkos",
        .aliases            = k_kokkos_aliases,
        .env_key            = env_vars::USE_KOKKOSP,
        .library_default    = false,
        .in_legacy_parallel = true,
        .include_token      = "kokkosp",
    },
    {
        .name            = "ucx",
        .env_key         = env_vars::USE_UCX,
        .library_default = false,
    },
    {
        .name            = "openshmem",
        .aliases         = k_openshmem_aliases,
        .env_key         = env_vars::USE_OPENSHMEM,
        .library_default = false,
    },
};

inline constexpr name_override k_trace_overrides[] = {
    { .name = "ompt", .canonical = "openmp" },
    { .name = "rccl_api", .canonical = "rccl" },
};

[[nodiscard]] inline constexpr std::span<const row>
rows() noexcept
{
    return k_rows;
}

[[nodiscard]] inline constexpr std::span<const name_override>
trace_overrides() noexcept
{
    return k_trace_overrides;
}

/// @brief Find a row by its canonical name or a legacy alias.
[[nodiscard]] inline const row*
find(std::string_view name) noexcept
{
    for(const auto& spec : rows())
    {
        if(spec.name == name)
        {
            return &spec;
        }
        for(const auto alias : spec.aliases)
        {
            if(alias == name)
            {
                return &spec;
            }
        }
    }
    return nullptr;
}

/// @brief Find a row by canonical name, alias, or an explicit trace override.
[[nodiscard]] inline const row*
find_trace_name(std::string_view name) noexcept
{
    for(const auto& item : trace_overrides())
    {
        if(item.name == name)
        {
            return find(item.canonical);
        }
    }
    return find(name);
}

/// @brief Env keys set to "true" by legacy `--parallel`. Empty and "all" enable
///        every @ref row::in_legacy_parallel row. Other tokens must match a name
///        or alias of those rows. Unknown tokens and `ucx` / `openshmem` are ignored.
[[nodiscard]] std::map<std::string, std::string>
expand_legacy_parallel(std::string_view runtimes);

}  // namespace rocprofsys::runtime_table
