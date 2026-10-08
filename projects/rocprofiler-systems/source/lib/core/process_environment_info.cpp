// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "process_environment_info.hpp"

#include "common/environment.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys
{
namespace
{
constexpr char             k_assignment     = '=';
constexpr char             k_path_separator = '/';
constexpr std::string_view k_home_alias     = "~";
constexpr std::string_view k_mpi_rank_name  = "MPI_COMM_WORLD_RANK";
constexpr std::string_view k_mpi_size_name  = "MPI_COMM_WORLD_SIZE";

// Single reviewable source of truth for which variables may be stored in a shared
// database. Everything not listed here is dropped, never filtered by keyword.
constexpr auto k_allowed_prefixes = std::to_array<std::string_view>({
    "ROCPROFSYS_",
    "LD_",
    "ROCM_",
});

constexpr auto k_allowed_names = std::to_array<std::string_view>({
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "SLURM_PROCID",
    "SLURM_JOB_ID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
});

}  // namespace

process_environment_info::process_environment_info()
{
    const auto end       = std::ranges::find(environ, std::unreachable_sentinel, nullptr);
    const auto env_block = std::span<const char* const>{ environ, end };

    auto home = rocprofsys::get_env<std::string>("HOME", "");

    // HOME unset or "/" must not be redacted: it would rewrite every path.
    while(home.ends_with(k_path_separator))
    {
        home.pop_back();
    }

    for(const char* raw_entry : env_block)
    {
        // Split at the first '=' so values may contain '='; entries without one, or with
        // an empty name, are not valid environment entries.
        const std::string_view entry{ raw_entry };
        const auto             assignment = entry.find(k_assignment);
        if(assignment == std::string_view::npos || assignment == 0)
        {
            continue;
        }

        const auto name       = entry.substr(0, assignment);
        const auto has_prefix = [name](std::string_view prefix) {
            return name.starts_with(prefix);
        };
        if(std::ranges::find(k_allowed_names, name) == k_allowed_names.end() &&
           !std::ranges::any_of(k_allowed_prefixes, has_prefix))
        {
            continue;
        }

        // Replace every occurrence of the home directory with "~" so the user name is
        // not leaked, including from free-form values. An empty home matches everywhere,
        // hence the guard in the loop condition.
        std::string value{ entry.substr(assignment + 1) };
        for(auto pos = value.find(home); !home.empty() && pos != std::string::npos;
            pos      = value.find(home, pos + k_home_alias.size()))
        {
            value.replace(pos, home.size(), k_home_alias);
        }
        m_entries[std::string{ name }] = std::move(value);
    }
}

void
process_environment_info::add_mpi_identity(int rank, int size)
{
    m_entries[std::string{ k_mpi_rank_name }] = rank;
    m_entries[std::string{ k_mpi_size_name }] = size;
}

std::string
process_environment_info::to_json() const
{
    // Environment values are arbitrary bytes; the default strict handler would throw on
    // invalid UTF-8, and this runs during process start-up.
    return m_entries.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}
}  // namespace rocprofsys
