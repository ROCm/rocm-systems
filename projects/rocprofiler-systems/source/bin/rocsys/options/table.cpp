// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "options/table.hpp"

#include "options/avail_traces.hpp"
#include "options/dry_run.hpp"
#include "options/traces.hpp"

#include <array>

namespace rocprofsys::rocsys::options
{

std::span<const option_spec>
option_table() noexcept
{
    static const auto k_options =
        std::to_array<option_spec>({ k_dry_run, k_traces, k_avail_traces });
    return k_options;
}

}  // namespace rocprofsys::rocsys::options
