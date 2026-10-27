// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/option_spec.hpp"
#include "trace/catalog.hpp"

#include <ostream>

namespace rocprofsys::rocsys::options
{

/// @brief `rocsys avail --traces`. The launcher prints the catalog and does not
///        run `rocprof-sys-avail`.
inline constexpr option_spec k_avail_traces{
    .name          = "traces",
    .kind          = value_kind::flag,
    .required_caps = static_cast<capability_set>(capability::query_only),
    .summary       = "List trace names and whether this install can use them",
};

/// @brief Print NAME / WHAT IT TRACES / EXPANDS TO / AVAILABLE, then the SDK domains.
void
write_avail_traces(std::ostream& out, const trace::trace_catalog& catalog);

}  // namespace rocprofsys::rocsys::options
