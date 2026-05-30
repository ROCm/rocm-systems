// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Public header — intentionally free of <perfetto.h> so that translation
// units which only need to install the filter do not pay for the full
// amalgamated SDK header. The Perfetto-typed surface (classify, filter_fn,
// filter_action) lives in perfetto_log_filter_detail.hpp and is included
// only by the implementation TU and the unit tests.

namespace rocprofsys::output::perfetto_log_filter
{

// Registers the filter as perfetto's log message callback so all
// subsequent perfetto-side log emissions flow through classify() and
// either drop or forward via the rocprof-sys logger. Idempotent
// (std::call_once); must run before any perfetto::Tracing::Initialize
// on the owning process.
void
register_with_perfetto_logger();

}  // namespace rocprofsys::output::perfetto_log_filter
