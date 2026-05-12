// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <perfetto.h>

namespace rocprofsys::output::perfetto_log_filter
{

// Drops kLogDebug + kLogInfo, forwards kLogImportant -> LOG_WARNING
// and kLogError -> LOG_ERROR. Signature matches the SDK typedef
// perfetto::base::LogMessageCallback (by value).
void
filter_fn(::perfetto::base::LogMessageCallbackArgs args);

// Idempotent (std::call_once); must run before any
// perfetto::Tracing::Initialize on the owning process.
void
install();

}  // namespace rocprofsys::output::perfetto_log_filter
