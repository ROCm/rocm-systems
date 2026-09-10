// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/* Stub implementations of logging functions for unit tests.
   Unit tests don't need actual logging functionality, so we provide
   no-op stubs to satisfy linker requirements. */

#include "log_level.h"

#include <fstream>

namespace amd::debug_agent
{

/* Global log level variable (not used in unit tests). */
log_level_t log_level = log_level_t::none;

/* Global output stream (not used in unit tests). */
std::ofstream agent_out;

namespace detail
{

/* No-op log function for unit tests. */
void
log (log_level_t level, const char *format, ...)
{
  /* No-op - unit tests don't need actual logging. */
}

} /* namespace detail */

} /* namespace amd::debug_agent */
