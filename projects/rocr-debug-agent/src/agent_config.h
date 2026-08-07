// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef ROCM_DEBUG_AGENT_CONFIG_H
#define ROCM_DEBUG_AGENT_CONFIG_H

#include "logging.h"

#include <optional>
#include <string>
#include <variant>

namespace amd::debug_agent
{

/* Configuration options for the debug agent.  */
struct debug_agent_options_t
{
  bool all_wavefronts = false;          /* Print all wavefronts on exception.  */
  bool disable_sigquit = false;         /* Disable SIGQUIT handler.  */
  bool precise_memory = false;          /* Enable precise memory exception mode.  */
  bool precise_alu_exceptions = false;  /* Enable precise ALU exceptions.  */
  bool lazy = true;                     /* Lazy code object loading.  */
  bool delay_loading = false;           /* Delay code object inspection.  */
  std::optional<std::string> code_objects_dir;  /* Directory to save code objects.  */
  std::optional<std::string> output_file;       /* Output file for agent logs.  */
  log_level_t log_level = log_level_t::warning; /* Logging verbosity level.  */
};

/* Print usage message to stderr.  */
void print_usage ();

/* Parse debug agent options from environment string.
   \param env_options Space-separated option string (e.g., "-a --log-level=info").
   \return Either a parsed options structure or an error/help message string.  */
std::variant<std::string, debug_agent_options_t>
parse_debug_agent_options (const char *env_options);

} /* namespace amd::debug_agent */

#endif /* ROCM_DEBUG_AGENT_CONFIG_H */
