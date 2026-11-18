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
  /* Print all wavefronts on exception.  */
  bool all_wavefronts = false;
  /* Disable SIGQUIT handler.  */
  bool disable_sigquit = false;
  /* Enable precise memory exception mode.  */
  bool precise_memory = false;
  /* Enable precise ALU exceptions.  */
  bool precise_alu_exceptions = false;
  /* Lazy code object loading.  */
  bool lazy = true;
  /* Delay code object inspection.  */
  bool delay_loading = false;
  /* Directory to save code objects.  */
  std::optional<std::string> code_objects_dir;
  /* Output file for agent logs.  */
  std::optional<std::string> output_file;
  /* Logging verbosity level.  */
  log_level_t log_level = log_level_t::warning;
  /* The usage message was requested (-h).  No other option is valid.  */
  bool help_requested = false;
};

/* Print usage message to stderr.  */
void print_usage ();

/* Parse debug agent options from environment string.
   \param env_options Space-separated option string
                      (e.g., "-a --log-level=info").
   \return Either a parsed options structure or an error message string.
           If help was requested, help_requested is set in the options.  */
std::variant<std::string, debug_agent_options_t>
parse_debug_agent_options (const char *env_options);

} /* namespace amd::debug_agent */

#endif /* ROCM_DEBUG_AGENT_CONFIG_H */
