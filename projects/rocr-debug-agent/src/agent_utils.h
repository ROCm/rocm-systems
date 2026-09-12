// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef ROCM_DEBUG_AGENT_UTILS_H
#define ROCM_DEBUG_AGENT_UTILS_H

#include "logging.h"

#include <amd-dbgapi/amd-dbgapi.h>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace amd::debug_agent
{

/* Hex and register value formatting functions.  */

/* Convert a byte vector to a hexadecimal string representation.
   Bytes are output in reverse order (big-endian).
   \param value The byte vector to convert.
   \return Hexadecimal string representation.  */
std::string hex_string (const std::vector<uint8_t> &value);

/* Format a register value as a string based on its type.
   For vector types (e.g., "int[4]"), formats each element separately.
   For scalar types, formats as hexadecimal.
   \param register_type The type string (e.g., "int", "float[8]").
   \param register_value The raw register bytes.
   \return Formatted string representation of the register value.  */
std::string register_value_string (const std::string &register_type,
                                     const std::vector<uint8_t> &register_value);

/* URI parsing and sanitization.  */

/* Result of parsing a code object URI.  */
struct parsed_uri_t
{
  std::string protocol;          /* Protocol scheme (e.g., "file").  */
  std::string decoded_path;      /* Path component with percent-decoding applied.  */
  std::unordered_map<std::string, std::string> params;  /* Query/fragment parameters.  */
};

/* Parse a code object URI into its components.
   Extracts protocol, path (percent-decoded), and parameters from query/fragment.
   \param uri The URI string to parse (e.g., "file:///path?param=value").
   \return Parsed URI structure.  */
parsed_uri_t parse_code_object_uri (const std::string &uri);

/* Convert a URI to a safe filename by replacing special characters.
   Replaces characters like :, /, #, ?, &, = with underscores.
   \param uri The URI string to sanitize.
   \return Filename-safe string.  */
std::string sanitize_uri_for_filename (const std::string &uri);

/* Path format token expansion.  */

/* Expand format tokens in a string with runtime values.
   Supported tokens: %p (PID), %h (hostname), %t (timestamp), %e (executable),
   %u (UID), %g (GID), %% (literal %).
   \param format Format string containing tokens (e.g., "/tmp/trace-%p-%t").
   \return String with tokens replaced by actual values.  */
std::string expand_format_tokens (const std::string &format);

/* Debug agent options parsing.  */

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

/* Exception bitmask mapping.  */

/* Map wave stop reason bits to exception bitmask for resume.
   Handles multi-bit stop reasons by processing each bit individually.
   \param stop_reason Wave stop reason bitmask (may contain multiple bits).
   \return Exception bitmask suitable for amd_dbgapi_wave_resume.  */
std::underlying_type_t<amd_dbgapi_exceptions_t>
map_stop_reason_to_exceptions (
  std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason);

/* Convert a single wave stop reason to its string name.
   \param reason Stop reason enumeration value (single bit expected).
   \return String name of the stop reason (e.g., "BREAKPOINT"), or empty string
           if unknown.  */
const char *
stop_reason_to_string (amd_dbgapi_wave_stop_reasons_t reason);

} /* namespace amd::debug_agent */

#endif /* ROCM_DEBUG_AGENT_UTILS_H */
