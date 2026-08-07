// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef ROCM_DEBUG_AGENT_UTILS_H
#define ROCM_DEBUG_AGENT_UTILS_H

#include <cstdint>
#include <string>
#include <unordered_map>
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

} /* namespace amd::debug_agent */

#endif /* ROCM_DEBUG_AGENT_UTILS_H */
