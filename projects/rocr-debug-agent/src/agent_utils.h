/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

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

/* Hex and register value formatting functions. */

std::string hex_string (const std::vector<uint8_t> &value);

std::string register_value_string (const std::string &register_type,
                                     const std::vector<uint8_t> &register_value);

/* URI parsing and sanitization. */

struct parsed_uri_t
{
  std::string protocol;
  std::string decoded_path;
  std::unordered_map<std::string, std::string> params;
};

parsed_uri_t parse_code_object_uri (const std::string &uri);

std::string sanitize_uri_for_filename (const std::string &uri);

/* Debug agent options parsing. */

struct debug_agent_options_t
{
  bool all_wavefronts = false;
  bool disable_sigquit = false;
  bool precise_memory = false;
  bool precise_alu_exceptions = false;
  bool lazy = true;
  bool delay_loading = false;
  std::optional<std::string> code_objects_dir;
  std::optional<std::string> output_file;
  log_level_t log_level = log_level_t::warning;
};

void print_usage ();

std::variant<std::string, debug_agent_options_t>
parse_debug_agent_options (const char *env_options);

/* Exception bitmask mapping. */

std::underlying_type_t<amd_dbgapi_exceptions_t>
map_stop_reason_to_exceptions (
  std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason);

const char *
stop_reason_to_string (amd_dbgapi_wave_stop_reasons_t reason);

} /* namespace amd::debug_agent */

#endif /* ROCM_DEBUG_AGENT_UTILS_H */
