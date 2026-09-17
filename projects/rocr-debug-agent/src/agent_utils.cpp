// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/* Function implementations - see agent_utils.h for documentation.  */

#include "agent_utils.h"
#include "debug.h"

#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

namespace amd::debug_agent
{

/* Hex and register value formatting functions.  */

std::string
hex_string (const std::vector<uint8_t> &value)
{
  std::string value_string;
  value_string.reserve (2 * value.size ());

  for (size_t pos = value.size (); pos > 0; --pos)
    {
      static constexpr char hex_digits[] = "0123456789abcdef";
      value_string.push_back (hex_digits[value[pos - 1] >> 4]);
      value_string.push_back (hex_digits[value[pos - 1] & 0xF]);
    }

  return value_string;
}

std::string
register_value_string (const std::string &register_type,
                       const std::vector<uint8_t> &register_value)
{
  /* Handle vector types.  */
  if (size_t pos = register_type.find_last_of ('['); pos != std::string::npos)
    {
      const std::string element_type = register_type.substr (0, pos);
      const size_t element_count = std::stoi (register_type.substr (pos + 1));
      const size_t element_size = register_value.size () / element_count;

      agent_assert ((register_value.size () % element_size) == 0);

      std::stringstream ss;
      for (size_t i = 0; i < element_count; ++i)
        {
          if (i != 0)
            ss << " ";
          ss << "[" << i << "] ";

          std::vector<uint8_t> element_value (
              &register_value[element_size * i],
              &register_value[element_size * (i + 1)]);

          ss << register_value_string (element_type, element_value);
        }
      return ss.str ();
    }

  return hex_string (register_value);
}

/* URI parsing and sanitization.  */

parsed_uri_t
parse_code_object_uri (const std::string &uri)
{
  parsed_uri_t result;

  const std::string protocol_delim{ "://" };

  size_t protocol_end = uri.find (protocol_delim);
  if (protocol_end == std::string::npos)
    {
      /* No protocol delimiter found, treat entire URI as path.  */
      result.protocol = "";
      protocol_end = 0;
    }
  else
    {
      result.protocol = uri.substr (0, protocol_end);
      protocol_end += protocol_delim.length ();

      std::transform (result.protocol.begin (), result.protocol.end (),
                      result.protocol.begin (),
                      [] (unsigned char c) { return std::tolower (c); });
    }

  std::string path;
  size_t path_end = uri.find_first_of ("#?", protocol_end);
  if (path_end != std::string::npos)
    path = uri.substr (protocol_end, path_end++ - protocol_end);
  else
    path = uri.substr (protocol_end);

  /* %-decode the string.  */
  result.decoded_path.reserve (path.length ());
  for (size_t i = 0; i < path.length (); ++i)
    if (path[i] == '%' && i + 2 < path.length ()
        && std::isxdigit (path[i + 1]) && std::isxdigit (path[i + 2]))
      {
        result.decoded_path += std::stoi (path.substr (i + 1, 2), 0, 16);
        i += 2;
      }
    else
      result.decoded_path += path[i];

  /* Tokenize the query/fragment.  */
  std::vector<std::string> tokens;
  size_t pos, last = path_end;
  while ((pos = uri.find ('&', last)) != std::string::npos)
    {
      tokens.emplace_back (uri.substr (last, pos - last));
      last = pos + 1;
    }
  if (last != std::string::npos)
    tokens.emplace_back (uri.substr (last));

  /* Create a tag-value map from the tokenized query/fragment.  */
  std::for_each (tokens.begin (), tokens.end (), [&] (std::string &token) {
    size_t delim = token.find ('=');
    if (delim != std::string::npos)
      result.params.emplace (token.substr (0, delim), token.substr (delim + 1));
  });

  return result;
}

std::string
sanitize_uri_for_filename (const std::string &uri)
{
  std::string name{ uri };

  size_t pos{};
  while ((pos = name.find_first_of (":/#?&="), pos) != std::string::npos)
    name[pos] = '_';

  return name;
}

/* Path format token expansion.  */

std::string
expand_format_tokens (const std::string &format)
{
  std::string result;
  result.reserve (format.size ());

  for (size_t i = 0; i < format.size (); ++i)
    {
      if (format[i] != '%' || i + 1 == format.size ())
        {
          result += format[i];
          continue;
        }

      switch (format[++i])
        {
        case 'p':
          result += std::to_string (::getpid ());
          break;

        case 'h':
          {
            char hostname[256] = {};
            if (::gethostname (hostname, sizeof (hostname) - 1) == 0)
              result += hostname;
          }
          break;

        case 't':
          result
              += std::to_string (static_cast<long long> (std::time (nullptr)));
          break;

        case 'e':
          {
            std::ifstream comm ("/proc/self/comm");
            std::string name;
            if (comm && std::getline (comm, name))
              result += name;
          }
          break;

        case 'u':
          result += std::to_string (::getuid ());
          break;

        case 'g':
          result += std::to_string (::getgid ());
          break;

        case '%':
          result += '%';
          break;

        default:
          /* Leave unrecognized tokens unchanged.  */
          result += '%';
          result += format[i];
          break;
        }
    }

  return result;
}

/* Exception bitmask mapping.  */

std::underlying_type_t<amd_dbgapi_exceptions_t>
map_stop_reason_to_exceptions (
    std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason)
{
  std::underlying_type_t<amd_dbgapi_exceptions_t> resume_exceptions = 0;
  auto stop_reason_bits{ stop_reason };

  do
    {
      auto one_bit
          = stop_reason_bits ^ (stop_reason_bits & (stop_reason_bits - 1));
      stop_reason_bits ^= one_bit;

      switch (one_bit)
        {
        case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
        case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
          /* AMD_DBGAPI_EXCEPTION_NONE is 0, no assignment needed. */
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
        case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
        case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
        case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_TRAP;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
          /* AMD_DBGAPI_EXCEPTION_NONE is 0, no assignment needed. */
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
        case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ADDRESS_ERROR;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
        case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ABORT;
          break;

#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
        case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
          break;
#endif
        }
    }
  while (stop_reason_bits != 0);

  return resume_exceptions;
}

const char *
stop_reason_to_string (amd_dbgapi_wave_stop_reasons_t reason)
{
  switch (reason)
    {
    case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
      return "NONE";
    case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
      return "BREAKPOINT";
    case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
      return "WATCHPOINT";
    case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
      return "SINGLE_STEP";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
      return "FP_INPUT_DENORMAL";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
      return "FP_DIVIDE_BY_0";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
      return "FP_OVERFLOW";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
      return "FP_UNDERFLOW";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
      return "FP_INEXACT";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
      return "FP_INVALID_OPERATION";
    case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
      return "INT_DIVIDE_BY_0";
    case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
      return "DEBUG_TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
      return "ASSERT_TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
      return "TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
      return "MEMORY_VIOLATION";
    case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
      return "ADDRESS_ERROR";
    case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
      return "ILLEGAL_INSTRUCTION";
    case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
      return "ECC_ERROR";
    case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
      return "FATAL_HALT";
#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
    case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
      return "RESERVED";
#endif
    }

  return "";
}

} /* namespace amd::debug_agent */
