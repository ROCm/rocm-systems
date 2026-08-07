// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/* Function implementations - see agent_utils.h for documentation.  */

#include "agent_utils.h"
#include "debug.h"

#include <algorithm>
#include <cctype>
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

} /* namespace amd::debug_agent */
