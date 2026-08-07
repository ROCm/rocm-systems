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

#include "agent_utils.h"
#include "debug.h"

#include <sstream>

namespace amd::debug_agent
{

/* Hex and register value formatting functions. */

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
  /* Handle vector types. */
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

} /* namespace amd::debug_agent */
