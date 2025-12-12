/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

#include "logging.h"

#include <amd-dbgapi/amd-dbgapi.h>
#include <cstdio>
#include <stdarg.h>

#include <string>

namespace amd::debug_agent
{

log_level_t log_level = log_level_t::warning;

std::ofstream agent_out;

namespace detail
{

void
log (log_level_t level, const char *format, ...)
{
  va_list va;

  agent_out << "rocm-debug-agent: ";

  if (level == log_level_t::error)
    agent_out << "error: ";
  else if (level == log_level_t::warning)
    agent_out << "warning: ";

  va_start (va, format);
  size_t size = vsnprintf (NULL, 0, format, va);
  va_end (va);

  va_start (va, format);
  std::string str (size, '\0');
  vsprintf (&str[0], format, va);
  va_end (va);

  agent_out << str << std::endl;
}

} /* namespace detail */

void
set_log_level (log_level_t level)
{
  log_level = level;
  switch (level)
    {
    case log_level_t::none:
      amd_dbgapi_set_log_level (AMD_DBGAPI_LOG_LEVEL_NONE);
      break;
    case log_level_t::verbose:
      amd_dbgapi_set_log_level (AMD_DBGAPI_LOG_LEVEL_VERBOSE);
      break;
    case log_level_t::info:
      amd_dbgapi_set_log_level (AMD_DBGAPI_LOG_LEVEL_INFO);
      break;
    case log_level_t::warning:
      amd_dbgapi_set_log_level (AMD_DBGAPI_LOG_LEVEL_WARNING);
      break;
    case log_level_t::error:
      amd_dbgapi_set_log_level (AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR);
      break;
    }
}

} /* namespace amd::debug_agent */
