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

#include "logging.h"

#include <amd-dbgapi/amd-dbgapi.h>

namespace amd::debug_agent
{

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
