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

#ifndef _ROCM_DEBUG_AGENT_DEBUG_H
#define _ROCM_DEBUG_AGENT_DEBUG_H 1

#include "logging.h"

namespace amd::debug_agent
{

/* A macro instead of a variadic template so that format is still a string
   literal when passed to agent_log.  */
#define agent_warning(format, ...)                                            \
  agent_log (log_level_t::warning, format, ##__VA_ARGS__)

#define agent_error(format, ...)                                              \
  do                                                                          \
    {                                                                         \
      agent_log (log_level_t::error, format, ##__VA_ARGS__);                  \
      abort ();                                                               \
    }                                                                         \
  while (false)

#define agent_assert_fail(assertion, file, line)                              \
  [] () {                                                                     \
    agent_error ("%s:%d: Assertion `%s' failed.", file, line, assertion);     \
  }()

#define DEBUG_AGENT_ASSERTION_ENABLED 1

#if defined(DEBUG_AGENT_ASSERTION_ENABLED)
#define agent_assert(expr)                                                    \
  ((void)((expr) ? 0 : (agent_assert_fail (#expr, __FILE__, __LINE__), 0)))
#else /* !defined (DEBUG_AGENT_ASSERTION_ENABLED) */
#define agent_assert(expr) ((void)0)
#endif /* !defined(DEBUG_AGENT_ASSERTION_ENABLED) */

} /* namespace amd::debug_agent */

#endif /* _ROCM_DEBUG_AGENT_DEBUG_H */