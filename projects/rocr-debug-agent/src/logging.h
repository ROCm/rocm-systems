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

#ifndef _ROCM_DEBUG_AGENT_LOGGING_H
#define _ROCM_DEBUG_AGENT_LOGGING_H 1

#include <fstream>

namespace amd::debug_agent
{

enum class log_level_t
{
  /* Print no messages.  */
  none = 0,
  /* Print error messages.  */
  error = 1,
  /* Print error, and warning messages.  */
  warning = 2,
  /* Print error, warning, and info messages.  */
  info = 3,
  /* Print error, warning, info, and verbose  messages.  */
  verbose = 4
};

extern log_level_t log_level;

extern std::ofstream agent_out;

namespace detail
{

/* A macro instead of a variadic template so that the __VAR_ARGS__ are not
   evaluated unless the log level indicated they are needed.  */
extern void log (log_level_t level, const char *format, ...)
#if defined(__GNUC__)
    __attribute__ ((format (printf, 2, 3)))
#endif /* defined (__GNUC__) */
    ;

} /* namespace detail */

#define agent_log(level, format, ...)                                         \
  do                                                                          \
    {                                                                         \
      if (level <= amd::debug_agent::log_level)                               \
        amd::debug_agent::detail::log (level, format, ##__VA_ARGS__);         \
    }                                                                         \
  while (0)

void set_log_level (log_level_t level);

} /* namespace amd::debug_agent */

#endif /* _ROCM_DEBUG_AGENT_LOGGING_H */
