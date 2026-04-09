/* Copyright (c) 2019-2024 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "debug.h"
#include "exception.h"
#include "logging.h"
#include "utils.h"

#include <cstdarg>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(HAVE_BACKTRACE)
#include <backtrace-supported.h>
#endif /* defined(HAVE_BACKTRACE) */

#if BACKTRACE_SUPPORTED
#include <backtrace.h>
#include <cxxabi.h>
#endif /* BACKTRACE_SUPPORTED */

namespace amd::dbgapi
{
#if BACKTRACE_SUPPORTED
namespace detail
{

struct backtrace_info
{
  struct backtrace_state *state = nullptr;
  std::stringstream sstream{};
  int depth = 0;
  int error = 0;
};

static void
error_callback (void *data, const char *message, int errnum)
{
  backtrace_info *info = static_cast<backtrace_info *> (data);
  info->sstream << "Error: " << message << '(' << errnum << ')';
  info->error = 1;
}

static void
syminfo_callback (void *data, uintptr_t /* pc  */, const char *symname,
                  uintptr_t /* symval  */, uintptr_t /* symsize  */)
{
  backtrace_info *info = static_cast<backtrace_info *> (data);
  int status;

  if (symname == nullptr)
    return;

  char *demangled = abi::__cxa_demangle (symname, nullptr, nullptr, &status);
  info->sstream << ' ' << (status == 0 ? demangled : symname);
  free (demangled);
}

static int
full_callback (void *data, uintptr_t pc, const char *filename, int lineno,
               const char *function)
{
  backtrace_info *info = static_cast<backtrace_info *> (data);
  int status;

  info->sstream << std::endl
                << "    #" << std::dec << info->depth++ << ' ' << "0x"
                << std::hex << std::setfill ('0')
                << std::setw (sizeof (pc) * 2) << pc;

  if (function == nullptr)
    backtrace_syminfo (info->state, pc, syminfo_callback, error_callback,
                       data);
  else
    {
      char *demangled
        = abi::__cxa_demangle (function, nullptr, nullptr, &status);
      info->sstream << ' ' << (status == 0 ? demangled : function);
      free (demangled);

      if (filename != nullptr)
        info->sstream << " in " << filename << ':' << std::dec << lineno;
    }

  return info->error;
}

} /* namespace detail */
#endif /* BACKTRACE_SUPPORTED */

void
warning (const char *format, ...)
{
  va_list va;
  va_start (va, format);
  vlog (AMD_DBGAPI_LOG_LEVEL_WARNING, format, va);
  va_end (va);
}

void
fatal_error (const char *format, ...)
{
  va_list va;
  va_start (va, format);

  std::string message = string_vprintf (format, va);
  va_end (va);

#if BACKTRACE_SUPPORTED
  detail::backtrace_info info;

  info.sstream << std::endl << "Backtrace:";
  info.state = backtrace_create_state ("/proc/self/exe", 0,
                                       detail::error_callback, &info);
  backtrace_full (info.state, 1, detail::full_callback, detail::error_callback,
                  &info);

  message += info.sstream.str ();
#endif /* BACKTRACE_SUPPORTED */

  /* TODO: We should return a FATAL error here, and put the API
     in a finalized state (maybe the catch should do that).  */
  throw fatal_error_t (message);
}

} /* namespace amd::dbgapi */
