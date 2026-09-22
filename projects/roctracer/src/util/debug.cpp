/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
#include "util.h"

#include <cstdarg>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(ENABLE_BACKTRACE)

#include <cxxabi.h>
#include <backtrace.h>

namespace {

struct BackTraceInfo {
  struct ::backtrace_state* state = nullptr;
  std::stringstream sstream{};
  int depth = 0;
  int error = 0;
};

void errorCallback(void* data, const char* message, int errnum) {
  BackTraceInfo* info = static_cast<BackTraceInfo*>(data);
  info->sstream << "ROCtracer error: " << message << '(' << errnum << ')';
  info->error = 1;
}

void syminfoCallback(void* data, uintptr_t /* pc  */, const char* symname, uintptr_t /* symval  */,
                     uintptr_t /* symsize  */) {
  BackTraceInfo* info = static_cast<BackTraceInfo*>(data);

  if (symname == nullptr) return;

  int status;
  char* demangled = abi::__cxa_demangle(symname, nullptr, nullptr, &status);
  info->sstream << ' ' << (status == 0 ? demangled : symname);
  free(demangled);
}

int fullCallback(void* data, uintptr_t pc, const char* filename, int lineno, const char* function) {
  BackTraceInfo* info = static_cast<BackTraceInfo*>(data);

  info->sstream << std::endl
                << "    #" << std::dec << info->depth++ << ' ' << "0x" << std::noshowbase
                << std::hex << std::setfill('0') << std::setw(sizeof(pc) * 2) << pc;
  if (function == nullptr)
    backtrace_syminfo(info->state, pc, syminfoCallback, errorCallback, data);
  else {
    int status;
    char* demangled = abi::__cxa_demangle(function, nullptr, nullptr, &status);
    info->sstream << ' ' << (status == 0 ? demangled : function);
    free(demangled);

    if (filename != nullptr) {
      info->sstream << " in " << filename;
      if (lineno) info->sstream << ':' << std::dec << lineno;
    }
  }

  return info->error;
}

}  // namespace
#endif  // defined (ENABLE_BACKTRACE)

namespace roctracer {

void warning(const char* format, ...) {
  va_list va;
  va_start(va, format);
  std::cerr << "ROCtracer warning: " << string_vprintf(format, va) << std::endl;
  va_end(va);
}

void error(const char* format, ...) {
  va_list va;
  va_start(va, format);
  std::cerr << "ROCtracer error: " << string_vprintf(format, va) << std::endl;
  va_end(va);
  exit(EXIT_FAILURE);
}

void fatal [[noreturn]] (const char* format, ...) {
  va_list va;
  va_start(va, format);
  std::string message = string_vprintf(format, va);
  va_end(va);

#if defined(ENABLE_BACKTRACE)
  BackTraceInfo info;

  info.sstream << std::endl << "Backtrace:";
  info.state = ::backtrace_create_state("/proc/self/exe", 0, errorCallback, &info);
  ::backtrace_full(info.state, 1, fullCallback, errorCallback, &info);

  message += info.sstream.str();
#endif /* defined (ENABLE_BACKTRACE) */

  std::cerr << "ROCtracer fatal error: " << message << std::endl;
  abort();
}

}  // namespace roctracer