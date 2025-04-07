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

#include "helper.h"

#include <cstdio>
#include <cstdarg>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <amd_comgr/amd_comgr.h>

#define amd_comgr_(call)                                                                           \
  do {                                                                                             \
    if (amd_comgr_status_t status = amd_comgr_##call; status != AMD_COMGR_STATUS_SUCCESS) {        \
      const char* reason = "";                                                                     \
      amd_comgr_status_string(status, &reason);                                                    \
      fatal(#call " failed: %s", reason);                                                          \
    }                                                                                              \
  } while (false)

#if defined(ENABLE_BACKTRACE)

#include <backtrace.h>
#include <cxxabi.h>

namespace {

struct BackTraceInfo {
  struct ::backtrace_state* state = nullptr;
  std::stringstream sstream{};
  int depth = 0;
  int error = 0;
};

void errorCallback(void* data, const char* message, int errnum) {
  BackTraceInfo* info = static_cast<BackTraceInfo*>(data);
  info->sstream << "ROCProfiler: error: " << message << '(' << errnum << ')';
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
  if (function == nullptr) {
    backtrace_syminfo(info->state, pc, syminfoCallback, errorCallback, data);
  } else {
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

namespace rocprofiler {

std::string string_vprintf(const char* format, va_list va) {
  va_list copy;

  va_copy(copy, va);
  size_t size = vsnprintf(NULL, 0, format, copy);
  va_end(copy);

  std::string str(size, '\0');
  vsprintf(&str[0], format, va);

  return str;
}

std::string string_printf(const char* format, ...) {
  va_list va;
  va_start(va, format);
  std::string str(string_vprintf(format, va));
  va_end(va);

  return str;
}

[[maybe_unused]] void warning(const char* format, ...) {
  va_list va;
  va_start(va, format);
  vfprintf(stderr, format, va);
  va_end(va);
}

[[maybe_unused]] void fatal [[noreturn]] (const char* format, ...) {
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

  std::string errmsg("ROCProfiler: fatal error: " + message);

  std::cerr << errmsg << std::endl;
  abort();
}

/* The function extracts the kernel name from
input string. By using the iterators it finds the
window in the string which contains only the kernel name.
For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'*/
std::string truncate_name(const std::string& name) {

  const char* path = getenv("ROCPROFILER_TRUNCATE_KERNEL_PATH");
  if(path != nullptr) {
      auto rit = name.rbegin();
      auto rend = name.rend();
      uint32_t counter = 0;
      char open_token = 0;
      char close_token = 0;
      while (rit != rend) {
        if (counter == 0) {
          switch (*rit) {
            case ')':
              counter = 1;
              open_token = ')';
              close_token = '(';
              break;
            case '>':
              counter = 1;
              open_token = '>';
              close_token = '<';
              break;
            case ']':
              counter = 1;
              open_token = ']';
              close_token = '[';
              break;
            case ' ':
              ++rit;
              continue;
          }
          if (counter == 0) break;
        } else {
          if (*rit == open_token) counter++;
          if (*rit == close_token) counter--;
        }
        ++rit;
      }
      auto rbeg = rit;
      while ((rit != rend) && (*rit != ' ') && (*rit != ':')) rit++;
      return name.substr(rend - rit, rit - rbeg);
  }
  else
    return name;
}

// C++ symbol demangle
std::string cxx_demangle(const std::string& symbol) {
  amd_comgr_data_t mangled_data;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data));
  amd_comgr_(set_data(mangled_data, symbol.size(), symbol.data()));

  amd_comgr_data_t demangled_data;
  amd_comgr_(demangle_symbol_name(mangled_data, &demangled_data));

  size_t demangled_size = 0;
  amd_comgr_(get_data(demangled_data, &demangled_size, nullptr));

  std::string demangled_str;
  demangled_str.resize(demangled_size);
  amd_comgr_(get_data(demangled_data, &demangled_size, demangled_str.data()));

  amd_comgr_(release_data(mangled_data));
  amd_comgr_(release_data(demangled_data));
  return demangled_str;
}

// check if string has special char
bool has_special_char(std::string const& str) {
  return std::find_if(str.begin(), str.end(), [](unsigned char ch) {
           return !(isalnum(ch) || ch == '_' || ch == ':' || ch == ' ');
         }) != str.end();
}

// check if string has correct counter format
bool has_counter_format(std::string const& str) {
  return std::find_if(str.begin(), str.end(),
                      [](unsigned char ch) { return (isalnum(ch) || ch == '_'); }) != str.end();
}

// trims the begining of the line for spaces
std::string left_trim(const std::string& s) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  size_t start = s.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : s.substr(start);
}

// trims begining and end of input line in place
void trim(std::string& str) {
  // Remove leading spaces.
  str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
              return !std::isspace(ch);
            }));
  // Remove trailing spaces.
  str.erase(
      std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) { return !std::isspace(ch); })
          .base(),
      str.end());
}

// replace unsuported specail chars with space
static void handle_special_chars(std::string& str) {
  std::set<char> specialChars = {'!', '@', '#', '$', '%', '&', '(', ')', ',',
                                 '*', '+', '-', '.', '/', ';', '<', '=', '>',
                                 '?', '@', '{', '}', '^', '`', '~', '|', ':'};

  // Iterate over the string and replace any special characters with a space.
  for (unsigned int i = 0; i < str.length(); i++) {
    if (specialChars.find(str[i]) != specialChars.end()) {
      str[i] = ' ';
    }
  }
}

// validate input coutners and correct format if needed
void validate_counters_format(std::vector<std::string>& counters, std::string line) {
  // trim line for any white spaces
  trim(line);

  if (!(line[0] == '#' || line.find("pmc") == std::string::npos)) {
    handle_special_chars(line);

    std::stringstream input_line(line);
    std::string counter;
    while (getline(input_line, counter, ' ')) {
      if (counter.substr(0, 3) != "pmc" && has_counter_format(counter)) {
        counters.push_back(counter);
      }
    }
  }

  // raise exception with correct usage if user still managed to corrupt input
  for (const auto& itr : counters) {
    if (!has_counter_format(itr)) {
      rocprofiler::fatal("Bad input metric. usage --> pmc: <counter1> <counter2>");
    }
  }
}

}  // namespace rocprofiler
