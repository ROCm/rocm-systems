/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#include "top.hpp"
#include "utils/flags.hpp"
#include "os/os.hpp"

#include <unordered_map>
#include <string>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <unistd.h>
#endif  // !_WIN32

namespace {

const char* removeQuotes(const char* Value) {
  const char *b, *e, *p;
  if (Value == NULL) {
    return Value;
  }

  // skip the leading blank
  for (p = Value; *p == ' '; ++p);
  if (*p != '"') {
    return Value;
  }
  b = p;
  e = NULL;
  for (++p; *p != '\0'; ++p) {
    if (*p == '"') {
      // e points to the last '"'
      e = p;
    } else if ((e != NULL) && (*p != ' ')) {
      // e isn't last '"' if there is any non-blank following e
      e = NULL;
    }
  }

  if (e == NULL) {
    return Value;
  }
  // Found a valid quoted string "<str>" with b=1st '"' and e=the last '"'
  size_t len = (e - b - 1) > 0 ? (e - b - 1) : 0;
#ifdef _WIN32
  char* p1 = _strdup(b + 1);
  p1[len] = '\0';
  p = p1;
#else
  p = strndup(b + 1, len);
#endif
  return p;
}
}  // namespace

namespace amd {

#ifdef __APPLE__
#include <crt_externs.h>
#endif  // __APPLE__

bool IS_HIP = false;

// static
char* Flag::envstr_;

void Flag::tearDown() {
#ifdef _WIN32
  FreeEnvironmentStringsA(envstr_);
#endif
}

bool Flag::init() {
  typedef std::unordered_map<std::string, const char*> vars_type;
  vars_type vars;

#ifdef _WIN32
  char* str = GetEnvironmentStringsA();
  envstr_ = str;

  for (; *str != '\0'; str += strlen(str) + 1) {
    // For all environment variables:
    std::string var = str;
    size_t pos = var.find('=');
    if ((pos == std::string::npos) || ((pos + 1) > var.size())) {
      continue;
    }

    std::string name = var.substr(0, pos);
    if ((pos + 1) == var.size()) {
      vars.insert(std::make_pair(name, " "));
    } else {
      vars.insert(std::make_pair(name, &str[pos + 1]));
    }
  }
#else  // !_WIN32
#ifdef __APPLE__
  char** environ = *_NSGetEnviron();
  if (environ == NULL) {
    return false;
  }
#endif  // __APPLE__

  for (const char** p = const_cast<const char**>(environ); *p != NULL; ++p) {
    std::string var = *p;
    size_t pos = var.find('=');
    if ((pos == std::string::npos) || ((pos + 1) > var.size())) {
      continue;
    }

    std::string name = var.substr(0, pos);
    if ((pos + 1) == var.size()) {
      vars.insert(std::make_pair(name, " "));
    } else {
      vars.insert(std::make_pair(name, &(*p)[pos + 1]));
    }
  }
#endif  // !_WIN32

  for (size_t i = 0; i < numFlags_; ++i) {
    Flag& flag = flags_[i];

    const auto it = vars.find(flag.name_);
    if (it != vars.cend()) {
      flag.setValue(it->second);
    }
  }
  if (!flagIsDefault(AMD_LOG_LEVEL)) {
    if (!flagIsDefault(AMD_LOG_LEVEL_FILE)) {
      std::string fileName = AMD_LOG_LEVEL_FILE;
      std::string pid = std::to_string(Os::getProcessId());
      fileName = fileName + "_" + pid;
      outFile = fopen(fileName.c_str(), "a");
      if (outFile == NULL) {
        outFile = fopen(("clr_logs_" + pid + ".txt").c_str(), "a");
      }
    }
  }

  return true;
}

bool Flag::setValue(const char* value) {
  if (value_ == NULL) {
    return false;  // flag is constant.
  }

  isDefault_ = false;

  switch (type_) {
    case Tbool:
      *(bool*)value_ = (strcmp(value, "true") == 0 || atoi(value) != 0) ? true : false;
      return true;

    case Tint:
    case Tuint:
      *(int*)value_ = atoi(value);
      return true;

    case Tsize_t:
      *(size_t*)value_ = atol(value);
      return true;

    case Tcstring:
      *(const char**)value_ = removeQuotes(value);
      return true;

    default:
      break;
  }
  ShouldNotReachHere();
  return false;
}

#define DEFINE_RELEASE_FLAG_STRUCT(type, name, value, help) {#name, &name, T##type, true},
#define DEFINE_DEBUG_FLAG_STRUCT(type, name, value, help)                                          \
  {#name, RELEASE_ONLY(NULL) DEBUG_ONLY(&name), T##type, true},

Flag Flag::flags_[] = {RUNTIME_FLAGS(DEFINE_DEBUG_FLAG_STRUCT, DEFINE_RELEASE_FLAG_STRUCT,
                                     DEFINE_DEBUG_FLAG_STRUCT){NULL, NULL, Tinvalid, true}};

#undef DEFINE_DEBUG_FLAG_STRUCT
#undef DEFINE_RELEASE_FLAG_STRUCT

}  // namespace amd

#ifndef _WIN32
namespace amd::flags {
#endif
#define DEFINE_RELEASE_FLAG_VALUE(type, name, value, help) type name = value;
#define DEFINE_DEBUG_FLAG_VALUE(type, name, value, help) DEBUG_ONLY(type name = value);

RUNTIME_FLAGS(DEFINE_DEBUG_FLAG_VALUE, DEFINE_RELEASE_FLAG_VALUE, DEFINE_DEBUG_FLAG_VALUE);

#undef DEFINE_DEBUG_FLAG_VALUE
#undef DEFINE_RELEASE_FLAG_VALUE
#ifndef _WIN32
}
#endif /*!_WIN32*/
