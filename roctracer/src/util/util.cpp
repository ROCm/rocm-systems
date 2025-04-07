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

#include "util.h"

#include <cstdio>
#include <cstdarg>
#include <string>

namespace roctracer {

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

}  // namespace roctracer
