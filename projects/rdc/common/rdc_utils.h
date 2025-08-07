
/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef COMMON_RDC_UTILS_H_
#define COMMON_RDC_UTILS_H_

#include <string>

namespace amd {
namespace rdc {

#ifdef NDEBUG
#define debug_print(fmt, ...) \
  do {                        \
  } while (false)
#else
#define debug_print(fmt, ...)            \
  do {                                   \
    fprintf(stderr, fmt, ##__VA_ARGS__); \
  } while (false)
#endif

bool FileExists(char const* filename);

int ReadFile(std::string path, std::string* retStr, bool chop_newline = false);
int ReadFile(const char* path, std::string* retStr, bool chop_newline = false);

bool IsNumber(const std::string& s);
bool IsIP(const std::string& s);

}  // namespace rdc
}  // namespace amd

#endif  // COMMON_RDC_UTILS_H_
