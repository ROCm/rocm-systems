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

#pragma once

#include <cstdio>
#include <cstdarg>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cxxabi.h>
// #include "exception.h"

namespace rocprofiler {

std::string string_vprintf(const char* format, va_list va);

std::string string_printf(const char* format, ...);

[[maybe_unused]] void warning(const char* format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif /* defined (__GNUC__) */
    ;
[[maybe_unused]] void fatal [[noreturn]] (const char* format, ...)
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif /* defined (__GNUC__) */
;

[[maybe_unused]] void warning(const char* format, ...);

[[maybe_unused]] void fatal [[noreturn]] (const char* format, ...);

/* The function extracts the kernel name from
input string. By using the iterators it finds the
window in the string which contains only the kernel name.
For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'*/
std::string truncate_name(const std::string& name);

// C++ symbol demangle
std::string cxx_demangle(const std::string& symbol);

// check if string has special char
bool has_special_char(std::string const& str);

// check if string has correct counter format
bool has_counter_format(std::string const& str);

// trims the begining of the line for spaces
std::string left_trim(const std::string& s);

}  // namespace rocprofiler
