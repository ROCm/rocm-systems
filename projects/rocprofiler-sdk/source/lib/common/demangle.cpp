// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/demangle.hpp"
#include "lib/common/logging.hpp"

#include <cxxabi.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

namespace rocprofiler
{
namespace common
{
std::string
cxa_demangle(std::string_view _mangled_name, int* _status)
{
    // return the mangled since there is no buffer
    if(_mangled_name.empty())
    {
        *_status = -2;
        return std::string{};
    }

    auto _demangled_name = std::string{_mangled_name};

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be NULL; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    size_t _demang_len = 0;
    char*  _demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr, &_demang_len, _status);
    switch(*_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang) _demangled_name = std::string{_demang};
            break;
        }
        case -1:
        {
            ROCP_ERROR << "memory allocation failure occurred demangling " << _demangled_name;
            break;
        }
        case -2: break;
        case -3:
        {
            ROCP_ERROR << "Invalid argument in: (\"" << _demangled_name << "\", nullptr, nullptr, "
                       << _status << ")";
            break;
        }
        default: break;
    };

    // if it "demangled" but the length is zero, set the status to -2
    if(_demang_len == 0 && *_status == 0) *_status = -2;

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
}

// C++ symbol demangle
std::string
cxx_demangle(std::string_view symbol)
{
    int  _status       = 0;
    auto demangled_str = cxa_demangle(symbol, &_status);

    if(_status == 0) return demangled_str;

    return std::string{symbol};
}

// The function extracts the kernel name from
// input string. By using the iterators it finds the
// window in the string which contains only the kernel name.
// For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'
std::string
truncate_name(std::string_view name)
{
    auto     rit         = name.rbegin();
    auto     rend        = name.rend();
    uint32_t counter     = 0;
    char     open_token  = 0;
    char     close_token = 0;
    while(rit != rend)
    {
        if(counter == 0)
        {
            switch(*rit)
            {
                case ')':
                    counter     = 1;
                    open_token  = ')';
                    close_token = '(';
                    break;
                case '>':
                    counter     = 1;
                    open_token  = '>';
                    close_token = '<';
                    break;
                case ']':
                    counter     = 1;
                    open_token  = ']';
                    close_token = '[';
                    break;
                case ' ': ++rit; continue;
            }
            if(counter == 0) break;
        }
        else
        {
            if(*rit == open_token) counter++;
            if(*rit == close_token) counter--;
        }
        ++rit;
    }
    auto rbeg = rit;
    while((rit != rend) && (*rit != ' ') && (*rit != ':'))
        rit++;
    return std::string{name.substr(rend - rit, rit - rbeg)};
}
}  // namespace common
}  // namespace rocprofiler
