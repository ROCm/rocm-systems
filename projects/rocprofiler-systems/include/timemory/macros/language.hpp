// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

//======================================================================================//
//
//      LANGUAGE
//
//======================================================================================//

// Define C++20
#if !defined(CXX20)
#    if __cplusplus > 201703L  // C++20
#        define CXX20
#    endif
#endif

//--------------------------------------------------------------------------------------//

// Define C++17
#if !defined(CXX17)
#    if __cplusplus > 201402L  // C++17
#        define CXX17
#    endif
#endif

//--------------------------------------------------------------------------------------//

// Define C++14
#if !defined(CXX14)
#    if __cplusplus > 201103L  // C++14
#        define CXX14
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(CXX14)
#    if !defined(TIMEMORY_WINDOWS)
#        error "timemory requires __cplusplus > 201103L (C++14)"
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(IF_CONSTEXPR)
#    if defined(CXX17)
#        define IF_CONSTEXPR(...) if constexpr(__VA_ARGS__)
#    else
#        define IF_CONSTEXPR(...) if(__VA_ARGS__)
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if defined(CXX17)
#    include <string_view>
//
#    if !defined(TIMEMORY_STRING_VIEW)
#        define TIMEMORY_STRING_VIEW 1
#    endif
//
namespace tim
{
using string_view_t = std::string_view;
}
//
#else
//
#    include <string>
//
#    if !defined(TIMEMORY_STRING_VIEW)
#        define TIMEMORY_STRING_VIEW 0
#    endif
//
namespace tim
{
using string_view_t = std::string;
}
//
#endif

//--------------------------------------------------------------------------------------//
