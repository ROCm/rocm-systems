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

#include "timemory/utility/macros.hpp"

#if defined(TIMEMORY_CORE_SOURCE)
#    define TIMEMORY_ENVIRONMENT_SOURCE
#elif defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_ENVIRONMENT_EXTERN
#endif
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_ENVIRONMENT_EXTERN)
#    define TIMEMORY_USE_ENVIRONMENT_EXTERN
#endif

#if defined(TIMEMORY_ENVIRONMENT_SOURCE)
#    define TIMEMORY_ENVIRONMENT_LINKAGE(...) __VA_ARGS__
#elif defined(TIMEMORY_USE_ENVIRONMENT_EXTERN)
#    define TIMEMORY_ENVIRONMENT_LINKAGE(...) extern __VA_ARGS__
#else
#    define TIMEMORY_ENVIRONMENT_LINKAGE(...) inline __VA_ARGS__
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_ENVIRONMENT_EXTERN_TEMPLATE)
#    if defined(TIMEMORY_ENVIRONMENT_SOURCE)
#        define TIMEMORY_ENVIRONMENT_EXTERN_TEMPLATE(TYPE)                               \
            namespace tim                                                                \
            {                                                                            \
            template TYPE get_env<TYPE>(const std::string&, TYPE);                       \
            template void set_env<TYPE>(const std::string&, const TYPE&, int);           \
            }
#    elif defined(TIMEMORY_USE_ENVIRONMENT_EXTERN)
#        define TIMEMORY_ENVIRONMENT_EXTERN_TEMPLATE(TYPE)                               \
            namespace tim                                                                \
            {                                                                            \
            extern template TYPE get_env<TYPE>(const std::string&, TYPE);                \
            extern template void set_env<TYPE>(const std::string&, const TYPE&, int);    \
            }
#    endif
#else
#    define TIMEMORY_ENVIRONMENT_EXTERN_TEMPLATE(...)
#endif
//
//======================================================================================//
