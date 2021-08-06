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

#include "timemory/mpl/concepts.hpp"

/// \macro TIMEMORY_DECLARE_NS_API(NS, NAME)
/// \brief Declare an API category. APIs are used to designate
/// different project implementations, different external library tools, etc.
///
#if !defined(TIMEMORY_DECLARE_NS_API)
#    define TIMEMORY_DECLARE_NS_API(NS, NAME)                                            \
        namespace tim                                                                    \
        {                                                                                \
        namespace NS                                                                     \
        {                                                                                \
        struct NAME;                                                                     \
        }                                                                                \
        }
#endif

#if !defined(TIMEMORY_DECLARE_API)
#    define TIMEMORY_DECLARE_API(NAME) TIMEMORY_DECLARE_NS_API(api, NAME)
#endif

//
/// \macro TIMEMORY_DEFINE_NS_API(NS, NAME)
/// \param NS sub-namespace within tim::
/// \param NAME the name of the API
///
/// \brief Define an API category within a namespace
///
#if !defined(TIMEMORY_DEFINE_NS_API)
#    define TIMEMORY_DEFINE_NS_API(NS, NAME)                                             \
        namespace tim                                                                    \
        {                                                                                \
        namespace NS                                                                     \
        {                                                                                \
        struct NAME : public concepts::api                                               \
        {};                                                                              \
        }                                                                                \
        }
#endif

///
/// \macro TIMEMORY_DEFINE_API
/// \brief Define an API category. APIs are used to designate
/// different project implementations, different external library tools, etc.
/// Note: this macro inherits from \ref concepts::api instead of specializing
/// is_api<...>, thus allowing specialization from tools downstream
///
#if !defined(TIMEMORY_DEFINE_API)
#    define TIMEMORY_DEFINE_API(NAME) TIMEMORY_DEFINE_NS_API(api, NAME)
#endif
