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

/** \file timemory/utility/bits/macros.hpp
 * \headerfile timemory/utility/bits/macros.hpp "timemory/utility/bits/macros.hpp"
 * Deprecated macros
 */

//======================================================================================//
//
//                      ALL OF THESE MACROS ARE DEPRECATED!
//
//======================================================================================//

#pragma once

//======================================================================================//
//
//                      OBJECT MACROS
//
//======================================================================================//

/// \deprecated
#define TIMEMORY_BLANK_OBJECT(type, ...) TIMEMORY_BLANK_MARKER(type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

/// \deprecated
#define TIMEMORY_BASIC_OBJECT(type, ...) TIMEMORY_BASIC_MARKER(type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

/// \deprecated
#define TIMEMORY_OBJECT(type, ...) TIMEMORY_MARKER(type, __VA_ARGS__)

//======================================================================================//
//
//                      INSTANCE MACROS
//
//======================================================================================//

/// \deprecated
#define TIMEMORY_BLANK_INSTANCE(type, ...) TIMEMORY_BLANK_HANDLE(type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

/// \deprecated
#define TIMEMORY_BASIC_INSTANCE(type, ...) TIMEMORY_BASIC_HANDLE(type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

/// \deprecated
#define TIMEMORY_INSTANCE(type, ...) TIMEMORY_HANDLE(type, __VA_ARGS__)

//======================================================================================//
//
//                      AUTO_TUPLE MACROS
//
//======================================================================================//

//  DEPRECATED use macros in timemory/variadic/macros.hpp!
/// \deprecated
#define TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_BLANK_MARKER(auto_tuple_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_BASIC_MARKER(auto_tuple_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_AUTO_TUPLE(auto_tuple_type, ...)                                        \
    TIMEMORY_MARKER(auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// caliper versions -- DEPRECATED use macros in timemory/variadic/macros.hpp!

/// \deprecated
#define TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(id, auto_tuple_type, ...)                      \
    TIMEMORY_BLANK_CALIPER(id, auto_tuple_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_BASIC_AUTO_TUPLE_CALIPER(id, auto_tuple_type, ...)                      \
    TIMEMORY_BASIC_CALIPER(id, auto_tuple_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_AUTO_TUPLE_CALIPER(id, auto_tuple_type, ...)                            \
    TIMEMORY_CALIPER(id, auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// instance versions -- DEPRECATED use macros in timemory/variadic/macros.hpp!

/// \deprecated
#define TIMEMORY_BLANK_AUTO_TUPLE_INSTANCE(auto_tuple_type, ...)                         \
    TIMEMORY_BLANK_HANDLE(auto_tuple_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_BASIC_AUTO_TUPLE_INSTANCE(auto_tuple_type, ...)                         \
    TIMEMORY_BASIC_HANDLE(auto_tuple_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_AUTO_TUPLE_INSTANCE(auto_tuple_type, ...)                               \
    TIMEMORY_HANDLE(auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// debug versions -- DEPRECATED use macros in timemory/variadic/macros.hpp!

/// \deprecated
#define TIMEMORY_DEBUG_BASIC_AUTO_TUPLE(auto_tuple_type, ...)                            \
    TIMEMORY_DEBUG_BASIC_MARKER(auto_tuple_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_DEBUG_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_DEBUG_MARKER(auto_tuple_type, __VA_ARGS__)

//======================================================================================//
//
//                      AUTO_LIST MACROS
//
//======================================================================================//

/// \deprecated
#define TIMEMORY_BLANK_AUTO_LIST(auto_list_type, ...)                                    \
    TIMEMORY_BLANK_MARKER(auto_list_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_BASIC_AUTO_LIST(auto_list_type, ...)                                    \
    TIMEMORY_BASIC_MARKER(auto_list_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_AUTO_LIST(auto_list_type, ...)                                          \
    TIMEMORY_MARKER(auto_list_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// caliper versions

/// \deprecated
#define TIMEMORY_BLANK_AUTO_LIST_CALIPER(id, auto_list_type, ...)                        \
    TIMEMORY_BLANK_CALIPER(id, auto_list_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_BASIC_AUTO_LIST_CALIPER(id, auto_list_type, ...)                        \
    TIMEMORY_BASIC_CALIPER(id, auto_list_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_AUTO_LIST_CALIPER(id, auto_list_type, ...)                              \
    TIMEMORY_CALIPER(id, auto_list_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// instance versions

/// \deprecated
#define TIMEMORY_BLANK_AUTO_LIST_INSTANCE(auto_list_type, ...)                           \
    TIMEMORY_BLANK_HANDLE(auto_list_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_BASIC_AUTO_LIST_INSTANCE(auto_list_type, ...)                           \
    TIMEMORY_BASIC_HANDLE(auto_list_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_AUTO_LIST_INSTANCE(auto_list_type, ...)                                 \
    TIMEMORY_HANDLE(auto_list_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// debug versions

/// \deprecated
#define TIMEMORY_DEBUG_BASIC_AUTO_LIST(auto_list_type, ...)                              \
    TIMEMORY_DEBUG_BASIC_MARKER(auto_list_type, __VA_ARGS__)

/// \deprecated
#define TIMEMORY_DEBUG_AUTO_LIST(auto_list_type, ...)                                    \
    TIMEMORY_DEBUG_MARKER(auto_list_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

/// \deprecated
#define TIMEMORY_CALIPER_MARK_STREAM_BEGIN(id, stream)                                   \
    TIMEMORY_CALIPER_APPLY(id, mark_begin, stream)

//--------------------------------------------------------------------------------------//

/// \deprecated
#define TIMEMORY_CALIPER_MARK_STREAM_END(id, stream)                                     \
    TIMEMORY_CALIPER_APPLY(id, mark_end, stream)
