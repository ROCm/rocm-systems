//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory/variadic/auto_macros.hpp
 * \headerfile timemory/variadic/auto_macros.hpp "timemory/variadic/macros.hpp"
 * Generic macros that are intended to be building-blocks for other macros, e.g.
 * TIMEMORY_MARKER, TIMEMORY_HANDLE, TIMEMORY_CALIPER, etc.
 *
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/compat/macros.h"
#include "timemory/general/source_location.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

//======================================================================================//
//
//                      CXX variadic macros
//
//======================================================================================//

namespace tim
{
// e.g. tim::string::join(...)
using string = tim::mpl::apply<std::string>;
}  // namespace tim

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_MACROS)

#    define TIMEMORY_MACROS

//======================================================================================//
//
//                      HELPER MACROS
//
//======================================================================================//

#    if !defined(TIMEMORY_WINDOWS)
#        define TIMEMORY_OS_PATH_DELIMITER '/'
#    else
#        define TIMEMORY_OS_PATH_DELIMITER '\\'
#    endif

//--------------------------------------------------------------------------------------//

#    if defined(__FILE_NAME__)
#        define _TIM_FILESTR __FILE_NAME__
#    else
#        define _TIM_FILESTR                                                             \
            std::string(__FILE__).substr(                                                \
                std::string(__FILE__).find_last_of(TIMEMORY_OS_PATH_DELIMITER) + 1)
#    endif

//--------------------------------------------------------------------------------------//

#    define _TIM_FILELINE ::tim::string::join(':', _TIM_FILESTR, _TIM_LINESTR)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_JOIN(delim, ...) ::tim::string::join(delim, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BLANK_LABEL(...) ::tim::string::join("", __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_LABEL(...) ::tim::string::join("", _TIM_FUNC, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_FULL_LABEL ::tim::string::join('/', _TIM_FUNC, _TIM_FILELINE)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_LABEL(...) TIMEMORY_JOIN("", TIMEMORY_FULL_LABEL, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_AUTO_TYPE(TYPE) ::tim::concepts::auto_type_t<TYPE>

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_COMP_TYPE(TYPE) ::tim::concepts::component_type_t<TYPE>

//======================================================================================//
//
//                      MARKER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_MARKER(TYPE, ...)                                             \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        TIMEMORY_AUTO_TYPE(TYPE)                                                         \
        _TIM_VARIABLE(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_MARKER(TYPE, ...)                                             \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        TIMEMORY_AUTO_TYPE(TYPE)                                                         \
        _TIM_VARIABLE(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_MARKER(TYPE, ...)                                                   \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        TIMEMORY_AUTO_TYPE(TYPE)                                                         \
        _TIM_VARIABLE(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//======================================================================================//
//
//                      CONDITIONAL MARKER MACROS
//
//======================================================================================//

#    define TIMEMORY_CONDITIONAL_BLANK_MARKER(COND, TYPE, ...)                           \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        std::unique_ptr<TIMEMORY_AUTO_TYPE(TYPE)> _TIM_VARIABLE(__LINE__) =              \
            std::unique_ptr<TIMEMORY_AUTO_TYPE(TYPE)>(                                   \
                ((COND))                                                                 \
                    ? new TIMEMORY_AUTO_TYPE(TYPE)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))   \
                    : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CONDITIONAL_BASIC_MARKER(COND, TYPE, ...)                           \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        std::unique_ptr<TIMEMORY_AUTO_TYPE(TYPE)> _TIM_VARIABLE(__LINE__) =              \
            std::unique_ptr<TIMEMORY_AUTO_TYPE(TYPE)>(                                   \
                ((COND))                                                                 \
                    ? new TIMEMORY_AUTO_TYPE(TYPE)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))   \
                    : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CONDITIONAL_MARKER(COND, TYPE, ...)                                 \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        std::unique_ptr<TIMEMORY_AUTO_TYPE(TYPE)> _TIM_VARIABLE(__LINE__) =              \
            std::unique_ptr<TIMEMORY_AUTO_TYPE(TYPE)>(                                   \
                ((COND))                                                                 \
                    ? new TIMEMORY_AUTO_TYPE(TYPE)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))   \
                    : nullptr)

//======================================================================================//
//
//                      POINTER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_POINTER(TYPE, ...)                                            \
        TIMEMORY_CONDITIONAL_BLANK_MARKER(::tim::settings::enabled(), TYPE, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_POINTER(TYPE, ...)                                            \
        TIMEMORY_CONDITIONAL_BASIC_MARKER(::tim::settings::enabled(), TYPE, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_POINTER(TYPE, ...)                                                  \
        TIMEMORY_CONDITIONAL_MARKER(::tim::settings::enabled(), TYPE, __VA_ARGS__)

//======================================================================================//
//
//                      CALIPER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_CALIPER(ID, TYPE, ...)                                        \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        TYPE _TIM_VARIABLE(ID)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_CALIPER(ID, TYPE, ...)                                        \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        TYPE _TIM_VARIABLE(ID)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER(ID, TYPE, ...)                                              \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        TYPE _TIM_VARIABLE(ID)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BLANK_CALIPER(ID, TYPE, ...)                                 \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        static TYPE _TIM_VARIABLE(ID)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BASIC_CALIPER(ID, TYPE, ...)                                 \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        static TYPE _TIM_VARIABLE(ID)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_CALIPER(ID, TYPE, ...)                                       \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        static TYPE _TIM_VARIABLE(ID)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_REFERENCE(ID) std::ref(_TIM_VARIABLE(ID)).get()

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_APPLY(ID, FUNC, ...) _TIM_VARIABLE(ID).FUNC(__VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_APPLY(ID, TYPE, FUNC, ...)                             \
        _TIM_VARIABLE(ID).type_apply<TYPE>(FUNC, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_APPLY0(ID, FUNC) _TIM_VARIABLE(ID).FUNC()

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_APPLY0(ID, TYPE, FUNC)                                 \
        _TIM_VARIABLE(ID).type_apply<TYPE>(FUNC)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_LAMBDA(ID, LAMBDA, ...)                                     \
        LAMBDA(_TIM_VARIABLE(ID), __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_LAMBDA(ID, TYPE, LAMBDA, ...)                          \
        LAMBDA(_TIM_VARIABLE(ID).get<TYPE>(), __VA_ARGS__)

//======================================================================================//
//
//                      HANDLE MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_HANDLE(TYPE, ...)                                             \
        TYPE(TIMEMORY_INLINE_SOURCE_LOCATION(blank, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_HANDLE(TYPE, ...)                                             \
        TYPE(TIMEMORY_INLINE_SOURCE_LOCATION(basic, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_HANDLE(TYPE, ...)                                                   \
        TYPE(TIMEMORY_INLINE_SOURCE_LOCATION(full, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BLANK_RAW_POINTER(TYPE, ...)                                        \
        (::tim::settings::enabled())                                                     \
            ? new TYPE(TIMEMORY_INLINE_SOURCE_LOCATION(blank, __VA_ARGS__))              \
            : nullptr

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_RAW_POINTER(TYPE, ...)                                        \
        (::tim::settings::enabled())                                                     \
            ? new TYPE(TIMEMORY_INLINE_SOURCE_LOCATION(basic, __VA_ARGS__))              \
            : nullptr

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_RAW_POINTER(TYPE, ...)                                              \
        (::tim::settings::enabled())                                                     \
            ? new TYPE(TIMEMORY_INLINE_SOURCE_LOCATION(full, __VA_ARGS__))               \
            : nullptr

//======================================================================================//
//
//                      DEBUG MACROS
//
//======================================================================================//

#    if defined(DEBUG)

//--------------------------------------------------------------------------------------//

#        define TIMEMORY_DEBUG_BLANK_MARKER(TYPE, ...)                                   \
            TIMEMORY_BASIC_MARKER(TYPE, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#        define TIMEMORY_DEBUG_BASIC_MARKER(TYPE, ...)                                   \
            TIMEMORY_BASIC_MARKER(TYPE, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#        define TIMEMORY_DEBUG_MARKER(TYPE, ...) TIMEMORY_MARKER(TYPE, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    else
#        define TIMEMORY_DEBUG_BLANK_MARKER(TYPE, ...)
#        define TIMEMORY_DEBUG_BASIC_MARKER(TYPE, ...)
#        define TIMEMORY_DEBUG_MARKER(TYPE, ...)
#    endif

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CONFIGURE(TYPE, ...) TYPE::configure(__VA_ARGS__)

//--------------------------------------------------------------------------------------//

// deprecated macros
#    include "timemory/utility/bits/macros.hpp"

//--------------------------------------------------------------------------------------//

#endif  // !defined(TIMEMORY_MACROS)

//--------------------------------------------------------------------------------------//
