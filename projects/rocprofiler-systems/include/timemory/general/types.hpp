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

/** \headerfile "timemory/general/types.hpp"
 */

#pragma once

#include <cstddef>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
class source_location;
//
template <size_t, size_t, typename... Args>
static inline auto&
get_static_source_location(Args&&... args);
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

#if !defined(TIMEMORY_SOURCE_LOCATION)

#    define _AUTO_LOCATION_COMBINE(X, Y) X##Y
#    define _AUTO_LOCATION(Y) _AUTO_LOCATION_COMBINE(timemory_source_location_, Y)

#    define TIMEMORY_SOURCE_LOCATION(MODE, ...)                                          \
        ::tim::source_location(MODE, __FUNCTION__, __LINE__, __FILE__, __VA_ARGS__)

#    define TIMEMORY_CAPTURE_MODE(MODE_TYPE) ::tim::source_location::mode::MODE_TYPE

#    define TIMEMORY_CAPTURE_ARGS(...) _AUTO_LOCATION(__LINE__).get_captured(__VA_ARGS__)

#    define TIMEMORY_INLINE_SOURCE_LOCATION(MODE, ...)                                   \
        ::tim::get_static_source_location<__LINE__, __COUNTER__>(                        \
            TIMEMORY_CAPTURE_MODE(MODE), __FUNCTION__, __LINE__, __FILE__, __VA_ARGS__)  \
            .get_captured(__VA_ARGS__)

#    define _TIM_STATIC_SRC_LOCATION(MODE, ...)                                          \
        static thread_local auto _AUTO_LOCATION(__LINE__) =                              \
            TIMEMORY_SOURCE_LOCATION(TIMEMORY_CAPTURE_MODE(MODE), __VA_ARGS__)

#endif
