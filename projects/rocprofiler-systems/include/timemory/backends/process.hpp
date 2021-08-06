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

/** \file backends/process.hpp
 * \headerfile backends/process.hpp "timemory/backends/process.hpp"
 * Defines process backend functions
 *
 */

#pragma once

#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <cstdio>

#if defined(TIMEMORY_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(TIMEMORY_MACOS)
#        include <libproc.h>
#        include <mach/mach.h>
#    endif
#elif defined(TIMEMORY_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    if !defined(WIN32_LEAN_AND_MEAN)
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

namespace tim
{
namespace process
{
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_UNIX)
using id_t = pid_t;
#elif defined(TIMEMORY_WINDOWS)
using id_t = DWORD;
#else
using id_t = int;
#endif
//
//--------------------------------------------------------------------------------------//
//
/// \fn pid_t tim::process::get_id()
/// \brief get the process id of this process
///
inline id_t
get_id()
{
#if defined(TIMEMORY_WINDOWS)
    static auto instance = GetCurrentProcessId();
#elif defined(TIMEMORY_UNIX)
    static auto instance = getpid();
#else
    static auto instance = 0;
#endif
    return instance;
}
//
//--------------------------------------------------------------------------------------//
//
/// \fn pid_t& tim::process::get_target_id()
/// \brief get the target process id of this process (may differ from process::get_id)
///
inline id_t&
get_target_id()
{
#if defined(TIMEMORY_WINDOWS)
    static auto instance = GetCurrentProcessId();
#elif defined(TIMEMORY_UNIX)
    static auto instance = getpid();
#else
    static auto instance = 0;
#endif
    return instance;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace process
}  // namespace tim
