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
//      Operating System
//
//======================================================================================//

#if defined(__x86_64__)
#    if !defined(TIMEMORY_64BIT)
#        define TIMEMORY_64BIT 1
#    endif
#else
#    if !defined(TIMEMORY_32BIT)
#        define TIMEMORY_32BIT 1
#    endif
#endif

//--------------------------------------------------------------------------------------//
// timemory prefixed os preprocessor definitions
//
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
#    if !defined(TIMEMORY_WINDOWS)
#        define TIMEMORY_WINDOWS 1
#    endif
#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(TIMEMORY_MACOS)
#        define TIMEMORY_MACOS 1
#    endif
#    if !defined(TIMEMORY_UNIX)
#        define TIMEMORY_UNIX 1
#    endif
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(TIMEMORY_LINUX)
#        define TIMEMORY_LINUX 1
#    endif
#    if !defined(TIMEMORY_UNIX)
#        define TIMEMORY_UNIX 1
#    endif
#elif defined(__unix__) || defined(__unix) || defined(unix)
#    if !defined(TIMEMORY_UNIX)
#        define TIMEMORY_UNIX 1
#    endif
#endif

//--------------------------------------------------------------------------------------//
// non-timemory prefixed os preprocessor definitions (deprecated)
//
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS 1
#    endif
#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(_MACOS)
#        define _MACOS 1
#    endif
#    if !defined(_UNIX)
#        define _UNIX 1
#    endif
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_LINUX)
#        define _LINUX 1
#    endif
#    if !defined(_UNIX)
#        define _UNIX 1
#    endif
#elif defined(__unix__) || defined(__unix) || defined(unix)
#    if !defined(_UNIX)
#        define _UNIX 1
#    endif
#endif

//--------------------------------------------------------------------------------------//
// this ensures that winnt.h never causes a 64-bit build to fail
// also solves issue with ws2def.h and winsock2.h:
//  https://www.zachburlingame.com/2011/05/
//      resolving-redefinition-errors-betwen-ws2def-h-and-winsock-h/
#if defined(_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    if !defined(WIN32_LEAN_AND_MEAN)
#        define WIN32_LEAN_AND_MEAN
#    endif
#    if !defined(WIN32)
#        define WIN32
#    endif
#    if defined(TIMEMORY_USE_WINSOCK)
#        include <WinSock2.h>
#    endif
#    include <Windows.h>
#endif

//--------------------------------------------------------------------------------------//
