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

#define _TIM_STRINGIZE(X) _TIM_STRINGIZE2(X)
#define _TIM_STRINGIZE2(X) #X
#define _TIM_VAR_NAME_COMBINE(X, Y) X##Y
#define _TIM_VARIABLE(Y) _TIM_VAR_NAME_COMBINE(timemory_variable_, Y)
#define _TIM_SCOPE_DTOR(Y) _TIM_VAR_NAME_COMBINE(timemory_scoped_dtor_, Y)
#define _TIM_TYPEDEF(Y) _TIM_VAR_NAME_COMBINE(timemory_typedef_, Y)
#define _TIM_STORAGE_INIT(Y) _TIM_VAR_NAME_COMBINE(timemory_storage_initializer_, Y)

#define _TIM_LINESTR _TIM_STRINGIZE(__LINE__)

#if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#    define _TIM_FUNC __PRETTY_FUNCTION__
#elif defined(TIMEMORY_PRETTY_FUNCTION) && defined(_WINDOWS)
#    define _TIM_FUNC __FUNCSIG__
#else
#    define _TIM_FUNC __FUNCTION__
#endif

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED)
#    if !defined(TIMEMORY_SPRINTF)
#        define TIMEMORY_SPRINTF(...)
#    endif
#else
#    if !defined(TIMEMORY_SPRINTF)
#        define TIMEMORY_SPRINTF(VAR, LEN, FMT, ...)                                     \
            char VAR[LEN];                                                               \
            snprintf(VAR, LEN, FMT, __VA_ARGS__);
#    endif
#endif

#if !defined(TIMEMORY_CODE)
#    if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED)
#        define TIMEMORY_CODE(...)
#    else
#        define TIMEMORY_CODE(...) __VA_ARGS__
#    endif
#endif

//======================================================================================//
//
//      Operating System
//
//======================================================================================//

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

//======================================================================================//
//
//      General attribute
//
//======================================================================================//

#if !defined(TIMEMORY_ATTRIBUTE)
#    if !defined(_MSC_VER)
#        define TIMEMORY_ATTRIBUTE(attr) __attribute__((attr))
#    else
#        define TIMEMORY_ATTRIBUTE(attr) __declspec(attr)
#    endif
#endif

//======================================================================================//
//
//      Windows DLL settings
//
//======================================================================================//

#if !defined(TIMEMORY_CDLL)
#    if defined(_WINDOWS)
#        if defined(TIMEMORY_CDLL_EXPORT)
#            define TIMEMORY_CDLL __declspec(dllexport)
#        elif defined(TIMEMORY_CDLL_IMPORT)
#            define TIMEMORY_CDLL __declspec(dllimport)
#        else
#            define TIMEMORY_CDLL
#        endif
#    else
#        define TIMEMORY_CDLL
#    endif
#endif

#if !defined(TIMEMORY_DLL)
#    if defined(_WINDOWS)
#        if defined(TIMEMORY_DLL_EXPORT)
#            define TIMEMORY_DLL __declspec(dllexport)
#        elif defined(TIMEMORY_DLL_IMPORT)
#            define TIMEMORY_DLL __declspec(dllimport)
#        else
#            define TIMEMORY_DLL
#        endif
#    else
#        define TIMEMORY_DLL
#    endif
#endif

//======================================================================================//
//
//      Visibility
//
//======================================================================================//

#if !defined(TIMEMORY_VISIBILITY)
#    if !defined(_MSC_VER)
#        define TIMEMORY_VISIBILITY(mode) TIMEMORY_ATTRIBUTE(visibility(mode))
#    else
#        define TIMEMORY_VISIBILITY(mode)
#    endif
#endif

#if !defined(TIMEMORY_VISIBLE)
#    define TIMEMORY_VISIBLE TIMEMORY_VISIBILITY("default")
#endif

#if !defined(TIMEMORY_HIDDEN)
#    define TIMEMORY_HIDDEN TIMEMORY_VISIBILITY("hidden")
#endif

//======================================================================================//
//
//      Instrumentation
//
//======================================================================================//

#if !defined(_WINDOWS)
#    if !defined(TIMEMORY_NEVER_INSTRUMENT)
#        if defined(__clang__)
#            define TIMEMORY_NEVER_INSTRUMENT                                            \
                TIMEMORY_ATTRIBUTE(no_instrument_function)                               \
                TIMEMORY_ATTRIBUTE(xray_never_instrument)
#        else
#            define TIMEMORY_NEVER_INSTRUMENT TIMEMORY_ATTRIBUTE(no_instrument_function)
#        endif
#    endif
//
#    if !defined(TIMEMORY_INSTRUMENT)
#        define TIMEMORY_INSTRUMENT TIMEMORY_ATTRIBUTE(xray_always_instrument)
#    endif
#else
#    if !defined(TIMEMORY_NEVER_INSTRUMENT)
#        define TIMEMORY_NEVER_INSTRUMENT
#    endif
//
#    if !defined(TIMEMORY_INSTRUMENT)
#        define TIMEMORY_INSTRUMENT
#    endif
#endif

//======================================================================================//
//
//      Symbol override
//
//======================================================================================//

#if !defined(TIMEMORY_WEAK_PREFIX)
#    if !defined(_MSC_VER)
#        if defined(__clang__) && defined(__APPLE__)
#            define TIMEMORY_WEAK_PREFIX
#        else
#            define TIMEMORY_WEAK_PREFIX TIMEMORY_ATTRIBUTE(weak)
#        endif
#    else
#        define TIMEMORY_WEAK_PREFIX
#    endif
#endif

#if !defined(TIMEMORY_WEAK_POSTFIX)
#    if !defined(_MSC_VER)
#        if defined(__clang__) && defined(__APPLE__)
#            define TIMEMORY_WEAK_POSTFIX TIMEMORY_ATTRIBUTE(weak_import)
#        else
#            define TIMEMORY_WEAK_POSTFIX
#        endif
#    else
#        define TIMEMORY_WEAK_POSTFIX
#    endif
#endif

//======================================================================================//
//
//      Library Constructor/Destructor
//
//======================================================================================//

#if !defined(TIMEMORY_CTOR)
#    if !defined(_WINDOWS)
#        define TIMEMORY_CTOR TIMEMORY_ATTRIBUTE(constructor)
#    else
#        define TIMEMORY_CTOR
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_DTOR)
#    if !defined(_WINDOWS)
#        define TIMEMORY_DTOR TIMEMORY_ATTRIBUTE(destructor)
#    else
#        define TIMEMORY_DTOR
#    endif
#endif

//======================================================================================//
//
//      WINDOWS WARNINGS (apply to C code)
//
//======================================================================================//

//  MSVC compiler
#if defined(_MSC_VER) && _MSC_VER > 0 && !defined(TIMEMORY_MSVC_COMPILER)
#    define TIMEMORY_MSVC_COMPILER
#endif

#if defined(TIMEMORY_MSVC_COMPILER) && !defined(TIMEMORY_MSVC_WARNINGS)

#    pragma warning(disable : 4996)  // function may be unsafe
#    pragma warning(disable : 5105)  // macro produce 'defined' has undefined behavior

#endif
