/* This content was copied and modified from
 * the HDF5 C library's src/H5warnings.h file
 *
 * (https://github.com/HDFGroup/hdf5)
 *
 * HDF5 copyright:
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the LICENSE file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * AMD changes:
 *
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
 * Macros for suppressing warnings
 *
 * These macros can be used to suppress compiler warnings that are difficult
 * or impossible to engineer around. By suppressing these warnings, we make
 * it easier to identify warnings created by new code and can build with
 * -Werror in CI to ensure new warnings don't get added to the code.
 *
 * USAGE:
 *
 * The macros are used in ON/OFF pairs. (i.e. H5_WARN_FOO_(ON|OFF)). To
 * suppress a warning, add the OFF macro before the offending line(s) of
 * code, and to turn it back on, add the ON macro.
 *
 *      HIPFILE_WARN_FOO_OFF
 *      HIPFILE_WARN_BAR_OFF
 *      code_that_raises_warnings();
 *      HIPFILE_WARN_BAR_ON
 *      HIPFILE_WARN_FOO_ON
 *
 * Since the warning macros work by pushing and popping diagnostic contexts,
 * warnings should be switched back on in reverse order. In practice, the ON
 * macro is just a generic pop, so it doesn't matter which one gets used or
 * in what order they get called, but using the wrong macro will make it
 * harder to reason about the code.
 *
 * Suppression macro pairs should span the minimum amount of code that
 * quiets the warning. Ideally, a single line of code. Compilers in the
 * past limited diagnostic pragmas to outside of functions, but this
 * is no longer the case in any compiler we care about.
 *
 * LIMITATIONS:
 *
 * The warning macros use compiler pragmas, which limits where the suppression
 * macros can be used. The most obvious limitation is that they often can't
 * be used inside macros.
 *
 * ADDING A NEW MACRO:
 *
 * - First, ask yourself if we really need a new warning macro. Do your best
 *   to actually correct warnings and not just suppress them. If a compiler
 *   update starts raising a bunch of new warnings we are unlikely to fix,
 *   consider shutting down the warning flag globally using the compiler
 *   flags (like -Wno-foo-warning).
 *
 * - The macro should have a helpful comment about why the warning should
 *   be suppressed and not corrected.
 *
 * - The macro name should reflect the actual problem, not the name of the
 *   compiler warning option. Names are of the form HIPFILE_WARN_<THING>_(ON|OFF).
 *
 * - Be careful with the ifdefs. clang defines __GNUC__, for example, so
 *   you can't simply check for that if you have a gcc-specific warning.
 *   Check for compiler version numbers in the macros to avoid warnings
 *   about undefined diagnostics in older compilers.
 *
 * - Add the new macro to the list in the .clang-format file.
 */

/* Macros for enabling/disabling particular gcc / clang warnings
 *
 * (see the following web-sites for more info:
 *      http://www.dbp-consulting.com/tutorials/SuppressingGCCWarnings.html
 *      http://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html#Diagnostic-Pragmas
 *
 * _Pragma is C++11/C99 and should work with all compilers, though permitted
 * placement may vary
 *
 * "GCC" in the pragma works for both clang/llvm and gcc
 */
#define HIPFILE_WARN_JOINSTR(x, y) x y
#define HIPFILE_WARN_DO_PRAGMA(x)  _Pragma(#x)
#define HIPFILE_WARN_PRAGMA(x)     HIPFILE_WARN_DO_PRAGMA(GCC diagnostic x)

/* clang-format off */
#if (defined(__GNUC__) || defined(__clang__))
    #define HIPFILE_WARN_OFF(x) HIPFILE_WARN_PRAGMA(push) HIPFILE_WARN_PRAGMA(ignored HIPFILE_WARN_JOINSTR("-W", x))
    #define HIPFILE_WARN_ON(x)  HIPFILE_WARN_PRAGMA(pop)
#endif
/* clang-format on */

/*********************
 * SPECIFIC WARNINGS *
 *********************/

/* Suppress warnings about using format strings that aren't string
 * literals
 */
#if defined(__clang__) || defined(__GNUC__)
#define HIPFILE_WARN_FORMAT_NONLITERAL_OFF HIPFILE_WARN_OFF("format-nonliteral")
#define HIPFILE_WARN_FORMAT_NONLITERAL_ON  HIPFILE_WARN_ON("format-nonliteral")
#else
#define HIPFILE_WARN_FORMAT_NONLITERAL_OFF
#define HIPFILE_WARN_FORMAT_NONLITERAL_ON
#endif

/* Suppress warnings about missing exit destructors */
#if defined(__clang__)
#define HIPFILE_WARN_NO_EXIT_DTOR_OFF HIPFILE_WARN_OFF("exit-time-destructors")
#define HIPFILE_WARN_NO_EXIT_DTOR_ON  HIPFILE_WARN_ON("exit-time-destructors")
#else
#define HIPFILE_WARN_NO_EXIT_DTOR_OFF
#define HIPFILE_WARN_NO_EXIT_DTOR_ON
#endif

/* Suppress warnings about missing global constructors */
#if defined(__clang__)
#define HIPFILE_WARN_NO_GLOBAL_CTOR_OFF HIPFILE_WARN_OFF("global-constructors")
#define HIPFILE_WARN_NO_GLOBAL_CTOR_ON  HIPFILE_WARN_ON("global-constructors")
#else
#define HIPFILE_WARN_NO_GLOBAL_CTOR_OFF
#define HIPFILE_WARN_NO_GLOBAL_CTOR_ON
#endif
