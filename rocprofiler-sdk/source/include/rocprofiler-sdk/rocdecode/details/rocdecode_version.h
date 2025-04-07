/*
Copyright (c) 2024 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ROCDECODE_VERSION_H
#define ROCDECODE_VERSION_H

/*!
 * \file
 * \brief rocDecode version
 * \defgroup group_rocdecode_version rocDecode Version
 * \brief rocDecode version
 */

#ifdef __cplusplus
extern "C" {
#endif
/* NOTE: Match version with CMakeLists.txt */
#define ROCDECODE_MAJOR_VERSION 0
#define ROCDECODE_MINOR_VERSION 10
#define ROCDECODE_MICRO_VERSION 0

/**
 * ROCDECODE_CHECK_VERSION:
 * @major: major version, like 1 in 1.2.3
 * @minor: minor version, like 2 in 1.2.3
 * @micro: micro version, like 3 in 1.2.3
 *
 * Evaluates to %TRUE if the version of rocDecode is greater than
 * @major, @minor and @micro
 */
#define ROCDECODE_CHECK_VERSION(major, minor, micro)                                               \
    (ROCDECODE_MAJOR_VERSION > (major) ||                                                          \
     (ROCDECODE_MAJOR_VERSION == (major) && ROCDECODE_MINOR_VERSION > (minor)) ||                  \
     (ROCDECODE_MAJOR_VERSION == (major) && ROCDECODE_MINOR_VERSION == (minor) &&                  \
      ROCDECODE_MICRO_VERSION >= (micro)))

#ifdef __cplusplus
}
#endif

#endif
