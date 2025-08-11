/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ROCJPEG_VERSION_H
#define ROCJPEG_VERSION_H

/*!
 * \file
 * \brief rocJPEG version
 * \defgroup group_rocjpeg_version rocJPEG Version
 * \brief rocJPEG version
 */

#ifdef __cplusplus
extern "C" {
#endif
#define ROCJPEG_MAJOR_VERSION 0
#define ROCJPEG_MINOR_VERSION 6
#define ROCJPEG_MICRO_VERSION 0

/**
 * ROCJPEG_CHECK_VERSION:
 * @major: major version, like 1 in 1.2.3
 * @minor: minor version, like 2 in 1.2.3
 * @micro: micro version, like 3 in 1.2.3
 *
 * Evaluates to %TRUE if the version of rocJPEG is greater than
 * @major, @minor and @micro
 */
#define ROCJPEG_CHECK_VERSION(major, minor, micro)                                                 \
    (ROCJPEG_MAJOR_VERSION > (major) ||                                                            \
     (ROCJPEG_MAJOR_VERSION == (major) && ROCJPEG_MINOR_VERSION > (minor)) ||                      \
     (ROCJPEG_MAJOR_VERSION == (major) && ROCJPEG_MINOR_VERSION == (minor) &&                      \
      ROCJPEG_MICRO_VERSION >= (micro)))

#ifdef __cplusplus
}
#endif

#endif
