/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_CONSTANTS_HPP_
#define LIBRARY_SRC_CONSTANTS_HPP_

/**
 * @file constants.hpp
 *
 * @brief Contains global constants for rocSHMEM library
 */

namespace rocshmem {

/**
 * @brief Minimum object alignment for symmetric heap.
 *
 * @note Cache line size on most systems is either 64 or 128.
 */
inline const unsigned ALIGNMENT{128};

/**
 * @brief Constant number which holds maximum workgroup size.
 *
 * @todo Remove this member from this class. It belongs in a class
 * that specifically holds device hardware information. If this
 * device class existed, we could consolidate the various flavours of
 * the Instinct cards into their own groups and then set these
 * hard-coded fields by querying the rocm runtime during our library
 * initialization.
 */
inline const unsigned MAX_WG_SIZE{1024};

/**
 * @brief Constant number which holds the wavefront size
 *
 * @note Wavefront size on most systems is either 32 or 64.
 */
#if defined(__gfx90a__) || defined(__gfx942__) || defined (__gfx950__)
inline const int WF_SIZE{64};
#else
inline const int WF_SIZE{32};
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONSTANTS_HPP_
