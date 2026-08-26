/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef CUID_TEST_COMMON_H_
#define CUID_TEST_COMMON_H_

#include <gtest/gtest.h>

#include "include/amd_cuid.h"

struct CUIDTstGlobals {
  uint32_t verbosity = 0;
  bool dont_fail = false;
};

extern CUIDTstGlobals sCUIDGlvalues;

// Conditionally execute a block at or above the given verbosity level.
#define IF_VERB(V) if (sCUIDGlvalues.verbosity >= (V))

// Assert that ret == AMDCUID_STATUS_SUCCESS, unless dont_fail is set.
#define CHK_ERR_ASRT(RET)                       \
  do {                                          \
    if (!sCUIDGlvalues.dont_fail) {             \
      ASSERT_EQ(AMDCUID_STATUS_SUCCESS, (RET)); \
    }                                           \
  } while (0)

void ProcessCmdline(CUIDTstGlobals* globals, int argc, char** argv);

#endif  // CUID_TEST_COMMON_H_
