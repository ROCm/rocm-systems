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

#include "test_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

CUIDTstGlobals sCUIDGlvalues;

static void print_help() {
  printf(
      "amdcuid_test — CUID test suite\n"
      "\n"
      "Usage: amdcuid_test [options] [gtest options]\n"
      "\n"
      "Options:\n"
      "  -v, --verbose      Increase output verbosity (may be repeated)\n"
      "  -f, --dont_fail    Continue on assertion failures instead of "
      "aborting\n"
      "  -h, --help         Show this help message\n"
      "\n"
      "GoogleTest flags must use --gtest_*; all other unrecognised options are ignored.\n");
}

void ProcessCmdline(CUIDTstGlobals* globals, int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
      globals->verbosity++;
    } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--dont_fail") == 0) {
      globals->dont_fail = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_help();
      exit(0);
    }
    // Unrecognised flags are currently ignored (GoogleTest is initialized before ProcessCmdline()).
  }
}
