// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
