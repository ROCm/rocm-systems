/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "getopt.h"

#include <string.h>

char* optarg = nullptr;
int optind = 1;

int getopt(int argc, char* const argv[], const char* optstring) {
  if ((optind >= argc) || (argv[optind][0] != '-') || (argv[optind][0] == 0)) {
    return -1;
  }

  int opt = argv[optind][1];
  const char* p = strchr(optstring, opt);

  if (p == nullptr) {
    return '?';
  }
  if (p[1] == ':') {
    optind++;
    if (optind >= argc) {
      return '?';
    }
    optarg = argv[optind];
  }

  optind++;
  return opt;
}
