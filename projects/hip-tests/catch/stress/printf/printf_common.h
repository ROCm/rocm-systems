/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _STRESSTEST_PRINTF_COMMON_H_
#define _STRESSTEST_PRINTF_COMMON_H_

#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

struct CaptureStream {
  int saved_fd;
  int orig_fd;
  int temp_fd;

  char tempname[13] = "mytestXXXXXX";

  explicit CaptureStream(FILE* original) {
    orig_fd = fileno(original);
    saved_fd = dup(orig_fd);

    if ((temp_fd = mkstemp(tempname)) == -1) {
      error(0, errno, "Error");
      assert(false);
    }

    fflush(nullptr);
    if (dup2(temp_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(temp_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  void restoreStream() {
    if (saved_fd == -1) return;
    fflush(nullptr);
    if (dup2(saved_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(saved_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
    saved_fd = -1;
  }

  const char* getTempFilename() { return (const char*)tempname; }

  std::ifstream getCapturedData() {
    restoreStream();
    std::ifstream temp(tempname);
    return temp;
  }

  ~CaptureStream() {
    restoreStream();
    if (remove(tempname) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }
};

#endif  // _STRESSTEST_PRINTF_COMMON_H_
