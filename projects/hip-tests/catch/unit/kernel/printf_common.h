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
#include <fstream>
#include <iostream>
#include <string>

class CaptureStream {
 public:
  CaptureStream() {
    if (pipe2(pipe, 0) == -1) {
      error(0, errno, "Error");
      assert(false);
    }

    orig_fd = dup(fileno(stdout));
    if (orig_fd == -1) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  ~CaptureStream() {
    close(orig_fd);
    close(pipe[READ]);
    close(pipe[WRITE]);
  }

  void beginCapture() {
    fflush(stdout);
    if (dup2(pipe[WRITE], fileno(stdout)) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  void endCapture() {
    // End Capture
    fflush(stdout);
    if (dup2(orig_fd, fileno(stdout)) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    result.clear();

    // Get string
    const int bufSize = 2048;
    char buf[bufSize];
    int bytesRead = 0;
    bytesRead = read(pipe[READ], &buf, bufSize);
    
    if (bytesRead < 0) {
      error(0, errno, "Error reading from pipe");
      assert(false);
      result.clear();
    } else {
      result = std::string(buf, bytesRead);
    }
  }

  std::string getCapturedData() { return result; }

 private:
  enum PIPES { READ, WRITE };
  int pipe[2] = {0, 0};
  int orig_fd;
  std::string result;
};

#endif  // _STRESSTEST_PRINTF_COMMON_H_
