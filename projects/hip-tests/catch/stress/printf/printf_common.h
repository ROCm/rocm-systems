/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _STRESSTEST_PRINTF_COMMON_H_
#define _STRESSTEST_PRINTF_COMMON_H_

#include <cstdlib>

#include <hip_test_common.hh>

// Best-effort heuristic: is there likely an interactive GPU compositor driving
// this session? The printf stress kernels hold every CU for several seconds; a
// compositor (e.g. GNOME Shell on Wayland/X11) trying to submit GFX work during
// that window can cause the kernel driver to interpret the stall as a hang and
// issue a GPU reset, killing the test.
//
// On Linux we treat the presence of DISPLAY / WAYLAND_DISPLAY as a likely
// compositor. This is intentionally loose and will also match SSH-with-X-
// forwarding or headless setups that happen to export these variables, which
// is why we only use it to *skip* the test, never to fail it. Users who know
// their environment is safe can bypass the check with
// HIP_PRINTF_STRESS_FORCE_RUN=1. Detection on other platforms is not
// implemented.
static inline bool gpuCompositorLikelyActive() {
  if (std::getenv("HIP_PRINTF_STRESS_FORCE_RUN")) return false;
#if defined(__linux__)
  return (std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr);
#else
  return false;
#endif
}

#define SKIP_IF_GPU_COMPOSITOR_ACTIVE()                                                         \
  if (gpuCompositorLikelyActive()) {                                                            \
    HipTest::HIP_SKIP_TEST(                                                                     \
        "likely running under a GPU compositor; long printf kernels may trigger a GPU reset. "  \
        "Run from a VT (chvt 3) or over SSH, or set HIP_PRINTF_STRESS_FORCE_RUN=1 to bypass."); \
    return;                                                                                     \
  }

#ifdef __linux__
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
#endif  // __linux__

#endif  // _STRESSTEST_PRINTF_COMMON_H_
