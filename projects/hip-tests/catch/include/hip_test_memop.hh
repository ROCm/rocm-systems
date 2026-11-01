/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdlib>
#include <cstring>

namespace HipTest {

// A single stream WAIT capability does not imply batch memop support: PAL
// implements the former, but not the latter (including captured stream ops).
// HIP has no public batch capability query. Follow explicit backend selection
// and platform defaults; pin auto-selection to GPU_ENABLE_PAL=0 to run these
// ROCr-only tests rather than assuming that an automatic fallback supports them.
inline bool IsRocrMemOpBackendSelected() {
  const char* pal = std::getenv("GPU_ENABLE_PAL");
  if (pal != nullptr && pal[0] != '\0') return std::strcmp(pal, "0") == 0;
#if defined(_WIN32)
  return false;  // Windows defaults to PAL; explicit ROCr above is supported.
#else
  return true;  // Linux defaults to ROCr.
#endif
}

}  // namespace HipTest
