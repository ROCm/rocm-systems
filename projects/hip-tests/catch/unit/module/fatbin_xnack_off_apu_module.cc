/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
*/

/*
Fixture for the Unit_FatBin_XnackOffApu_* tests (test_fatbin_xnack_off_apu.cc).

CMake compiles it into several fat binaries with fixed gfx9 targets (explicit xnack-, generic,
generic with xnack+/xnack-). The kernel writes 2 if the code object was compiled with an explicit
xnack setting (clang then defines __amdgcn_feature_xnack__), else 1.
*/

#include <hip/hip_runtime.h>

extern "C" __global__ void xnack_kernel(int* out) {
#if defined(__amdgcn_feature_xnack__)
  *out = 2;
#else
  *out = 1;
#endif
}
