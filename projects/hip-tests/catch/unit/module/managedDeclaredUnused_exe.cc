/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
// Declares __managed__ variables that the program never reads or writes. The
// declarations alone are enough for __hipRegisterManagedVar to publish them, so
// StatCO::PromoteManagedVars backs all of them during hip::init regardless of use.
//
// Needs to be its own executable: managed variables are process-global, and every
// other managed test in the module binary touches one, which would promote the whole
// set through a path this test is trying to avoid.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

// Three sizes and alignments, so a promotion that mishandles one shape still leaves
// the others to report success.
__managed__ int g_unusedScalar = 42;
__managed__ int g_unusedArray[1024];
__managed__ double g_unusedDouble = 3.5;

namespace {

constexpr char kModeNoApi[] = "no-api";
constexpr char kModeTrivialApi[] = "trivial-api";

// Enters HIP through the cheapest APIs available. Neither touches a managed symbol,
// so a failure here is attributable to promotion rather than to the call itself.
int runTrivialApi() {
  int deviceCount = 0;
  hipError_t err = hipGetDeviceCount(&deviceCount);
  if (err != hipSuccess) {
    printf("hipGetDeviceCount failed: %s (%d)\n", hipGetErrorString(err), err);
    return -1;
  }

  int device = -1;
  err = hipGetDevice(&device);
  if (err != hipSuccess) {
    printf("hipGetDevice failed: %s (%d)\n", hipGetErrorString(err), err);
    return -1;
  }

  printf("OK: deviceCount=%d device=%d\n", deviceCount, device);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : kModeTrivialApi;

  if (std::strcmp(mode, kModeNoApi) == 0) {
    // Returning without entering HIP leaves the variables registered but unpromoted,
    // which is the state RemoveAllFatBinaries has to release at teardown.
    printf("OK: no HIP API called\n");
    return 0;
  }

  if (std::strcmp(mode, kModeTrivialApi) == 0) {
    return runTrivialApi();
  }

  printf("unknown mode '%s'\n", mode);
  return -1;
}
