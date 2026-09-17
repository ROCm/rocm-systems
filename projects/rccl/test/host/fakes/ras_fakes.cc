/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// client.cc (via ras-client-test.cc) calls rasTimeoutFactorSec(). That symbol
// lives in ras_param.cc, which librccl does not export and which this binary
// does not compile. Identity-factor stubs match NCCL_RAS_TIMEOUT_FACTOR=1.

#include <stdint.h>

extern "C" {

float ncclParamRasTimeoutFactor(void) { return 1.0f; }

int64_t rasTimeoutFactorNs(int64_t baseSeconds) {
  return baseSeconds * 1000000000LL;
}

double rasTimeoutFactorSec(int baseSeconds) { return (double)baseSeconds; }

} // extern "C"
