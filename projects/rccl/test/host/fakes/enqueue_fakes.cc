/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See enqueue_fakes.h. This file holds only the reset chain for the
// `rccl-UnitTestsMicroEnqueue` binary; every seam it resets is defined in a
// fakes file named after the production TU that owns the symbol.

#include "enqueue_fakes.h"

void ResetEnqueueFakes() {
  ResetHipFakes();
  ResetNcclFakes();
  ResetCeFakes();
  ResetCommFakes();
  ResetDevRuntimeFakes();
  ResetProxyFakes();
  ResetRcclWrapFakes();
  ResetRecorderFakes();
  ResetSymKernelsFakes();
  ResetTransportStubs();
  ResetTuningFakes();
  ResetEnvFakes();

  // enqueue.cc has raw libc getenv() call sites, not just ncclGetEnv ones
  // (topoGetAlgoInfo:2687-2688 reads NCCL_PROTO and NCCL_ALGO that way). The
  // interposer in env_fakes.cc only intercepts names the map KNOWS; an unmapped
  // name falls through to the real environment. Mapping them absent is what
  // makes the suite hermetic against the CI machine's ambient environment
  // rather than merely capable of being made so.
  SetMicroEnvAbsent("NCCL_PROTO");
  SetMicroEnvAbsent("NCCL_ALGO");
}
