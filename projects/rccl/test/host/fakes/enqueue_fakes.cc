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
  ClearMicroEnv();
}
