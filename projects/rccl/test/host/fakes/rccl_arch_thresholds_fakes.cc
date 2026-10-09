/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fake for src/rccl_arch_thresholds.cc.
// Host-only test binaries do not link the real TU (it pulls in net.h / archinfo
// symbols that are already provided by other stubs), so this stub satisfies the
// linker. Returning nullptr is the correct "no arch table" sentinel -- every
// caller in production null-checks before dereferencing.

#include "rccl_arch_thresholds.h"

const rcclArchThresholds* rcclGetArchThresholds(const char* /*gcn*/) {
  return nullptr;
}
