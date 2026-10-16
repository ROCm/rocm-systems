/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fake for src/rccl_arch_thresholds.cc.
// Host-only test binaries do not link the real TU, so this stub satisfies the linker.
// By default it returns nullptr, the "no arch table" sentinel every production caller
// null-checks; a test can install a table per partition to see which one a comm got.

#include "rccl_arch_thresholds_fakes.h"

int g_archThresholdsCalls = 0;
rcclTuningPartition g_lastArchThresholdsPartition = RCCL_TUNING_PARTITION_SPX;
const rcclArchThresholds* g_archThresholdsTable[2] = {nullptr, nullptr};

const rcclArchThresholds* rcclGetArchThresholds(const char* /*gcn*/, rcclTuningPartition partition) {
  ++g_archThresholdsCalls;
  g_lastArchThresholdsPartition = partition;
  return g_archThresholdsTable[partition];
}

void ResetRcclArchThresholdsFakes() {
  g_archThresholdsCalls = 0;
  g_lastArchThresholdsPartition = RCCL_TUNING_PARTITION_SPX;
  g_archThresholdsTable[RCCL_TUNING_PARTITION_SPX] = nullptr;
  g_archThresholdsTable[RCCL_TUNING_PARTITION_DPX] = nullptr;
}
