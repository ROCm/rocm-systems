/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/rccl_arch_thresholds.cc fake.

#ifndef RCCL_TEST_HOST_RCCL_ARCH_THRESHOLDS_FAKES_H_
#define RCCL_TEST_HOST_RCCL_ARCH_THRESHOLDS_FAKES_H_

#include "rccl_arch_thresholds.h"

// Calls to rcclGetArchThresholds and the partition the last one asked for.
extern int g_archThresholdsCalls;
extern rcclTuningPartition g_lastArchThresholdsPartition;
// What rcclGetArchThresholds returns for each rcclTuningPartition (nullptr until a test sets it),
// so a test can check which table ended up in comm->archThresholds.
extern const rcclArchThresholds* g_archThresholdsTable[2];

void ResetRcclArchThresholdsFakes();

#endif  // RCCL_TEST_HOST_RCCL_ARCH_THRESHOLDS_FAKES_H_
