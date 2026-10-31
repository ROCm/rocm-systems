/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable stand-in for ncclDdaIpcNranksSupported(), whose real body lives in
// ipc_init.cu. Mirrors that gate: the 8-rank clique always, 2..8 when relaxed.

#include "dda_ipc_nranks_fakes.h"

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

bool g_ddaIpcNranksRelaxed = false;

bool ncclDdaIpcNranksSupported(int nRanks) {
  if (nRanks == 8) return true;
  return g_ddaIpcNranksRelaxed && nRanks >= 2 && nRanks <= 8;
}
