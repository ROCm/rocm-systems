/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <vector>

namespace GinAnvilPluginStubs {

void Reset();

void SetProbeResult(int result);
void SetBootstrapFail(bool fail);
void SetBootstrapNranks(int nranks);
// Queue one allgather payload. Each sizeof(int) allgather consumes the front
// entry, so a second call can inject a peer-reported missing vector.
void SetBootstrapIntResult(const int* values, int count);
void SetFactoryCreateFail(bool fail);
void SetFactoryNullHandles(bool nullHandles);
void SetLsaAddrFail(bool fail);
// Also clears the latched input base, so the next resolved address maps to `addr`.
void SetLsaSelfAddr(void* addr);
// Positive: simulate missing for the next N verify calls. Negative: always. Zero (default): never.
void SetConnCheckMissingCalls(int calls);
int GetConnCheckWriteCalls();
int GetConnCheckVerifyCalls();
unsigned long long GetConnCheckWriteStamp(int call);
const std::vector<int>& GetLastIntraNodeAllGatherRanks();
int GetLastIntraNodeAllGatherRank();
int GetLastIntraNodeAllGatherNranks();
int GetBootstrapAllGatherCalls();
// Element size of the last bootstrapAllGather that was not sizeof(int).
int GetLastWideAllGatherSize();
// Non-int bootstrapAllGather calls whose buffer could not hold one slot per world rank.
int GetWideAllGatherOverruns();
int GetIntraNodeAllGatherCalls();
const std::vector<int>& GetLastIntraNodeBarrierRanks();
int GetLastIntraNodeBarrierRank();
int GetLastIntraNodeBarrierTag();

}  // namespace GinAnvilPluginStubs
