/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

// size_t below; this header is included first by GinAnvilPlugin_test.cpp and by
// sibling targets built as LANGUAGE CXX, where the HIP compilation prelude that
// would otherwise declare it is absent.
#include <cstddef>

#include <vector>

namespace GinAnvilPluginStubs {

void Reset();

void SetProbeResult(int result);
void SetBootstrapFail(bool fail);
void SetBootstrapNranks(int nranks);
void SetBootstrapPeerThreshold(size_t threshold);
// Queue one allgather payload. Each sizeof(int) allgather consumes the front
// entry, so a second call can inject a peer-reported missing vector.
void SetBootstrapIntResult(const int* values, int count);
void SetFactoryCreateFail(bool fail);
void SetFactoryNullHandles(bool nullHandles);
void SetLsaAddrFail(bool fail);
// Also clears the latched input base, so the next resolved address maps to `addr`.
void SetLsaSelfAddr(void* addr);
void SetUseFabricMem(bool use);
void SetFabricLsaCapable(bool capable);
void SetFabricAddSelfFail(bool fail);
void SetFabricExchangeFail(bool fail);
void SetFabricPeerBase(void* ptr);
void SetFabricPeerStride(size_t stride);
void SetFabricVmmQueryOk(bool ok);
void SetFabricRetainOk(bool ok);
/** ncclFabricMemHandler instances constructed and not yet destroyed. */
int LiveFabricHandlers();
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

// hipMalloc / hipExtMallocWithFlags / hipFree are interposed and forwarded to the
// HIP runtime. Pointer attribute queries cannot detect a free because the HIP
// allocator keeps freed blocks mapped, so cleanup is checked against this ledger.
// Reset() clears it, so allocations from earlier tests are not tracked.
bool IsLiveHipAllocation(const void* ptr);
int GetLiveHipAllocationCount();
// Fail the next hipExtMallocWithFlags call once with hipErrorOutOfMemory.
void SetFailNextExtMalloc(bool fail);

}  // namespace GinAnvilPluginStubs
