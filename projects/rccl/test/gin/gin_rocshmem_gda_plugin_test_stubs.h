/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace GinRocshmemGdaPluginStubs {

struct RegMrCall {
  void* addr;
  size_t size;
  int atomic;
  uint32_t rkey;
};

void Reset();

void SetProbeResult(int result);
// The bootstrap allgather fills every slot except `rank` with PeerRkey/PeerAddr.
void SetBootstrapLayout(int nranks, int rank);
void SetCreateQpsFail(bool fail);
void SetRegMrFail(bool fail);

uint32_t PeerRkey(int pe);
uintptr_t PeerAddr(int pe);
void** FakeGpuQps();

const std::vector<RegMrCall>& GetRegMrCalls();
int GetLiveMrCount();
int GetDestroyQpsCalls();

// hipMalloc / hipExtMallocWithFlags / hipFree are interposed and forwarded to the
// HIP runtime. Pointer attribute queries cannot detect a free because the HIP
// allocator keeps freed blocks mapped, so cleanup is checked against this ledger.
bool IsLiveHipAllocation(const void* ptr);
int GetLiveHipAllocationCount();

}  // namespace GinRocshmemGdaPluginStubs
