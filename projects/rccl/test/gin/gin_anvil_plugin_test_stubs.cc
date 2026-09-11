/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Test doubles for gin_plugin_anvil_sdma.cc (bootstrap, devr, factory, debug).

#include "gin_anvil_plugin_test_stubs.h"

#include <gin_anvil/sdma_factory.h>

#include "alloc.h"
#include "bootstrap.h"
#include "debug.h"
#include "dev_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace GinAnvilPluginStubs {

struct State {
  int probeResult = 1;
  bool bootstrapFail = false;
  int bootstrapNranks = 1;
  bool factoryCreateFail = false;
  bool factoryNullHandles = false;
  bool lsaAddrFail = false;
  void* lsaSelfAddr = reinterpret_cast<void*>(0x70001000ULL);
  bool useFabricMem = false;
  bool fabricAddSelfFail = false;
  bool fabricExchangeFail = false;
  void* fabricPeerBase = reinterpret_cast<void*>(0x80000000ULL);
  size_t fabricPeerStride = 0x1000;
  bool fabricVmmQueryOk = true;
  bool fabricRetainOk = true;
};

struct FakeSdmaOpaque {
  int nRanks;
  int numChannels;
  int sdmaChannelStride;
  void** deviceHandles_d;
  uint64_t* sdmaDirty_d;
};

static State g;

void Reset() { g = State{}; }

void SetProbeResult(int result) { g.probeResult = result; }
void SetBootstrapFail(bool fail) { g.bootstrapFail = fail; }
void SetBootstrapNranks(int nranks) { g.bootstrapNranks = nranks; }
void SetFactoryCreateFail(bool fail) { g.factoryCreateFail = fail; }
void SetFactoryNullHandles(bool nullHandles) { g.factoryNullHandles = nullHandles; }
void SetLsaAddrFail(bool fail) { g.lsaAddrFail = fail; }
void SetLsaSelfAddr(void* addr) { g.lsaSelfAddr = addr; }
void SetUseFabricMem(bool use) { g.useFabricMem = use; }
void SetFabricAddSelfFail(bool fail) { g.fabricAddSelfFail = fail; }
void SetFabricExchangeFail(bool fail) { g.fabricExchangeFail = fail; }
void SetFabricPeerBase(void* ptr) { g.fabricPeerBase = ptr; }
void SetFabricPeerStride(size_t stride) { g.fabricPeerStride = stride; }
void SetFabricVmmQueryOk(bool ok) { g.fabricVmmQueryOk = ok; }
void SetFabricRetainOk(bool ok) { g.fabricRetainOk = ok; }

}  // namespace GinAnvilPluginStubs

namespace GinAnvilPluginTestHooks {

ncclResult_t queryVmmRange(void* data, size_t size, CUdeviceptr* base, size_t* memSize, int* numSegments) {
  if (!GinAnvilPluginStubs::g.fabricVmmQueryOk) return ncclSystemError;
  if (base == nullptr || memSize == nullptr || numSegments == nullptr) return ncclInvalidArgument;
  *base = reinterpret_cast<CUdeviceptr>(GinAnvilPluginStubs::g.fabricPeerBase);
  *memSize = size + 0x1000;
  *numSegments = 1;
  (void)data;
  return ncclSuccess;
}

CUresult retainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) {
  if (!GinAnvilPluginStubs::g.fabricRetainOk) return CUDA_ERROR_UNKNOWN;
  if (handle == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *handle = CUmemGenericAllocationHandle{};
  (void)addr;
  return CUDA_SUCCESS;
}

}  // namespace GinAnvilPluginTestHooks

int ncclDebugLevel = NCCL_LOG_VERSION;
uint64_t ncclDebugMask = NCCL_INIT;
thread_local int ncclDebugNoWarn = 0;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line,
                  const char* fmt, ...) {
  (void)level;
  (void)flags;
  (void)filefunc;
  (void)line;
  (void)fmt;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  (void)commState;
  if (GinAnvilPluginStubs::g.bootstrapFail) return ncclInternalError;
  if (size == static_cast<int>(sizeof(int))) {
    int* devs = static_cast<int*>(allData);
    for (int i = 0; i < GinAnvilPluginStubs::g.bootstrapNranks; ++i) {
      if (devs[i] < 0) devs[i] = 0;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclDevrGetLsaSelfAddr(struct ncclDevrState* devr, void* addr, void** outAddr) {
  (void)devr;
  (void)addr;
  if (GinAnvilPluginStubs::g.lsaAddrFail) {
    *outAddr = nullptr;
    return ncclSuccess;
  }
  *outAddr = GinAnvilPluginStubs::g.lsaSelfAddr;
  return ncclSuccess;
}

extern "C" int gin_anvil_sdma_probe(void) { return GinAnvilPluginStubs::g.probeResult; }

extern "C" int gin_anvil_sdma_create(int nRanks, int myRank, int my_device_id,
                                     int (*allgather)(void*, void*, size_t), void* allgather_ctx,
                                     int num_channels, gin_anvil_sdma_handle_t* out_handle,
                                     void** out_gpu_handles, uint64_t** out_sdma_dirty) {
  if (!out_handle || !out_gpu_handles || !out_sdma_dirty || !allgather || nRanks < 1 || myRank < 0 ||
      myRank >= nRanks)
    return -1;
  if (GinAnvilPluginStubs::g.factoryCreateFail) return -1;

  std::vector<int> devs(static_cast<size_t>(nRanks), -1);
  devs[static_cast<size_t>(myRank)] = my_device_id;
  if (allgather(allgather_ctx, devs.data(), sizeof(int)) != 0) return -1;
  for (int i = 0; i < nRanks; ++i) {
    if (devs[static_cast<size_t>(i)] < 0) return -1;
  }

  const int numChannels = num_channels < 1 ? 1 : (num_channels > 8 ? 8 : num_channels);
  auto* impl = new GinAnvilPluginStubs::FakeSdmaOpaque{};
  impl->nRanks = nRanks;
  impl->numChannels = numChannels;
  impl->sdmaChannelStride = 1;

  if (GinAnvilPluginStubs::g.factoryNullHandles) {
    impl->deviceHandles_d = nullptr;
    impl->sdmaDirty_d = nullptr;
    *out_handle = reinterpret_cast<gin_anvil_sdma_handle_t>(impl);
    *out_gpu_handles = nullptr;
    *out_sdma_dirty = nullptr;
    return 0;
  }

  void** row = nullptr;
  uint64_t* dirty = nullptr;
  if (hipMalloc(&row, static_cast<size_t>(nRanks * numChannels) * sizeof(void*)) != hipSuccess) {
    delete impl;
    return -1;
  }
  if (hipExtMallocWithFlags(reinterpret_cast<void**>(&dirty), sizeof(uint64_t),
                            hipDeviceMallocFinegrained) != hipSuccess) {
    hipFree(row);
    delete impl;
    return -1;
  }
  (void)hipMemset(dirty, 0, sizeof(uint64_t));
  impl->deviceHandles_d = row;
  impl->sdmaDirty_d = dirty;

  *out_handle = reinterpret_cast<gin_anvil_sdma_handle_t>(impl);
  *out_gpu_handles = row;
  *out_sdma_dirty = dirty;
  return 0;
}

extern "C" void gin_anvil_sdma_destroy(gin_anvil_sdma_handle_t handle) {
  if (!handle) return;
  auto* impl = reinterpret_cast<GinAnvilPluginStubs::FakeSdmaOpaque*>(handle);
  if (impl->deviceHandles_d) hipFree(impl->deviceHandles_d);
  if (impl->sdmaDirty_d) hipFree(impl->sdmaDirty_d);
  delete impl;
}

extern "C" int gin_anvil_sdma_get_n_ranks(gin_anvil_sdma_handle_t handle) {
  return handle ? reinterpret_cast<GinAnvilPluginStubs::FakeSdmaOpaque*>(handle)->nRanks : 0;
}

extern "C" int gin_anvil_sdma_get_num_channels(gin_anvil_sdma_handle_t handle) {
  return handle ? reinterpret_cast<GinAnvilPluginStubs::FakeSdmaOpaque*>(handle)->numChannels : 0;
}

extern "C" int gin_anvil_sdma_get_channel_stride(gin_anvil_sdma_handle_t handle) {
  return handle ? reinterpret_cast<GinAnvilPluginStubs::FakeSdmaOpaque*>(handle)->sdmaChannelStride : 0;
}

#include "algorithms/dda/fabric/fabric_init.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"

bool ginAnvilUseFabricMem(struct ncclComm* comm) {
  (void)comm;
  return GinAnvilPluginStubs::g.useFabricMem;
}

ncclFabricMemHandler::ncclFabricMemHandler(void* bootstrap, int rank, int nranks, struct ncclMemManager* manager)
  : bootstrap_(bootstrap), rank_(rank), nranks_(nranks), manager_(manager), selfPtr_(nullptr), selfHandle_{},
    selfSize_(0), memPtrs_(static_cast<size_t>(nranks), nullptr), exchanged_(false) {}

ncclFabricMemHandler::~ncclFabricMemHandler() {}

ncclResult_t ncclFabricMemHandler::addSelfDeviceMem(void* deviceMemPtr, CUmemGenericAllocationHandle handle,
                                                    size_t size) {
  if (GinAnvilPluginStubs::g.fabricAddSelfFail) return ncclSystemError;
  selfPtr_ = deviceMemPtr;
  selfHandle_ = handle;
  selfSize_ = size;
  if (rank_ >= 0 && rank_ < nranks_) memPtrs_[static_cast<size_t>(rank_)] = deviceMemPtr;
  return ncclSuccess;
}

ncclResult_t ncclFabricMemHandler::exchangeMemPtrs() {
  if (GinAnvilPluginStubs::g.fabricExchangeFail) return ncclSystemError;
  exchanged_ = true;
  return ncclSuccess;
}

ncclResult_t ncclFabricMemHandler::getPeerDeviceMemPtr(int peerRank, void** outPeerPtr) const {
  if (!outPeerPtr || peerRank < 0 || peerRank >= nranks_) return ncclInvalidArgument;
  uintptr_t base = reinterpret_cast<uintptr_t>(GinAnvilPluginStubs::g.fabricPeerBase);
  *outPeerPtr = reinterpret_cast<void*>(base + static_cast<uintptr_t>(peerRank) * GinAnvilPluginStubs::g.fabricPeerStride);
  return ncclSuccess;
}
