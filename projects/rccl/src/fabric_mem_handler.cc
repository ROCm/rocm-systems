/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "fabric_mem_handler.h"

#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "debug.h"
#include "p2p.h"

#include <cstring>

namespace {

struct FabricExchEntry {
  ncclCuDesc desc;
  size_t size;
};
} // namespace

ncclFabricMemHandler::ncclFabricMemHandler(
    void* bootstrap, int rank, int nranks, struct ncclMemManager* manager)
    : bootstrap_(bootstrap),
      rank_(rank),
      nranks_(nranks),
      manager_(manager),
      selfPtr_(nullptr),
      selfHandle_{},
      selfSize_(0),
      memPtrs_(static_cast<size_t>(nranks), nullptr),
      exchanged_(false) {}

ncclFabricMemHandler::~ncclFabricMemHandler() {
  if (!exchanged_) {
    return;
  }
  // Free only the imported peer mappings; the local buffer is owned by the
  // caller and freed separately.
  for (int i = 0; i < nranks_; ++i) {
    if (i == rank_ || memPtrs_[static_cast<size_t>(i)] == nullptr) {
      continue;
    }
    (void)ncclCuMemFreeAddr(memPtrs_[static_cast<size_t>(i)], manager_);
  }
}

ncclResult_t ncclFabricMemHandler::addSelfDeviceMem(
    void* deviceMemPtr, CUmemGenericAllocationHandle handle, size_t size) {
  selfPtr_ = deviceMemPtr;
  selfHandle_ = handle;
  selfSize_ = size;
  memPtrs_[static_cast<size_t>(rank_)] = deviceMemPtr;
  return ncclSuccess;
}

ncclResult_t ncclFabricMemHandler::exchangeMemPtrs() {
  if (exchanged_) {
    return ncclSuccess;
  }
  if (selfPtr_ == nullptr) {
    WARN("ncclFabricMemHandler::exchangeMemPtrs: local device pointer was not set");
    return ncclInvalidUsage;
  }

  std::vector<FabricExchEntry> entries(static_cast<size_t>(nranks_));
  memset(entries.data(), 0, entries.size() * sizeof(FabricExchEntry));

  // Export this rank's allocation handle into an opaque fabric descriptor.
  CUCHECK(cuMemExportToShareableHandle(
      &entries[static_cast<size_t>(rank_)].desc,
      selfHandle_,
      ncclCuMemHandleType,
      0));
  entries[static_cast<size_t>(rank_)].size = selfSize_;

  NCCLCHECK(bootstrapAllGather(
      bootstrap_, entries.data(), static_cast<int>(sizeof(FabricExchEntry))));

  for (int i = 0; i < nranks_; ++i) {
    if (i == rank_) {
      continue;
    }
    CUmemGenericAllocationHandle peerHandle{};
    CUCHECK(cuMemImportFromShareableHandle(
        &peerHandle,
        (void*)&entries[static_cast<size_t>(i)].desc,
        ncclCuMemHandleType));

    // Reserve + map + set access for the imported handle
    void* peerPtr = nullptr;
    ncclResult_t res = ncclCuMemAllocAddr(
        &peerPtr, &peerHandle, entries[static_cast<size_t>(i)].size);
    if (res != ncclSuccess) {
      (void)cuMemRelease(peerHandle);
      WARN("ncclFabricMemHandler::exchangeMemPtrs: ncclCuMemAllocAddr failed for peer %d", i);
      return res;
    }
    CUCHECK(cuMemRelease(peerHandle));
    memPtrs_[static_cast<size_t>(i)] = peerPtr;
  }

  exchanged_ = true;
  return ncclSuccess;
}

ncclResult_t ncclFabricMemHandler::getPeerDeviceMemPtr(
    int peerRank, void** outPeerPtr) const {
  if (!exchanged_) {
    WARN("ncclFabricMemHandler::getPeerDeviceMemPtr: handles not exchanged yet");
    return ncclInvalidUsage;
  }
  if (peerRank < 0 || peerRank >= nranks_) {
    return ncclInvalidArgument;
  }
  if (outPeerPtr == nullptr) {
    return ncclInvalidArgument;
  }
  *outPeerPtr = memPtrs_[static_cast<size_t>(peerRank)];
  return ncclSuccess;
}
