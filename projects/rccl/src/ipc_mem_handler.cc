/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms IpcMemHandler.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "ipc_mem_handler.h"

#include "bootstrap.h"
#include "checks.h"

#include <cuda_runtime.h>

ncclIpcMemHandler::ncclIpcMemHandler(void* bootstrap, int rank, int nranks)
    : bootstrap_(bootstrap),
      rank_(rank),
      nranks_(nranks),
      memPtrs_(static_cast<size_t>(nranks), nullptr),
      exchanged_(false) {}

ncclIpcMemHandler::~ncclIpcMemHandler() {
  if (!exchanged_) {
    return;
  }
  for (int i = 0; i < nranks_; ++i) {
    if (i == rank_) {
      continue;
    }
    CUDACHECKIGNORE(cudaIpcCloseMemHandle(memPtrs_[static_cast<size_t>(i)]));
  }
}

ncclResult_t ncclIpcMemHandler::addSelfDeviceMemPtr(void* deviceMemPtr) {
  memPtrs_[static_cast<size_t>(rank_)] = deviceMemPtr;
  return ncclSuccess;
}

ncclResult_t ncclIpcMemHandler::exchangeMemPtrs() {
  if (exchanged_) {
    return ncclSuccess;
  }
  if (memPtrs_[static_cast<size_t>(rank_)] == nullptr) {
    WARN("ncclIpcMemHandler::exchangeMemPtrs: local device pointer was not set");
    return ncclInvalidUsage;
  }

  const int handleBytes = static_cast<int>(sizeof(cudaIpcMemHandle_t));

  std::vector<cudaIpcMemHandle_t> ipcHandles(static_cast<size_t>(nranks_));
  CUDACHECK(cudaIpcGetMemHandle(&ipcHandles[rank_], memPtrs_[static_cast<size_t>(rank_)]));
  NCCLCHECK(bootstrapAllGather(bootstrap_, ipcHandles.data(), handleBytes));

  for (int i = 0; i < nranks_; ++i) {
    if (i == rank_) {
      continue;
    }
    CUDACHECK(cudaIpcOpenMemHandle(
        &memPtrs_[static_cast<size_t>(i)],
        ipcHandles[i],
        cudaIpcMemLazyEnablePeerAccess));
  }
  exchanged_ = true;
  return ncclSuccess;
}

ncclResult_t ncclIpcMemHandler::getPeerDeviceMemPtr(int peerRank, void** outPeerPtr) const {
  if (!exchanged_) {
    WARN("ncclIpcMemHandler::getPeerDeviceMemPtr: handles not exchanged yet");
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
