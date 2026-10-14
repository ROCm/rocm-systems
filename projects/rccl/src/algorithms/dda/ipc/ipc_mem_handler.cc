/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms IpcMemHandler.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/ipc/ipc_mem_handler.h"

#include "bootstrap.h"
#include "checks.h"

#include <cuda_runtime.h>

ncclIpcMemHandler::ncclIpcMemHandler(void* bootstrap, int rank, int nranks)
  : bootstrap_(bootstrap), rank_(rank), nranks_(nranks), memPtrs_(static_cast<size_t>(nranks), nullptr),
    exchanged_(false) {}

ncclIpcMemHandler::~ncclIpcMemHandler() {
  (void)closePeerMappings();
}

ncclResult_t ncclIpcMemHandler::closePeerMappings() {
  // Close every mapping that was opened, including a partial exchange that
  // returned before exchanged_ could be set.
  ncclResult_t result = ncclSuccess;
  for (int i = 0; i < nranks_; ++i) {
    if (i == rank_ || memPtrs_[static_cast<size_t>(i)] == nullptr) {
      continue;
    }
    cudaError_t err = cudaIpcCloseMemHandle(memPtrs_[static_cast<size_t>(i)]);
    if (err != cudaSuccess) {
      WARN("ncclIpcMemHandler::closePeerMappings: cudaIpcCloseMemHandle(rank %d) failed (%s)", i,
           cudaGetErrorString(err));
      result = ncclUnhandledCudaError;
    }
    memPtrs_[static_cast<size_t>(i)] = nullptr;
  }
  exchanged_ = false;
  return result;
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
    CUDACHECK(cudaIpcOpenMemHandle(&memPtrs_[static_cast<size_t>(i)], ipcHandles[i], cudaIpcMemLazyEnablePeerAccess));
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
