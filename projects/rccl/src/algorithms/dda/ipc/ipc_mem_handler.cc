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

  // Every rank reaches the allgather, even without a handle to publish, so a local failure fails the
  // exchange instead of leaving the peers blocked in it.
  struct PublishedHandle {
    cudaIpcMemHandle_t handle;
    int valid;
  };
  std::vector<PublishedHandle> published(static_cast<size_t>(nranks_));
  void* self = memPtrs_[static_cast<size_t>(rank_)];
  if (self == nullptr) {
    WARN("ncclIpcMemHandler::exchangeMemPtrs: local device pointer was not set");
  } else {
    cudaError_t err = cudaIpcGetMemHandle(&published[rank_].handle, self);
    if (err == cudaSuccess) {
      published[rank_].valid = 1;
    } else {
      WARN("ncclIpcMemHandler::exchangeMemPtrs: cudaIpcGetMemHandle failed (%s)", cudaGetErrorString(err));
      (void)cudaGetLastError();
    }
  }
  NCCLCHECK(bootstrapAllGather(bootstrap_, published.data(), static_cast<int>(sizeof(PublishedHandle))));

  for (int i = 0; i < nranks_; ++i) {
    if (!published[i].valid) {
      if (i != rank_) INFO(NCCL_INIT, "ncclIpcMemHandler::exchangeMemPtrs: rank %d published no handle", i);
      return self == nullptr ? ncclInvalidUsage : ncclRemoteError;
    }
  }
  for (int i = 0; i < nranks_; ++i) {
    if (i == rank_) {
      continue;
    }
    CUDACHECK(cudaIpcOpenMemHandle(&memPtrs_[static_cast<size_t>(i)], published[i].handle,
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
