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

#include <cstdint>
#include <vector>

ncclIpcMemHandler::ncclIpcMemHandler(void* bootstrap, int rank, int nranks)
  : bootstrap_(bootstrap), rank_(rank), nranks_(nranks), memPtrs_(static_cast<size_t>(nranks), nullptr),
    exchanged_(false) {}

namespace {

// One rank's slot in the handle all-gather; exportOk lets a rank whose export failed still take part.
struct ncclIpcExchangeSlot {
  cudaIpcMemHandle_t handle;
  int32_t exportOk;
};

// Closes the peer mappings in [0, end), skipping this rank's own pointer.
void closePeerMappings(std::vector<void*>& memPtrs, int rank, int end) {
  for (int i = 0; i < end; ++i) {
    if (i == rank || memPtrs[static_cast<size_t>(i)] == nullptr) {
      continue;
    }
    CUDACHECKIGNORE(cudaIpcCloseMemHandle(memPtrs[static_cast<size_t>(i)]));
    memPtrs[static_cast<size_t>(i)] = nullptr;
  }
}

} // namespace

ncclIpcMemHandler::~ncclIpcMemHandler() {
  if (!exchanged_) {
    return;
  }
  closePeerMappings(memPtrs_, rank_, nranks_);
}

ncclResult_t ncclIpcMemHandler::addSelfDeviceMemPtr(void* deviceMemPtr) {
  memPtrs_[static_cast<size_t>(rank_)] = deviceMemPtr;
  return ncclSuccess;
}

ncclResult_t ncclIpcMemHandler::exchangeMemPtrs() {
  if (exchanged_) {
    return ncclSuccess;
  }

  // A rank that cannot export still enters the all-gather, so its failure reaches every peer instead of stranding them.
  std::vector<ncclIpcExchangeSlot> slots(static_cast<size_t>(nranks_));
  ncclIpcExchangeSlot& self = slots[static_cast<size_t>(rank_)];
  ncclResult_t selfRes = ncclSuccess;
  void* selfPtr = memPtrs_[static_cast<size_t>(rank_)];
  if (selfPtr == nullptr) {
    WARN("ncclIpcMemHandler::exchangeMemPtrs: local device pointer was not set");
    selfRes = ncclInvalidUsage;
  } else {
    cudaError_t err = cudaIpcGetMemHandle(&self.handle, selfPtr);
    if (err != cudaSuccess) {
      WARN("ncclIpcMemHandler::exchangeMemPtrs: cudaIpcGetMemHandle failed (%s)", cudaGetErrorString(err));
      selfRes = ncclUnhandledCudaError;
    } else {
      self.exportOk = 1;
    }
  }
  NCCLCHECK(bootstrapAllGather(bootstrap_, slots.data(), static_cast<int>(sizeof(ncclIpcExchangeSlot))));
  if (selfRes != ncclSuccess) {
    return selfRes;
  }
  for (int i = 0; i < nranks_; ++i) {
    if (slots[static_cast<size_t>(i)].exportOk == 0) {
      INFO(NCCL_INIT, "ncclIpcMemHandler::exchangeMemPtrs: rank %d could not export its IPC handle", i);
      return ncclRemoteError;
    }
  }

  for (int i = 0; i < nranks_; ++i) {
    if (i == rank_) {
      continue;
    }
    cudaError_t err = cudaIpcOpenMemHandle(&memPtrs_[static_cast<size_t>(i)], slots[static_cast<size_t>(i)].handle,
                                           cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
      WARN("ncclIpcMemHandler::exchangeMemPtrs: cudaIpcOpenMemHandle(rank %d) failed (%s)", i,
           cudaGetErrorString(err));
      memPtrs_[static_cast<size_t>(i)] = nullptr;
      closePeerMappings(memPtrs_, rank_, i);
      return ncclUnhandledCudaError;
    }
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
