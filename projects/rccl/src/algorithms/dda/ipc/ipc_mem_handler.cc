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
#include <cstring>
#include <vector>

ncclIpcMemHandler::ncclIpcMemHandler(void* bootstrap, int rank, int nranks)
  : bootstrap_(bootstrap), rank_(rank), nranks_(nranks), memPtrs_(static_cast<size_t>(nranks), nullptr),
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

  // The all-gather below is a bootstrap collective, so every rank has to reach
  // it: a rank that returned early -- local pointer never set, handle export
  // failed -- would leave its peers blocked there. Such a rank still takes part
  // and reports the failure in its slot instead, and every rank then fails
  // together after the exchange.
  struct Slot {
    cudaIpcMemHandle_t handle;
    uint8_t ok;
  };
  std::vector<Slot> slots(static_cast<size_t>(nranks_));
  Slot& self = slots[static_cast<size_t>(rank_)];
  std::memset(&self, 0, sizeof(self));
  if (memPtrs_[static_cast<size_t>(rank_)] == nullptr) {
    WARN("ncclIpcMemHandler::exchangeMemPtrs: local device pointer was not set");
  } else {
    cudaError_t err = cudaIpcGetMemHandle(&self.handle, memPtrs_[static_cast<size_t>(rank_)]);
    if (err == cudaSuccess) {
      self.ok = 1;
    } else {
      WARN("ncclIpcMemHandler::exchangeMemPtrs: cudaIpcGetMemHandle failed (%s)", cudaGetErrorString(err));
    }
  }
  NCCLCHECK(bootstrapAllGather(bootstrap_, slots.data(), static_cast<int>(sizeof(Slot))));

  for (int i = 0; i < nranks_; ++i) {
    if (slots[static_cast<size_t>(i)].ok == 0) {
      if (i != rank_) {
        INFO(NCCL_INIT, "ncclIpcMemHandler::exchangeMemPtrs: rank %d could not export its IPC handle", i);
      }
      return ncclSystemError;
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
      // The destructor only closes peer mappings once exchanged_ is set, so
      // close the ones this loop already opened before failing.
      for (int j = 0; j < i; ++j) {
        if (j != rank_) {
          CUDACHECKIGNORE(cudaIpcCloseMemHandle(memPtrs_[static_cast<size_t>(j)]));
          memPtrs_[static_cast<size_t>(j)] = nullptr;
        }
      }
      memPtrs_[static_cast<size_t>(i)] = nullptr;
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
