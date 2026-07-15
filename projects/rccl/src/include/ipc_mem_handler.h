/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms IpcMemHandler (same-node GPU IPC exchange).
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef NCCL_IPC_MEM_HANDLER_H_
#define NCCL_IPC_MEM_HANDLER_H_

#include "nccl.h"

#include <vector>

/*
 * On a single machine, processes can use ncclIpcMemHandler to exchange device
 * memory pointers via HIP/CUDA IPC, using the NCCL bootstrap allgather for
 * handle distribution (host memory only, like the original implementation).
 *
 * This class is not thread-safe; only one thread per process should use it.
 *
 * Example (each rank):
 *   ncclIpcMemHandler handler(comm->bootstrap, comm->rank, comm->nRanks);
 *   NCCLCHECK(handler.addSelfDeviceMemPtr(localDevPtr));
 *   NCCLCHECK(handler.exchangeMemPtrs());
 *   void* peerPtr;
 *   NCCLCHECK(handler.getPeerDeviceMemPtr(peerRank, &peerPtr));
 */
class ncclIpcMemHandler {
 public:
  ncclIpcMemHandler(void* bootstrap, int rank, int nranks);
  ncclIpcMemHandler(const ncclIpcMemHandler&) = delete;
  ncclIpcMemHandler& operator=(const ncclIpcMemHandler&) = delete;
  ncclIpcMemHandler(ncclIpcMemHandler&&) = delete;
  ncclIpcMemHandler& operator=(ncclIpcMemHandler&&) = delete;
  ~ncclIpcMemHandler();

  ncclResult_t addSelfDeviceMemPtr(void* deviceMemPtr);
  ncclResult_t exchangeMemPtrs();
  ncclResult_t getPeerDeviceMemPtr(int peerRank, void** outPeerPtr) const;

 private:
  void* bootstrap_;
  const int rank_;
  const int nranks_;
  std::vector<void*> memPtrs_;
  bool exchanged_;
};

#endif
