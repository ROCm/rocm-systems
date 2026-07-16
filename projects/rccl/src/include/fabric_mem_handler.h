/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef NCCL_FABRIC_MEM_HANDLER_H_
#define NCCL_FABRIC_MEM_HANDLER_H_

#include "nccl.h"

#include <cuda.h>

#include <cstddef>
#include <vector>

struct ncclMemManager;

class ncclFabricMemHandler {
 public:
  ncclFabricMemHandler(void* bootstrap, int rank, int nranks,
                       struct ncclMemManager* manager);
  ncclFabricMemHandler(const ncclFabricMemHandler&) = delete;
  ncclFabricMemHandler& operator=(const ncclFabricMemHandler&) = delete;
  ncclFabricMemHandler(ncclFabricMemHandler&&) = delete;
  ncclFabricMemHandler& operator=(ncclFabricMemHandler&&) = delete;
  ~ncclFabricMemHandler();

  // Register this rank's local VMM allocation: mapped pointer, its cuMem
  // allocation handle (for export), and the requested size (the import side
  // re-aligns to the allocation granularity).
  ncclResult_t addSelfDeviceMem(
      void* deviceMemPtr,
      CUmemGenericAllocationHandle handle,
      size_t size);
  ncclResult_t exchangeMemPtrs();
  ncclResult_t getPeerDeviceMemPtr(int peerRank, void** outPeerPtr) const;

 private:
  void* bootstrap_;
  const int rank_;
  const int nranks_;
  struct ncclMemManager* manager_;
  void* selfPtr_;
  CUmemGenericAllocationHandle selfHandle_;
  size_t selfSize_;
  // Mapped device pointer per rank (self = selfPtr_; peers = imported mapping).
  std::vector<void*> memPtrs_;
  bool exchanged_;
};

#endif
