/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_p2p_fakes.h"

#include <cstdlib>
#include <cstring>

#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclP2pAllocateShareableBuffer, ncclP2pAllocateShareableBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclP2pImportShareableBuffer, ncclP2pImportShareableBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclP2pFreeShareableBuffer, ncclP2pFreeShareableBuffer);

static ncclResult_t DefaultNcclP2pAllocateShareableBuffer(size_t size, int, ncclIpcDesc* ipcDesc, void** ptr, int,
                                                          struct ncclMemManager*, ncclMemType_t) {
  if (size == 0 || ipcDesc == nullptr || ptr == nullptr) {
    return ncclInvalidArgument;
  }
  *ptr = std::calloc(1, size);
  if (*ptr == nullptr) {
    return ncclSystemError;
  }
  std::memset(ipcDesc, 0, sizeof(*ipcDesc));
  std::memcpy(ipcDesc, ptr, sizeof(*ptr));
  return ncclSuccess;
}
std::function<ncclResult_t(size_t, int, ncclIpcDesc*, void**, int, struct ncclMemManager*, ncclMemType_t)>
    g_ncclP2pAllocateShareableBuffer = DefaultNcclP2pAllocateShareableBuffer;

static ncclResult_t DefaultNcclP2pImportShareableBuffer(struct ncclComm*, int, size_t, ncclIpcDesc* ipcDesc,
                                                        void** devMemPtr, void* ownerPtr, ncclMemType_t) {
  if (devMemPtr == nullptr || ownerPtr == nullptr || ipcDesc == nullptr ||
      std::memcmp(ipcDesc, &ownerPtr, sizeof(ownerPtr)) != 0) {
    return ncclInvalidArgument;
  }
  *devMemPtr = ownerPtr;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, int, size_t, ncclIpcDesc*, void**, void*, ncclMemType_t)>
    g_ncclP2pImportShareableBuffer = DefaultNcclP2pImportShareableBuffer;

static ncclResult_t DefaultNcclP2pFreeShareableBuffer(ncclIpcDesc*) { return ncclSuccess; }
std::function<ncclResult_t(ncclIpcDesc*)> g_ncclP2pFreeShareableBuffer = DefaultNcclP2pFreeShareableBuffer;

ncclResult_t micro_ncclP2pAllocateShareableBuffer(size_t size, int directMap, ncclIpcDesc* ipcDesc, void** ptr,
                                                  int peerRank, struct ncclMemManager* manager, ncclMemType_t memtype) {
  return g_ncclP2pAllocateShareableBuffer(size, directMap, ipcDesc, ptr, peerRank, manager, memtype);
}

ncclResult_t micro_ncclP2pImportShareableBuffer(struct ncclComm* comm, int peer, size_t size, ncclIpcDesc* ipcDesc,
                                                void** devMemPtr, void* ownerPtr, ncclMemType_t memType) {
  return g_ncclP2pImportShareableBuffer(comm, peer, size, ipcDesc, devMemPtr, ownerPtr, memType);
}

ncclResult_t micro_ncclP2pFreeShareableBuffer(ncclIpcDesc* ipcDesc) { return g_ncclP2pFreeShareableBuffer(ipcDesc); }

#ifndef RCCL_TRANSPORT_P2P_FAKES_OMIT_SHAREABLE_BUFFERS
ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int directMap, ncclIpcDesc* ipcDesc, void** ptr,
                                            int peerRank, struct ncclMemManager* manager, ncclMemType_t memtype) {
  return micro_ncclP2pAllocateShareableBuffer(size, directMap, ipcDesc, ptr, peerRank, manager, memtype);
}

ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm* comm, int peer, size_t size, ncclIpcDesc* ipcDesc,
                                          void** devMemPtr, void* ownerPtr, ncclMemType_t memType) {
  return micro_ncclP2pImportShareableBuffer(comm, peer, size, ipcDesc, devMemPtr, ownerPtr, memType);
}

ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc* ipcDesc) { return micro_ncclP2pFreeShareableBuffer(ipcDesc); }
#endif

void ResetTransportP2pFakes() {
  g_ncclP2pAllocateShareableBuffer = DefaultNcclP2pAllocateShareableBuffer;
  g_ncclP2pImportShareableBuffer = DefaultNcclP2pImportShareableBuffer;
  g_ncclP2pFreeShareableBuffer = DefaultNcclP2pFreeShareableBuffer;
}
