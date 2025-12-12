/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "core.h"
#include "mem_manager.h"

// [RCCL] Preserve AMD's HIP_FABRIC_API hook in case it ever gets enabled.
// When the runtime exposes the FABRIC API natively we map the upstream
// CUmemFabricHandle name onto the real hipMemFabricHandle_t. Otherwise we
// fall back to the 64-byte compat shim that mem_manager.h provides as
// hipMemFabricHandle_compat_t (aliased to CUmemFabricHandle), and stub out
// the FABRIC sentinel constants so upstream NCCL 2.29.7 code paths that
// reference CU_MEM_HANDLE_TYPE_FABRIC / CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
// compile on HIP without exercising any real fabric API.
#ifdef HIP_FABRIC_API
#undef CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
#undef CU_MEM_HANDLE_TYPE_FABRIC
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED hipDeviceAttributeHandleTypeFabricSupported
#define CU_MEM_HANDLE_TYPE_FABRIC hipMemHandleTypeFabric
typedef hipMemFabricHandle_t CUmemFabricHandle;
// HIP equivalents for CU vmm attribute/location constants used across alloc.h and transport files
#define CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID hipDeviceAttributeHostNumaId
#define CU_MEM_LOCATION_TYPE_DEVICE      hipMemLocationTypeDevice
#elif CUDART_VERSION < 12030
// MNNVL: FABRIC handle support lifted from CUDA 12.3 (and used as the HIP
// fallback whenever HIP_FABRIC_API is not enabled).
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ((CUdevice_attribute)128)
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
typedef hipMemFabricHandle_compat_t CUmemFabricHandle;
#endif

typedef union {
  uint64_t data; // Needs to hold a CUmemGenericAllocationHandle for UDS fd support
  CUmemFabricHandle handle;
} ncclCuDesc;

typedef union {
  // Legacy CUDA IPC
  cudaIpcMemHandle_t devIpc;
  // cuMem API support
  struct {
    ncclCuDesc cuDesc;
    CUmemGenericAllocationHandle memHandle;
  };
} ncclIpcDesc;

enum ncclIpcRegType {
  NCCL_IPC_SENDRECV = 0,
  NCCL_IPC_COLLECTIVE = 1
};

struct ncclIpcImpInfo {
  void* rmtRegAddr;
  bool legacyIpcCap;
  uintptr_t offset;
  int numSegments;
};

struct ncclIpcRegInfo {
  int peerRank;
  void* baseAddr;
  struct ncclProxyConnector* ipcProxyconn;
  struct ncclIpcImpInfo impInfo;
};

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int directMap, ncclIpcDesc *ipcDesc, void **ptr, int peerRank = -1, struct ncclMemManager* manager = nullptr, ncclMemType_t memtype = ncclMemPersist);
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int peer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr, void* ownerPtr = nullptr, ncclMemType_t memType = ncclMemPersist);
ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut);
ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts);

ncclResult_t ncclIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo);

#endif
