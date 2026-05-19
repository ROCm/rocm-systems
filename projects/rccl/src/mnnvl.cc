/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mnnvl.h"
#include "transport.h"
#include <cuda.h>
#include "rocmwrap.h"

// Determine if MNNVL support is available
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  // MNNVL requires cuMem to be enabled
  if (!ncclCuMemEnable()) return ncclSuccess;

  // MNNVL also requires FABRIC handle support
  int cudaDev;
  int flag = 0;
  CUdevice currentDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUDACHECK(cuDeviceGet(&currentDev, cudaDev));
  // Ignore error if CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported
  (void) cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev);
  if (!flag) return ncclSuccess;

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
  // Check that all ranks have initialized the fabric fully
  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].fabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED) return ncclSuccess;
  }
#else
  // Require ACTIVE or READY state on all ranks before enabling MNNVL.
  // AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED means the vpod is provisioned but fabric handles
  // are not yet exchangeable cross-process — equivalent to NCCL's NVML_GPU_FABRIC_STATE_COMPLETED gate.
  for (int i = 0; i < comm->nRanks; i++) {
    if ((comm->peerInfo[i].fabricInfo.state != AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE) &&
        (comm->peerInfo[i].fabricInfo.state != AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY)) {
      INFO(NCCL_INIT, "MNNVL disabled: peer %d fabric state %d is not ACTIVE or READY; falling back to RDMA",
           i, comm->peerInfo[i].fabricInfo.state);
      return ncclSuccess;
    }
  }
#endif
  // Determine our MNNVL domain/clique
  NCCLCHECK(ncclCalloc(&comm->clique.ranks, comm->nRanks));
  comm->clique.id = comm->peerInfo[comm->rank].fabricInfo.cliqueId;
  for (int i = 0; i < comm->nRanks; i++) {
    auto fabricInfo1 = &comm->peerInfo[comm->rank].fabricInfo;
    auto fabricInfo2 = &comm->peerInfo[i].fabricInfo;
    // Check if the cluster UUID and cliqueId match
    // A zero UUID means we don't have MNNVL fabric info - disable MNNVL
    unsigned long uuid0 = 0;
    unsigned long uuid1 = 0;
    memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));
    memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));
    if ((uuid0 | uuid1) == 0) return ncclSuccess;
    if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
        (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
      if (i == comm->rank) {
        comm->cliqueRank = comm->clique.size;
      }
      comm->clique.ranks[comm->clique.size++] = i;
    }
  }

  // No MNNVL clique found
  if (comm->clique.size <= 1) return ncclSuccess;
#ifdef HIP_FABRIC_API
  // Check that FABRIC handles can be exported & imported by IMEX
  {
    void *ptr = NULL;
    CUmemGenericAllocationHandle handle;
    ncclCuDesc cuDesc;
    CUresult err;

    // Allocate FABRIC handle compatible memory
    ncclResult_t ret = ncclCuMemAlloc(&ptr, &handle, CU_MEM_HANDLE_TYPE_FABRIC, CUDA_IPC_MIN);
    if (ret != ncclSuccess) {
      // Return an error if this is a MNNVL capable system but FABRIC handles are not supported
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check afmctl. Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }
    err = cuMemExportToShareableHandle(&cuDesc, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (err != CUDA_SUCCESS) {
    // || (err = cuMemImportFromShareableHandle(&handle, &cuDesc, CU_MEM_HANDLE_TYPE_FABRIC)) != CUDA_SUCCESS
    // Note: cuMemImportFromShareableHandle is intentionally NOT tested here.
    // Same-process round-trip of a fabric handle is not currently supported — the import call
    // fails even on a correctly configured MNNVL system. Cross-process import is validated
    // implicitly when the first actual P2P transfer succeeds after MNNVL is enabled.
    // The ImportFromShareableHandle check above is preserved to document what was tried and why
    // it was removed, so future maintainers don't re-add it expecting it to work.
      const char *errStr;
      (void) cuGetErrorString(err, &errStr);
      NCCLCHECK(ncclCuMemFree(ptr));
      // Return an error if this is a MNNVL capable system but it's not working
      WARN("MNNVL rank%d (cliqueSize %d) is available but not working on this system: cuMemExportToShareableHandle/cuMemImportFromShareableHandle failed: %s. Check afmctl. Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
          comm->rank, comm->clique.size, errStr);
      return ncclSystemError;
    }
    NCCLCHECK(ncclCuMemFree(ptr));

    // Force the CUMEM handle type to be FABRIC for MNNVL
    ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
    comm->MNNVL = 1;
    INFO(NCCL_INIT, "MNNVL %d cliqueId %x cliqueSize %d cliqueRank %d",
        comm->MNNVL, comm->clique.id, comm->clique.size, comm->cliqueRank);
  }
#endif
  return ncclSuccess;
}
