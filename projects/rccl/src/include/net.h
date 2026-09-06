/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"
#include "comm.h"
#include "checks.h"
#include "os.h"   // strcasecmp

#define NCCL_UNDEF_DEV_COUNT -1

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetInit(struct ncclComm* comm);
ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclNetFinalize(struct ncclComm* comm);
ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDev);
ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDev);

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

extern ncclResult_t rcclNetP2pPolicy(void* handle, int isP2p);
extern int64_t ncclParamDmaBufEnable();

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
extern ncclNet_t netIbCast;
extern ncclResult_t rcclCastNetP2pPolicy(void* handle, int isP2p);

enum rcclIBNicType {
  rcclIBNicTypeUnknown = -1,
  rcclIBNicTypeDefault = 0,
  rcclIBNicTypeMLX = 1,
  rcclIBNicTypeAINIC = 2,
  rcclIBNicTypeBNXT2 = 3,
  rcclIBNicTypeMax = 4
};

struct rcclIBNicInfo {
  rcclIBNicType type;
  int rate;
  int count;
};

/**
 * @brief Map an NCCL_NET value to the plugin name it selects.
 *
 * Plugin selection only: ROCM-IB and IB-CAST load the same plugin but are not
 * the same mode. The other ROCM-IB / IB-CAST comparisons (transport/net.cc,
 * plugin/gin.cc, plugin/rma.cc) must keep matching their own literal spelling.
 *
 * NCCL_NET= (set but empty) is passed through, as upstream does.
 *
 * inline: the unit-test binaries compile init.cc without transport/net.cc.
 */
inline const char* rcclCanonicalNetName(const char* envNetName) {
  if (envNetName == nullptr) return nullptr;
  if (strcasecmp(envNetName, "ROCM-IB") == 0) return "IB-CAST";
  return envNetName;
}

/**
 * @brief Get the primary NIC info
 *
 * @return rcclIBNicInfo (type, rate, count)
 */
extern rcclIBNicInfo rcclPrimaryNic();
/**
 * @brief Check if the primary NIC is AINIC
 *
 * @return bool
 */
extern bool rcclUseAinic();
#endif

#endif
