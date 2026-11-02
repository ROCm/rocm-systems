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

#if defined(NCCL_OS_LINUX)
#define NCCL_NET_DATA_IMPORT
#elif defined(NCCL_OS_WINDOWS)
#if defined(nccl_EXPORTS)
#define NCCL_NET_DATA_IMPORT
#else
#define NCCL_NET_DATA_IMPORT __declspec(dllimport)
#endif
#endif

extern NCCL_NET_DATA_IMPORT ncclNet_t ncclNetIb;
extern NCCL_NET_DATA_IMPORT ncclNet_t ncclNetSocket;
extern NCCL_NET_DATA_IMPORT ncclNet_t ncclNetNd;

#undef NCCL_NET_DATA_IMPORT

// RCCL_IB_GDR_FLUSH_FLAG=1: whether this process can use the flag (the comm
// enables it only when every rank can), and the comm-wide decision for an
// ncclNetIb context, which listen copies into each listen comm. A comm sharing
// its parent's context sets its own decision on its listen comms instead.
bool ncclIbGdrFlushFlagLocalSupport();
void ncclIbSetGdrFlushFlag(void* netContext, int enable);
void ncclIbSetListenGdrFlushFlag(void* listenComm, int enable);
// The ordering part of ncclIbGdrFlushFlagLocalSupport, over explicit settings:
// NCCL_IB_OOO_RQ, NCCL_IB_RESILIENCY_PORT_FAILOVER,
// NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME and the adaptive routing state of each
// IB device.
bool ncclIbGdrFlushFlagOrdered(int64_t oooRq, int64_t portFailover, int64_t matchingScheme, int nDevs,
                               const int* adaptiveRouting);
// GPU-visible per-QP flags of an ncclNetIb recv comm (NULL if this connection
// keeps the RDMA read flush), and the packed gdrFlushSeq entry (see device.h)
// the GPU waits on for the last completed receive.
uint64_t* ncclIbGetGpuFlushFlag(void* recvComm);
uint64_t ncclIbGetGpuFlushFlagSeq(void* recvComm);
// Whether the following irecv needs the flag write (SIMPLE GDR receives only).
void ncclIbSetGpuFlushFlagHint(void* recvComm, int need);

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
