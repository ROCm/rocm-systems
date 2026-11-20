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
extern ncclNet_t ncclNetIbMseg; // RCCL: IB-MSEG, via NCCL_MULTI_SEGMENT_REGISTER=1 or NCCL_NET=IB-MSEG
extern ncclResult_t IbMsegNetP2pPolicy(void* handle, int isP2p); // RCCL: net_ib_multiseg's rcclNetP2pPolicy
// RCCL: net_ib_multiseg's copies of the ncclIb* GDR flush flag entry points above.
bool IbMsegGdrFlushFlagLocalSupport();
void IbMsegSetGdrFlushFlag(void* netContext, int enable);
void IbMsegSetListenGdrFlushFlag(void* listenComm, int enable);
uint64_t* IbMsegGetGpuFlushFlag(void* recvComm);
uint64_t IbMsegGetGpuFlushFlagSeq(void* recvComm);
void IbMsegSetGpuFlushFlagHint(void* recvComm, int need);
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
 * the same mode. plugin/rma.cc uses it for its internal RMA backend, which also
 * selects a plugin (IB-CAST's RMA backend). The other ROCM-IB / IB-CAST
 * comparisons (transport/net.cc, plugin/gin.cc) must keep matching their own
 * literal spelling.
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

extern int64_t ncclParamMultiSegmentRegister();
// RCCL: whether IB-MSEG takes the internal IB slot (NET and RMA) unless IB-CAST leads; envNet is canonical.
inline bool rcclIbMsegRequested(const char* envNet) {
  return ncclParamMultiSegmentRegister() || (envNet && strcasecmp(envNet, "IB-MSEG") == 0);
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

// RCCL: RCCL_IB_GDR_FLUSH_FLAG entry points of the transport behind a net vtable (net_ib or its IB-MSEG
// copy), or NULL for a net without the flag. inline: the unit-test binaries compile init.cc without net.cc.
struct rcclIbGdrFlushOps {
  bool (*localSupport)();
  void (*setFlag)(void* netContext, int enable);
  void (*setListenFlag)(void* listenComm, int enable);
  uint64_t* (*gpuFlag)(void* recvComm);
  uint64_t (*gpuFlagSeq)(void* recvComm);
  void (*setGpuFlagHint)(void* recvComm, int need);
};
inline const struct rcclIbGdrFlushOps* rcclIbGdrFlushOpsFor(const ncclNet_t* net) {
  static const struct rcclIbGdrFlushOps ib = {ncclIbGdrFlushFlagLocalSupport, ncclIbSetGdrFlushFlag,
                                              ncclIbSetListenGdrFlushFlag,    ncclIbGetGpuFlushFlag,
                                              ncclIbGetGpuFlushFlagSeq,       ncclIbSetGpuFlushFlagHint};
  if (net == &ncclNetIb) return &ib;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  static const struct rcclIbGdrFlushOps mseg = {IbMsegGdrFlushFlagLocalSupport, IbMsegSetGdrFlushFlag,
                                                IbMsegSetListenGdrFlushFlag,    IbMsegGetGpuFlushFlag,
                                                IbMsegGetGpuFlushFlagSeq,       IbMsegSetGpuFlushFlagHint};
  if (net == &ncclNetIbMseg) return &mseg;
#endif
  return nullptr;
}

#endif
