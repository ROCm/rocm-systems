/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_GID_INSPECT_MSEG_H_
#define RCCL_NET_IB_GID_INSPECT_MSEG_H_

#ifdef ENABLE_FAULT_INJECTION

#include <stdint.h>
#include "net_ib_limits.h"
#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Test-only GID cache introspection for IBV_EVENT_GID_CHANGE handling.
 * IbMsegGid* targets net_ib, IbMsegCastGid* targets net_ib_cast; both sets
 * have identical semantics.
 */

struct IbMsegGidState {
  int linkLayer;
  int gidIndex;
  uint8_t gid[16];
  int ibDev; /* physical device backing a comm's devIndex; -1 for device queries */
};

struct IbMsegGidQpState {
  int nqps;
  int devIndex[NCCL_IB_MAX_QPS];
  int rtrGidIndex[NCCL_IB_MAX_QPS]; /* localGidIndex the QP was last moved to RTR with */
  int sgidIndex[NCCL_IB_MAX_QPS];   /* ah_attr.grh.sgid_index read back via ibv_query_qp */
  bool queryOk[NCCL_IB_MAX_QPS];
};

ncclResult_t IbMsegGidGetDev(int ibDev, struct IbMsegGidState* out);
ncclResult_t IbMsegGidSetDev(int ibDev, const struct IbMsegGidState* in);
ncclResult_t IbMsegGidGetComm(void* comm, int devIndex, struct IbMsegGidState* out);
ncclResult_t IbMsegGidSetComm(void* comm, int devIndex, const struct IbMsegGidState* in);
ncclResult_t IbMsegGidChangeEvent(int ibDev);
ncclResult_t IbMsegGidGetQpState(void* comm, struct IbMsegGidQpState* out);
ncclResult_t IbMsegGidGetDevState(void* comm, int devIndex, int* state);
ncclResult_t IbMsegGidGetRecoveryGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t IbMsegGidGetProbingGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t IbMsegGidDriveQpToError(void* comm, int qpIdx);
/* net_ib only: *supported = 1 if ibDev can create the UC QPs net_ib port recovery uses. */
ncclResult_t IbMsegGidUcSupported(int ibDev, int* supported);

ncclResult_t IbMsegCastGidGetDev(int ibDev, struct IbMsegGidState* out);
ncclResult_t IbMsegCastGidSetDev(int ibDev, const struct IbMsegGidState* in);
ncclResult_t IbMsegCastGidGetComm(void* comm, int devIndex, struct IbMsegGidState* out);
ncclResult_t IbMsegCastGidSetComm(void* comm, int devIndex, const struct IbMsegGidState* in);
ncclResult_t IbMsegCastGidChangeEvent(int ibDev);
ncclResult_t IbMsegCastGidGetQpState(void* comm, struct IbMsegGidQpState* out);
ncclResult_t IbMsegCastGidGetDevState(void* comm, int devIndex, int* state);
ncclResult_t IbMsegCastGidGetRecoveryGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t IbMsegCastGidGetProbingGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t IbMsegCastGidDriveQpToError(void* comm, int qpIdx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENABLE_FAULT_INJECTION */

#endif /* RCCL_NET_IB_GID_INSPECT_MSEG_H_ */
