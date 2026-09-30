/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_GID_INSPECT_H_
#define RCCL_NET_IB_GID_INSPECT_H_

#ifdef ENABLE_FAULT_INJECTION

#include <stdint.h>
#include "net_ib_limits.h"
#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Test-only GID cache introspection for IBV_EVENT_GID_CHANGE handling.
 * ncclIbGid* targets net_ib, ncclIbCastGid* targets net_ib_cast; both sets
 * have identical semantics.
 */

struct ncclIbGidState {
  int linkLayer;
  int gidIndex;
  uint8_t gid[16];
  int ibDev; /* physical device backing a comm's devIndex; -1 for device queries */
};

struct ncclIbGidQpState {
  int nqps;
  int devIndex[NCCL_IB_MAX_QPS];
  int rtrGidIndex[NCCL_IB_MAX_QPS]; /* localGidIndex the QP was last moved to RTR with */
  int sgidIndex[NCCL_IB_MAX_QPS];   /* ah_attr.grh.sgid_index read back via ibv_query_qp */
  bool queryOk[NCCL_IB_MAX_QPS];
};

ncclResult_t ncclIbGidGetDev(int ibDev, struct ncclIbGidState* out);
ncclResult_t ncclIbGidSetDev(int ibDev, const struct ncclIbGidState* in);
ncclResult_t ncclIbGidGetComm(void* comm, int devIndex, struct ncclIbGidState* out);
ncclResult_t ncclIbGidSetComm(void* comm, int devIndex, const struct ncclIbGidState* in);
ncclResult_t ncclIbGidChangeEvent(int ibDev);
ncclResult_t ncclIbGidGetQpState(void* comm, struct ncclIbGidQpState* out);
ncclResult_t ncclIbGidGetDevState(void* comm, int devIndex, int* state);
ncclResult_t ncclIbGidGetRecoveryGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t ncclIbGidGetProbingGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t ncclIbGidDriveQpToError(void* comm, int qpIdx);

ncclResult_t ncclIbCastGidGetDev(int ibDev, struct ncclIbGidState* out);
ncclResult_t ncclIbCastGidSetDev(int ibDev, const struct ncclIbGidState* in);
ncclResult_t ncclIbCastGidGetComm(void* comm, int devIndex, struct ncclIbGidState* out);
ncclResult_t ncclIbCastGidSetComm(void* comm, int devIndex, const struct ncclIbGidState* in);
ncclResult_t ncclIbCastGidChangeEvent(int ibDev);
ncclResult_t ncclIbCastGidGetQpState(void* comm, struct ncclIbGidQpState* out);
ncclResult_t ncclIbCastGidGetDevState(void* comm, int devIndex, int* state);
ncclResult_t ncclIbCastGidGetRecoveryGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t ncclIbCastGidGetProbingGidIndex(void* comm, int devIndex, int* gidIndex);
ncclResult_t ncclIbCastGidDriveQpToError(void* comm, int qpIdx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENABLE_FAULT_INJECTION */

#endif /* RCCL_NET_IB_GID_INSPECT_H_ */
