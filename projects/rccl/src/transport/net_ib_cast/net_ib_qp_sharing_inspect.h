/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_QP_SHARING_INSPECT_H_
#define RCCL_NET_IB_QP_SHARING_INSPECT_H_

#include <stdint.h>
#include "net_ib_limits.h"

#ifdef __cplusplus
#include "nccl.h"  /* ncclResult_t */
extern "C" {
#else
#include "nccl.h"
#endif

/*
 * Test-only introspection API for QP sharing (RCCL_IB_COMM_NGROUPS).
 * Shared between librccl.so and the unit tests so the struct layout
 * and signatures cannot diverge.
 */

struct ncclIbQpSharingState {
  uint16_t commId;             /* 0 = not shared */
  bool     isSharedQpPrimary;
  int      sharedGroupIdx;     /* -1 = not shared */
  int      refcount;           /* comms sharing this physical QP (0 if not shared) */
  int      cqRefcount;         /* comms sharing this group's CQ (0 if not shared) */
  int      nqps;
  uint32_t qpn[NCCL_IB_MAX_QPS];
};

/* Copy QP-sharing state out of a connected sendComm or recvComm.
 * Returns ncclInvalidArgument on null pointers. */
ncclResult_t ncclIbCastGetQpSharingState(void* comm, struct ncclIbQpSharingState* out);

/* Effective QP-sharing configuration as decided by the net plugin's init(),
 * after its own adjustments (e.g. sharing is turned off when NGROUPS < 1).
 * Only meaningful once init() has run. */
struct ncclIbQpSharingConfig {
  bool enabled;            /* IbCastQpSharingEnabled() */
  int  ngroups;            /* RCCL_IB_COMM_NGROUPS */
  int  depthMultiplier;    /* IbCastQpSharingDepthMultiplier() */
  bool resiliencyEnabled;  /* NCCL_IB_RESILIENCY_PORT_FAILOVER or _RECOVERY */
};

/* Returns ncclInvalidArgument on a null pointer. */
ncclResult_t ncclIbCastGetQpSharingConfig(struct ncclIbQpSharingConfig* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_NET_IB_QP_SHARING_INSPECT_H_ */
