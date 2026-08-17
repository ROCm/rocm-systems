/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_CAST_INSPECT_H_
#define RCCL_NET_IB_CAST_INSPECT_H_

#include <stdint.h>
#include "net_ib_limits.h"
#include "plugin/nccl_net.h"  /* NCCL_NET_MAX_DEVS_PER_NIC */

#ifdef __cplusplus
#include "nccl.h"  /* ncclResult_t */
extern "C" {
#else
#include "nccl.h"
#endif

/*
 * Test-only introspection API for the net-ib-cast WRR scheduler.
 * Shared between librccl.so and the unit tests so the struct layout
 * and signatures cannot diverge.
 */

struct ncclIbCastSchedState {
  int nqps;
  int qpIndex;          /* current WRR cursor */
  int initTotTokens;
  int initQpTokens[NCCL_IB_MAX_QPS];
  int activeTotTokens;
  int activeQpTokens[NCCL_IB_MAX_QPS];
  uint32_t splitDataMin;
  bool schedInit;        /* true once IbCastQpSchedUpdateTx has fired */
  bool schedEnable;
  bool doWrr;
  bool splitData;
};

/* Copy scheduler state out of a connected sendComm.
 * Returns ncclInvalidArgument on null pointers. */
ncclResult_t ncclIbCastGetSchedState(void* sendComm, struct ncclIbCastSchedState* out);

/* Force-initialize the WRR token table, bypassing RTT-driven scheduling.
 * nqps must match the connection's nqps. */
ncclResult_t ncclIbCastSetTokens(void* sendComm, const int* qpTokens, int nqps);

/* Override schedParms; takes effect on the next isend, no reconnect needed. */
ncclResult_t ncclIbCastSetSchedParms(void* sendComm, bool schedEnable, bool doWrr, bool splitData,
                                     uint32_t splitDataMin);

/* ── Test-only wrappers over the backported feature logic (host-only, no HW) ──
 * These forward to the internal static helpers in init.cc / connect.cc so the
 * unit tests exercise the real code, not a copy. GIDs are passed as raw 16-byte
 * arrays to keep this header free of <infiniband/verbs.h>. */

/* plane/rail: dedup a plane ID into a compact plane index. Mirrors the sort-loop
 * call IbCastGetPlaneIndex(). Returns ncclInvalidUsage on overflow / bad ID. */
ncclResult_t ncclIbCastTestGetPlaneIndex(int devPlane, int16_t* count, int16_t* planes, int16_t* idx);

/* plane/rail: derive the merged railId/planeId of a fused vNIC from its physical
 * devices' (railId, planeId, planeIdx) triples. Forwards to the real
 * IbCastMergedRailPlane() used by IbCastMakeVDeviceInternal. Returns
 * ncclInvalidArgument on null args or ndevs outside [1, NCCL_NET_MAX_DEVS_PER_NIC]. */
ncclResult_t ncclIbCastTestMergedRailPlane(const int16_t* railIds, const int16_t* planeIds, const int16_t* planeIdxs,
                                           int ndevs, int16_t* outRailId, int16_t* outPlaneId);

/* subnet detection: 1 if two RoCE GIDs share a subnet, else 0. prefixLen in
 * [1,32] applies to IPv4-mapped GIDs; native IPv6 compares the 64-bit prefix. */
int ncclIbCastTestGidSameSubnet(const uint8_t localGid[16], const uint8_t remoteGid[16], int prefixLen);

/* subnet detection: 1 if localGid shares a subnet with ANY of nRemote GIDs
 * (each 16 bytes, laid out contiguously), else 0. */
int ncclIbCastTestSubnetMatchesAny(const uint8_t localGid[16], const uint8_t* remoteGids, int nRemote, int prefixLen);

/* ── Resiliency state introspection (requires ENABLE_FAULT_INJECTION) ── */
#ifdef ENABLE_FAULT_INJECTION

struct ncclIbCastResiliencyState {
  bool recoveryEnabled;
  bool inProgress;
  int outstandingRequests;
  int outstandingRecovery;
  int ndevs;
  int devState[NCCL_NET_MAX_DEVS_PER_NIC];
  int recoveryCount[NCCL_NET_MAX_DEVS_PER_NIC];
};

/* Fills out with the current resiliency state of the communicator.
 * sendComm must have resiliency enabled (NCCL_IB_RESILIENCY_PORT_FAILOVER=1).
 * Returns ncclInvalidArgument if resiliency context is NULL. */
ncclResult_t ncclIbCastGetResiliencyState(void* sendComm, struct ncclIbCastResiliencyState* out);

/* Returns the number of times IbCastResiliencyRepostRequest was called
 * (i.e. selective retransmit count). Counter is stored in the resiliency
 * context and incremented in p2p_resiliency.cc. */
ncclResult_t ncclIbCastGetRepostCount(void* sendComm, int* out);

#endif /* ENABLE_FAULT_INJECTION */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_NET_IB_CAST_INSPECT_H_ */
