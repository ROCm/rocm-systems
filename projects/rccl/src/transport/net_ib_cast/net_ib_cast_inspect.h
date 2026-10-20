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

/* *out is 1 when optional recv completion is enabled. */
ncclResult_t ncclIbCastGetOptRecvCompletion(void* comm, int* out);

/* Force-initialize the WRR token table, bypassing RTT-driven scheduling.
 * nqps must match the connection's nqps. */
ncclResult_t ncclIbCastSetTokens(void* sendComm, const int* qpTokens, int nqps);

/* Override schedParms; takes effect on the next isend, no reconnect needed. */
ncclResult_t ncclIbCastSetSchedParms(void* sendComm, bool schedEnable, bool doWrr, bool splitData,
                                     uint32_t splitDataMin);

/* ── Hardware capabilities of a vNic device, probed once and cached. ── */
struct ncclIbCastDeviceCaps {
  bool udSupported;
  bool rdmaReadSupported;
};

ncclResult_t ncclIbCastGetDeviceCaps(int dev, struct ncclIbCastDeviceCaps* out);

/* ── Test-only wrappers over internal static helpers (host-only, no HW). ── */
ncclResult_t ncclIbCastTestGetPlaneIndex(int devPlane, int16_t* count, int16_t* planes, int16_t* idx);
int ncclIbCastTestGidSameSubnet(const uint8_t localGid[16], const uint8_t remoteGid[16], int prefixLen);
int ncclIbCastTestSubnetMatchesAny(const uint8_t localGid[16], const uint8_t* remoteGids, int nRemote, int prefixLen);

/* IbCastSpeed(active_speed|active_speed_ex bitmask) -> Mb/s, via the static ibvSpeeds[] table. */
int ncclIbCastTestSpeed(int speed);

/* *fromQuery is 1 if IbCastDevs[dev].speed came from ibv_query_port_speed(),
 * 0 if it came from the active_speed(_ex)/active_width fallback.
 * Returns ncclInvalidArgument if dev is out of range or fromQuery is null. */
ncclResult_t ncclIbCastTestSpeedSource(int dev, int* fromQuery);

/* Number of raw (pre-merge) devices in IbCastDevs[], i.e. the valid range
 * [0, ncclIbCastTestGetNDevs()) for ncclIbCastTestSpeedSource()'s dev arg. */
int ncclIbCastTestGetNDevs(void);

/* GRH (global route header) state per active QP, read back from the driver.
 * isGlobal is queried from the live QP via ibv_query_qp (ah_attr.is_global), so
 * it reflects what the driver actually programmed at RTR, not just intent.
 * linkLayer is the rtrAttr.linkLayer used to configure the QP. */
struct ncclIbCastGrhState {
  int      nqps;
  uint8_t  linkLayer[NCCL_IB_MAX_QPS];  /* IBV_LINK_LAYER_ETHERNET (RoCE) / _INFINIBAND */
  uint8_t  isGlobal[NCCL_IB_MAX_QPS];   /* ah_attr.is_global read back from the QP */
  bool     queryOk[NCCL_IB_MAX_QPS];    /* false if ibv_query_qp failed for this QP */
  uint32_t qpNum[NCCL_IB_MAX_QPS];      /* ibv_qp.qp_num for CI diagnostics */
};

/* Copy per-QP GRH state out of a connected send or recv comm.
 * Returns ncclInvalidArgument on null pointers. */
ncclResult_t ncclIbCastGetGrhState(void* sendComm, struct ncclIbCastGrhState* out);

/* ── Multiplane test-only wrappers (host-only, no HW). ── */

/* Convert an IPv4 or IPv6 address string to the 16-byte ibv_gid representation.
 * IPv4 addresses are stored as IPv4-mapped IPv6 (::ffff:x.x.x.x).
 * Returns ncclInvalidArgument on unparseable input. */
ncclResult_t ncclIbCastTestIpToGid(const char* ipStr, uint8_t gid[16]);

/* Convert a 16-byte GID to colon-separated hex string
 * (e.g. "fe80:0000:...:0002").  buf must be at least 40 bytes. */
void ncclIbCastTestGidToString(const uint8_t gid[16], char* buf, int bufLen);

/* Reset multiplane module state so IbCastMultiplaneLoad() can be called again.
 * Test-only: not thread-safe, must not be called while other threads may be
 * loading or querying multiplane state. */
void ncclIbCastTestMultiplaneReset(void);

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
