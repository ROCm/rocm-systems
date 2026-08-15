/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * QP Sharing infrastructure for IB CAST transport layer.
 * Allows multiple RCCL communicator channels to share physical IB QPs,
 * reducing QP count and improving scalability.
 * Controlled via NCCL_IB_COMM_NGROUPS (0=disabled).
 ************************************************************************/

#ifndef NET_IB_CAST_QP_SHARING_H_
#define NET_IB_CAST_QP_SHARING_H_

#include "common_cast.h"
#include "param.h"
#include <mutex>

// QP sharing configuration parameters
// Accessible as RCCL_IB_COMM_NGROUPS / NCCL_IB_COMM_NGROUPS
// 0 = disabled, >0 = enable QP sharing with N groups
extern int64_t rcclParamIbCastCommNGroups();
// CQ/WR depth scaling for primary QPs
extern int64_t rcclParamIbCastQpDepthMultiplier();

#define IBCAST_MAX_SHARED_QPS 1024
#define IBCAST_MAX_COMMS      4096
#define IBCAST_CTS_QP_SLOT_INVALID 0xFF
#define IBCAST_FLUSH_QP_IDX  -1   // sentinel qpIdx for flush QPs in the shared pool

// Pool key -- uniquely identifies a shared QP
struct IbCastSharedQpKey {
    int             ibDevN;              // local IB device index
    union ncclSocketAddress peerAddr;    // remote peer address (port stripped)
    int             remIbDevIdx;         // remote IB device index for disambiguation
    bool            isSend;              // send vs recv direction
    int             groupIdx;            // sharing group (0..m-1)
    int             qpIdx;              // QP slot within the group (0..nqps-1)
};

// Pool entry -- one per physical shared QP
struct IbCastSharedQp {
    IbCastSharedQpKey key;
    struct ibv_qp*    qp;                  // the physical QP
    struct ibv_cq*    primaryCq;           // CQ for this device in this group
    struct ncclIbNetCommDevBase* primaryDevBase;  // device base of primary comm's CQ
    int      devIndex;                     // local device index within comm->devs[]
    int      refcount;                     // number of comms using this QP
    int      cqRefcount;                   // comms using this group's CQs (tracked on qpIdx==0 only)
    bool     used;                         // slot in use
    int8_t   ctsQpSlot;                    // CTS signaling slot from primary
};

// Global comm table for completion routing (commId -> comm pointer)
struct IbCastCommTableEntry {
    void*    comm;          // ncclIbSendComm* or ncclIbRecvComm*
    bool     isSend;
    bool     used;
};

// Pool and comm table globals (defined in qp_sharing.cc)
extern struct IbCastSharedQp       g_IbCastSharedQpPool[IBCAST_MAX_SHARED_QPS];
extern int                         g_IbCastSharedQpPoolCount;
extern struct IbCastCommTableEntry g_IbCastCommTable[IBCAST_MAX_COMMS];
extern uint16_t                    g_IbCastNextCommId;
extern std::mutex                  g_IbCastSharedQpMutex;

// Strip port from socket address for peer matching
void IbCastStripPort(union ncclSocketAddress* addr);

// Compare two pool keys
bool IbCastSharedQpKeyMatch(const IbCastSharedQpKey* a, const IbCastSharedQpKey* b);

// Find a shared QP by key
struct IbCastSharedQp* IbCastFindSharedQp(const IbCastSharedQpKey* key);

// Find a shared QP by QP number and direction (for teardown)
struct IbCastSharedQp* IbCastFindSharedQpByQpn(uint32_t qpn, bool isSend);

// Register a new shared QP in the pool
struct IbCastSharedQp* IbCastRegisterSharedQp(const IbCastSharedQpKey* key,
    struct ibv_qp* qp, struct ibv_cq* primaryCq,
    struct ncclIbNetCommDevBase* primaryDevBase, int devIndex, int initialRefcount);

// Count QP slots registered by the primary for a given group
int IbCastCountGroupQpSlots(const union ncclSocketAddress* peerAddr,
    int remIbDevIdx, bool isSend, int groupIdx);

// Count total comms connected to a peer (for channelSeq computation)
int IbCastCountPeerTotalRefcount(int ibDevN, const union ncclSocketAddress* peerAddr,
    int remIbDevIdx, bool isSend);

// Allocate a commId and register in the global comm table (mutex-protected)
uint16_t IbCastAllocCommId(void* comm, bool isSend);

// Free a commId (self-locking; for callers NOT holding g_IbCastSharedQpMutex)
void IbCastFreeCommId(uint16_t commId);

// Free a commId; caller MUST already hold g_IbCastSharedQpMutex (teardown paths)
void IbCastFreeCommIdLocked(uint16_t commId);

// Destroy all CQs for a group when cqRefcount reaches 0
void IbCastCleanupGroupCqs(struct IbCastSharedQp* slot0Entry);

// Strip the commId from wr_id[63:48], recovering the original index. Safe when
// sharing is disabled (commId==0, so the mask is a no-op).
static inline uint64_t IbCastStripCommId(uint64_t wr_id) {
  return wr_id & ~((uint64_t)WR_ID_RX_COMM_ID_MASK << WR_ID_RX_COMM_ID_SHIFT);
}

// Look up the target comm from the commId encoded in wr_id[63:48]. Returns NULL
// if sharing is disabled or the commId is invalid.
struct ncclIbNetCommBase* IbCastRouteCommFromWrId(uint64_t wr_id);

// Look up the target comm from the commId encoded in immData.
// Returns target base based on commId in immData if sharing is enabled or
// returns NULL if sharing is disabled or commId is invalid.
struct ncclIbNetCommBase* IbCastRouteCommFromImmData(
    struct ncclIbNetCommBase* base, uint32_t immDataHost);

#endif // NET_IB_CAST_QP_SHARING_H_
