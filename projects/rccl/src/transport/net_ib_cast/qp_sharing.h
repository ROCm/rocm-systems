/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * QP Sharing infrastructure for IB CAST transport layer.
 * Allows multiple RCCL communicator channels to share physical IB QPs,
 * reducing QP count and improving scalability.
 * Controlled via RCCL_IB_QP_SHARING_ENABLE (0=disabled, 1=enabled).
 * When enabled, RCCL_IB_COMM_NGROUPS (default 4) and
 * RCCL_IB_QP_DEPTH_MULTIPLIER (default 8) tune sharing behavior.
 ************************************************************************/

#ifndef NET_IB_CAST_QP_SHARING_H_
#define NET_IB_CAST_QP_SHARING_H_

#include "common_cast.h"
#include "param.h"
#include <mutex>

// QP sharing configuration parameters
// Master switch: RCCL_IB_QP_SHARING_ENABLE
// 0 = disabled (default), 1 = enabled with sensible defaults
extern int64_t rcclParamIbCastQpSharingEnable();
// Number of sharing groups (default 4, effective only when master switch is on)
extern int64_t rcclParamIbCastCommNGroups();
// CQ/WR depth scaling for primary QPs (default 8, effective only when master switch is on)
extern int64_t rcclParamIbCastQpDepthMultiplier();

// QP sharing role for a comm during connection setup.
// Returned by IbCastQpSharing{Sender,Receiver}Setup() to tell the caller
// whether to create QPs, register them, or skip both.
enum IbCastQpSharingRole {
  QP_SHARING_NONE,       // sharing disabled or fallback — normal QP creation
  QP_SHARING_PRIMARY,    // primary — create QPs with scaled depth, then register
  QP_SHARING_SECONDARY,  // secondary — QPs already assigned, skip creation
};

// Global runtime enable flag — set from master switch param during init,
// can be cleared if configuration is invalid (e.g. NGROUPS < 1).
extern bool IbCastQpSharingGlobalEnable;

// Returns true when QP sharing is enabled at runtime.
static inline bool IbCastQpSharingEnabled(void) {
  return IbCastQpSharingGlobalEnable;
}

// Returns true when this comm is actively sharing QPs (sharing enabled AND
// commId was successfully allocated; commId==0 means fallback to non-sharing).
static inline bool IbCastCommIsSharing(const struct ncclIbNetCommBase* base) {
  return IbCastQpSharingEnabled() && base->commId != 0;
}

// Returns true when this comm is the primary (owner) of shared QPs in its group.
static inline bool IbCastCommIsPrimary(const struct ncclIbNetCommBase* base) {
  return base->commId != 0 && base->isSharedQpPrimary;
}

// Returns true when this comm is a secondary (reuses QPs owned by a primary).
static inline bool IbCastCommIsSecondary(const struct ncclIbNetCommBase* base) {
  return base->commId != 0 && !base->isSharedQpPrimary;
}

// Initialize QP sharing fields on a comm base to defaults (sharing disabled).
static inline void IbCastCommInitSharingFields(struct ncclIbNetCommBase* base) {
  base->commId = 0;
  base->isSharedQpPrimary = false;
  base->sharedGroupIdx = -1;
  base->remIbDevIdx = -1;
  base->sharedPrimaryNqps = 0;
}

// Compute CQ/WR depth multiplier for shared QPs.
// Returns 1 when sharing is disabled.
static inline int IbCastQpSharingDepthMultiplier(void) {
  return IbCastQpSharingEnabled() ? std::max((int)rcclParamIbCastQpDepthMultiplier(), 1) : 1;
}

#define IBCAST_MAX_SHARED_QPS 1024
#define IBCAST_MAX_COMMS      4096
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
    int      primaryIbDevN;                // local IB device index of the primary comm's device,
                                            // captured by value: the primary comm (and its inline
                                            // devs[]) may be freed while secondaries or this pool
                                            // entry still reference the group
    int      devIndex;                     // local device index within comm->devs[]
    int      refcount;                     // number of comms using this QP
    int      cqRefcount;                   // comms using this group's CQs (tracked on qpIdx==0 only)
    bool     used;                         // slot in use
    // Note: ctsQpSlot is NOT tracked in the shared pool.  CTS offload is
    // mutually exclusive with QP sharing (disabled at init when sharing is on).
    // In the sharing-enabled, non-offload path, CTS signaling uses
    // (slot == ctsQp->devIndex) to decide when to set IBV_SEND_SIGNALED,
    // which relies only on the per-QP devIndex already stored here — no
    // additional per-pool ctsQpSlot is needed.
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
extern int                         g_IbCastSharedQpFreeStack[IBCAST_MAX_SHARED_QPS];
extern int                         g_IbCastSharedQpFreeTop;
extern struct IbCastCommTableEntry g_IbCastCommTable[IBCAST_MAX_COMMS];
extern uint16_t                    g_IbCastNextCommId;
extern uint16_t                    g_IbCastCommIdFreeStack[IBCAST_MAX_COMMS];
extern int                         g_IbCastCommIdFreeTop;
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
    int primaryIbDevN, int devIndex, int initialRefcount);

// Undo a registration for a slot whose last user is closing outside of
// IbCastCleanupGroupCqs (the flush-QP path).
void IbCastUnregisterSharedQpLocked(struct IbCastSharedQp* entry);

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

// Validate that all shared QP pool entries and commIds are cleaned up.
// Logs leaked entries and asserts on non-zero counts. Call at finalize time.
// Gated by RCCL_IB_QP_SHARING_VALIDATE_POOL=1.
extern int64_t rcclParamIbCastQpSharingValidatePool();
void IbCastValidateSharedQpPool(void);

// Encode commId into wr_id[63:48]. When commId==0 (sharing disabled or
// fallback) this is a no-op (OR with zero).
static inline uint64_t IbCastEncodeCommId(uint64_t wr_id, uint16_t commId) {
  return wr_id | ((uint64_t)commId << WR_ID_RX_COMM_ID_SHIFT);
}

// Encode receiver commId into immData for BY_ID matching scheme:
//   bits[7:0]  = reqId,  bits[23:8] = remCommId.
static inline uint32_t IbCastEncodeCommIdImmData(uint32_t reqId, uint16_t remCommId) {
  return (reqId & WR_IMM_BYID_REQ_ID_MASK) |
         (((uint32_t)remCommId & WR_IMM_BYID_COMM_ID_MASK) << WR_IMM_BYID_COMM_ID_SHIFT);
}

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
