/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * QP Sharing pool management implementation for IB CAST transport layer.
 ************************************************************************/

#include "qp_sharing.h"
#include <cassert>

// QP sharing configuration parameters
RCCL_PARAM(IbCastQpSharingEnable, "IB_QP_SHARING_ENABLE", 0);  // master switch: 0=off, 1=on
RCCL_PARAM(IbCastCommNGroups, "IB_COMM_NGROUPS", 4);            // number of sharing groups (effective only when master switch is on)
RCCL_PARAM(IbCastQpDepthMultiplier, "IB_QP_DEPTH_MULTIPLIER", 8); // CQ/WR depth multiplier for shared QPs
RCCL_PARAM(IbCastQpSharingValidatePool, "IB_QP_SHARING_VALIDATE_POOL", 0); // 1 = validate pool at shutdown for leaks

// Global runtime enable flag — initialized from master switch param,
// can be cleared during init if configuration is invalid (e.g. NGROUPS < 1).
bool IbCastQpSharingGlobalEnable = false;

// Pool and comm table globals
struct IbCastSharedQp       g_IbCastSharedQpPool[IBCAST_MAX_SHARED_QPS];
int                         g_IbCastSharedQpPoolCount = 0;
int                         g_IbCastSharedQpFreeStack[IBCAST_MAX_SHARED_QPS];
int                         g_IbCastSharedQpFreeTop = 0;
struct IbCastCommTableEntry g_IbCastCommTable[IBCAST_MAX_COMMS];
uint16_t                    g_IbCastNextCommId = 1;   // 0 reserved for "not shared"
uint16_t                    g_IbCastCommIdFreeStack[IBCAST_MAX_COMMS];
int                         g_IbCastCommIdFreeTop = 0;
std::mutex                  g_IbCastSharedQpMutex;

void IbCastStripPort(union ncclSocketAddress* addr) {
    if (addr->sa.sa_family == AF_INET) {
        addr->sin.sin_port = 0;
    } else if (addr->sa.sa_family == AF_INET6) {
        addr->sin6.sin6_port = 0;
    }
}

bool IbCastSharedQpKeyMatch(const IbCastSharedQpKey* a, const IbCastSharedQpKey* b) {
    if (a->ibDevN != b->ibDevN) return false;
    if (a->isSend != b->isSend) return false;
    if (a->groupIdx != b->groupIdx) return false;
    if (a->qpIdx != b->qpIdx) return false;
    if (a->remIbDevIdx != b->remIbDevIdx) return false;
    if (memcmp(&a->peerAddr, &b->peerAddr, sizeof(union ncclSocketAddress)) != 0) return false;
    return true;
}

struct IbCastSharedQp* IbCastFindSharedQp(const IbCastSharedQpKey* key) {
    for (int i = 0; i < g_IbCastSharedQpPoolCount; i++) {
        if (g_IbCastSharedQpPool[i].used && IbCastSharedQpKeyMatch(&g_IbCastSharedQpPool[i].key, key)) {
            return &g_IbCastSharedQpPool[i];
        }
    }
    return NULL;
}

struct IbCastSharedQp* IbCastFindSharedQpByQpn(uint32_t qpn, bool isSend) {
    for (int i = 0; i < g_IbCastSharedQpPoolCount; i++) {
        if (g_IbCastSharedQpPool[i].used && g_IbCastSharedQpPool[i].qp &&
            g_IbCastSharedQpPool[i].key.isSend == isSend &&
            g_IbCastSharedQpPool[i].qp->qp_num == qpn) {
            return &g_IbCastSharedQpPool[i];
        }
    }
    return NULL;
}

struct IbCastSharedQp* IbCastRegisterSharedQp(const IbCastSharedQpKey* key,
    struct ibv_qp* qp, struct ibv_cq* primaryCq,
    int primaryIbDevN, int devIndex, int initialRefcount) {

    std::lock_guard<std::mutex> lock(g_IbCastSharedQpMutex);
    int idx;
    if (g_IbCastSharedQpFreeTop > 0) {
        // Reuse a slot freed by IbCastCleanupGroupCqs -- O(1)
        idx = g_IbCastSharedQpFreeStack[--g_IbCastSharedQpFreeTop];
    } else if (g_IbCastSharedQpPoolCount < IBCAST_MAX_SHARED_QPS) {
        idx = g_IbCastSharedQpPoolCount++;
    } else {
        WARN("IB CAST QP Sharing: pool full (%d entries)", IBCAST_MAX_SHARED_QPS);
        return NULL;
    }
    struct IbCastSharedQp* entry = &g_IbCastSharedQpPool[idx];
    entry->key = *key;
    entry->qp = qp;
    entry->primaryCq = primaryCq;
    entry->primaryIbDevN = primaryIbDevN;
    entry->devIndex = devIndex;
    entry->refcount = initialRefcount;
    entry->cqRefcount = 0;
    entry->used = true;
    return entry;
}

void IbCastUnregisterSharedQpLocked(struct IbCastSharedQp* entry) {
    if (entry == NULL) return;
    int idx = (int)(entry - g_IbCastSharedQpPool);
    entry->used = false;
    g_IbCastSharedQpFreeStack[g_IbCastSharedQpFreeTop++] = idx;
}

int IbCastCountGroupQpSlots(const union ncclSocketAddress* peerAddr,
    int remIbDevIdx, bool isSend, int groupIdx) {
    int count = 0;
    for (int i = 0; i < g_IbCastSharedQpPoolCount; i++) {
        if (!g_IbCastSharedQpPool[i].used) continue;
        if (g_IbCastSharedQpPool[i].key.isSend != isSend) continue;
        if (g_IbCastSharedQpPool[i].key.groupIdx != groupIdx) continue;
        if (g_IbCastSharedQpPool[i].key.remIbDevIdx != remIbDevIdx) continue;
        if (memcmp(&g_IbCastSharedQpPool[i].key.peerAddr, peerAddr, sizeof(union ncclSocketAddress)) == 0) {
            count++;
        }
    }
    return count;
}

int IbCastCountPeerTotalRefcount(int ibDevN, const union ncclSocketAddress* peerAddr,
    int remIbDevIdx, bool isSend) {
    int total = 0;
    for (int i = 0; i < g_IbCastSharedQpPoolCount; i++) {
        if (!g_IbCastSharedQpPool[i].used) continue;
        if (g_IbCastSharedQpPool[i].key.qpIdx != 0) continue;
        if (g_IbCastSharedQpPool[i].key.isSend != isSend) continue;
        if (g_IbCastSharedQpPool[i].key.remIbDevIdx != remIbDevIdx) continue;
        if (memcmp(&g_IbCastSharedQpPool[i].key.peerAddr, peerAddr, sizeof(union ncclSocketAddress)) == 0) {
            total += g_IbCastSharedQpPool[i].refcount;
        }
    }
    return total;
}

uint16_t IbCastAllocCommId(void* comm, bool isSend) {
    std::lock_guard<std::mutex> lock(g_IbCastSharedQpMutex);
    uint16_t id;
    if (g_IbCastCommIdFreeTop > 0) {
        // Reuse a previously freed commId — O(1)
        id = g_IbCastCommIdFreeStack[--g_IbCastCommIdFreeTop];
    } else if (g_IbCastNextCommId < IBCAST_MAX_COMMS) {
        // Allocate a fresh commId — O(1)
        id = g_IbCastNextCommId++;
    } else {
        WARN("NET/IB: commId pool exhausted (max %d), falling back to non-sharing", IBCAST_MAX_COMMS);
        return 0;
    }
    g_IbCastCommTable[id].comm = comm;
    g_IbCastCommTable[id].isSend = isSend;
    g_IbCastCommTable[id].used = true;
    return id;
}

// Caller MUST hold g_IbCastSharedQpMutex. Used by the teardown paths, which take
// the mutex across the whole shared-QP cleanup block.
void IbCastFreeCommIdLocked(uint16_t commId) {
    if (commId > 0 && commId < IBCAST_MAX_COMMS) {
        g_IbCastCommTable[commId].used = false;
        g_IbCastCommTable[commId].comm = NULL;
        g_IbCastCommIdFreeStack[g_IbCastCommIdFreeTop++] = commId;
    }
}

struct ncclIbNetCommBase* IbCastRouteCommFromWrId(uint64_t wr_id) {
  uint16_t commId = (wr_id >> WR_ID_RX_COMM_ID_SHIFT) & WR_ID_RX_COMM_ID_MASK;
  if (commId == 0 || commId >= IBCAST_MAX_COMMS || !g_IbCastCommTable[commId].used) return NULL;
  return g_IbCastCommTable[commId].isSend
    ? &((struct ncclIbSendComm*)g_IbCastCommTable[commId].comm)->base
    : &((struct ncclIbRecvComm*)g_IbCastCommTable[commId].comm)->base;
}

struct ncclIbNetCommBase* IbCastRouteCommFromImmData(struct ncclIbNetCommBase* base, uint32_t immDataHost) {
  if (IbCastQpSharingEnabled()) {
    uint16_t immCommId = (immDataHost >> WR_IMM_BYID_COMM_ID_SHIFT) & WR_IMM_BYID_COMM_ID_MASK;
    //uint8_t reqSlot = immDataHost & WR_IMM_BYID_REQ_ID_MASK;
    if (immCommId != 0 && immCommId < IBCAST_MAX_COMMS && g_IbCastCommTable[immCommId].used) {
      return g_IbCastCommTable[immCommId].isSend
        ? &((struct ncclIbSendComm*)g_IbCastCommTable[immCommId].comm)->base
        : &((struct ncclIbRecvComm*)g_IbCastCommTable[immCommId].comm)->base;
    }
  }
  return NULL;
}

// Self-locking variant for callers that do NOT already hold the mutex
// (e.g. the connect/accept non-sharing fallback paths).
void IbCastFreeCommId(uint16_t commId) {
    if (commId > 0 && commId < IBCAST_MAX_COMMS) {
        std::lock_guard<std::mutex> lock(g_IbCastSharedQpMutex);
        IbCastFreeCommIdLocked(commId);
    }
}

void IbCastCleanupGroupCqs(struct IbCastSharedQp* slot0Entry) {
    struct ibv_cq* destroyedCqs[NCCL_IB_MAX_DEVS_PER_NIC];
    int nDestroyed = 0;

    for (int i = 0; i < g_IbCastSharedQpPoolCount; i++) {
        if (!g_IbCastSharedQpPool[i].used) continue;
        // Skip flush QP slots — they are torn down separately before CQ cleanup
        if (g_IbCastSharedQpPool[i].key.qpIdx == IBCAST_FLUSH_QP_IDX) continue;
        if (g_IbCastSharedQpPool[i].key.isSend != slot0Entry->key.isSend) continue;
        if (g_IbCastSharedQpPool[i].key.groupIdx != slot0Entry->key.groupIdx) continue;
        if (g_IbCastSharedQpPool[i].key.remIbDevIdx != slot0Entry->key.remIbDevIdx) continue;
        if (memcmp(&g_IbCastSharedQpPool[i].key.peerAddr, &slot0Entry->key.peerAddr,
                   sizeof(union ncclSocketAddress)) != 0) continue;

        struct ibv_cq* cq = g_IbCastSharedQpPool[i].primaryCq;
        bool alreadyDestroyed = false;
        for (int j = 0; j < nDestroyed; j++) {
            if (destroyedCqs[j] == cq) { alreadyDestroyed = true; break; }
        }
        if (!alreadyDestroyed && cq != NULL) {
            INFO(NCCL_NET, "IB CAST TEARDOWN: destroying CQ %p group=%d ibDevN=%d isSend=%d remIbDev=%d qpIdx=%d refcount=%d",
                 (void*)cq, g_IbCastSharedQpPool[i].key.groupIdx, g_IbCastSharedQpPool[i].key.ibDevN,
                 g_IbCastSharedQpPool[i].key.isSend, g_IbCastSharedQpPool[i].key.remIbDevIdx,
                 g_IbCastSharedQpPool[i].key.qpIdx, g_IbCastSharedQpPool[i].refcount);
            ncclResult_t cqRet = wrap_ibv_destroy_cq(cq);
            if (cqRet != ncclSuccess) {
                WARN("IB CAST TEARDOWN: ibv_destroy_cq FAILED cq=%p errno=%d group=%d ibDevN=%d qpIdx=%d refcount=%d",
                     (void*)cq, errno, g_IbCastSharedQpPool[i].key.groupIdx, g_IbCastSharedQpPool[i].key.ibDevN,
                     g_IbCastSharedQpPool[i].key.qpIdx, g_IbCastSharedQpPool[i].refcount);
            }
            destroyedCqs[nDestroyed++] = cq;
            {
                int ibDevN2 = g_IbCastSharedQpPool[i].primaryIbDevN;
                std::lock_guard<std::mutex> lock(IbCastDevs[ibDevN2].mutex);
                INFO(NCCL_NET, "IB CAST TEARDOWN: pdRefs ibDevN=%d: %d->%d %s",
                     ibDevN2, IbCastDevs[ibDevN2].pdRefs, IbCastDevs[ibDevN2].pdRefs - 1,
                     (IbCastDevs[ibDevN2].pdRefs == 1) ? "DEALLOC" : "");
                if (0 == --IbCastDevs[ibDevN2].pdRefs) {
                    wrap_ibv_dealloc_pd(IbCastDevs[ibDevN2].pd);
                }
            }
        }
        g_IbCastSharedQpPool[i].used = false;
        g_IbCastSharedQpFreeStack[g_IbCastSharedQpFreeTop++] = i;
    }
}

void IbCastValidateSharedQpPool(void) {
    if (!IbCastQpSharingEnabled()) return;

    std::lock_guard<std::mutex> lock(g_IbCastSharedQpMutex);
    int leakedSlots = 0;
    int leakedCommIds = 0;

    // Check QP pool for leaked entries
    for (int i = 0; i < g_IbCastSharedQpPoolCount; i++) {
        struct IbCastSharedQp* slot = &g_IbCastSharedQpPool[i];
        if (slot->used) {
            WARN("IB CAST POOL LEAK: slot=%d qpIdx=%d group=%d isSend=%d refcount=%d cqRefcount=%d qp=%p",
                 i, slot->key.qpIdx, slot->key.groupIdx, slot->key.isSend,
                 slot->refcount, slot->cqRefcount, (void*)slot->qp);
            leakedSlots++;
        }
    }

    // Check comm table for leaked commIds
    for (int i = 1; i < IBCAST_MAX_COMMS; i++) {
        if (g_IbCastCommTable[i].used) {
            WARN("IB CAST POOL LEAK: commId=%d isSend=%d comm=%p still in use",
                 i, g_IbCastCommTable[i].isSend, g_IbCastCommTable[i].comm);
            leakedCommIds++;
        }
    }

    if (leakedSlots == 0 && leakedCommIds == 0) {
        INFO(NCCL_NET, "IB CAST POOL VALIDATE: clean shutdown — pool=%d/%d slots used, freeStack=%d, nextCommId=%u",
             0, g_IbCastSharedQpPoolCount, g_IbCastCommIdFreeTop, g_IbCastNextCommId);
    } else {
        WARN("IB CAST POOL VALIDATE: UNCLEAN shutdown — %d leaked QP slots, %d leaked commIds", leakedSlots, leakedCommIds);
        assert(leakedSlots == 0 && "QP sharing pool has leaked QP slots at shutdown");
        assert(leakedCommIds == 0 && "QP sharing pool has leaked commIds at shutdown");
    }
}
