/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * QP Sharing pool management implementation for IB CAST transport layer.
 ************************************************************************/

#include "qp_sharing.h"

// QP sharing configuration parameters
RCCL_PARAM(IbCastCommNGroups, "IB_COMM_NGROUPS", 0);
RCCL_PARAM(IbCastQpDepthMultiplier, "IB_QP_DEPTH_MULTIPLIER", 1);

// Pool and comm table globals
struct IbCastSharedQp       g_IbCastSharedQpPool[IBCAST_MAX_SHARED_QPS];
int                         g_IbCastSharedQpPoolCount = 0;
struct IbCastCommTableEntry g_IbCastCommTable[IBCAST_MAX_COMMS];
uint16_t                    g_IbCastNextCommId = 1;   // 0 reserved for "not shared"
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
    struct ncclIbNetCommDevBase* primaryDevBase, int devIndex, int initialRefcount) {

    if (g_IbCastSharedQpPoolCount >= IBCAST_MAX_SHARED_QPS) {
        WARN("IB CAST QP Sharing: pool full (%d entries)", IBCAST_MAX_SHARED_QPS);
        return NULL;
    }
    struct IbCastSharedQp* entry = &g_IbCastSharedQpPool[g_IbCastSharedQpPoolCount++];
    entry->key = *key;
    entry->qp = qp;
    entry->primaryCq = primaryCq;
    entry->primaryDevBase = primaryDevBase;
    entry->devIndex = devIndex;
    entry->refcount = initialRefcount;
    entry->cqRefcount = 0;
    entry->used = true;
    entry->ctsQpSlot = IBCAST_CTS_QP_SLOT_INVALID;
    return entry;
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
    if (g_IbCastNextCommId >= IBCAST_MAX_COMMS) {
        WARN("NET/IB: commId pool exhausted (max %d), falling back to non-sharing", IBCAST_MAX_COMMS);
        return 0;
    }
    uint16_t id = g_IbCastNextCommId++;
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
  if (rcclParamIbCastCommNGroups() > 0) {
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
            if (g_IbCastSharedQpPool[i].primaryDevBase) {
                int ibDevN2 = g_IbCastSharedQpPool[i].primaryDevBase->ibDevN;
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
    }
}
