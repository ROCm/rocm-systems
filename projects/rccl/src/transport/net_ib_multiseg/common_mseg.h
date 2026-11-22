/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_COMMON_MSEG_H_
#define NET_IB_COMMON_MSEG_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "ib_gid_utils.h"
#include "profiler/net_ib.h"
#include "ib_multiseg_mseg.h" // RCCL: struct rcclIbSegStats.

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"
#include <sys/utsname.h>

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"
#include "wqe_lat_mon_mseg.h"
#include "seg_p2p_mseg.h" // RCCL: shared CTS layout, per-segment MRs and the layout cache.

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
extern char IbMsegIfName[MAX_IF_NAME_SIZE + 1];
extern union ncclSocketAddress IbMsegIfAddr;

enum IbMsegRequestMatchingScheme {
  BY_INDEX = 0,
  BY_ID = 1,
};

struct IbMsegMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr* mr;
};

struct IbMsegMrCache {
  struct IbMsegMr* slots;
  int capacity, population;
};

extern int IbMsegNMergedDevs;
#define NCCL_IB_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE * NCCL_IB_MAX_DEVS_PER_NIC) + NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) IbMsegMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  int16_t railId;
  int16_t planeId;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct IbMsegStats {
  int fatalErrorCount;
};

enum IbMsegProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

struct IbMsegGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

extern int IbMsegNDevs;

enum IbMsegRailPolicy {
  NCCL_IB_RAIL_POLICY_NONE,
  NCCL_IB_RAIL_POLICY_CX9_FLIP, // Enable CX9 rail flipped layout
  NCCL_IB_RAIL_POLICY_CX9_ALT, // Enable CX9 rail alternate layout
  NCCL_IB_RAIL_POLICY_CX9_BLOCK, // Enable CX9 rail blocked layout
};

struct alignas(64) IbMsegDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint32_t vendorId;
  uint32_t vendorPartId;
  char fwVer[64];
  uint8_t portNum;
  uint8_t link;
  int speed;
  uint64_t currSpeed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char fullPciPath[PATH_MAX];
  char* pciPath;
  int realPort;
  int maxQp;
  int maxMr; // RCCL: devAttr.max_mr, for rcclIbMrAdmit
  float latency;
  struct IbMsegMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  uint32_t oooRqSize;  // valid only when ar=1
  struct ibv_port_attr portAttr;
  struct IbMsegStats stats;
  int dmaBufSupported;
  int16_t railId;
  int16_t planeId;
  int16_t planeIdx;
  enum IbMsegProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
  struct IbMsegGidInfo gidInfo;
};

#define MAX_IB_DEVS 32
#define MAX_IB_VDEVS MAX_IB_DEVS * 8
extern struct IbMsegMergedDev IbMsegMergedDevs[MAX_IB_VDEVS];
extern struct IbMsegDev IbMsegDevs[MAX_IB_DEVS];
extern int IbMsegRelaxedOrderingEnabled;
extern uint64_t IbMsegSpeedChangeCounter;
extern int64_t ncclParamIbEventBasedLb();
extern int64_t ncclParamIbEventBasedLbRemote();

#define NCCL_IB_LLSTR(ll) \
  (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

struct alignas(32) IbMsegRemoteSpeedBuf {
  volatile uint64_t counter;
  uint16_t speedGbps[NCCL_IB_MAX_DEVS_PER_NIC];
};
static_assert(sizeof(IbMsegRemoteSpeedBuf) == 32);

// Per-Dev connection metadata
struct IbMsegDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // The key used for remote access to the addr exchanged by the peers
  // in IbMsegConnectionMetadata::addr
  // This member is populated differently on the sender and on the receiver
  // side.
  // The sender side populates this member with the RKey obtained after it
  // registered the CTS FIFO (on the specific device).
  // The receiver side populates this member with the RKey obtained after it
  // registered the completion records structure (on the specific device).
  uint32_t rkey;

  // remote dev info
  union ibv_gid remoteGid;
  int ibv_dev_index;

  uint64_t currSpeed;
  uint32_t remSpeedBufRkey;
};

#define MAX_QPS_PER_REQ 8
struct IbMsegProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  ncclProfilerNetIbDescr_v1_t data;
  void* pHandle;
};

#define NCCL_NET_IB_MAX_RECVS 8

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
#define NCCL_NET_IB_REQ_GIN_IGET 5
#define NCCL_NET_IB_REQ_FAILED 6
extern const char* IbMsegReqTypeStr[];

// Maximal number of QPs a communicator can have for data transfers
#define NCCL_IB_MAX_QPS 128

// Tracks data transfers between sender and receiver. A multi-recv/send uses a
// single record.
struct IbMsegRequestCompletionRecord {
  // This array communicates data transfer sizes from the sender to the
  // receiver. The sender writes the size of each completed data transfer to
  // this array. The receiver reads these sizes before reporting completion of
  // the corresponding receive request to the user.
  int sizes[NCCL_NET_IB_MAX_RECVS];
  // The receiver fills this array to signal the completion of a data transfer.
  // The sender can then read this array to see the receiver's status. If the
  // sender detects an error or device failure, it reads this array to
  // determine if the receiver considered the transfer complete. This prevents
  // the sender from retransmitting data if the failure was only visible on the
  // sender's side. Based on the array's contents, the sender decides if, how,
  // and on which QPs/devices to replay the transfer.
  bool completions[NCCL_IB_MAX_QPS];
};

struct IbMsegMrHandle; // RCCL: defined below with one MR per segment.

struct IbMsegRequest {
  struct IbMsegNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  // Array of counters. Each element in the array is populated with the expected
  // number of completion events that the request is expecting to be generated
  // on the device corresponding to the index of the element. After the request
  // is initialized and posted, for every completion event generated by a
  // device, the corresponding counter is decremented. When the counter reaches
  // zero it means that the request was fully completed on that device.
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to the per-device base structures to make it easier to
  // poll the device's CQ when the request is tested for progress.
  // The pointers are initialized only for the devices that the request expects
  // to receive completions from.
  struct IbMsegNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct IbMsegProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  uint64_t id;
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      struct IbMsegMrHandle* mh; // RCCL: segmented send reads per-segment lkeys from this handle.
      // Tracks whether data was transmitted on a QP for this request.
      bool sentData[NCCL_IB_MAX_QPS];
      // Per-device LB weights used for chunk computation.
      uint8_t weights[NCCL_IB_MAX_DEVS_PER_NIC];
    } send;
    struct {
      struct IbMsegRequestCompletionRecord* cmplsRecords;
      // Aggregates the size of a send request when sender does not write to the
      // completion records array.
      int aggSize;
      bool flagFlush; // The sender writes the GDR flush flag for this request
    } recv;
    struct {
      int rank;
    } iput;
    struct {
      int rank;
    } iget;
  };
  void* rmaProxyCtx;
};

struct IbMsegNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct IbMsegGidInfo gidInfo;
  // Resolved once at device init and reused by every QP (like the GID index above).
  int pkeyIndex;
};

// Snapshot the device-wide GID info into a comm's per-device base under a mutex.
static inline void IbMsegGidInfoSnapshot(struct IbMsegNetCommDevBase* base, struct IbMsegDev* ibDev) {
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  base->gidInfo = ibDev->gidInfo;
}

// The sender polls idx and then reads the other fields, so every field must share
// idx's 64-byte line: a field in a second line can still be stale when idx is seen.
struct alignas(64) IbMsegSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint16_t nreqs;
  uint16_t flagFlush; // Receiver asks for the GDR flush flag write after this payload
  uint32_t tag;
  uint64_t idx;
};
static_assert(sizeof(struct IbMsegSendFifo) == 64, "IbMsegSendFifo must fit in one 64-byte line");

// RCCL: multi-segment CTS slots. The layout prefix sits immediately before the 64-byte
// entries, so one RDMA write ends at idx. A single-segment write starts at cts.
struct IbMsegCtsSlot {
  struct IbMsegSegLayout layout[NCCL_NET_IB_MAX_RECVS];
  struct IbMsegSendFifo cts[NCCL_NET_IB_MAX_RECVS];
};
static_assert(offsetof(struct IbMsegCtsSlot, cts) == sizeof(struct IbMsegSegLayout) * NCCL_NET_IB_MAX_RECVS,
              "CTS entries must follow the layout prefix with no gap");
static_assert(offsetof(struct IbMsegCtsSlot, cts) % 64 == 0, "CTS entries stay 64-byte aligned");

// RCCL: byte size of the legacy CTS FIFO, and of one multi-segment slot array.
#define NCCL_IB_CTS_FIFO_BYTES \
  (sizeof(struct IbMsegSendFifo) * (size_t)NET_IB_MAX_REQUESTS * (size_t)NCCL_NET_IB_MAX_RECVS)
#define NCCL_IB_CTS_SLOTS_BYTES (sizeof(struct IbMsegCtsSlot) * (size_t)NET_IB_MAX_REQUESTS)

// RCCL: a segmented send posts one slice of each receive per QP and resumes on a later test.
enum rcclIbSendQpState : uint8_t { RCCL_IB_SEND_QP_IDLE, RCCL_IB_SEND_QP_INFLIGHT, RCCL_IB_SEND_QP_DONE };
// RCCL: slots post in fifo order. queued slots follow slot and wait until it posts every tail.
struct rcclIbSendCursor {
  int slot;
  int active; // slot still has slices to post
  int queued;
  int nqps;
  uint64_t wrId;
  uint32_t immData;
  bool needSizesWr;
  bool extraImmWr;
  bool flagFlush;
  struct IbMsegQp* qp[NCCL_IB_MAX_QPS];
  int qpIndex[NCCL_IB_MAX_QPS];
  enum rcclIbSendQpState qpState[NCCL_IB_MAX_QPS];
  uint32_t off[NCCL_IB_MAX_QPS][NCCL_NET_IB_MAX_RECVS];
  uint32_t end[NCCL_IB_MAX_QPS][NCCL_NET_IB_MAX_RECVS];
};

struct IbMsegQpInitAttr {
  ibv_qp_state state;
  int pkeyIndex;
  uint8_t portNum;
  int qpAccessFlags;
};

struct IbMsegQpRtrAttr {
  enum ibv_mtu mtu;
  uint8_t linkLayer;
  uint8_t tc;
  int sl;

  uint32_t remoteQpNum;
  uint32_t remoteLid;
  union ibv_gid remoteGid;

  uint8_t localIbPort;
  uint8_t localPortFlags;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

struct IbMsegQpRtsAttr {
  int timeout;
  int retryCnt;
};

struct IbMsegQp {
  struct ibv_qp* qp;
  // The index of the device on which this QP was created on.
  int devIndex;

  // The ECE (enhanced connection establishment) used on this QP.
  // Note: This is the reduced ECE exchanged between the sender and receiver.
  struct ibv_ece ece;
  int eceSupported;

  // Stores the attributes used to configure the QP to allow QP restore after
  // failure.
  struct IbMsegQpInitAttr initAttr;
  struct IbMsegQpRtrAttr rtrAttr;
  struct IbMsegQpRtsAttr rtsAttr;

  // The index of the device on the remote side to which this QP is connected
  // to.
  int remDevIdx;
  struct IbMsegWqeLatMon latMon;
};

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256,
              "request id are encoded in wr_id and we need up to 8 requests ids per completion");

// Structure to describe the completion records on the sender side.
struct IbMsegRemCompletionsRecords {
  // A "shadow" structure of the receiver's completion records in which the
  // sender tracks the completion records locally on its side. Sender uses this
  // memory to place the records it writes/reads to/from the receiver's
  // completion records.
  int elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  // Address in memory of the completion records structure on the receiver side.
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the sender chooses the
  // RKey (depending on the device being used) when it accesses the receiver's
  // completion records structure.
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
};

// A per-dev struct for netIbSendComm
struct alignas(8) IbMsegSendCommDev {
  struct IbMsegNetCommDevBase base;
  struct ibv_mr* ctsFifoMr;
  struct ibv_mr* putSignalScratchpadMr;
  struct ibv_mr* cmplsRecordsMr;
  struct ibv_mr* flagFlushSrcMr;
  struct ibv_sge sge;
};

// Wrapper to track an MR per-device, if needed
// RCCL: multi-segment buffers add one MR per segment; mrs[] aliases segment 0.
struct IbMsegMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct rcclIbSegMr seg; // RCCL: per-segment MRs; seg.nSegments 1 is the legacy single MR
};
static_assert(NCCL_IB_MAX_DEVS_PER_NIC == RCCL_IB_MSEG_MAX_DEVS && NCCL_NET_IB_MAX_RECVS == RCCL_IB_MSEG_MAX_RECVS,
              "the shared multi-segment tables use the plugin's device and receive counts");

// Forward declaration
struct IbMsegResiliency;

struct alignas(32) IbMsegNetCommBase {
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  struct IbMsegRequest reqs[NET_IB_MAX_REQUESTS];
  struct IbMsegQp qps[NCCL_IB_MAX_QPS];
  // Array of pointers to the "actual" QPs that are used for data transfers.
  // The pointers point to QPs in the IbMsegNetCommBase::qps[] array.
  struct IbMsegQp* activeQps[NCCL_IB_MAX_QPS];
  uint64_t fifoHead;
  int nqps;
  int splitDataOnQps;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  bool remOooRq;
  bool localOooRq;
  int recvMatchingScheme;
  int nDataQps;
  struct IbMsegDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct IbMsegStats stats;
  struct IbMsegResiliency* resiliency;
  uint64_t speedChangeCounter;
  uint64_t totalSpeed;
  uint8_t weights[NCCL_IB_MAX_DEVS_PER_NIC];
  uint64_t devSpeeds[NCCL_IB_MAX_DEVS_PER_NIC];
  bool isRma; // RCCL: GIN/RMA proxy comm (connected with nQpsPerDev > 0)
};

// Compute per-device LB weights (1-100); weight is never 0 since it is not considered a speed update but rather a port down.
static inline void IbMsegComputeLbWeights(struct IbMsegNetCommBase* base) {
  int ndevs = base->vProps.ndevs;
  // totalSpeed can not be 0: devices with inactive ports (speed 0) are
  // skipped at init, and speed-to-zero events are skipped in
  // IbMsegUpdateDeviceSpeed (port-failover handles those).
  uint8_t totalWeight = 0;
  for (int d = 0; d < ndevs; d++) {
    base->weights[d] = ncclParamIbEventBasedLb() ? (base->devSpeeds[d] * 100 / base->totalSpeed) : (100 / ndevs);
    totalWeight += base->weights[d];
  }
  if (totalWeight < 100) {
    base->weights[ndevs - 1] += (100 - totalWeight);
  }
}

struct IbMsegNetCommDevBase* IbMsegGetNetCommDevBase(IbMsegNetCommBase* base, int devIndex);
// RCCL: false when nMr more MRs would pass any rail's max_mr. Counts this process's cached MRs only.
bool rcclIbMrAdmit(IbMsegNetCommBase* base, int nMr);

static inline void IbMsegComputeDevSpeeds(struct IbMsegNetCommBase* base) {
  uint64_t totalSpeed = 0;
  for (int d = 0; d < base->vProps.ndevs; d++) {
    int ibDevN = IbMsegGetNetCommDevBase(base, d)->ibDevN;
    int remDevIdx = base->qps[d].remDevIdx;
    uint64_t localSpeed = COMPILER_ATOMIC_LOAD(&IbMsegDevs[ibDevN].currSpeed, std::memory_order_relaxed);
    uint64_t remoteSpeed = (base->remDevs[remDevIdx].currSpeed > 0) ? base->remDevs[remDevIdx].currSpeed : localSpeed;
    base->devSpeeds[d] = std::min(localSpeed, remoteSpeed);
    totalSpeed += base->devSpeeds[d];
  }
  base->totalSpeed = totalSpeed;
}

// qpIndex is the index relative to a device.
// For example, if a device has 2 QPs, qpIndex can be 0 or 1.
static inline ncclResult_t IbMsegCommBaseGetQpByIndex(struct IbMsegNetCommBase* commBase, int devIndex, int qpIndex,
                                                      IbMsegQp** qp) {
  if (devIndex < 0 || devIndex >= commBase->vProps.ndevs) {
    WARN("NET/IB: Invalid device index %d, expected [0, %d)", devIndex, commBase->vProps.ndevs);
    return ncclInternalError;
  }
  *qp = commBase->activeQps[commBase->vProps.ndevs * qpIndex + devIndex];
  return ncclSuccess;
}

// Each request is transferred over all devices, and depending on the
// "splitDataOnQps" configuration parameter, a request may be transferred over
// a single QP per device or on all QPs of each device.
static inline ncclResult_t IbMsegCommBaseGetNqpsPerRequest(struct IbMsegNetCommBase* baseComm, int* nQps) {
  if (nQps == NULL) {
    WARN("NET/IB: nQps output parameter is NULL");
    return ncclInternalError;
  }
  if (baseComm->nDataQps == -1) {
    WARN("NET/IB: nDataQps is not initialized");
    return ncclInternalError;
  }
  if (baseComm->nqps == -1) {
    WARN("NET/IB: nqps is not initialized");
    return ncclInternalError;
  }
  *nQps = (baseComm->splitDataOnQps == 1) ? baseComm->nqps : baseComm->nDataQps;
  return ncclSuccess;
}

// The function selects the QP to be used for the request. The QP selected
// based on the request ID and also based on the provided QP index. A request
// can be posted on multiple QPs. For example, if a request is posted on 4
// QPs, this function should be called 4 times, each time with a different
// qpIndex, ranging from 0 to 3.
// The function outputs the selected QP in the outQp argument and populates the
// outQpIndex argument with the index of the selected QP. Note that the
// outQpIndex is the index of the QP in the base::qps[] array.
static inline ncclResult_t IbMsegCommBaseGetQpForRequest(struct IbMsegNetCommBase* baseComm, const uint64_t id,
                                                         const uint8_t qpIndex, IbMsegQp** outQp, int* outQpIndex) {
  int nQps = 0;
  NCCLCHECK(IbMsegCommBaseGetNqpsPerRequest(baseComm, &nQps));
  *outQpIndex = (id * nQps + qpIndex) % baseComm->nqps;
  *outQp = baseComm->activeQps[*outQpIndex];
  if (*outQp == NULL) {
    WARN("NET/IB: QP is NULL for request id %lu, QP index %d", id, *outQpIndex);
    return ncclInternalError;
  }
  return ncclSuccess;
}

// Get a QP object from a QP number. If not NULL, qpIndex will also return the
// index of the QP in the IbMsegNetCommBase::qps[] array.
static inline ncclResult_t IbMsegCommBaseGetQpByQpNum(struct IbMsegNetCommBase* commBase, int devIndex, uint32_t qpNum,
                                                      IbMsegQp** qp, int* qpIndex) {
  if (devIndex < 0 || devIndex >= commBase->vProps.ndevs) {
    WARN("NET/IB: Invalid device index %d, expected [0, %d)", devIndex, commBase->vProps.ndevs);
    return ncclInternalError;
  }
  if (qp == NULL) {
    WARN("NET/IB: QP output pointer is NULL");
    return ncclInternalError;
  }
  TRACE(NCCL_NET, "NET/IB: %s: Looking for QP num %u on devIndex %d among %d QPs", __func__, qpNum, devIndex,
        commBase->nqps / commBase->vProps.ndevs);
  for (int qpIndexInDev = 0; qpIndexInDev < (commBase->nqps / commBase->vProps.ndevs); qpIndexInDev++) {
    *qp = &(commBase->qps[commBase->vProps.ndevs * qpIndexInDev + devIndex]);
    if ((*qp)->qp->qp_num == qpNum) {
      if (qpIndex != NULL) {
        *qpIndex = *qp - commBase->qps;
      }
      return ncclSuccess;
    }
  }
  *qp = NULL;
  return ncclInternalError;
}

static inline ncclResult_t IbMsegPostRecvWorkRequest(struct ibv_qp* qp, struct ibv_recv_wr* wr) {
  struct ibv_recv_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(qp, wr, &bad_wr));
  return ncclSuccess;
}

struct IbMsegSendComm {
  struct IbMsegNetCommBase base;
  // CTS FIFO from which the sender reads the Clear-to-Send (CTS) messages that
  // are written by the receiver (The receiver side writes into it upon
  // issuing a (multi-)receive request). Each row in the 2D array corresponds
  // to a single CTS message but can describe multiple recv-requests issued
  // on the receiver side.
  // RCCL: legacy FIFO when useMultiSeg is 0. ctsSlots replaces it otherwise,
  // and the published address is that array's first CTS row.
  struct IbMsegSendFifo (*ctsFifo)[NCCL_NET_IB_MAX_RECVS];
  struct IbMsegCtsSlot* ctsSlots;
  alignas(32) struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  // Data writes, plus the completion records write and the GDR flush flag write.
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 2];
  // Each dev correlates to a mergedIbDev
  struct IbMsegSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the send requests for faster access. The
  // pointers are pointing into requests stored in IbMsegNetCommBase::reqs[]
  // array. The requests are inserted to this array based on the "slot" they
  // are associated with.
  struct IbMsegRequest* sendReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

  // Counter per "slot" on how many send request were called for a multi-recv
  int sendReqsCnt[NET_IB_MAX_REQUESTS];
  struct IbMsegRemCompletionsRecords remCmplsRecords;
  int ar; // Use adaptive routing when all merged devices have it enabled
  uint64_t putSignalScratchpad;
  int useMultiSeg; // RCCL: cached connect decision; connect fails unless both ends agree, 1 uses ctsSlots.
  // RCCL: allocated with ctsSlots; a legacy-CTS comm allocates the cursor on its first segmented send.
  struct rcclIbSendCursor* sendCursor; // RCCL: the slot whose segmented send is still posting.
  struct rcclIbLayoutCache* layoutCache; // RCCL: receiver layouts that CTS entries referenced.
  struct rcclIbSegStats segStats;        // RCCL: classic and RMA segmented posts on this comm.

  struct {
    uint64_t posted;
    uint64_t completed;
  } ginSeq;

  struct IbMsegRemoteSpeedBuf remoteSpeedBuf;
  struct ibv_mr* remoteSpeedMr;
  uint64_t remoteSpeedCounter;
  // GDR flush flag (RCCL_IB_GDR_FLUSH_FLAG): after the payload of a slot, every
  // QP that carried it writes the slot sequence number from flagFlushSrc[slot]
  // into its own RO=0 GPU flag, at flagFlushRemAddr + qpIndex * NCCL_GDR_FLUSH_FLAG_STRIDE.
  int flagFlush;
  uint64_t flagFlushRemAddr;
  uint32_t flagFlushRkey;
  struct ibv_sge flagFlushSge;
  uint64_t flagFlushSrc[NET_IB_MAX_REQUESTS];
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct IbMsegNetCommBase) % 32) == 0,
              "IbMsegNetCommBase size must be 32-byte multiple to ensure ctsFifo is at proper offset");
static_assert((sizeof(struct IbMsegSendFifo) % 32) == 0, "IbMsegSendFifo element size must be 32-byte multiples");
static_assert(NCCL_IB_CTS_FIFO_BYTES % 64 == 0, "CTS FIFO size must keep the following layout lines 64-byte aligned");
static_assert((offsetof(struct IbMsegSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct IbMsegSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct IbMsegGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_mr* gpuMr;
  int* gpuFlushGpuMem;
  struct ibv_sge sge;
  struct IbMsegQp qp;
  int dmabuf_fd;
  // gpuFlushGpuMem comes from ncclMemAlloc on the cuMem path but from
  // hipExtMallocWithFlags on the HSA fallback, so the free has to match the allocator.
  bool gpuFlushMemIsHipAlloc;
};

// This structure describes the FIFO which the receiver uses when it sends CTS
// messages to the sender.
struct IbMsegRemCtsFifo {
  // A "shadow" structure of the sender's CTS FIFO in which the receiver tracks
  // the CTS FIFO locally on its side. Receiver uses this memory to place the
  // CTS messages and populates the RDMA message "gather address" with the
  // memory of the CTS message that is sent.
  // RCCL: a separate NCCL_IB_CTS_FIFO_BYTES block, NULL when useMultiSeg because ctsSlots holds the rows.
  struct IbMsegSendFifo (*elems)[NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the receiver chooses the
  // RKey (depending on the device being used) when it posts a CTS to the
  // sender
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
};

struct alignas(16) IbMsegRecvCommDev {
  struct IbMsegNetCommDevBase base;
  struct IbMsegGpuFlush gpuFlush;
  // MR that is obtained after registering the "shadow" CTS FIFO on the
  // receiver's side. The LKey of this MR allows RDMA operations on the receiver
  // side to gather CTS messages (formatted by the receiver) and write them to
  // the sender's CTS FIFO.
  struct ibv_mr* ctsFifoMr;
  // MR that is obtained after registering the completion records on the
  // receiver side. The RKey of this MR is provided to the sender side, to allow
  // the sender side to access receiver's completion records using RDMA
  // operations.
  struct ibv_mr* cmplsRecordsMr;
  // SGE to avoid allocation of SGE structures on the stack when receiver
  // posts RDMA operations. The SGE is populated by the address of the memory
  // in which the CTS message formatted on the receiver is placed.
  struct ibv_sge sge;
  struct ibv_mr* speedUpdateMr;
};

#define NCCL_IB_RECV_WR_ID_DUMMY UINT64_MAX
#define NCCL_IB_SPEED_UPDATE_WR_ID (UINT64_MAX - 1)

struct IbMsegRecvComm {
  struct IbMsegNetCommBase base;
  struct IbMsegRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the recv requests to allow faster access. The
  // pointers are pointing into requests stored in IbMsegNetCommBase::reqs[]
  // array. The requests are inserted to this array using a hash (modulo) on
  // their ID.
  struct IbMsegRequest* recvReqs[NET_IB_MAX_REQUESTS];
  // Structure to hold all the related structures regarding the CTS FIFO
  // structure.
  struct IbMsegRemCtsFifo remCtsFifo;
  // RCCL: local combined slots when useMultiSeg. NULL keeps the legacy elems shadow.
  struct IbMsegCtsSlot* ctsSlots;
  uint64_t ctsSlotStride; // RCCL: bytes from one remote CTS row to the next. 0 means the legacy stride.
  int useMultiSeg; // RCCL: cached connect decision; connect fails unless both ends agree, 1 uses ctsSlots.
  // Structure to hold all the completion records of all the outstanding
  // receive requests on the receiver side.
  struct IbMsegRequestCompletionRecord cmplsRecords[NET_IB_MAX_REQUESTS];
  int gpuFlushHostMem;
  int flushEnabled;
  int flushQpSl;
  int flushQpTc;
  bool flushQpsCreated;
  // GDR flush flag (RCCL_IB_GDR_FLUSH_FLAG): the sender writes an RO=0 flag in
  // GPU memory after the payload and the GPU kernel waits on it, replacing
  // the RDMA read flush.
  int flagFlush;
  // Whether the next receives need the flag (only SIMPLE GDR receives are
  // flushed); set by net.cc before each irecv.
  int flagFlushHint;
  uint64_t* flagFlushGpuMem;
  struct ibv_mr* flagFlushMr;
  int flagFlushDmabufFd;
  // Packed gdrFlushSeq entry of the receive that requested the flag and was
  // completed last: its sequence and the per-QP flags that carried it.
  uint64_t flagFlushLastDesc;
  bool prepostReceiveWorkRequests;
  // To avoid allocation and memset on the data-path a single structure is used
  // and only the wr_id is updated before posting a receive work request.
  struct ibv_recv_wr ibRecvWorkRequest;

  uint64_t remSpeedBufAddr;
  struct IbMsegRemoteSpeedBuf speedUpdateBuf;
  uint16_t lastSentSpeeds[NCCL_IB_MAX_DEVS_PER_NIC];
  uint64_t lastSentCounter;
  bool postedSpeedUpdate;
  struct {
    uint64_t posted;
    uint64_t completed;
  } ginSeq;
};
static_assert((offsetof(struct IbMsegRecvComm, remCtsFifo) % 32) == 0,
              "IbMsegRecvComm ctsFifo must be 32-byte aligned");

ncclResult_t IbMsegBaseCommInit(struct IbMsegNetCommBase* baseComm, bool isSend);
ncclResult_t IbMsegRecvCommInit(struct IbMsegRecvComm* recvComm);
ncclResult_t IbMsegSendCommInit(struct IbMsegSendComm* sendComm);

struct IbMsegListenComm {
  int dev;
  struct ncclSocket sock;
  struct IbMsegCommStage* stage;
  // Accepted receivers set up the GDR flush flag (ncclNetIbMseg only, see IbMsegNetCtx).
  int gdrFlushFlag;
  bool isGin; // RCCL: set by rcclIbMarkGinListen; accepted GIN comms skip the multi-segment layout.
};

static inline void IbMsegCheckSpeedChanges(struct IbMsegSendComm* sendComm, struct IbMsegNetCommBase* base) {
  bool speedChanged = false;
  // Local speed change detection
  if (base->speedChangeCounter != COMPILER_ATOMIC_LOAD(&IbMsegSpeedChangeCounter, std::memory_order_acquire)) {
    base->speedChangeCounter = COMPILER_ATOMIC_LOAD(&IbMsegSpeedChangeCounter, std::memory_order_acquire);
    speedChanged = true;
  }
  // Remote speed change detection
  if (sendComm->remoteSpeedCounter !=
      COMPILER_ATOMIC_LOAD(&sendComm->remoteSpeedBuf.counter, std::memory_order_acquire)) {
    sendComm->remoteSpeedCounter = COMPILER_ATOMIC_LOAD(&sendComm->remoteSpeedBuf.counter, std::memory_order_acquire);
    for (int i = 0; i < base->nRemDevs; i++) {
      base->remDevs[i].currSpeed = (uint64_t)sendComm->remoteSpeedBuf.speedGbps[i] * 1000;
    }
    speedChanged = true;
  }
  if (speedChanged) {
    IbMsegComputeDevSpeeds(base);
    IbMsegComputeLbWeights(base);
  }
}

static ncclResult_t IbMsegStatsInit(struct IbMsegStats* stat) {
  COMPILER_ATOMIC_STORE(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void IbMsegStatsFatalError(struct IbMsegStats* stat) {
  COMPILER_ATOMIC_FETCH_ADD(&stat->fatalErrorCount, 1, std::memory_order_relaxed);
}
static void IbMsegQpFatalError(struct ibv_qp* qp) {
  IbMsegStatsFatalError((struct IbMsegStats*)qp->qp_context);
}
static void IbMsegCqFatalError(struct ibv_cq* cq) {
  IbMsegStatsFatalError((struct IbMsegStats*)cq->cq_context);
}
static void IbMsegDevFatalError(struct IbMsegDev* dev) {
  IbMsegStatsFatalError(&dev->stats);
}
ncclResult_t IbMsegStatsCheckFatalCount(struct IbMsegStats* stat, const char* funcName);

extern ncclProfilerCallback_t IbMsegProfilerFunction;

extern std::thread IbMsegAsyncThread;
void* IbMsegAsyncThreadMain(void* args);

ncclResult_t IbMsegGdrSupport();
ncclResult_t IbMsegPeerMemSupport();
ncclResult_t IbMsegDmaBufSupport(int dev);

void IbMsegAddEvent(struct IbMsegRequest* req, int devIndex);

// Check if request has any pending events (IBV operations in flight)
static inline bool IbMsegRequestHasEvents(struct IbMsegRequest* r) {
  for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
    if (r->events[i] != 0) return true;
  }
  return false;
}

ncclResult_t IbMsegGetGidIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                               int* gidIndex);
ncclResult_t IbMsegGetPkeyIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                int* pkeyIndex);
ncclResult_t IbMsegGidInfoQuery(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                struct IbMsegGidInfo* gidInfo);
ncclResult_t IbMsegGetRequest(struct IbMsegNetCommBase* base, struct IbMsegRequest** req);
ncclResult_t IbMsegFreeRequest(struct IbMsegRequest* r);

ncclResult_t IbMsegRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle);

int IbMsegGetTrafficClass(void* ctx);
void IbMsegSetTrafficClass(void* ctx, int trafficClass);

// ncclNetIbMseg per-communicator context. config stays first: the traffic class
// accessors cast the context to it. Any context passed to ncclNetIbMseg.listen,
// including GIN/RMA's, must be a full IbMsegNetCtx.
struct IbMsegNetCtx {
  ncclNetCommConfig_t config;
  // Every rank of the communicator can use RCCL_IB_GDR_FLUSH_FLAG (set at init).
  int gdrFlushFlag;
};
// ncclNetIbMseg listen: IbMsegListen plus the context's GDR flush flag decision.
ncclResult_t IbMsegNetListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);

// CX9 rail policy helper. Returns NCCL_NET_ID_UNDEF when nDevs is not a positive multiple of eight.
int IbMsegCx9RailId(enum IbMsegRailPolicy policy, int devId, int nDevs);

// Net IB plugin entry functions.
ncclResult_t IbMsegInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t IbMsegInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction,
                        ncclProfilerCallback_t profFunction);
ncclResult_t IbMsegDevices(int* ndev);
ncclResult_t IbMsegGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t IbMsegGetPhysProperties(int dev, ncclNetProperties_t* props);
ncclResult_t IbMsegListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
// RCCL: mark a GIN listener; the flag rides in the handle so both ends skip the multi-segment layout.
void rcclIbMarkGinListen(void* opaqueHandle, void* listenComm);
// nQpsPerDev > 0: that many QPs per device. <= 0: NCCL_IB_QPS_PER_CONNECTION / RCCL_IB_QPS_PER_P2P.
ncclResult_t IbMsegConnectImpl(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                               ncclNetDeviceHandle_t** sendDevComm, int nQpsPerDev, int envTrafficClass);
ncclResult_t IbMsegConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                           ncclNetDeviceHandle_t** sendDevComm);
ncclResult_t IbMsegAcceptImpl(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm, int nQpsPerDev);
ncclResult_t IbMsegAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm);
ncclResult_t IbMsegRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t IbMsegRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t IbMsegDeregMr(void* comm, void* mhandle);
ncclResult_t IbMsegIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                         void** request);
ncclResult_t IbMsegIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                         void** request);
ncclResult_t IbMsegIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t IbMsegCreateFlushQp(struct IbMsegRecvComm* comm);
ncclResult_t IbMsegTest(void* request, int* done, int* sizes);
ncclResult_t IbMsegCloseSend(void* sendComm);
ncclResult_t IbMsegCloseRecv(void* recvComm);
ncclResult_t IbMsegCloseListen(void* listenComm);
ncclResult_t IbMsegMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t IbMsegFinalizeDevices(void);
ncclResult_t IbMsegFinalize(void* ctx);
ncclResult_t IbMsegSetNetAttr(void* ctx, ncclNetAttr_t* netAttr);

static inline void printIbWcStatusHint(int status) {
  switch (status) {
  case IBV_WC_LOC_PROT_ERR:
    INFO(NCCL_NET,
         "HINT: In many cases this error indicates that ACS is enabled, which breaks the GPU Direct RDMA protocol.");
    INFO(NCCL_NET, "HINT: To confirm, set NCCL_NET_GDR_LEVEL=0; if that resolves it, "
                   "disable ACS following your vendor documentation.");
    return;
  case IBV_WC_WR_FLUSH_ERR:
    INFO(NCCL_NET, "HINT: In many cases this error indicates that NICs on the same node cannot talk to each other.");
    INFO(NCCL_NET, "HINT: To confirm, use a lower level tool like ib_write_bw to communicate across NICs on the same "
                   "node.");
    return;
  case IBV_WC_RETRY_EXC_ERR:
    INFO(NCCL_NET, "HINT: In many cases this error indicates that NCCL_IB_TIMEOUT is set too short (the default value "
                   "is 20, which is ~30 seconds before timing out).");
    INFO(NCCL_NET, "HINT: To confirm, increase NCCL_IB_TIMEOUT (see "
                   "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-timeout).");
    return;
  default:
    break;
  }
}

static uint16_t IbMsegExtractLocalSubnetPrefix(uint64_t subnet_prefix) {
  return ncclIbGidSubnetPrefix16(subnet_prefix);
}

static int IbMsegExtractFlid(union ibv_gid* gid) {
  return ncclIbGidFlid(gid->raw);
}

// RCCL: relaxed, since tests read the counters from another thread only after the transfer completed.
static inline void rcclIbSegStatsAdd(struct rcclIbSegStats* s, int nWrs, bool laterSeg) {
  COMPILER_ATOMIC_FETCH_ADD(&s->posts, (uint64_t)1, std::memory_order_relaxed);
  COMPILER_ATOMIC_FETCH_ADD(&s->wrs, (uint64_t)nWrs, std::memory_order_relaxed);
  if (laterSeg) COMPILER_ATOMIC_FETCH_ADD(&s->laterSegPosts, (uint64_t)1, std::memory_order_relaxed);
}

static inline void rcclIbSegStatsRead(const struct rcclIbSegStats* s, struct rcclIbSegStats* out) {
  out->posts = COMPILER_ATOMIC_LOAD(&s->posts, std::memory_order_relaxed);
  out->wrs = COMPILER_ATOMIC_LOAD(&s->wrs, std::memory_order_relaxed);
  out->laterSegPosts = COMPILER_ATOMIC_LOAD(&s->laterSegPosts, std::memory_order_relaxed);
}

#endif
