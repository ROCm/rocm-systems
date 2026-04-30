/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_COMMON_H_
#define NET_IB_COMMON_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"

#include <climits>
#include <iostream>
#include <cfloat>
#include <sys/utsname.h>
#include "ionic/ionicdvwrap.h"
#include "graph/xml.h"
#include "nccl_common.h"

// NCCL 2.30 MR flag: suppress relaxed-ordering for a specific registration.
// Callers in this codebase always pass 0 for mrFlags, so this is never set.
#ifndef NCCL_NET_MR_FLAG_FORCE_SO
#define NCCL_NET_MR_FLAG_FORCE_SO 0x1ULL
#endif

// Map NCCL 2.30 C++ atomic-style macros to GCC built-ins used in this codebase
#ifndef COMPILER_ATOMIC_STORE
#define COMPILER_ATOMIC_STORE(ptr, val, order) __atomic_store_n((ptr), (val), __ATOMIC_RELAXED)
#endif
#ifndef COMPILER_ATOMIC_LOAD
#define COMPILER_ATOMIC_LOAD(ptr, order) __atomic_load_n((ptr), __ATOMIC_RELAXED)
#endif
#ifndef COMPILER_ATOMIC_FETCH_ADD
#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add((ptr), (val), __ATOMIC_RELAXED)
#endif

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
extern char IbCastIfName[MAX_IF_NAME_SIZE+1];
extern union ncclSocketAddress IbCastIfAddr;

enum ncclIbRequestMatchingScheme {
  BY_INDEX=0,
  BY_ID=1,
  BY_ORDER=2, // send requests are posted in the order they are received (CTS OFFLOAD)
};

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

extern int IbCastNMergedDevs;
#define NCCL_IB_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct ncclIbStats {
  int fatalErrorCount;
};

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

extern int IbCastNDevs;
struct alignas(64) ncclIbDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char fullPciPath[PATH_MAX];
  char* pciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  uint32_t oooRqSize;  // valid only when ar=1
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
  enum ncclIbProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
};

#define MAX_IB_DEVS  32
#define MAX_IB_VDEVS MAX_IB_DEVS*8
extern struct ncclIbMergedDev IbCastMergedDevs[MAX_IB_VDEVS];
extern struct ncclIbDev IbCastDevs[MAX_IB_DEVS];
extern int IbCastRelaxedOrderingEnabled;
extern bool IbCastUseInline;


#define WR_IMM_RX_REQ_IDX_MASK  0xff
#define WR_IMM_RX_REQ_IDX_SHIFT 24
#define WR_IMM_SPLIT_DATA_FLAG  0x00800000
#define WR_IMM_SIZE_MASK        0x007fffff
extern int IbCastGdrFlushDisable;
extern bool IbCastAinicRoce;
extern bool rcclCtsInlineData;
extern bool IbCastOffloadEnabled;
extern int64_t rcclParamIbCastP2pDisableCts();

#define NCCL_IB_LLSTR(ll) (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // The key used for remote access to the addr exchanged by the peers
  // in ncclIbConnectionMetadata::addr
  // This member is populated differently on the sender and on the receiver
  // side.
  // The sender side populates this member with the RKey obtained after it
  // registered the CTS FIFO (on the specific device).
  // The receiver side populates this member with the RKey obtained after it
  // registered the completion records structure (on the specific device).
  uint32_t rkey;

  //remote dev info
  union ibv_gid remoteGid;
  int ibv_dev_index;
};

// Retain local RoCE address for error logging
struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

#define MAX_QPS_PER_REQ 8
struct ncclProfilerInfo {
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
extern const char* IbCastReqTypeStr[];

// Maximal number of QPs a communicator can have for data transfers
#define NCCL_IB_MAX_QPS 128

struct ncclIbQpSchedParms {
    bool enable;
    bool wrrEnable;
    uint64_t updateInterval; // in nsec
    uint64_t resetInterval;  // in nsec
    double weightNew;        // fractional weight applied to most recent RTT sample
    uint32_t splitDataMin;   // in bytes
    bool splitData;          // init from NCCL_IB_SPLIT_DATA_ON_QPS
    bool doWrr;
    bool resetRtt;
    bool logEnable;
    uint64_t logInterval;    // in nsec
  };
  extern struct ncclIbQpSchedParms castGlobalQpSchedParms;
  
  // Data about QP transmission
  struct ncclIbQpTxData {
    uint64_t startTimeNs;
    uint64_t bytes;
  };
  
  // For remapping work request ID so that additional info is avail at
  // completion time of sends
  struct ncclIbRemapWrId {
    int state; // in use or unused
    uint64_t origWrId;
    int qpIndex;
    struct ncclIbQpTxData tx;
    struct ncclIbQpSchedParms parms;
  };
  
  // Stats for scheduling QP transmissions
  struct ncclIbQpTxStats {
    uint64_t minRttSample;
    uint64_t totRtt;
    uint64_t numMeasurements;
    double rtt;
  };
  
  // Scratchpad for computing scheduler weights
  struct ncclIbQpTxSchedScratchpad {
    double rtt[NCCL_IB_MAX_QPS];
  };
  
  // Scheduler for QP transmissions
  struct ncclIbQpTxSched {
    double weight;    // fraction of sub-chunk to be transmitted on QP
    double minWeight; // min value of weight used
    double maxWeight; // max value of weight used
  };
  
  #define NCCL_IB_TARGET_TOT_TOKENS 100
  
  // Tokens for weighted round-robin QP scheduler
  struct ncclIbRrTokens {
    int totTokens;
    int qpTokens[NCCL_IB_MAX_QPS];
  };
  
  // Scheduler for weighted round-robin QP transmissions
  struct ncclIbRrQpTxSched {
    struct ncclIbRrTokens initTokens;
    struct ncclIbRrTokens activeTokens;
    int qpIndex;
  };
  
  // QP scheduling descriptor
  struct IbCastQpSchedDesc {
    bool wrrSched;
    int nqps;
    int startQpIndex;
    struct ncclIbQpSchedParms parms;
};


// Tracks data transfers between sender and receiver. A multi-recv/send uses a
// single record.
struct ncclIbRequestCompletionRecord {
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

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  // Array of counters. Each element in the array is populated with the expected
  // number of completion events that the request is expecting to be generated
  // on the device corresponding to the index of the element. After the request
  // is initialized and posted, for every completion event generated by a
  // device, the corresponding counter is decremented. When the counter reaches
  // zero it means that the request was fully completed on that device.
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  int ctsEvents[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to the per-device base structures to make it easier to
  // poll the device's CQ when the request is tested for progress.
  // The pointers are initialized only for the devices that the request expects
  // to receive completions from.
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  uint64_t id;
  int nreqs;
  struct IbCastQpSchedDesc desc;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      // Tracks whether data was transmitted on a QP for this request.
      bool sentData[NCCL_IB_MAX_QPS];
    } send;
    struct {
      struct ncclIbRequestCompletionRecord* cmplsRecords;
      // Aggregates the size of a send request when sender does not write to the
      // completion records array.
      int aggSize;
    } recv;
    struct {
      int rank;
    } iput;
    struct {
      int rank;
    } iget;
  };
  void* ginProxyCtx;
};

struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
};

struct alignas(64) ncclIbSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
  uint16_t rxReqIndex;
  char padding[14];
};

#define MAX_INLINE_DATA_SIZE 24
struct alignas(32) ncclIbSendFifoCtsInline {
  uint64_t addr;
  uint32_t rkeys[1];
  int size;
  uint8_t nreqs;
  uint16_t rxReqIndex;
  uint16_t tag;
  uint32_t idx;
  char padding[9];
} __attribute__((packed));

struct ncclIbQpInitAttr {
  ibv_qp_state state;
  int pkeyIndex;
  uint8_t portNum;
  int qpAccessFlags;
};

struct ncclIbQpRtrAttr {
  enum ibv_mtu mtu;
  uint8_t linkLayer;
  uint8_t tc;
  int sl;

  uint32_t remoteQpNum;
  uint32_t remoteLid;
  union ibv_gid remoteGid;

  uint8_t localIbPort;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

struct ncclIbQpRtsAttr {
  int timeout;
  int retryCnt;
};

struct ncclIbQp {
  struct ibv_qp* qp;
  // The index of the device on which this QP was created on.
  int devIndex;

  // The ECE (enhanced connection establishment) used on this QP.
  // Note: This is the reduced ECE exchanged between the sender and receiver.
  struct ibv_ece ece;
  int eceSupported;

  // Stores the attributes used to configure the QP to allow QP restore after
  // failure.
  struct ncclIbQpInitAttr initAttr;
  struct ncclIbQpRtrAttr rtrAttr;
  struct ncclIbQpRtsAttr rtsAttr;

  // The index of the device on the remote side to which this QP is connected
  // to.
  int remDevIdx;
  int8_t ctsQpSlot;
};

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

// Structure to describe the completion records on the sender side.
struct ncclIbRemCompletionsRecords {
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
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* ctsFifoMr;
  struct ibv_mr* putSignalScratchpadMr;
  struct ibv_mr* cmplsRecordsMr;
  struct ibv_sge sge;
};


// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Forward declaration
struct ncclIbResiliency;

struct alignas(32) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  struct ncclIbRequest reqs[NET_IB_MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  // Array of pointers to the "actual" QPs that are used for data transfers.
  // The pointers point to QPs in the ncclIbNetCommBase::qps[] array.
  struct ncclIbQp* activeQps[NCCL_IB_MAX_QPS];

  //NET-IB-CAST: QP scheduler state
  struct ncclIbRemapWrId remapWrId[NET_IB_MAX_REQUESTS];
  struct ncclIbQpTxStats qpTxStats[NCCL_IB_MAX_QPS];
  uint64_t nextQpTxStatsResetNs;
  struct ncclIbQpTxSched qpTxSched[NCCL_IB_MAX_QPS];
  struct ncclIbRrQpTxSched rrQpTxSched;
  bool qpTxSchedInit;
  uint64_t nextQpTxSchedUpdateNs;
  uint64_t nextSchedLogNs;
  int remapHead;
  uint64_t stagedParmsConEpoch;
  bool schedParmsInit;
  struct ncclIbQpSchedParms schedParms;
  bool resetRttDone;
  int qpIndex;
  int rxPosts[NCCL_IB_MAX_QPS * NCCL_NET_IB_MAX_RECVS];

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
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct ncclIbStats stats;
  struct ncclIbResiliency* resiliency;
};

struct ncclIbNetCommDevBase* IbCastGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex);

// qpIndex is the index relative to a device.
// For example, if a device has 2 QPs, qpIndex can be 0 or 1.
static inline ncclResult_t IbCastCommBaseGetQpByIndex(struct ncclIbNetCommBase* commBase, int devIndex, int qpIndex, ncclIbQp** qp) {
  assert(devIndex >= 0 && devIndex < commBase->vProps.ndevs);
  *qp = commBase->activeQps[commBase->vProps.ndevs*qpIndex + devIndex];
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
static inline ncclResult_t IbCastCommBaseGetQpForRequest(struct ncclIbNetCommBase* baseComm, const uint64_t id, const uint8_t qpIndex, ncclIbQp** outQp, int* outQpIndex) {
  *outQpIndex = (id + qpIndex) % baseComm->nqps;
  *outQp = baseComm->activeQps[*outQpIndex];
  assert(*outQp != NULL);
  return ncclSuccess;
}

// Get a QP object from a QP number. If not NULL, qpIndex will also return the
// index of the QP in the ncclIbNetCommBase::qps[] array.
static inline ncclResult_t IbCastCommBaseGetQpByQpNum(struct ncclIbNetCommBase* commBase, int devIndex, uint32_t qpNum, ncclIbQp** qp, int* qpIndex) {
  assert(devIndex >= 0 && devIndex < commBase->vProps.ndevs);
  assert(qp != NULL);
  TRACE(NCCL_NET, "NET/IB: %s: Looking for QP num %u on devIndex %d among %d QPs", __func__, qpNum, devIndex, commBase->nqps / commBase->vProps.ndevs);
  for (int qpIndexInDev = 0; qpIndexInDev < (commBase->nqps / commBase->vProps.ndevs); qpIndexInDev++) {
    *qp = &(commBase->qps[commBase->vProps.ndevs*qpIndexInDev + devIndex]);
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

// Each request is transfered over all devices, and depending on the
// "splitDataOnQps" configuration parameter, a request may be transffered over
// a single QP per device or on all QPs of each device.
static inline int IbCastCommBaseGetNqpsPerRequest(struct ncclIbNetCommBase* baseComm) {
  assert(baseComm->nDataQps != -1);
  assert(baseComm->nqps != -1);
  return (baseComm->splitDataOnQps == 1) ? baseComm->nqps : baseComm->nDataQps;
}

static inline ncclResult_t IbCastPostRecvWorkRequest(struct ibv_qp* qp, struct ibv_recv_wr* wr) {
  struct ibv_recv_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(qp, wr, &bad_wr));
  return ncclSuccess;
}

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // Start with CTS FIFO and ibv structs as they have alignment restrictions

  // CTS FIFO from which the sender reads the Clear-to-Send (CTS) messages that
  // are written by the receiver (The receiver side writes into it upon
  // issuing a (multi-)receive request). Each row in the 2D array corresponds
  // to a single CTS message but can describe multiple recv-requests issued
  // on the receiver side.
  struct ncclIbSendFifo ctsFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the send requests for faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array based on the "slot" they
  // are associated with.
  struct ncclIbRequest* sendReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

  // Counter per "slot" on how many send request were called for a multi-recv
  int sendReqsCnt[NET_IB_MAX_REQUESTS];
  struct ncclIbRemCompletionsRecords remCmplsRecords;
  int ar; // Use adaptive routing when all merged devices have it enabled
  uint64_t putSignalScratchpad;
  bool useCtsOffload;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure ctsFifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, ctsFifo) % 32) == 0, "ncclIbSendComm ctsFifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((sizeof(struct ncclIbSendFifoCtsInline) % 32) == 0, "ncclIbSendFifoCtsInline element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_mr* gpuMr;
  int* gpuFlushGpuMem;
  struct ibv_sge sge;
  struct ncclIbQp qp;
  int dmabufFd;
};

// This structure describes the FIFO which the receiver uses when it sends CTS
// messages to the sender.
struct ncclIbRemCtsFifo {
  // A "shadow" structure of the sender's CTS FIFO in which the receiver tracks
  // the CTS FIFO locally on its side. Receiver uses this memory to place the
  // CTS messages and populates the RDMA message "gather address" with the
  // memory of the CTS message that is sent.
  struct ncclIbSendFifo elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the receiver chooses the
  // RKey (depending on the device being used) when it posts a CTS to the
  // sender
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
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
};

#define NCCL_IB_RECV_WR_ID_DUMMY UINT64_MAX

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the recv requests to allow faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array using a hash (modulo) on
  // their ID.
  struct ncclIbRequest* recvReqs[NET_IB_MAX_REQUESTS];
  // Structure to hold all the related structures regarding the CTS FIFO
  // structure.
  struct ncclIbRemCtsFifo remCtsFifo;
  // Structure to hold all the completion records of all the outstanding
  // receive requests on the receiver side.
  struct ncclIbRequestCompletionRecord cmplsRecords[NET_IB_MAX_REQUESTS];
  int gpuFlushHostMem;
  int flushEnabled;
  bool prepostReceiveWorkRequests;
  // To avoid allocation and memset on the data-path a single structure is used
  // and only the wr_id is updated before posting a receive work request.
  struct ibv_recv_wr ibRecvWorkRequest;
  bool useCtsOffload;
};
static_assert((offsetof(struct ncclIbRecvComm, remCtsFifo) % 32) == 0, "ncclIbRecvComm ctsFifo must be 32-byte aligned");

ncclResult_t IbCastBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend);
ncclResult_t IbCastRecvCommInit(struct ncclIbRecvComm* recvComm);
ncclResult_t IbCastSendCommInit(struct ncclIbSendComm* sendComm);

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage* stage;
};

static ncclResult_t IbCastStatsInit(struct ncclIbStats* stat) {
  COMPILER_ATOMIC_STORE(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void IbCastStatsFatalError(struct ncclIbStats* stat){
  COMPILER_ATOMIC_FETCH_ADD(&stat->fatalErrorCount, 1, std::memory_order_relaxed);
}
static void IbCastQpFatalError(struct ibv_qp* qp) {
  IbCastStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void IbCastCqFatalError(struct ibv_cq* cq) {
  IbCastStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void IbCastDevFatalError(struct ncclIbDev* dev) {
  IbCastStatsFatalError(&dev->stats);
}
ncclResult_t IbCastStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName);

extern ncclProfilerCallback_t IbCastProfilerFunction;

extern std::thread IbCastAsyncThread;
void* IbCastAsyncThreadMain(void* args);

ncclResult_t IbCastGdrSupport();
ncclResult_t IbCastPeerMemSupport();
ncclResult_t IbCastDmaBufSupport(int dev);

void IbCastAddEvent(struct ncclIbRequest* req, int devIndex);
void IbCastAddEventCTS(struct ncclIbRequest* req, int devIndex);
ncclResult_t IbCastGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex);
ncclResult_t IbCastGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t IbCastFreeRequest(struct ncclIbRequest* r);

ncclResult_t IbCastRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle);

int IbCastGetTrafficClass(void* ctx);
void IbCastSetTrafficClass(void* ctx, int trafficClass);

// Net IB plugin entry functions.
ncclResult_t IbCastInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t IbCastInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t IbCastDevices(int* ndev);
ncclResult_t IbCastGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t IbCastGetPhysProperties(int dev, ncclNetProperties_t* props);
ncclResult_t IbCastListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
ncclResult_t IbCastConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/);
ncclResult_t IbCastAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/);
ncclResult_t IbCastRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t IbCastRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t IbCastDeregMr(void* comm, void* mhandle);
ncclResult_t IbCastIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request);
ncclResult_t IbCastIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request);
ncclResult_t IbCastIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t IbCastTest(void* request, int* done, int* sizes);
ncclResult_t IbCastCloseSend(void* sendComm);
ncclResult_t IbCastCloseRecv(void* recvComm);
ncclResult_t IbCastCloseListen(void* listenComm);
ncclResult_t IbCastMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t IbCastFinalizeDevices(void);
ncclResult_t IbCastFinalize(void* ctx);
ncclResult_t IbCastSetNetAttr(void *ctx, ncclNetAttr_t *netAttr);


// AINIC-specific infrastructure
#define NCCL_CTS_QP_SLOT_INVALID 0xFF

// IB-CAST specific infrastructure
#define NCCL_IB_MAX_QPS 128

#define NSEC_PER_USEC           1000ULL
#define NSEC_PER_MSEC           (NSEC_PER_USEC * 1000)
#define NSEC_PER_SEC            (NSEC_PER_MSEC * 1000)
#define NSEC_PER_MIN            (NSEC_PER_SEC * 60)

#define QP_SCHED_RESET_MIN      NSEC_PER_MSEC
#define QP_SCHED_RESET_NEVER    0

#define QP_SCHED_UPDATE_MIN     NSEC_PER_USEC
#define QP_SCHED_UPDATE_MAX     NSEC_PER_MIN

#define QP_SCHED_WEIGHT_NONE    0
#define QP_SCHED_WEIGHT_MIN     QP_SCHED_WEIGHT_NONE
#define QP_SCHED_WEIGHT_MAX     1.0

#define QP_SCHED_DISABLE        0
#define QP_SCHED_ENABLE         1

#define QP_SCHED_ENABLE_DEF             QP_SCHED_ENABLE
#define QP_SCHED_WRR_ENABLE_DEF         QP_SCHED_ENABLE
#define QP_SCHED_RESET_DEF              (NSEC_PER_SEC * 60)
#define QP_SCHED_UPDATE_DEF             (NSEC_PER_USEC * 50)
#define QP_SCHED_WEIGHT_DEF             QP_SCHED_WEIGHT_NONE
#define QP_SCHED_SPLIT_DATA_MIN_DEF     (64 * 1024)
#define QP_SCHED_LOG_DEF                NSEC_PER_SEC

// CAST – each variable is accessible as RCCL_IB_QP_SCHED_* or NCCL_IB_QP_SCHED_*
// Declarations only — definitions live in scheduler.cc to avoid duplicate symbols.
extern int64_t rcclParamIbCastQpSchedEnable();
extern int64_t rcclParamIbQpSchedWrrEnable();
extern int64_t rcclParamIbQpSchedResetInterval();
extern int64_t rcclParamIbQpSchedUpdateInterval();
extern int64_t rcclParamIbQpSchedSplitDataMin();
extern int64_t rcclParamIbQpSchedLogInterval();

#define QP_SCHED_WEIGHT_ENV_VAR           "RCCL_IB_QP_SCHED_WEIGHT"
#define QP_SCHED_WEIGHT_ENV_VAR_ALIAS     "NCCL_IB_QP_SCHED_WEIGHT"
#define QP_SCHED_LOG_PATH_ENV_VAR         "RCCL_IB_QP_SCHED_LOG_PATH"
#define QP_SCHED_LOG_PATH_ENV_VAR_ALIAS   "NCCL_IB_QP_SCHED_LOG_PATH"

#define QP_SCHED_LOG_FILE_NAME_PREFIX	"cast_log_"
#define NCCL_NET_IB_REMAP_UNUSED 0
#define NCCL_NET_IB_REMAP_USED   1

#define BITS_PER_BYTE 8
#define MEG           1000000

#define TIMESPEC_TO_NSEC(TS_PTR)                   \
  ((((uint64_t) (TS_PTR)->tv_sec) * NSEC_PER_SEC) + \
   ((uint64_t) (TS_PTR)->tv_nsec))

  
// Control block for staged dynamic scheduling parameters
struct ncclIbQpSchedParmsCB {
  ncclFunc_t collType;
  size_t msgSz;
  uint64_t prodEpoch;
  struct ncclIbQpSchedParms parms;
};

ncclResult_t IbCastQpSchedInitParms(struct ncclIbQpSchedParms *parms);
void IbCastLogSched(struct ncclIbSendComm *comm);
void IbCastUpdateSchedParmsTry(struct ncclIbNetCommBase *base, int nreqs, int size);
void IbCastQpSchedUpdateTx(struct ncclIbNetCommBase *base);
void IbCastQpSchedUpdateTxStats(struct ncclIbRemapWrId *remap,
  struct ncclIbNetCommBase *base);
int IbCastQpSchedGetEffectiveTxNqps(struct ncclIbRequest* req, int *startQpIndex, bool *wrrSched);
ncclResult_t IbCastQpSchedGetRemap(struct ncclIbNetCommBase* base, uint64_t wrId, int qpIndex, struct ncclIbRemapWrId** remap);
ncclResult_t IbCastQpSchedFreeRemap(struct ncclIbRemapWrId* r);
#endif

