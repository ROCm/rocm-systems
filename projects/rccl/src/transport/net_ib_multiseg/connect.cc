/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "connect_mseg.h"
#include "common_mseg.h"
#include "p2p_mseg.h"
#include "p2p_resiliency_mseg.h"
#include "ib_multiseg_mseg.h" // RCCL: the connect-time multi-segment decision and RMA QP depths.
#include "p2p_multiseg_mseg.h" // RCCL: the send comm's read-layout cache.
#include "rccl_common.h" // RCCL: rcclMultiSegOn gates the multi-segment CTS.

#include <mutex>

// RCCL: cached once per comm. The core's fd -1 gate (switch and DMA-BUF) off, a GIN comm or a resilient comm keeps
// the legacy CTS.
static int IbMsegResolveUseMultiSeg(bool forGin, bool resilient) {
  const bool reg = rcclMultiSegOn(/*pluginRegistersPerSegment=*/true);
  const int ok = IbMsegUseMultiSeg(reg, forGin, resilient) ? 1 : 0;
  TRACE(NCCL_INIT | NCCL_NET, "NET/IB: multi-segment CTS %s (register=%d gin=%d resilient=%d)", ok ? "on" : "off",
        (int)reg, (int)forGin, (int)resilient);
  return ok;
}

// RCCL: ranks with different multi-segment settings would read each other's CTS in the wrong layout.
static ncclResult_t IbMsegCheckPeerUseMultiSeg(int local, int remote, const char* peerDevName) {
  if (local == remote) return ncclSuccess;
  WARN("NET/IB: multi-segment CTS is %s here but %s on the peer (%s); set NCCL_MULTI_SEGMENT_REGISTER, "
       "NCCL_DMABUF_ENABLE and NCCL_IB_RESILIENCY_PORT_FAILOVER the same on every rank",
       local ? "on" : "off", remote ? "on" : "off", peerDevName);
  return ncclInvalidUsage;
}

// RCCL: one page-aligned block. Multi-segment comms use combined slots, the send cursor and the layout cache;
// otherwise the legacy FIFO.
static ncclResult_t IbMsegSendCommAllocCts(struct IbMsegSendComm* comm, bool combinedSlots) {
  void* region = NULL;
  if (combinedSlots) {
    NCCLCHECK(ncclIbMalloc(&region, NCCL_IB_CTS_SLOTS_BYTES));
    comm->ctsSlots = (struct IbMsegCtsSlot*)region;
    comm->ctsFifo = NULL;
    NCCLCHECK(ncclCalloc(&comm->sendCursor, 1));
    NCCLCHECK(ncclCalloc(&comm->layoutCache, 1));
  } else {
    NCCLCHECK(ncclIbMalloc(&region, NCCL_IB_CTS_FIFO_BYTES));
    comm->ctsFifo = (struct IbMsegSendFifo (*)[NCCL_NET_IB_MAX_RECVS])region;
    comm->ctsSlots = NULL;
  }
  return ncclSuccess;
}

// RCCL: the CTS block is one allocation, either the legacy FIFO or the combined slots.
static void IbMsegSendCommFreeCts(struct IbMsegSendComm* comm) {
  if (comm == NULL) return;
  free(comm->ctsSlots);
  free(comm->ctsFifo);
  free(comm->sendCursor);
  free(comm->layoutCache);
  comm->ctsSlots = NULL;
  comm->ctsFifo = NULL;
  comm->sendCursor = NULL;
  comm->layoutCache = NULL;
}

// RCCL: seg_rma_mseg.h cannot include common.h, so it repeats NET_IB_MAX_REQUESTS.
static_assert(RCCL_RMA_MAX_REQUESTS == NET_IB_MAX_REQUESTS, "RCCL_RMA_MAX_REQUESTS must match NET_IB_MAX_REQUESTS");
static_assert(RCCL_RMA_SEND_QP_WRS >= 2 * NET_IB_MAX_REQUESTS, "multi-segment send QP must not shrink the base");

// RCCL: sender QP depth. A multi-segment RMA comm holds a full chain per in-flight request; NET keeps the base.
static inline int rcclIbSendQpWrs(bool isRma) {
  return isRma && rcclRmaMultiSegEnabled() ? RCCL_RMA_SEND_QP_WRS : 2 * NET_IB_MAX_REQUESTS;
}

USE_NCCL_PARAM(ncclParamIbGidIndex, int32_t);
USE_NCCL_PARAM(ncclParamIbRoutableFlidIbGidIndex, int32_t);
USE_NCCL_PARAM(ncclParamIbRoceVersionNum, int32_t);
USE_NCCL_PARAM(ncclParamIbTimeout, uint8_t);
USE_NCCL_PARAM(ncclParamIbRetryCnt, uint8_t);
USE_NCCL_PARAM(ncclParamIbPkey, uint16_t);
extern int64_t ncclParamIbPkeyValue();
USE_NCCL_PARAM(ncclParamIbUseInline, bool);
// per IB spec, SL value can be 0-15
USE_NCCL_PARAM(ncclParamIbSl, int32_t);
// per IB spec, TC value can be 0-0xFF
USE_NCCL_PARAM(ncclParamIbTc, int32_t);
// per IB spec, TC value can be 0-0xFF
USE_NCCL_PARAM(ncclParamIbFifoTc, int32_t);
USE_NCCL_PARAM(ncclParamIbEceEnable, bool);

extern int64_t ncclParamIbOooRq();

struct IbMsegDevExtraProps {
  bool oooRq;
};

extern int64_t rcclParamIbGdrFlushGpuMemNoRelaxedOrdering();

// The flush scratchpad is registered on every accept; report the outcome once per IB device.
static std::once_flag gpuFlushRegLogOnce[MAX_IB_DEVS];
static std::once_flag gpuFlushRegWarnOnce[MAX_IB_DEVS];

enum IbMsegCommState {
  IbMsegCommStateStart = 0,
  IbMsegCommStateConnect = 1,
  IbMsegCommStateAccept = 3,
  IbMsegCommStateSend = 4,
  IbMsegCommStateRecv = 5,
  IbMsegCommStateConnecting = 6,
  IbMsegCommStateConnected = 7,
  IbMsegCommStatePendingReady = 8,
  IbMsegCommStateSendDevList = 9,
  IbMsegCommStateRecvDevList = 10,
};

struct IbMsegCommStage {
  enum IbMsegCommState state;
  int offset;
  void* buffer;
  void* comm;
};

struct IbMsegHandle {
  union ncclSocketAddress connectAddr; // Filled by the target
  uint64_t magic; // random number to help debugging
  int isP2p; // P2P flag
  struct IbMsegCommStage stage; // Used by the other side when connecting
  // GIDs of the listener's device PFs, used by the connector to find a local
  // NIC on the same subnet (for multi-subnet RoCE direct-connect topologies).
  // Zero-valued slots are ignored (validGid() returns false).
  union ibv_gid listenGids[2];
  int isGin; // RCCL: set by rcclIbMarkGinListen; GIN comms skip the multi-segment layout.
};

USE_NCCL_PARAM(ncclParamIbQpsPerConn, int32_t);
extern int64_t rcclParamIbQpsPerP2p();

// nQpsPerDev > 0 wins (GIN passes 1). Otherwise use the IB QP env policy.
static int rcclIbQpsPerDev(int nQpsPerDev, int isP2p) {
  if (nQpsPerDev > 0) {
    // GIN opens a link per context per peer, so log once per process.
    static std::once_flag once;
    if (ncclParamIbQpsPerConn() > nQpsPerDev) {
      std::call_once(once, [&]() {
        INFO(NCCL_ENV | NCCL_NET,
             "NET/IB: NCCL_IB_QPS_PER_CONNECTION=%d is not applied to GIN connections, which use %d QP per device",
             ncclParamIbQpsPerConn(), nQpsPerDev);
      });
    }
    return nQpsPerDev;
  }
  if (rcclParamIbQpsPerP2p() > 0 && isP2p) return (int)rcclParamIbQpsPerP2p();
  return (int)ncclParamIbQpsPerConn();
}

// Explicit GIN count sizes the CQ. The env path keeps NCCL_IB_QPS_PER_CONNECTION.
static int rcclIbCqQpsPerDev(int nQpsPerDev) {
  return nQpsPerDev > 0 ? nQpsPerDev : (int)ncclParamIbQpsPerConn();
}

static int rcclIbMaxNqps(int qpPerDev, int localNdevs, int remoteNdevs, const char* funcName) {
  int localNqps = qpPerDev * localNdevs;
  int remoteNqps = qpPerDev * remoteNdevs;
  int maxNqps = (remoteNqps > localNqps) ? remoteNqps : localNqps;
  INFO(NCCL_NET, "NET/IB: %s Max Nqps=%d, localNqps=%d, remoteNqps=%d, qpPerDev=%d", funcName, maxNqps,
       localNqps, remoteNqps, qpPerDev);
  return maxNqps;
}

static int rcclIbSelectNqps(int nQpsPerDev, int isP2p, int localNdevs, int remoteNdevs, const char* funcName) {
  return rcclIbMaxNqps(rcclIbQpsPerDev(nQpsPerDev, isP2p), localNdevs, remoteNdevs, funcName);
}

// Opt-in GDR flush through RO=0 GPU flags written by the sender after the
// payload, one flag per QP. Flagged requests rotate over the QPs like the rest
// (all QPs at once with NCCL_IB_SPLIT_DATA_ON_QPS=1); the GPU waits only on the
// flags of the QPs that carried each step, and net.cc skips the proxy flush.
// The communicator enables it at init only when every rank passes
// IbMsegGdrFlushFlagLocalSupport(), so a receiver that publishes a flag always
// has a sender that writes it.
extern int64_t rcclParamIbGdrFlushFlag();
extern int64_t ncclParamIbOooRq();
extern int64_t ncclParamIbReceiverSideMatchingScheme();
extern int64_t ncclParamIbResiliencyPortFailover();

static_assert(NCCL_IB_MAX_QPS * NCCL_GDR_FLUSH_FLAG_STRIDE <= GPU_PAGE_SIZE, "per-QP GDR flush flags must fit in one page");
static_assert((NCCL_IB_MAX_QPS + NCCL_GDR_FLUSH_MAX_WAIT_QPS - 1) * NCCL_GDR_FLUSH_FLAG_STRIDE <= GPU_PAGE_SIZE,
              "the GPU polls NCCL_GDR_FLUSH_MAX_WAIT_QPS flags from any first QP, all within the page");
static_assert(NCCL_IB_MAX_QPS < 256, "per-QP GDR flush flag first index and count are packed in 8 bits each");

// The RO=0 flag write is only ordered behind the payload posted ahead of it on
// the same QP, delivered in order: no adaptive routing, no out-of-order receive
// queues (ID-based matching) and no resiliency retransmits. NIC fusion is
// checked per connection at accept: fused vNICs are only created by the
// topology search, after the init all-gather.
bool IbMsegGdrFlushFlagOrdered(int64_t oooRq, int64_t portFailover, int64_t matchingScheme, int nDevs,
                               const int* adaptiveRouting) {
  if (nDevs == 0 || oooRq != 0 || portFailover != 0) return false;
  if (matchingScheme != -2 && matchingScheme != BY_INDEX) return false;
  for (int d = 0; d < nDevs; d++)
    if (adaptiveRouting[d]) return false;
  return true;
}

bool IbMsegGdrFlushFlagLocalSupport() {
  if (rcclParamIbGdrFlushFlag() == 0) return false;
  int adaptiveRouting[MAX_IB_DEVS];
  for (int d = 0; d < IbMsegNDevs; d++) adaptiveRouting[d] = IbMsegDevs[d].ar;
  return IbMsegGdrFlushFlagOrdered(ncclParamIbOooRq(), ncclParamIbResiliencyPortFailover(),
                                   ncclParamIbReceiverSideMatchingScheme(), IbMsegNDevs, adaptiveRouting);
}

void IbMsegSetGdrFlushFlag(void* netContext, int enable) {
  ((struct IbMsegNetCtx*)netContext)->gdrFlushFlag = enable;
}

void IbMsegSetListenGdrFlushFlag(void* listenComm, int enable) {
  ((struct IbMsegListenComm*)listenComm)->gdrFlushFlag = enable;
}

extern int64_t ncclParamIbSubnetAwareRouting();
extern int64_t ncclParamIbSubnetPrefixLen();

ncclResult_t IbMsegInitCommDevBase(int ibDevN, struct IbMsegNetCommDevBase* base, void* cq_context, int cqSize) {
  base->ibDevN = ibDevN;
  IbMsegDev* ibDev = IbMsegDevs + ibDevN;
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (0 == ibDev->pdRefs++) {
      NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context));
    }
    base->pd = ibDev->pd;
  }

  IbMsegGidInfoSnapshot(base, ibDev);

  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, cqSize, cq_context, NULL, 0));

  NCCLCHECK(IbMsegGetPkeyIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &base->pkeyIndex));

  return ncclSuccess;
}

ncclResult_t IbMsegDestroyBase(struct IbMsegNetCommDevBase* base) {
  NCCLCHECK(wrap_ibv_destroy_cq(base->cq));

  std::lock_guard<std::mutex> lock(IbMsegDevs[base->ibDevN].mutex);
  if (0 == --IbMsegDevs[base->ibDevN].pdRefs) {
    NCCLCHECK(wrap_ibv_dealloc_pd(IbMsegDevs[base->ibDevN].pd));
  }
  return ncclSuccess;
}

static sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  const char* env = ncclGetEnv("NCCL_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

static void* envIbAddrRange(sa_family_t af, int* mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void* ret = (af == AF_INET) ? (void*)&addr : (void*)&addr6;

  const char* env = ncclGetEnv("NCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_RANGE set by environment to %s", env);

  char addrString[128] = {0};
  snprintf(addrString, 128, "%s", env);
  char* addrStrPtr = addrString;
  char* maskStrPtr = strstr(addrString, "/");
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *(maskStrPtr++) = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    INFO(NCCL_INIT | NCCL_NET, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address", addrStrPtr,
         (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    INFO(NCCL_INIT | NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask,
         (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    INFO(NCCL_INIT | NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask,
         (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

static sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast =
    (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

static bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid) {
  struct in_addr* base = NULL;
  struct in6_addr* base6 = NULL;
  struct in6_addr* addr6 = NULL;
  if (af == AF_INET) {
    base = (struct in_addr*)prefix;
  } else {
    base6 = (struct in6_addr*)prefix;
  }
  addr6 = (struct in6_addr*)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

static bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

static bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

static bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

static ncclResult_t IbMsegRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version) {
  char gidRoceVerStr[16] = {0};
  char roceTypePath[PATH_MAX] = {0};
  snprintf(roceTypePath, sizeof(roceTypePath), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName,
           portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    WARN("NET/IB: open failed in IbMsegRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    // In containerized environments, read could return EINVAL if the GID index is not mapped to the
    // container sysfs. In this case return ncclSuccess and let the caller move to next GID index.
    if (errno == EINVAL) return ncclSuccess;
    WARN("NET/IB: read failed in IbMsegRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }

  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 ||
        strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return ncclSuccess;
}

static ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix,
                                       int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex) {
  union ibv_gid gid, gidCandidate;
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, *gidIndex, &gid));
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate));

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet = matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet) {
      return ncclSuccess;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate = -1;
    const char* deviceName = wrap_ibv_get_device_name(context->device);
    NCCLCHECK(IbMsegRoceGetVersionNum(deviceName, portNum, *gidIndex, &gidRoceVerNum));
    NCCLCHECK(IbMsegRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return ncclSuccess;
}

ncclResult_t IbMsegGetGidIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                               int* gidIndex) {
  int gidTblLen = portAttr->gid_tbl_len;

  // for IB, choose GID Index that will have routable FLID if present
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    union ibv_gid gid;
    int routableGidIndex = ncclParamIbRoutableFlidIbGidIndex();
    if (routableGidIndex < gidTblLen) {
      NCCLCHECK(wrap_ibv_query_gid(context, portNum, routableGidIndex, &gid));
      if (IbMsegExtractFlid(&gid) != 0) {
        *gidIndex = routableGidIndex;
        return ncclSuccess;
      }
    }
    *gidIndex = 0;
    return ncclSuccess;
  }

  // for ROCE
  *gidIndex = ncclParamIbGidIndex();
  if (*gidIndex > NCCL_PARAM_VAL_AUTO) {
    return ncclSuccess;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = ncclParamIbRoceVersionNum();
  int prefixlen;
  void* prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    NCCLCHECK(ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext,
                                 gidIndex));
  }

  return ncclSuccess;
}

ncclResult_t IbMsegGidInfoQuery(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                struct IbMsegGidInfo* gidInfo) {
  if (context == NULL || portAttr == NULL || gidInfo == NULL) return ncclInternalError;
  gidInfo->link_layer = portAttr->link_layer;
  NCCLCHECK(IbMsegGetGidIndex(context, portNum, portAttr, &gidInfo->localGidIndex));
  if (gidInfo->localGidIndex < 0) return ncclInternalError;
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, gidInfo->localGidIndex, &gidInfo->localGid));
  return ncclSuccess;
}

// Resolve NCCL_IB_PKEY_VALUE to a PKey-table index, else fall back to the NCCL_IB_PKEY index.
// ibv_query_pkey() returns network byte order with the MSB as the membership bit, so compare low 15 bits.
ncclResult_t IbMsegGetPkeyIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                int* pkeyIndex) {
  int64_t pkeyValue = ncclParamIbPkeyValue();
  if (pkeyValue < 0) {
    *pkeyIndex = ncclParamIbPkey();
    return ncclSuccess;
  }
  if (ncclParamIbPkey() != 0) {
    INFO(NCCL_NET | NCCL_ENV, "NET/IB: NCCL_IB_PKEY_VALUE=0x%x takes precedence over NCCL_IB_PKEY index %d",
         (unsigned)pkeyValue, (int)ncclParamIbPkey());
  }
  uint16_t target = (uint16_t)(pkeyValue & 0x7fff);
  for (int index = 0; index < portAttr->pkey_tbl_len; index++) {
    uint16_t pkey = 0;
    NCCLCHECK(wrap_ibv_query_pkey(context, portNum, index, &pkey));
    if ((uint16_t)(ntohs(pkey) & 0x7fff) == target) {
      *pkeyIndex = index;
      return ncclSuccess;
    }
  }
  WARN("NET/IB: NCCL_IB_PKEY_VALUE=0x%x not found in PKey table (pkey_tbl_len=%d, port=%d)", (unsigned)pkeyValue,
       portAttr->pkey_tbl_len, portNum);
  return ncclInvalidUsage;
}

ncclResult_t IbMsegQpInit(struct IbMsegQp* qp) {
  struct IbMsegQpInitAttr* initAttr = &qp->initAttr;
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = initAttr->state;
  qpAttr.pkey_index = initAttr->pkeyIndex;
  qpAttr.port_num = initAttr->portNum;
  qpAttr.qp_access_flags = initAttr->qpAccessFlags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

static ncclResult_t IbMsegCreateQpMlx5(struct IbMsegQpCreateAttr* createQpAttrs, struct IbMsegQp* qp) {
  struct ibv_qp_init_attr_ex qpInitAttr;
  struct mlx5dv_qp_init_attr dvAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr_ex));
  memset(&dvAttr, 0, sizeof(struct mlx5dv_qp_init_attr));
  qpInitAttr.qp_context = createQpAttrs->qpContext;
  qpInitAttr.send_cq = createQpAttrs->cq;
  qpInitAttr.recv_cq = createQpAttrs->cq;
  qpInitAttr.qp_type = createQpAttrs->type;
  qpInitAttr.cap.max_recv_wr = createQpAttrs->maxRecvWorkRequest;
  qpInitAttr.cap.max_send_wr = createQpAttrs->maxSendWorkRequest;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = std::max<uint32_t>(createQpAttrs->maxInlineData,
                                                      ncclParamIbUseInline() ? sizeof(struct IbMsegSendFifo) : 0);

  qpInitAttr.comp_mask = IBV_QP_INIT_ATTR_PD;
  qpInitAttr.pd = createQpAttrs->pd;

  if (createQpAttrs->oooRq) {
    dvAttr.create_flags |= MLX5DV_QP_CREATE_OOO_DP;
    dvAttr.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
  }
  qp->qp = wrap_mlx5dv_create_qp(createQpAttrs->pd->context, &qpInitAttr, &dvAttr);
  if (qp->qp == NULL) {
    WARN("NET/IB: %s: mlx5dv_create_qp failed to create QP: %m", __func__);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t IbMsegQpCreate(struct IbMsegQp* qp, struct IbMsegQpCreateAttr* createQpAttrs) {
  IbMsegWqeLatMonInit(&qp->latMon);
  if (createQpAttrs->oooRq) {
    NCCLCHECK(IbMsegCreateQpMlx5(createQpAttrs, qp));
    return ncclSuccess;
  }
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.qp_context = createQpAttrs->qpContext;
  qpInitAttr.send_cq = createQpAttrs->cq;
  qpInitAttr.recv_cq = createQpAttrs->cq;
  qpInitAttr.qp_type = createQpAttrs->type;
  qpInitAttr.cap.max_recv_wr = createQpAttrs->maxRecvWorkRequest;
  qpInitAttr.cap.max_send_wr = createQpAttrs->maxSendWorkRequest;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = std::max<uint32_t>(createQpAttrs->maxInlineData,
                                                      ncclParamIbUseInline() ? sizeof(struct IbMsegSendFifo) : 0);
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, createQpAttrs->pd, &qpInitAttr));
  return ncclSuccess;
}

ncclResult_t IbMsegQpRtr(struct IbMsegQp* qp) {
  struct IbMsegQpRtrAttr* rtrAttr = &qp->rtrAttr;
  struct ibv_qp_attr qpAttr;
  int attrMask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = rtrAttr->mtu;
  qpAttr.dest_qp_num = rtrAttr->remoteQpNum;
  qpAttr.rq_psn = 0;
  if (qp->qp->qp_type != IBV_QPT_UC) {
    qpAttr.max_dest_rd_atomic = 1;
    qpAttr.min_rnr_timer = 12;
    attrMask |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  }
  if (rtrAttr->linkLayer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = rtrAttr->remoteGid.global.subnet_prefix;
    qpAttr.ah_attr.grh.dgid.global.interface_id = rtrAttr->remoteGid.global.interface_id;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = rtrAttr->localGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = rtrAttr->tc;
  } else {
    // Path-local if same subnet and GRH not required; else global addressing. FLID only when subnets differ.
    bool sameSubnet = (IbMsegExtractLocalSubnetPrefix(rtrAttr->localGid.global.subnet_prefix) ==
                       IbMsegExtractLocalSubnetPrefix(rtrAttr->remoteGid.global.subnet_prefix));
    bool needGlobal = !sameSubnet || (rtrAttr->localPortFlags & IBV_QPF_GRH_REQUIRED);
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = rtrAttr->remoteLid;
    if (needGlobal) {
      if (!sameSubnet) {
        uint16_t flid = IbMsegExtractFlid(&rtrAttr->remoteGid);
        if (flid == 0) {
          WARN("Warning: remote FLID configured as zero even when endpoints are on different subnets, using dlid as "
               "fallback");
          qpAttr.ah_attr.dlid = rtrAttr->remoteLid;
        } else {
          qpAttr.ah_attr.dlid = flid;
        }
      }
      qpAttr.ah_attr.is_global = 1;
      qpAttr.ah_attr.grh.dgid.global.subnet_prefix = rtrAttr->remoteGid.global.subnet_prefix;
      qpAttr.ah_attr.grh.dgid.global.interface_id = rtrAttr->remoteGid.global.interface_id;
      qpAttr.ah_attr.grh.sgid_index = rtrAttr->localGidIndex;
      qpAttr.ah_attr.grh.hop_limit = 255;
    }
  }
  qpAttr.ah_attr.sl = rtrAttr->sl;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = rtrAttr->localIbPort;
  TRACE(NCCL_NET, "NET/IB: %s: qpn=%u mtu=%d dst=%u ll=%u port=%u sl: %d tc: %d", __func__, qp->qp->qp_num,
        qpAttr.path_mtu, qpAttr.dest_qp_num, rtrAttr->linkLayer, qpAttr.ah_attr.port_num, qpAttr.ah_attr.sl,
        qpAttr.ah_attr.grh.traffic_class);
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, attrMask));
  return ncclSuccess;
}

ncclResult_t IbMsegQpRts(struct IbMsegQp* qp) {
  struct IbMsegQpRtsAttr* rtsAttr = &qp->rtsAttr;
  struct ibv_qp_attr qpAttr;
  int attrMask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  if (qp->qp->qp_type != IBV_QPT_UC) {
    qpAttr.timeout = rtsAttr->timeout;
    qpAttr.retry_cnt = rtsAttr->retryCnt;
    qpAttr.rnr_retry = 7;
    qpAttr.max_rd_atomic = 1;
    attrMask |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
  }
  qpAttr.sq_psn = 0;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, attrMask));
  return ncclSuccess;
}

ncclResult_t IbMsegQpReset(struct IbMsegQp* qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}

ncclResult_t IbMsegQpError(struct IbMsegQp* qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_ERR;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}

// Check if two RoCE GIDs are on the same subnet.
// For IPv4-mapped GIDs (::ffff:a.b.c.d), uses the given prefix length (1..32).
// For native IPv6 GIDs, compares the 64-bit subnet prefix.
static bool gidSameSubnet(union ibv_gid* local, union ibv_gid* remote, int prefixLen) {
  sa_family_t localFam = getGidAddrFamily(local);
  sa_family_t remoteFam = getGidAddrFamily(remote);
  if (localFam != remoteFam) return false;
  if (localFam == AF_INET) {
    // IPv4-mapped: compare using configured prefix length.
    // IPv4 address is in bytes 12-15 of the raw GID.
    uint32_t localIp, remoteIp;
    memcpy(&localIp, local->raw + 12, 4);
    memcpy(&remoteIp, remote->raw + 12, 4);
    uint32_t mask = htonl(~((1U << (32 - prefixLen)) - 1));
    return (localIp & mask) == (remoteIp & mask);
  } else {
    // IPv6: compare subnet prefix (first 64 bits)
    return local->global.subnet_prefix == remote->global.subnet_prefix;
  }
}

// check if a local GID matches ANY of the remote GIDs.
static bool subnetMatchesAny(union ibv_gid* localGid, union ibv_gid* remoteGids, int nRemoteGids, int prefixLen) {
  for (int r = 0; r < nRemoteGids; r++) {
    if (validGid(&remoteGids[r]) && gidSameSubnet(localGid, &remoteGids[r], prefixLen)) return true;
  }
  return false;
}

// Given remote GIDs (one per PF on the remote side), find a local merged IB
// device that shares a subnet with any of them. Writes defaultDev to *foundDev
// if no better match is found, preserving existing behavior for single-subnet
// and IB deployments.
// Checks the default device first to preserve NIC Fusion when all PFs in
// the fused device can reach the peer (e.g., 2-node or switch setup).
static ncclResult_t IbMsegFindDevBySubnet(union ibv_gid* remoteGids, int nRemoteGids, int defaultDev, int* foundDev) {
  *foundDev = defaultDev;

  int prefixLen = ncclParamIbSubnetPrefixLen();
  if (prefixLen < 1 || prefixLen > 32) {
    WARN("NET/IB: NCCL_IB_SUBNET_PREFIX_LEN=%d is out of range [1,32]", prefixLen);
    return ncclInvalidArgument;
  }

  // Quick check: if no remote GID is valid, nothing to do.
  bool anyValid = false;
  for (int r = 0; r < nRemoteGids; r++) {
    if (validGid(&remoteGids[r])) {
      anyValid = true;
      break;
    }
  }
  if (!anyValid) return ncclSuccess;

  // First: check if the default device already works. If ALL its RoCE PFs
  // match some remote GID's subnet, keep it — this preserves NIC Fusion
  // bandwidth when both ports connect to the same destination.
  if (defaultDev >= 0 && defaultDev < IbMsegNMergedDevs) {
    struct IbMsegMergedDev* mDev = IbMsegMergedDevs + defaultDev;
    int checked = 0, matched = 0;
    for (int i = 0; i < mDev->vProps.ndevs; i++) {
      int ibDevN = mDev->vProps.devs[i];
      IbMsegDev* ibDev = IbMsegDevs + ibDevN;
      if (ibDev->portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
      int gidIndex = 0;
      union ibv_gid localGid;
      memset(&localGid, 0, sizeof(localGid));
      if (IbMsegGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &gidIndex) != ncclSuccess) continue;
      if (wrap_ibv_query_gid(ibDev->context, ibDev->portNum, gidIndex, &localGid) != ncclSuccess) continue;
      checked++;
      if (validGid(&localGid) && subnetMatchesAny(&localGid, remoteGids, nRemoteGids, prefixLen)) matched++;
    }
    if (checked > 0 && matched == checked) return ncclSuccess;
  }

  // Default device can't fully reach the peer (e.g., NIC Fusion fused PFs on
  // different subnets, or the device is on the wrong subnet entirely).
  // Search for a device whose RoCE PFs all match a remote GID's subnet.
  // Same "all PFs must match" criterion as the defaultDev check: NCCL takes
  // a merged-device index and spreads QPs across all its PFs, so a partial
  // match would leave some QPs on PFs with no L2 path to the peer.
  for (int devIdx = 0; devIdx < IbMsegNMergedDevs; devIdx++) {
    if (devIdx == defaultDev) continue;
    struct IbMsegMergedDev* mDev = IbMsegMergedDevs + devIdx;
    int checked = 0, matched = 0;
    for (int i = 0; i < mDev->vProps.ndevs; i++) {
      int ibDevN = mDev->vProps.devs[i];
      IbMsegDev* ibDev = IbMsegDevs + ibDevN;
      if (ibDev->portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
      int gidIndex = 0;
      union ibv_gid localGid;
      memset(&localGid, 0, sizeof(localGid));
      if (IbMsegGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &gidIndex) != ncclSuccess) continue;
      if (wrap_ibv_query_gid(ibDev->context, ibDev->portNum, gidIndex, &localGid) != ncclSuccess) continue;
      checked++;
      if (validGid(&localGid) && subnetMatchesAny(&localGid, remoteGids, nRemoteGids, prefixLen)) matched++;
    }
    if (checked > 0 && matched == checked) {
      INFO(NCCL_NET, "NET/IB: Subnet-aware routing: overriding dev %d with dev %d", defaultDev, devIdx);
      *foundDev = devIdx;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t IbMsegListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct IbMsegListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  struct IbMsegHandle* handle = (struct IbMsegHandle*)opaqueHandle;
  static_assert(sizeof(struct IbMsegHandle) < NCCL_NET_HANDLE_MAXSIZE, "IbMsegHandle size too large");
  memset(handle, 0, sizeof(struct IbMsegHandle));
  comm->dev = dev;
  handle->magic = ncclSocketDefaultMagic();
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &IbMsegIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);

  // Embed GIDs of all PFs in the handle so the connector can find a local NIC
  // on the same subnet as any of our ports.
  if (ncclParamIbSubnetAwareRouting() && dev < IbMsegNMergedDevs) {
    struct IbMsegMergedDev* mDev = IbMsegMergedDevs + dev;
    int gidSlot = 0;
    for (int i = 0; i < mDev->vProps.ndevs && gidSlot < 2; i++) {
      int ibDevN = mDev->vProps.devs[i];
      IbMsegDev* ibDev = IbMsegDevs + ibDevN;
      if (ibDev->portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
      int gidIndex;
      NCCLCHECKGOTO(IbMsegGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &gidIndex), ret, fail);
      NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, gidIndex, &handle->listenGids[gidSlot]), ret,
                    fail);
      gidSlot++;
    }
  }

  *listenComm = comm;
exit:
  return ret;
fail:
  (void)ncclSocketClose(&comm->sock);
  free(comm);
  goto exit;
}

void rcclIbMarkGinListen(void* opaqueHandle, void* listenComm) {
  ((struct IbMsegHandle*)opaqueHandle)->isGin = 1;
  ((struct IbMsegListenComm*)listenComm)->isGin = true;
}

ncclResult_t IbMsegNetListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  NCCLCHECK(IbMsegListen(ctx, dev, opaqueHandle, listenComm));
  ((struct IbMsegListenComm*)*listenComm)->gdrFlushFlag = ctx ? ((struct IbMsegNetCtx*)ctx)->gdrFlushFlag : 0;
  return ncclSuccess;
}

#define NCCL_IB_SL_DEFAULT 0
#define NCCL_IB_TC_DEFAULT 0

// The function creates and initializes QPs (modifies the QPs to INIT) on the
// sender side. Afterwards it populates the metadata structure, provided to the
// function (meta), with the QPs' information. Note that after the QPs'
// creation, the QPs are also queried for ECE support and the metadata structure
// is updated accordingly. The meta data structure is then expected to be
// delivered to the remote side (receiver) as part of the connection
// establishment process.
static ncclResult_t IbMsegSenderQpsCreate(IbMsegSendComm* comm, struct IbMsegConnectionMetadata* meta) {
  uint nqps = comm->base.nqps;
  struct IbMsegQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(struct IbMsegQpCreateAttr));
  qpCreateAttrs.type = IBV_QPT_RC;
  qpCreateAttrs.maxRecvWorkRequest = 0;
  // Send requests are sent using at most 2 messages (RDMA Write and RDMA Write with Immediate)
  qpCreateAttrs.maxSendWorkRequest = rcclIbSendQpWrs(comm->base.isRma); // RCCL: full RMA chains with multi-segment on.
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % comm->base.vProps.ndevs;
    IbMsegSendCommDev* commDev = &comm->devs[devIndex];
    IbMsegDev* ibDev = &IbMsegDevs[commDev->base.ibDevN];
    IbMsegQp* localQp = &comm->base.qps[qpIndex];
    IbMsegQpInfo* localQpInfo = &meta->qpInfo[qpIndex];

    qpCreateAttrs.cq = commDev->base.cq;
    qpCreateAttrs.pd = commDev->base.pd;
    qpCreateAttrs.qpContext = &comm->base.stats;

    if (ibDev->ibProvider == IB_PROVIDER_MLX5 && ncclParamIbOooRq()) {
      if (ibDev->ar == 0) {
        WARN("NET/IB: %s: OOO RQ is force enabled but AR is not enabled, which is required for OOO RQ (device=%s)",
             __func__, ibDev->devName);
        return ncclInternalError;
      }
      qpCreateAttrs.oooRq = (comm->base.remOooRq && comm->base.localOooRq);
      if (!qpCreateAttrs.oooRq) {
        WARN("NET/IB: %s: OOO RQ is force enabled but not supported on both sides of the connection (device=%s, "
             "localOooRq=%d, remOooRq=%d)",
             __func__, ibDev->devName, comm->base.localOooRq, comm->base.remOooRq);
        return ncclInternalError;
      }
    }

    NCCLCHECK(IbMsegQpCreate(localQp, &qpCreateAttrs));
    INFO(NCCL_NET,
         "NET/IB: %s: QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p oooRq=%d",
         __func__, ibDev->portNum, commDev->base.ibDevN, IbMsegDevs[commDev->base.ibDevN].devName, IbMsegNDevs,
         IbMsegNMergedDevs, localQp->qp->qp_num, (uint16_t)commDev->base.pkeyIndex, commDev->base.pd,
         qpCreateAttrs.oooRq);
    localQp->devIndex = devIndex;

    // Populate the metadata that will be delivered to the remote peer
    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    // Transition the QP to INIT state
    struct IbMsegQpInitAttr* initAttr = &localQp->initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = commDev->base.pkeyIndex;
    initAttr->portNum = ibDev->portNum;
    initAttr->qpAccessFlags = IBV_ACCESS_REMOTE_WRITE;
    NCCLCHECK(IbMsegQpInit(localQp));

    if (ncclParamIbEceEnable()) {
      // Query ECE (Enhanced Connection Establishment) capabilities and
      // populate the initial ECE into the metadata structure that is sent to
      // the remote (receiver) side.
      NCCLCHECK(wrap_ibv_query_ece(localQp->qp, &localQpInfo->ece, &localQp->eceSupported));
      localQpInfo->ece_supported = localQp->eceSupported;
    } else {
      // Declare to the remote side that ECE is not supported
      localQpInfo->ece_supported = 0;
      // Store locally that ECE is not supported
      localQp->ece = {0};
      localQp->eceSupported = 0;
    }
  }

  if (comm->base.resiliency) {
    IbMsegResiliencySenderCreateQps(comm->base.resiliency, &meta->resiliencyInfo);
  }

  return ncclSuccess;
}

// The function modifies the QPs on the sender side to RTR and RTS states. It
// uses the remote metadata (remMeta) provided to the function to get the remote
// QPs' information. The remote metadata is expected to be obtained from the
// remote side (receiver) as part of the connection establishment process.
// Note that if ECE is supported, the function sets up the reduced ECE (which
// was delivered from the receiver side) on the QPs before modifying the QPs
// to RTR.
static ncclResult_t IbMsegSenderQpsToRts(IbMsegSendComm* comm, struct IbMsegConnectionMetadata* remMeta) {
  uint nqps = comm->base.nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    IbMsegQp* localQp = &comm->base.qps[qpIndex];
    IbMsegSendCommDev* commDev = &comm->devs[localQp->devIndex];
    IbMsegDev* ibDev = &IbMsegDevs[commDev->base.ibDevN];
    IbMsegQpInfo* remQpInfo = &remMeta->qpInfo[qpIndex];
    // RCCL: the peer sends devIndex, and it indexes remDevs and the RMA rkey tables, filled up to its ndevs.
    const int nRemDevs = std::min(remMeta->ndevs, NCCL_IB_MAX_DEVS_PER_NIC);
    if (remQpInfo->devIndex < 0 || remQpInfo->devIndex >= nRemDevs) {
      WARN("NET/IB: peer QP %d reports device index %d, expected [0, %d)", qpIndex, remQpInfo->devIndex, nRemDevs);
      return ncclInternalError;
    }
    IbMsegDevInfo* remDevInfo = &remMeta->devs[remQpInfo->devIndex];

    localQp->remDevIdx = remQpInfo->devIndex;

    if (localQp->eceSupported && remQpInfo->ece_supported) {
      INFO(NCCL_NET,
           "NET/IB: %s: Set ECE: IbDev %d Port %d qp_num %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, "
           "comp_mask=0x%x}",
           __func__, commDev->base.ibDevN, ibDev->portNum, localQp->qp->qp_num, remQpInfo->ece_supported,
           remQpInfo->ece.vendor_id, remQpInfo->ece.options, remQpInfo->ece.comp_mask);
      // Set the reduced ECE received from the receiver side
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &remQpInfo->ece, &localQp->eceSupported));
      // Store the reduced ECE locally as well
      localQp->ece = remQpInfo->ece;
    } else {
      // If remote does not support ECE, disable it locally as well
      localQp->eceSupported = 0;
      localQp->ece = {0};
    }

    struct IbMsegQpRtrAttr* rtrAttr = &localQp->rtrAttr;
    rtrAttr->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu);
    rtrAttr->linkLayer = remDevInfo->link_layer;
    rtrAttr->tc = remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET ? remMeta->tc : -1;
    rtrAttr->sl = remMeta->sl;
    rtrAttr->remoteQpNum = remQpInfo->qpn;
    rtrAttr->remoteLid = remDevInfo->lid;
    rtrAttr->remoteGid = remDevInfo->gid;
    rtrAttr->localIbPort = remDevInfo->ib_port;
    rtrAttr->localPortFlags = ibDev->portAttr.flags;
    rtrAttr->localGid = commDev->base.gidInfo.localGid;
    rtrAttr->localGidIndex = commDev->base.gidInfo.localGidIndex;
    NCCLCHECK(IbMsegQpRtr(localQp));
    struct IbMsegQpRtsAttr* rtsAttr = &localQp->rtsAttr;
    rtsAttr->timeout = ncclParamIbTimeout();
    rtsAttr->retryCnt = ncclParamIbRetryCnt();
    NCCLCHECK(IbMsegQpRts(localQp));
  }

  if (comm->base.resiliency) {
    NCCLCHECK(IbMsegResiliencySenderQpsToRts(comm->base.resiliency, remMeta));
  }

  return ncclSuccess;
}

int IbMsegGetTrafficClass(void* ctx) {
  ncclNetCommConfig_t* config = (ncclNetCommConfig_t*)ctx;
  if (ctx == NULL) return NCCL_NET_TRAFFIC_CLASS_UNDEF;
  return config->trafficClass;
}
void IbMsegSetTrafficClass(void* ctx, int trafficClass) {
  ncclNetCommConfig_t* config = (ncclNetCommConfig_t*)ctx;
  if (config) config->trafficClass = trafficClass;
}

ncclResult_t IbMsegConnectImpl(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                               ncclNetDeviceHandle_t** /*sendDevComm*/, int nQpsPerDev, int envTrafficClass) {
  ncclResult_t ret = ncclSuccess;
  struct IbMsegHandle* handle = (struct IbMsegHandle*)opaqueHandle;

  // Subnet-aware device selection: use the listener's GIDs (embedded in the
  // handle) to find a local NIC on the same subnet as the remote peer.
  // For single-subnet or IB deployments, all GIDs are zero → dev stays unchanged.
  if (ncclParamIbSubnetAwareRouting()) NCCLCHECK(IbMsegFindDevBySubnet(handle->listenGids, 2, dev, &dev));

  struct IbMsegCommStage* stage = &handle->stage;
  struct IbMsegSendComm* comm = (struct IbMsegSendComm*)stage->comm;
  int ready;
  int isP2p = 0;
  uint8_t link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *sendComm = NULL;

  if (stage->state == IbMsegCommStateConnect) goto ib_connect_check;
  if (stage->state == IbMsegCommStateSendDevList) goto ib_send_dev_list;
  if (stage->state == IbMsegCommStateRecvDevList) goto ib_recv_dev_list;
  if (stage->state == IbMsegCommStateSend) goto ib_send;
  if (stage->state == IbMsegCommStateConnecting) goto ib_connect;
  if (stage->state == IbMsegCommStateConnected) goto ib_send_ready;
  if (stage->state != IbMsegCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }
  stage->buffer = NULL;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct IbMsegSendComm)));
  NCCLCHECKGOTO(IbMsegSendCommInit(comm), ret, fail);
  comm->useMultiSeg = IbMsegResolveUseMultiSeg(handle->isGin != 0, comm->base.resiliency != NULL);
  // RCCL: combined CTS slots only when multi-segment. The published address is the first CTS row.
  NCCLCHECKGOTO(IbMsegSendCommAllocCts(comm, comm->useMultiSeg != 0), ret, fail);
  NCCLCHECKGOTO(IbMsegStatsInit(&comm->base.stats), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1),
                ret, fail);
  stage->comm = comm;
  stage->state = IbMsegCommStateConnect;
  NCCLCHECKGOTO(ncclSocketConnect(&comm->base.sock), ret, fail);

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECKGOTO(ncclSocketReady(&comm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;

  // IB Setup
  struct IbMsegMergedDev* mergedDev;
  if (dev >= IbMsegNMergedDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", dev);
    return ncclInternalError;
  }

  mergedDev = IbMsegMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  stage->state = IbMsegCommStateSendDevList;
  stage->offset = 0;
  struct IbMsegConnectionMetadata meta;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(meta)), ret, fail);
  memcpy(stage->buffer, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));

  struct IbMsegDevExtraProps exProps;
  exProps.oooRq = true;
  for (int i = 0; i < mergedDev->vProps.ndevs; i++) {
    int ibDevN = mergedDev->vProps.devs[i];
    exProps.oooRq = exProps.oooRq && IbMsegDevs[ibDevN].oooRqSize;
  }
  comm->base.localOooRq = exProps.oooRq;
  memcpy((char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), &exProps, sizeof(struct IbMsegDevExtraProps));

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number
// of RC qps
ib_send_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer,
                               sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps))) return ncclSuccess;

  stage->state = IbMsegCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer,
                               sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps))) return ncclSuccess;
  stage->offset = 0;
  ncclNetVDeviceProps_t remoteVProps;
  int trafficClass;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  memcpy(&exProps, (char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), sizeof(exProps));
  comm->base.remOooRq = exProps.oooRq;

  mergedDev = IbMsegMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  // Read isP2p from handle
  isP2p = handle->isP2p;
  INFO(NCCL_NET, "NET/IB: IbMsegConnect isP2p=%d", isP2p);
  comm->base.nqps = rcclIbSelectNqps(nQpsPerDev, isP2p, comm->base.vProps.ndevs, remoteVProps.ndevs, __func__);
  // RCCL: only GIN/RMA connects pass nQpsPerDev. GIN connects that never post multi-segment chains (the collComm
  // ring, GDAKI) also get the deeper send queue; that costs memory only.
  comm->base.isRma = nQpsPerDev > 0;

  comm->base.nDataQps = std::max(comm->base.vProps.ndevs, remoteVProps.ndevs);

  if (comm->base.resiliency) {
    NCCLCHECK(IbMsegResiliencyDeviceNumSet(comm->base.resiliency, comm->base.vProps.ndevs, remoteVProps.ndevs));
  }

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  // Sender's CQ size needs to accomodate the upper bound of number of send
  // requests multiplied by the number of QPs used per request.
  int cqSize;
  // RCCL: the deeper RMA send queue keeps this CQ size, since a request has at most one signaled WR (its round's
  // tail) outstanding. Signaling more WRs per request would need a deeper CQ.
  cqSize = NET_IB_MAX_REQUESTS * rcclIbCqQpsPerDev(nQpsPerDev);
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    int ibDevN = comm->base.vProps.devs[i];
    if (comm->base.resiliency) {
      NCCLCHECKGOTO(IbMsegResiliencyDataCqSizeGet(comm->base.resiliency, i, &cqSize), ret, fail);
    }
    NCCLCHECKGOTO(IbMsegInitCommDevBase(ibDevN, &comm->devs[i].base, &comm->base.stats, cqSize), ret, fail);
    comm->ar = comm->ar && IbMsegDevs[ibDevN].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
    if (comm->base.resiliency) {
      NCCLCHECKGOTO(IbMsegResiliencyDevInit(comm->base.resiliency, i, &IbMsegDevs[ibDevN]), ret, fail);
    }
  }

  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.vProps.ndevs;
  meta.isP2p = isP2p;
  meta.useMultiSeg = comm->useMultiSeg;

  // Create QPs on the sender side
  NCCLCHECKGOTO(IbMsegSenderQpsCreate(comm, &meta), ret, fail);

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    IbMsegSendCommDev* commDev = comm->devs + i;
    IbMsegDev* ibDev = IbMsegDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    IbMsegDevInfo* devInfo = meta.devs + i;
    devInfo->ib_port = ibDev->portNum;
    devInfo->mtu = ibDev->portAttr.active_mtu;
    devInfo->lid = ibDev->portAttr.lid;

    // Prepare GIN Put Signal scratchpad (for RDMA Atomic result)
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->putSignalScratchpadMr, commDev->base.pd, &comm->putSignalScratchpad,
                                  sizeof(comm->putSignalScratchpad), IBV_ACCESS_LOCAL_WRITE),
                  ret, fail);

    // RCCL: one MR over the legacy FIFO or the combined slots. The layout prefix is in that MR.
    void* ctsBase = comm->ctsSlots != NULL ? (void*)comm->ctsSlots : (void*)comm->ctsFifo;
    size_t ctsBytes = comm->ctsSlots != NULL ? NCCL_IB_CTS_SLOTS_BYTES : NCCL_IB_CTS_FIFO_BYTES;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->ctsFifoMr, commDev->base.pd, ctsBase, ctsBytes,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    devInfo->rkey = commDev->ctsFifoMr->rkey;

    // Pack local GID info
    devInfo->link_layer = commDev->base.gidInfo.link_layer;
    devInfo->gid.global.subnet_prefix = commDev->base.gidInfo.localGid.global.subnet_prefix;
    devInfo->gid.global.interface_id = commDev->base.gidInfo.localGid.global.interface_id;

    // info logging
    for (int q = 0; q < comm->base.nqps; q++) {
      // Print just the QPs for this dev
      if (comm->base.qps[q].devIndex == i) {
        if (devInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
          INFO(NCCL_NET,
               "NET/IB: %s: %s %d IbDev %d Port %d qp_num %d mtu %d LID %d subnet-prefix %lu  FLID %d ctsFifoRkey=0x%x "
               "ctsFifoLkey=0x%x",
               __func__, comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev, commDev->base.ibDevN,
               ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid, devInfo->gid.global.subnet_prefix,
               IbMsegExtractFlid(&devInfo->gid), commDev->ctsFifoMr->rkey, commDev->ctsFifoMr->lkey);
        } else { // RoCE
          INFO(
            NCCL_NET,
            "NET/IB: %s: %s %d IbDev %d Port %d qp_num %d mtu %d GID %ld (%lX/%lX) ctsFifoRkey=0x%x ctsFifoLkey=0x%x",
            __func__, comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev, commDev->base.ibDevN,
            ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, (int64_t)commDev->base.gidInfo.localGidIndex,
            devInfo->gid.global.subnet_prefix, devInfo->gid.global.interface_id, commDev->ctsFifoMr->rkey,
            commDev->ctsFifoMr->lkey);
        }
        // Log ECE info
        if (meta.qpInfo[q].ece_supported) {
          INFO(NCCL_NET,
               "NET/IB: %s: IbDev %d Port %d qp_num %d query_ece={supported=%d, vendor_id=0x%x, options=0x%x, "
               "comp_mask=0x%x}",
               __func__, commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, meta.qpInfo[q].ece_supported,
               meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options, meta.qpInfo[q].ece.comp_mask);
        }
      }
    }
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = devInfo->link_layer;
    if (link_layer != devInfo->link_layer) {
      int ibDev0 = comm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of "
           "only one link type using NCCL_IB_HCA",
           commDev->base.ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0,
           IbMsegDevs[ibDev0].devName, IbMsegDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  if (ncclParamIbEventBasedLb() && ncclParamIbEventBasedLbRemote()) {
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&comm->remoteSpeedMr, comm->devs[0].base.pd, &comm->remoteSpeedBuf,
                                  sizeof(comm->remoteSpeedBuf), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE),
                  ret, fail);
  }
  trafficClass = IbMsegGetTrafficClass(ctx);
  // RCCL: peers add ctsSlotStride to reach the next row. Multi-segment comms publish the first CTS row.
  meta.addr = comm->ctsSlots != NULL ? (uint64_t)comm->ctsSlots[0].cts : (uint64_t)comm->ctsFifo;
  meta.sl = (ncclParamIbSl() != NCCL_PARAM_VAL_AUTO)       ? ncclParamIbSl() :
            (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass :
                                                             NCCL_IB_SL_DEFAULT;
  meta.tc = (envTrafficClass != NCCL_PARAM_VAL_AUTO)       ? envTrafficClass :
            (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass :
                                                             NCCL_IB_TC_DEFAULT;
  meta.remSpeedBufAddr = comm->remoteSpeedMr ? (uint64_t)&comm->remoteSpeedBuf : 0;
  meta.remSpeedBufRkey = comm->remoteSpeedMr ? comm->remoteSpeedMr->rkey : 0;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = IbMsegCommStateSend;
  stage->offset = 0;

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta), &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(meta)) return ncclSuccess;

  stage->state = IbMsegCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct IbMsegConnectionMetadata remMeta;
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(IbMsegConnectionMetadata),
                                   &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(IbMsegConnectionMetadata));
  // RCCL: defensive; the accepting peer checks first and closes the socket on a mismatch, so the read above fails.
  NCCLCHECKGOTO(IbMsegCheckPeerUseMultiSeg(comm->useMultiSeg, remMeta.useMultiSeg, remMeta.devName), ret, fail);

  // ensure that the remote devices have the same link layer than the local devices used in the connection.
  if (comm->base.vProps.ndevs > 0) {
    int ibDev0 = comm->devs[0].base.ibDevN;
    link_layer = IbMsegDevs[ibDev0].portAttr.link_layer;
    for (int i = 0; i < remMeta.ndevs; i++) {
      if (remMeta.devs[i].link_layer != link_layer) {
        WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one "
             "link type using NCCL_IB_HCA",
             NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, IbMsegDevs[ibDev0].devName, IbMsegDevs[ibDev0].portNum,
             NCCL_IB_LLSTR(link_layer));
        return ncclInternalError;
      }
    }
  }

  // Store the number of remote devices
  comm->base.nRemDevs = remMeta.ndevs;

  // Store the remote GID information per-device provided by the remote peer
  for (int i = 0; i < comm->base.nRemDevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id = comm->base.remDevs[i].gid.global.interface_id;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix = comm->base.remDevs[i].gid.global.subnet_prefix;
  }

  // Store the completion records info provided by the remote
  comm->remCmplsRecords.addr = remMeta.addr;
  for (int i = 0; i < comm->base.nRemDevs; i++) {
    comm->remCmplsRecords.rkeys[i] = remMeta.devs[i].rkey;
    if (comm->base.resiliency) {
      NCCLCHECKGOTO(IbMsegResiliencyRemoteCompletionRecordsSet(comm->base.resiliency, comm->remCmplsRecords.rkeys[i],
                                                               comm->remCmplsRecords.addr, i),
                    ret, fail);
    }
  }

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    IbMsegSendCommDev* commDev = comm->devs + i;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->cmplsRecordsMr, comm->devs[i].base.pd, &comm->remCmplsRecords.elems,
                                  sizeof(comm->remCmplsRecords.elems),
                                  IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    comm->devs[i].sge.lkey = comm->devs[i].cmplsRecordsMr->lkey;
  }

  if (remMeta.flagFlushAddr != 0) {
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&comm->devs[0].flagFlushSrcMr, comm->devs[0].base.pd, comm->flagFlushSrc,
                                  sizeof(comm->flagFlushSrc), IBV_ACCESS_LOCAL_WRITE),
                  ret, fail);
    comm->flagFlushSge.lkey = comm->devs[0].flagFlushSrcMr->lkey;
    comm->flagFlushSge.length = sizeof(uint64_t);
    comm->flagFlushRemAddr = remMeta.flagFlushAddr;
    comm->flagFlushRkey = remMeta.flagFlushRkey;
    comm->flagFlush = 1;
    INFO(NCCL_NET, "NET/IB: %s: GDR flush flag enabled (remote flag 0x%lx rkey 0x%x)", __func__,
         comm->flagFlushRemAddr, comm->flagFlushRkey);
  }

  NCCLCHECKGOTO(IbMsegSenderQpsToRts(comm, &remMeta), ret, fail);
  IbMsegComputeDevSpeeds(&comm->base);
  IbMsegComputeLbWeights(&comm->base);

  comm->base.ready = 1;
  stage->state = IbMsegCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *sendComm = comm;
exit:
  if (stage->buffer) free(stage->buffer);
  stage->state = IbMsegCommStateStart;
  return ret;
fail:
  if (comm && comm->devs[0].flagFlushSrcMr != nullptr) {
    (void)wrap_ibv_dereg_mr(comm->devs[0].flagFlushSrcMr);
    comm->devs[0].flagFlushSrcMr = nullptr;
  }
  IbMsegSendCommFreeCts(comm); // RCCL: the CTS block is a separate allocation.
  free(comm);
  goto exit;
}

ncclResult_t IbMsegConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                           ncclNetDeviceHandle_t** sendDevComm) {
  return IbMsegConnectImpl(ctx, dev, opaqueHandle, sendComm, sendDevComm, /*nQpsPerDev*/ 0, ncclParamIbTc());
}

USE_NCCL_PARAM(ncclParamIbWarnRailLocal, bool);

ncclResult_t IbMsegCheckVProps(ncclNetVDeviceProps_t* vProps1, ncclNetVDeviceProps_t* vProps2) {
  ncclNetVDeviceProps_t outVProps = {0};
  ncclNetVDeviceProps_t* minVProps = vProps2;
  ncclNetVDeviceProps_t* maxVProps = vProps1;
  if (vProps2->ndevs > vProps1->ndevs) {
    minVProps = vProps1;
    maxVProps = vProps2;
  }

  // Find the intersection of devices
  for (int i = 0; i < minVProps->ndevs; i++) {
    int dev = minVProps->devs[i];
    for (int j = 0; j < maxVProps->ndevs; j++) {
      // Found
      if (maxVProps->devs[j] == dev) {
        outVProps.devs[outVProps.ndevs++] = dev;
      }
    }
  }

  // In the case that at least one side has a fused NIC but there are no matching physical NICs, we should check if
  // the user wants this
  if (ncclParamIbWarnRailLocal() && outVProps.ndevs < maxVProps->ndevs) {
    char local[128];
    int cursor = 1;
    snprintf(local, sizeof(local), "%d", vProps1->devs[0]);
    for (int i = 1; i < vProps1->ndevs; i++) {
      snprintf(local + cursor, sizeof(local) - cursor, ",%d", vProps1->devs[i]);
      cursor += 2;
    }
    char remote[128];
    snprintf(remote, sizeof(remote), "%d", vProps2->devs[0]);
    cursor = 1;
    for (int i = 1; i < vProps2->ndevs; i++) {
      snprintf(remote + cursor, sizeof(remote) - cursor, ",%d", vProps2->devs[i]);
      cursor += 2;
    }
    INFO(NCCL_NET,
         "NET/IB : There are mismatched physical devices between local (%s) and remote (%s). To disable this warning, "
         "set NCCL_IB_WARN_RAIL_LOCAL=0",
         local, remote);
  }

  return ncclSuccess;
}

// The function creates and modifies QPs to RTS state on the receiver side
// using remote information from the sender side (remMeta). It also populates
// the remote metadata structure, provided to the function (remMeta), with the
// QPs' information so that data structure could be delivered to the remote
// side (sender) as part of the connection establishment process.
static ncclResult_t IbMsegReceiverQpsCreateToRts(IbMsegRecvComm* rComm, struct IbMsegConnectionMetadata* remMeta,
                                                 struct IbMsegConnectionMetadata* meta) {
  uint nqps = rComm->base.nqps;
  struct IbMsegQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(struct IbMsegQpCreateAttr));
  qpCreateAttrs.type = IBV_QPT_RC;
  qpCreateAttrs.maxRecvWorkRequest = NET_IB_MAX_REQUESTS;
  // CTS messages are posted using send work requests.
  // Note that because only specific CTS messages are signaled, the send queue
  // size needs to be double the number of max requests.
  // When resiliency is enabled, the number of send work requests is same as the
  // number of max requests because every CTS message is signaled.
  // +1 reserves space for one in-flight speed update RDMA write on qps[0].
  qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS * (rComm->base.resiliency ? 1 : 2) +
                                     ((ncclParamIbEventBasedLb() && ncclParamIbEventBasedLbRemote()) ? 1 : 0);
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % rComm->base.vProps.ndevs;
    IbMsegRecvCommDev* rCommDev = &rComm->devs[devIndex];
    IbMsegDev* ibDev = &IbMsegDevs[rCommDev->base.ibDevN];
    IbMsegQpInfo* remQpInfo = &remMeta->qpInfo[qpIndex];
    IbMsegQpInfo* localQpInfo = &meta->qpInfo[qpIndex];
    int remDevIndex = remQpInfo->devIndex;
    // RCCL: the peer sends devIndex, and it indexes remDevs and the RMA rkey tables, filled up to its ndevs.
    const int nRemDevs = std::min(remMeta->ndevs, NCCL_IB_MAX_DEVS_PER_NIC);
    if (remDevIndex < 0 || remDevIndex >= nRemDevs) {
      WARN("NET/IB: peer QP %d reports device index %d, expected [0, %d)", qpIndex, remDevIndex, nRemDevs);
      return ncclInternalError;
    }
    IbMsegDevInfo* remDevInfo = &remMeta->devs[remDevIndex];
    IbMsegQp* localQp = &rComm->base.qps[qpIndex];

    localQp->remDevIdx = remDevIndex;
    localQp->devIndex = devIndex;

    qpCreateAttrs.cq = rCommDev->base.cq;
    qpCreateAttrs.pd = rCommDev->base.pd;
    qpCreateAttrs.qpContext = &rComm->base.stats;
    if (rComm->base.resiliency) {
      NCCLCHECK(IbMsegResiliencyDataRqSizeGet(rComm->base.resiliency, devIndex, &qpCreateAttrs.maxRecvWorkRequest));
    }
    if (ibDev->ibProvider == IB_PROVIDER_MLX5 && ncclParamIbOooRq()) {
      if (ibDev->ar == 0) {
        WARN("NET/IB: %s: OOO RQ is force enabled but AR is not enabled, which is required for OOO RQ (device=%s)",
             __func__, ibDev->devName);
        return ncclInternalError;
      }
      qpCreateAttrs.oooRq = (rComm->base.remOooRq && rComm->base.localOooRq);
      // out-of-order recv prerequisite: oooRq is supported on both sides
      if (!qpCreateAttrs.oooRq) {
        WARN("NET/IB: %s: OOO RQ is force enabled but not supported on both sides of the connection (device=%s, "
             "localOooRq=%d, remOooRq=%d)",
             __func__, ibDev->devName, rComm->base.localOooRq, rComm->base.remOooRq);
        return ncclInternalError;
      }
      // out-of-order recv prerequisite: oooRq size requirements are met
      if (ibDev->oooRqSize < qpCreateAttrs.maxRecvWorkRequest) {
        WARN("NET/IB: %s: OOO RQ is force enabled but size %u is less than the required recv work request size %u on "
             "device:%s",
             __func__, ibDev->oooRqSize, qpCreateAttrs.maxRecvWorkRequest, ibDev->devName);
        return ncclInternalError;
      }
    }
    NCCLCHECK(IbMsegQpCreate(localQp, &qpCreateAttrs));
    INFO(NCCL_NET,
         "NET/IB: %s: QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p oooRq=%d",
         __func__, ibDev->portNum, rCommDev->base.ibDevN, IbMsegDevs[rCommDev->base.ibDevN].devName, IbMsegNDevs,
         IbMsegNMergedDevs, localQp->qp->qp_num, (uint16_t)rCommDev->base.pkeyIndex, rCommDev->base.pd,
         qpCreateAttrs.oooRq);

    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    // Transition the QP to INIT state
    struct IbMsegQpInitAttr* initAttr = &localQp->initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = rCommDev->base.pkeyIndex;
    initAttr->portNum = ibDev->portNum;
    // Remote Atomic operations are used for GIN! REMOTE_READ is required for GIN Get (RDMA READ).
    initAttr->qpAccessFlags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_READ;
    NCCLCHECK(IbMsegQpInit(localQp));

    if (remQpInfo->ece_supported) {
      // Set the ECE received from the remote (sender) side.
      // coverity[copy_paste_error]
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &remQpInfo->ece, &localQpInfo->ece_supported));
    } else {
      localQpInfo->ece_supported = 0;
      localQp->ece = {0};
      localQp->eceSupported = 0;
    }

    // Reduce the local MTU to match the remote MTU if needed
    ibDev->portAttr.active_mtu = std::min(ibDev->portAttr.active_mtu, remDevInfo->mtu);

    struct IbMsegQpRtrAttr* rtrAttr = &localQp->rtrAttr;
    rtrAttr->mtu = ibDev->portAttr.active_mtu;
    rtrAttr->linkLayer = remDevInfo->link_layer;
    rtrAttr->tc = (remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET && ncclParamIbFifoTc() != NCCL_PARAM_VAL_AUTO) ?
                    ncclParamIbFifoTc() :
                    remMeta->tc;
    rtrAttr->sl = remMeta->sl;
    rtrAttr->remoteQpNum = remQpInfo->qpn;
    rtrAttr->remoteLid = remDevInfo->lid;
    rtrAttr->remoteGid = remDevInfo->gid;
    rtrAttr->localIbPort = remDevInfo->ib_port;
    rtrAttr->localPortFlags = ibDev->portAttr.flags;
    rtrAttr->localGid = rCommDev->base.gidInfo.localGid;
    rtrAttr->localGidIndex = rCommDev->base.gidInfo.localGidIndex;
    NCCLCHECK(IbMsegQpRtr(localQp));
    struct IbMsegQpRtsAttr* rtsAttr = &localQp->rtsAttr;
    rtsAttr->timeout = ncclParamIbTimeout();
    rtsAttr->retryCnt = ncclParamIbRetryCnt();
    NCCLCHECK(IbMsegQpRts(localQp));

    // Query the reduced ECE by the device and storing it in the local QP info
    // to return it to the requestor (sender).
    if (remQpInfo->ece_supported && localQpInfo->ece_supported) {
      NCCLCHECK(wrap_ibv_query_ece(localQp->qp, &localQpInfo->ece, &localQpInfo->ece_supported));
      // Store the reduced ECE locally as well
      localQp->ece = localQpInfo->ece;
      localQp->eceSupported = localQpInfo->ece_supported;
    } else {
      localQp->ece = {0};
      localQp->eceSupported = 0;
    }
  }

  // Flush QPs are created lazily on the first flush (see IbMsegCreateFlushQp), so
  // here we only stash the remote-derived RTR inputs the loopback QP needs.
  if (rComm->flushEnabled) {
    rComm->flushQpSl = remMeta->sl;
    rComm->flushQpTc = remMeta->tc;
  }

  if (rComm->base.resiliency) {
    NCCLCHECK(IbMsegResiliencyReceiverQpsCreateToRts(rComm->base.resiliency, remMeta, &meta->resiliencyInfo));
  }

  return ncclSuccess;
}

// Build the flush QP and its host buffer on the first flush. NCCL only issues a
// network-read flush when the topology needs one, so connections that never flush
// never allocate these resources.
ncclResult_t IbMsegCreateFlushQp(struct IbMsegRecvComm* comm) {
  if (comm->flushEnabled == 0 || comm->flushQpsCreated) return ncclSuccess;

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    IbMsegRecvCommDev* rCommDev = &comm->devs[i];
    IbMsegDev* ibDev = &IbMsegDevs[rCommDev->base.ibDevN];
    IbMsegQp* flushQp = &rCommDev->gpuFlush.qp;

    struct IbMsegQpCreateAttr qpCreateAttrs;
    memset(&qpCreateAttrs, 0, sizeof(struct IbMsegQpCreateAttr));
    qpCreateAttrs.type = IBV_QPT_RC;
    qpCreateAttrs.cq = rCommDev->base.cq;
    qpCreateAttrs.pd = rCommDev->base.pd;
    qpCreateAttrs.maxRecvWorkRequest = 0;
    // Each flush may post a scratchpad write in addition to the read.
    qpCreateAttrs.maxSendWorkRequest = 2 * NET_IB_MAX_REQUESTS;
    // IbMsegIflush posts the scratchpad write inline.
    qpCreateAttrs.maxInlineData = sizeof(comm->gpuFlushHostMem);
    qpCreateAttrs.qpContext = &comm->base.stats;
    NCCLCHECK(IbMsegQpCreate(flushQp, &qpCreateAttrs));
    // RCCL: loopback flush QP, so local and remote device indices are this recv device.
    flushQp->devIndex = i;
    flushQp->remDevIdx = i;
    INFO(NCCL_NET, "NET/IB: %s: Flush QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p",
         __func__, ibDev->portNum, rCommDev->base.ibDevN, IbMsegDevs[rCommDev->base.ibDevN].devName, IbMsegNDevs,
         IbMsegNMergedDevs, flushQp->qp->qp_num, (uint16_t)rCommDev->base.pkeyIndex, rCommDev->base.pd);

    struct IbMsegQpInitAttr* initAttr = &flushQp->initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = rCommDev->base.pkeyIndex;
    initAttr->portNum = ibDev->portNum;
    initAttr->qpAccessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    NCCLCHECK(IbMsegQpInit(flushQp));

    // The flush QP is a loopback QP (connected to itself), so the "remote" target is
    // this same QP on the local port; only sl/tc come from the peer metadata.
    struct IbMsegQpRtrAttr* rtrAttr = &flushQp->rtrAttr;
    rtrAttr->mtu = ibDev->portAttr.active_mtu;
    rtrAttr->linkLayer = ibDev->portAttr.link_layer;
    rtrAttr->tc = ibDev->portAttr.link_layer == IBV_LINK_LAYER_ETHERNET ? comm->flushQpTc : -1;
    rtrAttr->sl = comm->flushQpSl;
    rtrAttr->remoteQpNum = flushQp->qp->qp_num;
    rtrAttr->remoteLid = ibDev->portAttr.lid;
    rtrAttr->remoteGid = rCommDev->base.gidInfo.localGid;
    rtrAttr->localIbPort = ibDev->portNum;
    rtrAttr->localPortFlags = ibDev->portAttr.flags;
    rtrAttr->localGid = rCommDev->base.gidInfo.localGid;
    rtrAttr->localGidIndex = rCommDev->base.gidInfo.localGidIndex;
    NCCLCHECK(IbMsegQpRtr(flushQp));

    struct IbMsegQpRtsAttr* rtsAttr = &flushQp->rtsAttr;
    rtsAttr->timeout = ncclParamIbTimeout();
    rtsAttr->retryCnt = ncclParamIbRetryCnt();
    NCCLCHECK(IbMsegQpRts(flushQp));

    NCCLCHECK(wrap_ibv_reg_mr(&rCommDev->gpuFlush.hostMr, rCommDev->base.pd, &comm->gpuFlushHostMem, sizeof(int),
                              IBV_ACCESS_LOCAL_WRITE));
    rCommDev->gpuFlush.sge.addr = (uint64_t)&comm->gpuFlushHostMem;
    rCommDev->gpuFlush.sge.length = 1;
    rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
  }

  comm->flushQpsCreated = true;
  return ncclSuccess;
}

ncclResult_t IbMsegPostReceiveWorkRequestsOnQp(struct IbMsegRecvComm* recvComm, IbMsegQp* dataQp) {
  uint32_t nRecvWorkRequestsPerQp = NET_IB_MAX_REQUESTS;
  if (recvComm->base.resiliency) {
    NCCLCHECK(IbMsegResiliencyDataRqSizeGet(recvComm->base.resiliency, dataQp->devIndex, &nRecvWorkRequestsPerQp));
  }
  INFO(NCCL_NET, "NET/IB: %s: Pre-posting %d Receive WQEs on QP (qp_num=%d, comm=%p)", __func__, nRecvWorkRequestsPerQp,
       dataQp->qp->qp_num, recvComm);
  for (int j = 0; j < nRecvWorkRequestsPerQp; j++) {
    NCCLCHECK(IbMsegPostRecvWorkRequest(dataQp->qp, &recvComm->ibRecvWorkRequest));
  }
  return ncclSuccess;
}

ncclResult_t IbMsegReceiverPrePostReceiveWorkRequests(struct IbMsegRecvComm* recvComm) {
  int nqps = recvComm->base.nqps;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(IbMsegPostReceiveWorkRequestsOnQp(recvComm, &recvComm->base.qps[i]));
  }
  return ncclSuccess;
}

USE_NCCL_PARAM(ncclParamIbGdrFlushDisable, bool);

static ncclResult_t IbMsegFreeGpuFlushMem(struct IbMsegGpuFlush* gpuFlush) {
  if (gpuFlush->gpuFlushGpuMem == nullptr) return ncclSuccess;
  if (gpuFlush->gpuFlushMemIsHipAlloc) {
    CUDACHECK(hipFree(gpuFlush->gpuFlushGpuMem));
  } else {
    NCCLCHECK(ncclCudaFree(gpuFlush->gpuFlushGpuMem, /*manager=*/nullptr));
  }
  gpuFlush->gpuFlushGpuMem = nullptr;
  gpuFlush->gpuFlushMemIsHipAlloc = false;
  return ncclSuccess;
}

uint64_t* IbMsegGetGpuFlushFlag(void* recvComm) {
  struct IbMsegRecvComm* rComm = (struct IbMsegRecvComm*)recvComm;
  return (rComm && rComm->flagFlush) ? rComm->flagFlushGpuMem : nullptr;
}

void IbMsegSetGpuFlushFlagHint(void* recvComm, int need) {
  ((struct IbMsegRecvComm*)recvComm)->flagFlushHint = need;
}

uint64_t IbMsegGetGpuFlushFlagSeq(void* recvComm) {
  return ((struct IbMsegRecvComm*)recvComm)->flagFlushLastDesc;
}

static ncclResult_t IbMsegFreeFlagFlush(struct IbMsegRecvComm* rComm) {
  if (rComm->flagFlushMr != nullptr) NCCLCHECK(wrap_ibv_dereg_mr(rComm->flagFlushMr));
  rComm->flagFlushMr = nullptr;
  if (rComm->flagFlushDmabufFd >= 0) close(rComm->flagFlushDmabufFd);
  rComm->flagFlushDmabufFd = -1;
#if defined(__HIP_PLATFORM_AMD__)
  if (rComm->flagFlushGpuMem != nullptr) CUDACHECK(hipFree((void*)rComm->flagFlushGpuMem));
#endif
  rComm->flagFlushGpuMem = nullptr;
  rComm->flagFlush = 0;
  return ncclSuccess;
}

// Allocates the receiver's GDR flush flags, one uint64_t per QP in one page,
// NCCL_GDR_FLUSH_FLAG_STRIDE bytes apart.
// The page is registered without IBV_ACCESS_RELAXED_ORDERING so the NIC issues
// flag writes with RO=0, which cannot pass the preceding (RO=1) payload writes
// on the same QP. The proxy writes the flags and the per-step sequences through
// the CPU mapping of fine-grained VRAM, which is only available on
// gfx942/gfx950 (checked at communicator init). The page is registered through
// dma-buf when the connection uses dma-buf for GDR, and through peermem
// otherwise. Any failure leaves this connection on the RDMA read flush: the
// sender only writes flags it was given an address for.
static ncclResult_t IbMsegSetupFlagFlush(struct IbMsegRecvComm* rComm, struct IbMsegConnectionMetadata* meta,
                                         bool useDmaBuf) {
#if defined(__HIP_PLATFORM_AMD__)
  void* ptr = nullptr;
#if defined(HIP_UNCACHED_MEMORY)
  const unsigned int flagFlags = hipDeviceMallocUncached;
#else
  const unsigned int flagFlags = hipDeviceMallocFinegrained;
#endif
  if (hipExtMallocWithFlags(&ptr, GPU_PAGE_SIZE, flagFlags) != hipSuccess) {
    (void)hipGetLastError();
    INFO(NCCL_NET, "NET/IB: GDR flush flag allocation failed, using RDMA read flush");
    return ncclSuccess;
  }
  rComm->flagFlushGpuMem = (uint64_t*)ptr;
  rComm->flagFlushDmabufFd = -1;
  for (int q = 0; q < rComm->base.nqps; q++)
    __atomic_store_n(rComm->flagFlushGpuMem + q * (NCCL_GDR_FLUSH_FLAG_STRIDE / sizeof(uint64_t)), 0, __ATOMIC_RELAXED);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  ncclResult_t regRes = ncclSystemError;
  if (useDmaBuf) {
    uint64_t exportOffset = 0;
    void* alignedPtr = nullptr;
    size_t alignedSize = 0;
    get_aligned_ptr_and_size(ptr, GPU_PAGE_SIZE, &alignedPtr, &alignedSize);
    if (hsa_amd_portable_export_dmabuf(alignedPtr, alignedSize, &rComm->flagFlushDmabufFd, &exportOffset) ==
          HSA_STATUS_SUCCESS &&
        rComm->flagFlushDmabufFd >= 0) {
      regRes = wrap_ibv_reg_dmabuf_mr(&rComm->flagFlushMr, rComm->devs[0].base.pd, exportOffset, GPU_PAGE_SIZE,
                                      (uint64_t)ptr, rComm->flagFlushDmabufFd,
                                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }
  } else {
    regRes = wrap_ibv_reg_mr(&rComm->flagFlushMr, rComm->devs[0].base.pd, ptr, GPU_PAGE_SIZE,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  }
  if (regRes != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: GDR flush flag %s registration failed, using RDMA read flush",
         useDmaBuf ? "dma-buf" : "peermem");
    rComm->flagFlushMr = nullptr;
    NCCLCHECK(IbMsegFreeFlagFlush(rComm));
    return ncclSuccess;
  }
  rComm->flagFlush = 1;
  rComm->flagFlushHint = 0;
  rComm->flagFlushLastDesc = 0;
  meta->flagFlushAddr = (uint64_t)ptr;
  meta->flagFlushRkey = rComm->flagFlushMr->rkey;
  INFO(NCCL_NET, "NET/IB: GDR flush flag enabled via %s (flag %p rkey 0x%x)", useDmaBuf ? "dma-buf" : "peermem", ptr,
       meta->flagFlushRkey);
#else
  (void)rComm;
  (void)meta;
  (void)useDmaBuf;
#endif
  return ncclSuccess;
}

ncclResult_t IbMsegAcceptImpl(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/,
                              int nQpsPerDev) {
  ncclResult_t ret = ncclSuccess;
  struct IbMsegListenComm* lComm = (struct IbMsegListenComm*)listenComm;
  struct IbMsegCommStage* stage = lComm->stage;
  if (stage == NULL) {
    NCCLCHECK(ncclCalloc(&lComm->stage, 1));
    stage = lComm->stage;
  }
  struct IbMsegRecvComm* rComm = (struct IbMsegRecvComm*)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  // Pre-declare dmabuf variables because of goto
  bool useDmaBuf = false;
  bool peermemAvailable = false;
  bool dmabufSupported = false;
  bool dmabufEnabled = false;
  bool dmabufAvailable = false;
  bool forceDmaBuf = false;
  *recvComm = NULL;

  if (stage->state == IbMsegCommStateAccept) goto ib_accept_check;
  if (stage->state == IbMsegCommStateRecvDevList) goto ib_recv_dev_list;
  if (stage->state == IbMsegCommStateSendDevList) goto ib_send_dev_list;
  if (stage->state == IbMsegCommStateRecv) goto ib_recv;
  if (stage->state == IbMsegCommStateSend) goto ib_send;
  if (stage->state == IbMsegCommStatePendingReady) goto ib_recv_ready;
  if (stage->state != IbMsegCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct IbMsegRecvComm)));
  NCCLCHECKGOTO(ncclSocketInit(&rComm->base.sock), ret, fail);
  NCCLCHECKGOTO(IbMsegRecvCommInit(rComm), ret, fail);
  rComm->useMultiSeg = IbMsegResolveUseMultiSeg(lComm->isGin, rComm->base.resiliency != NULL);
  NCCLCHECKGOTO(IbMsegStatsInit(&rComm->base.stats), ret, fail);
  stage->comm = rComm;
  stage->state = IbMsegCommStateAccept;
  NCCLCHECKGOTO(ncclSocketAccept(&rComm->base.sock, &lComm->sock), ret, fail);

  // Alloc stage->buffer here to be used for all following steps
  struct IbMsegConnectionMetadata remMeta;
  struct IbMsegDevExtraProps exProps;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)));

ib_accept_check:
  NCCLCHECKGOTO(ncclSocketReady(&rComm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;
  stage->state = IbMsegCommStateRecvDevList;
  stage->offset = 0;

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number
// of RC qps
ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer,
                               sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps))) return ncclSuccess;
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  if (lComm->dev >= IbMsegNMergedDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", lComm->dev);
    return ncclInternalError;
  }

  memcpy(&exProps, (char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), sizeof(exProps));
  rComm->base.remOooRq = exProps.oooRq;

  // Reduce the physical device list and store in the connection base
  struct IbMsegMergedDev* mergedDev;
  mergedDev = IbMsegMergedDevs + lComm->dev;
  NCCLCHECK(IbMsegCheckVProps(&mergedDev->vProps, &remoteVProps));
  rComm->base.vProps = mergedDev->vProps;
  memcpy(stage->buffer, &rComm->base.vProps, sizeof(ncclNetVDeviceProps_t));

  rComm->base.nDataQps = std::max(rComm->base.vProps.ndevs, remoteVProps.ndevs);

  if (rComm->base.resiliency) {
    NCCLCHECK(IbMsegResiliencyDeviceNumSet(rComm->base.resiliency, rComm->base.vProps.ndevs, remoteVProps.ndevs));
  }

  stage->offset = 0;
  stage->state = IbMsegCommStateSendDevList;

  exProps.oooRq = true;
  for (int i = 0; i < mergedDev->vProps.ndevs; i++) {
    int ibDevN = mergedDev->vProps.devs[i];
    exProps.oooRq = exProps.oooRq && IbMsegDevs[ibDevN].oooRqSize;
  }
  rComm->base.localOooRq = exProps.oooRq;
  memcpy((char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), &exProps, sizeof(struct IbMsegDevExtraProps));

ib_send_dev_list:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer,
                                   sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps), &stage->offset),
                ret, fail);
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct IbMsegDevExtraProps))) return ncclSuccess;

  stage->offset = 0;
  stage->state = IbMsegCommStateRecv;

ib_recv:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta), &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct IbMsegConnectionMetadata));
  NCCLCHECKGOTO(IbMsegCheckPeerUseMultiSeg(rComm->useMultiSeg, remMeta.useMultiSeg, remMeta.devName), ret, fail);
  rComm->base.nqps = rcclIbSelectNqps(nQpsPerDev, remMeta.isP2p, rComm->base.vProps.ndevs, remMeta.ndevs, __func__);

  // Subnet-aware device selection: use the remote sender's GIDs to find a local
  // NIC on the same subnet. Override lComm->dev and update vProps if a
  // better device is found.
  if (ncclParamIbSubnetAwareRouting() && remMeta.ndevs > 0) {
    union ibv_gid remoteGids[NCCL_IB_MAX_DEVS_PER_NIC];
    int nRemoteGids = 0;
    for (int i = 0; i < remMeta.ndevs && i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      if (remMeta.devs[i].link_layer == IBV_LINK_LAYER_ETHERNET) {
        remoteGids[nRemoteGids++] = remMeta.devs[i].gid;
      }
    }
    int effectiveDev = lComm->dev;
    NCCLCHECKGOTO(IbMsegFindDevBySubnet(remoteGids, nRemoteGids, lComm->dev, &effectiveDev), ret, fail);
    if (effectiveDev != lComm->dev) {
      lComm->dev = effectiveDev;
      rComm->base.vProps = IbMsegMergedDevs[effectiveDev].vProps;
    }
  }

  // IB setup
  // Pre-declare variables because of goto
  struct IbMsegDev* ibDev;
  int ibDevN;
  struct IbMsegRecvCommDev* rCommDev;

  mergedDev = IbMsegMergedDevs + lComm->dev;

  if (remMeta.ndevs != rComm->base.vProps.ndevs) {
    INFO(NCCL_NET, "NET/IB : Local mergedDev %s has a different number of devices=%d as remote %s %d",
         mergedDev->devName, rComm->base.vProps.ndevs, remMeta.devName, remMeta.ndevs);
  }

  // Metadata to send back to requestor (sender)
  struct IbMsegConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  // Receiver's CQ size needs to accomodate receive requests that can generate
  // up to 2 completions (one for the CTS message and one for the completion
  // of a receive request) per QP, in the worst case.
  // +1 reserves space for one in-flight speed update RDMA write completion.
  int cqSize;
  cqSize = 3 * NET_IB_MAX_REQUESTS * rcclIbCqQpsPerDev(nQpsPerDev) +
           ((ncclParamIbEventBasedLb() && ncclParamIbEventBasedLbRemote()) ? 1 : 0);
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rComm->base.vProps.devs[i];
    if (rComm->base.resiliency) {
      NCCLCHECKGOTO(IbMsegResiliencyDataCqSizeGet(rComm->base.resiliency, i, &cqSize), ret, fail);
    }
    NCCLCHECKGOTO(IbMsegInitCommDevBase(ibDevN, &rCommDev->base, &rComm->base.stats, cqSize), ret, fail);
    if (rComm->base.resiliency) {
      NCCLCHECKGOTO(IbMsegResiliencyDevInit(rComm->base.resiliency, i, &IbMsegDevs[ibDevN]), ret, fail);
    }
    ibDev = IbMsegDevs + ibDevN;
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = ibDev->portAttr.link_layer;
    if (link_layer != ibDev->portAttr.link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of "
           "only one link type using NCCL_IB_HCA",
           ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0,
           IbMsegDevs[ibDev0].devName, IbMsegDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Before assigning information about remote devices provided by the remote,
  // ensure that they are compatible with local devices
  for (int i = 0; i < remMeta.ndevs; i++) {
    if (remMeta.devs[i].link_layer != link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link "
           "type using NCCL_IB_HCA",
           NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, IbMsegDevs[ibDev0].devName, IbMsegDevs[ibDev0].portNum,
           NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Store the number of remote devices provided by the remote peer
  rComm->base.nRemDevs = remMeta.ndevs;
  // Store the sender's speed buffer address and rkey for RDMA speed updates
  rComm->remSpeedBufAddr = remMeta.remSpeedBufAddr;

  // Store the remote GID information per-device provided by the remote peer
  for (int i = 0; i < rComm->base.nRemDevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id = rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].gid.global.subnet_prefix;
    rComm->base.remDevs[i].remSpeedBufRkey = remMeta.remSpeedBufRkey;
  }

  // Determine if Flush is enabled for this Comm. Must be done before creating
  // QPs. If Flush is enabled, extra QPs will be created for Flush operations.
  // GDR mode selection logic
  peermemAvailable = (IbMsegGdrSupport() == ncclSuccess);
  dmabufSupported = (IbMsegDmaBufSupport(lComm->dev) == ncclSuccess);
  dmabufEnabled = (ncclParamDmaBufEnable() != 0);
  dmabufAvailable = dmabufSupported && dmabufEnabled;
  forceDmaBuf = rcclParamForceEnableDMABUF();

  if (forceDmaBuf) {
    // RCCL_FORCE_ENABLE_DMABUF=1: always try DMAbuf, skip peermem
    useDmaBuf = dmabufAvailable;
    if (dmabufAvailable) {
      INFO(NCCL_INIT | NCCL_NET, "NET/IB: Using GPU Direct RDMA (DMAbuf - forced) for device %d", lComm->dev);
    } else {
      WARN("NET/IB: RCCL_FORCE_ENABLE_DMABUF=1 but DMAbuf not available, GPU Direct RDMA disabled for device %d",
           lComm->dev);
    }
  } else {
    // Normal path: prefer peermem over dmabuf
    useDmaBuf = !peermemAvailable && dmabufAvailable;
    if (peermemAvailable) {
      INFO(NCCL_INIT | NCCL_NET, "NET/IB: Using GPU Direct RDMA (peermem) for device %d", lComm->dev);
    } else if (useDmaBuf) {
      WARN("NET/IB: Peermem not available, falling back to GPU Direct RDMA (DMAbuf) for device %d", lComm->dev);
    } else if (!peermemAvailable && dmabufSupported && !dmabufEnabled) {
      WARN("NET/IB: Peermem not available and DMAbuf is disabled (NCCL_DMABUF_ENABLE=0), GPU Direct RDMA disabled for "
           "device %d",
           lComm->dev);
    } else {
      WARN("NET/IB: GPU Direct RDMA not available for device %d", lComm->dev);
    }
  }

  rComm->flushEnabled = ((peermemAvailable || useDmaBuf) && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;

  // RCCL: the sender's useMultiSeg was checked to match, so it selects the remote stride and the local slots or
  // legacy FIFO shadow.
  if (rComm->useMultiSeg) {
    NCCLCHECKGOTO(ncclIbMalloc((void**)&rComm->ctsSlots, NCCL_IB_CTS_SLOTS_BYTES), ret, fail);
    rComm->ctsSlotStride = sizeof(struct IbMsegCtsSlot);
  } else {
    NCCLCHECKGOTO(ncclIbMalloc((void**)&rComm->remCtsFifo.elems, NCCL_IB_CTS_FIFO_BYTES), ret, fail);
    rComm->ctsSlots = NULL;
    rComm->ctsSlotStride = NCCL_NET_IB_MAX_RECVS * sizeof(struct IbMsegSendFifo);
  }

  NCCLCHECKGOTO(IbMsegReceiverQpsCreateToRts(rComm, &remMeta, &meta), ret, fail);
  if (rComm->prepostReceiveWorkRequests) {
    NCCLCHECKGOTO(IbMsegReceiverPrePostReceiveWorkRequests(rComm), ret, fail);
  }

  // Store the remote CTS FIFO info provided by the remote peer
  rComm->remCtsFifo.addr = remMeta.addr;
  for (int i = 0; i < rComm->base.nRemDevs; i++) {
    rComm->remCtsFifo.rkeys[i] = remMeta.devs[i].rkey;
  }

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;

    // RCCL: one local MR. A multi-segment peer writes the layout and the CTS from the combined slots.
    void* ctsLocal = rComm->ctsSlots != NULL ? (void*)rComm->ctsSlots : (void*)rComm->remCtsFifo.elems;
    size_t ctsLocalBytes = rComm->ctsSlots != NULL ? NCCL_IB_CTS_SLOTS_BYTES : NCCL_IB_CTS_FIFO_BYTES;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->ctsFifoMr, rCommDev->base.pd, ctsLocal, ctsLocalBytes,
                                  IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    rCommDev->sge.lkey = rCommDev->ctsFifoMr->lkey;

    // Register completion records
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->cmplsRecordsMr, rCommDev->base.pd, &rComm->cmplsRecords,
                                  sizeof(rComm->cmplsRecords),
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    meta.devs[i].rkey = rCommDev->cmplsRecordsMr->rkey;

    // Register speed update buffer
    if (ncclParamIbEventBasedLb() && ncclParamIbEventBasedLbRemote()) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->speedUpdateMr, rCommDev->base.pd, &rComm->speedUpdateBuf,
                                    sizeof(rComm->speedUpdateBuf), IBV_ACCESS_LOCAL_WRITE),
                    ret, fail);
    }
  }
  if (ncclParamIbUseInline()) rComm->remCtsFifo.flags = IBV_SEND_INLINE;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = IbMsegDevs + rCommDev->base.ibDevN;

    // Allocate Flush dummy buffer for GPU Direct RDMA. The flush QP itself and its host
    // scratch MR are created lazily by IbMsegCreateFlushQp() on the first flush, but the
    // RCCL relaxed-ordering scratchpad is registered here while the DMA-BUF/peermem
    // decision for this accept is still available.
    if (rComm->flushEnabled) {
      bool gpuFlushRegistered = false;
      rCommDev->gpuFlush.gpuFlushGpuMem = nullptr;
      rCommDev->gpuFlush.gpuMr = nullptr;
      rCommDev->gpuFlush.dmabuf_fd = -1;
      rCommDev->gpuFlush.gpuFlushMemIsHipAlloc = false;

      if (rcclParamIbGdrFlushGpuMemNoRelaxedOrdering()) {
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
        if (ncclCuMemEnable()) {
          NCCLCHECKGOTO(ncclMemAlloc((void**)&rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int)), ret, cumem_flush_hsa);
          CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&rCommDev->gpuFlush.dmabuf_fd,
                                                    (CUdeviceptr)rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int),
                                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                      ret, cumem_flush_hsa);
          NCCLCHECKGOTO(
            wrap_ibv_reg_dmabuf_mr(&rCommDev->gpuFlush.gpuMr, rCommDev->base.pd, 0, sizeof(int),
                                   (uint64_t)rCommDev->gpuFlush.gpuFlushGpuMem, rCommDev->gpuFlush.dmabuf_fd,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
            ret, cumem_flush_hsa);
          gpuFlushRegistered = true;
          goto flush_reg_done;
        cumem_flush_hsa:
          // The HSA fallback below allocates its own non-VMM buffer, so every cuMem failure
          // here is recoverable and must not leave ret poisoned for the eventual `return ret`.
          ret = ncclSuccess;
          if (rCommDev->gpuFlush.dmabuf_fd >= 0) {
            close(rCommDev->gpuFlush.dmabuf_fd);
            rCommDev->gpuFlush.dmabuf_fd = -1;
          }
        }
#endif
#if defined(__HIP_PLATFORM_AMD__)
        if (!gpuFlushRegistered) {
          // The cuMem attempt above may have left a buffer behind before failing.
          (void)IbMsegFreeGpuFlushMem(&rCommDev->gpuFlush);
#if defined(HIP_UNCACHED_MEMORY)
          const unsigned int gpuFlushFlags = hipDeviceMallocUncached;
#else
          const unsigned int gpuFlushFlags = hipDeviceMallocFinegrained;
#endif
          // Allocate directly through HIP rather than ncclCudaCalloc: with cuMem enabled the
          // latter routes to the VMM allocator, silently dropping these flags and handing the
          // HSA exporter another mapping from the allocator whose export just failed.
          // Proxy-thread HIP ops must use a non-blocking stream: hipMemset on the
          // legacy stream conflicts with ThreadLocal graph capture on the collective.
          hipError_t hipFlushSt =
            hipExtMallocWithFlags((void**)&rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int), gpuFlushFlags);
          if (hipFlushSt != hipSuccess) {
            ret = rcclCudaErrorHandler(hipFlushSt);
            goto fail;
          }
          rCommDev->gpuFlush.gpuFlushMemIsHipAlloc = true;
          cudaStreamCaptureMode capMode = cudaStreamCaptureModeRelaxed;
          bool capModeExchanged = false;
          cudaStream_t zeroStream = nullptr;
          hipFlushSt = cudaThreadExchangeStreamCaptureMode(&capMode);
          if (hipFlushSt == hipSuccess) {
            capModeExchanged = true;
            hipFlushSt = cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking);
          }
          if (hipFlushSt == hipSuccess)
            hipFlushSt = cudaMemsetAsync(rCommDev->gpuFlush.gpuFlushGpuMem, 0, sizeof(int), zeroStream);
          if (hipFlushSt == hipSuccess) hipFlushSt = cudaStreamSynchronize(zeroStream);
          if (zeroStream != nullptr) {
            cudaError_t destroySt = cudaStreamDestroy(zeroStream);
            if (hipFlushSt == hipSuccess) hipFlushSt = destroySt;
          }
          if (capModeExchanged) {
            cudaError_t restoreSt = cudaThreadExchangeStreamCaptureMode(&capMode);
            if (hipFlushSt == hipSuccess) hipFlushSt = restoreSt;
          }
          if (hipFlushSt != hipSuccess) {
            ret = rcclCudaErrorHandler(hipFlushSt);
            goto fail;
          }
          if (useDmaBuf) {
            uint64_t export_offset = 0;
            void* aligned_ptr = NULL;
            size_t aligned_size = 0;
            get_aligned_ptr_and_size(rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int), &aligned_ptr, &aligned_size);
            HSACHECKGOTO(hsa_amd_portable_export_dmabuf(aligned_ptr, aligned_size, &rCommDev->gpuFlush.dmabuf_fd,
                                                        &export_offset),
                         ret, peermem_flush);
            if (wrap_ibv_reg_dmabuf_mr(&rCommDev->gpuFlush.gpuMr, rCommDev->base.pd, export_offset, sizeof(int),
                                       (uint64_t)rCommDev->gpuFlush.gpuFlushGpuMem, rCommDev->gpuFlush.dmabuf_fd,
                                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ) !=
                ncclSuccess) {
              goto peermem_flush;
            }
            gpuFlushRegistered = true;
            goto flush_reg_done;
          peermem_flush:
            // Flush registration is optional; losing it degrades GDR rather than failing accept.
            ret = ncclSuccess;
            if (rCommDev->gpuFlush.dmabuf_fd >= 0) {
              close(rCommDev->gpuFlush.dmabuf_fd);
              rCommDev->gpuFlush.dmabuf_fd = -1;
            }
          } else if (peermemAvailable) {
            // Registered without IBV_ACCESS_RELAXED_ORDERING so the flush write to it is strictly ordered.
            if (wrap_ibv_reg_mr(&rCommDev->gpuFlush.gpuMr, rCommDev->base.pd, rCommDev->gpuFlush.gpuFlushGpuMem,
                                sizeof(int),
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ) ==
                ncclSuccess) {
              gpuFlushRegistered = true;
              std::call_once(gpuFlushRegLogOnce[rCommDev->base.ibDevN], [&] {
                INFO(NCCL_INIT | NCCL_NET, "NET/IB: GDR flush scratchpad registered via peermem on device %d (%s)",
                     rCommDev->base.ibDevN, ibDev->devName);
              });
            }
          }
          if (!gpuFlushRegistered) {
            std::call_once(gpuFlushRegWarnOnce[rCommDev->base.ibDevN], [&] {
              WARN("NET/IB: GDR flush scratchpad registration failed on device %d (%s); flush falls back to "
                   "reading the receive buffer, which does not fence relaxed-ordering writes",
                   rCommDev->base.ibDevN, ibDev->devName);
            });
          }
        }
#endif
      flush_reg_done:
        if (!gpuFlushRegistered) {
          (void)IbMsegFreeGpuFlushMem(&rCommDev->gpuFlush);
          rCommDev->gpuFlush.gpuMr = nullptr;
        }
      }
    }

    // Fill Handle
    meta.devs[i].lid = ibDev->portAttr.lid;
    meta.devs[i].link_layer = rCommDev->base.gidInfo.link_layer;
    meta.devs[i].ib_port = ibDev->portNum;
    meta.devs[i].gid.global.subnet_prefix = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].gid.global.interface_id = rCommDev->base.gidInfo.localGid.global.interface_id;
    meta.devs[i].mtu = ibDev->portAttr.active_mtu;
    meta.devs[i].ibv_dev_index = rCommDev->base.ibDevN;
    meta.devs[i].currSpeed =
      COMPILER_ATOMIC_LOAD(&IbMsegDevs[rCommDev->base.ibDevN].currSpeed, std::memory_order_relaxed);
    rComm->lastSentSpeeds[i] = (uint16_t)(meta.devs[i].currSpeed / 1000);
  }
  // The flag WR carries devs[0]'s keys, so both sides must use a single NIC.
  // Check vProps after subnet-aware routing, which can replace vProps with a
  // fused vNIC while leaving nDataQps at its pre-fusion value.
  if (lComm->gdrFlushFlag && rComm->flushEnabled && rComm->base.vProps.ndevs == 1 && remMeta.ndevs == 1) {
    int nQpsPerRequest = 0;
    NCCLCHECKGOTO(IbMsegCommBaseGetNqpsPerRequest(&rComm->base, &nQpsPerRequest), ret, fail);
    if (nQpsPerRequest <= NCCL_GDR_FLUSH_MAX_WAIT_QPS)
      NCCLCHECKGOTO(IbMsegSetupFlagFlush(rComm, &meta, useDmaBuf), ret, fail);
  }
  meta.addr = (uint64_t)rComm->cmplsRecords;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

  meta.ndevs = rComm->base.vProps.ndevs;
  meta.isP2p = remMeta.isP2p;
  meta.useMultiSeg = rComm->useMultiSeg;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = IbMsegCommStateSend;
  stage->offset = 0;
  if (stage->buffer) {
    free(stage->buffer);
    stage->buffer = NULL;
  }
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(struct IbMsegConnectionMetadata)), ret, fail);
  memcpy(stage->buffer, &meta, sizeof(struct IbMsegConnectionMetadata));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer,
                                   sizeof(struct IbMsegConnectionMetadata), &stage->offset),
                ret, fail);
  if (stage->offset < sizeof(struct IbMsegConnectionMetadata)) return ncclSuccess;

  stage->offset = 0;
  stage->state = IbMsegCommStatePendingReady;

ib_recv_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, &rComm->base.ready, sizeof(int),
                                   &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *recvComm = rComm;
exit:
  /* reset lComm stage */
  if (stage->buffer) free(stage->buffer);
  free(stage);
  lComm->stage = NULL;
  return ret;
fail:
  if (rComm) (void)IbMsegFreeFlagFlush(rComm);
  if (rComm) free(rComm->ctsSlots); // RCCL: combined CTS slots, only when useMultiSeg.
  if (rComm) free(rComm->remCtsFifo.elems); // RCCL: legacy FIFO shadow, only without useMultiSeg.
  (void)ncclSocketClose(&rComm->base.sock);
  free(rComm);
  goto exit;
}

ncclResult_t IbMsegAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  return IbMsegAcceptImpl(listenComm, recvComm, recvDevComm, /*nQpsPerDev*/ 0);
}

ncclResult_t IbMsegCloseSend(void* sendComm) {
  struct IbMsegSendComm* comm = (struct IbMsegSendComm*)sendComm;
  if (comm) {
    for (int q = 0; q < comm->base.nqps; q++)
      IbMsegWqeLatReportQpSummary(&comm->base, comm->base.qps[q].devIndex, &comm->base.qps[q]);

    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++) {
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));
    }

    if (comm->base.resiliency) {
      NCCLCHECK(IbMsegResiliencyClose(comm->base.resiliency));
    }

    if (comm->remoteSpeedMr) {
      NCCLCHECK(wrap_ibv_dereg_mr(comm->remoteSpeedMr));
    }
    NCCLCHECK(rcclIbSendLayoutCacheFree(comm)); // RCCL: read layouts' MRs go before their PDs.

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct IbMsegSendCommDev* commDev = comm->devs + i;
      if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
      if (commDev->putSignalScratchpadMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->putSignalScratchpadMr));
      if (commDev->flagFlushSrcMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->flagFlushSrcMr));
      if (comm->base.resiliency) {
        NCCLCHECK(IbMsegResiliencyDevDestroy(comm->base.resiliency, i));
      }
      NCCLCHECK(IbMsegDestroyBase(&commDev->base));
    }
    if (comm->base.resiliency) {
      NCCLCHECK(IbMsegResiliencyDestroy(&comm->base.resiliency));
    }
    IbMsegSendCommFreeCts(comm); // RCCL: the CTS block is a separate allocation.
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

ncclResult_t IbMsegCloseRecv(void* recvComm) {
  struct IbMsegRecvComm* comm = (struct IbMsegRecvComm*)recvComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++) {
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));
    }

    if (comm->base.resiliency) {
      NCCLCHECK(IbMsegResiliencyClose(comm->base.resiliency));
    }

    NCCLCHECK(IbMsegFreeFlagFlush(comm));
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct IbMsegRecvCommDev* commDev = comm->devs + i;
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.gpuFlushGpuMem != nullptr) {
          NCCLCHECK(IbMsegFreeGpuFlushMem(&commDev->gpuFlush));
          if (commDev->gpuFlush.gpuMr != nullptr) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.gpuMr));
          commDev->gpuFlush.gpuMr = nullptr;
          if (commDev->gpuFlush.dmabuf_fd > 0) {
            close(commDev->gpuFlush.dmabuf_fd);
          }
        }
        if (commDev->gpuFlush.qp.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.hostMr));
      }
      if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
      if (commDev->speedUpdateMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->speedUpdateMr));
      if (comm->base.resiliency) {
        IbMsegResiliencyDevDestroy(comm->base.resiliency, i);
      }
      NCCLCHECK(IbMsegDestroyBase(&commDev->base));
    }
    if (comm->base.resiliency) {
      NCCLCHECK(IbMsegResiliencyDestroy(&comm->base.resiliency));
    }
    free(comm->ctsSlots); // RCCL: combined CTS slots, only when useMultiSeg.
    free(comm->remCtsFifo.elems); // RCCL: legacy FIFO shadow, only without useMultiSeg.
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t IbMsegCloseListen(void* listenComm) {
  struct IbMsegListenComm* comm = (struct IbMsegListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t IbMsegNetP2pPolicy(void* handle, int isP2p) {
  if (!handle) return ncclInvalidArgument;
  struct IbMsegHandle* ibHandle = (struct IbMsegHandle*)handle;
  if (ibHandle->magic != ncclSocketDefaultMagic()) return ncclInvalidArgument;
  ibHandle->isP2p = isP2p;
  return ncclSuccess;
}
