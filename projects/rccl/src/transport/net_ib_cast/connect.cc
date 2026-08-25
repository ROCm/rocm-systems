/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "connect_cast.h"
#include "common_cast.h"
#include "p2p_cast.h"
#include "p2p_resiliency_cast.h"
#include "net_telemetry.h"
#include "qp_sharing.h"

NCCL_PARAM(IbCastGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM(IbCastRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbCastRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM(IbCastTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(IbCastRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbCastPkey, "IB_PKEY", 0);
NCCL_PARAM(IbCastUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbCastGdrFlushDisable, "GDR_FLUSH_DISABLE", 1);
NCCL_PARAM(IbCastSl, "IB_SL", -1);
NCCL_PARAM(IbCastTc, "IB_TC", -1);
NCCL_PARAM(IbCastFifoTc, "IB_FIFO_TC", -1);
NCCL_PARAM(IbCastEceEnable, "IB_ECE_ENABLE", 1);
NCCL_PARAM(IbCastSubnetAwareRouting, "IB_SUBNET_AWARE_ROUTING", 0);
NCCL_PARAM(IbCastSubnetPrefixLen, "IB_SUBNET_PREFIX_LEN", 24);

extern int64_t ncclParamIbCastOooRq();
extern int64_t ncclParamIbCastResiliencyPortFailover();
extern int64_t ncclParamIbCastReceiverSideMatchingScheme();

struct ncclIbDevExtraProps {
  bool oooRq;
};

NCCL_PARAM(IbCastQpsPerConn, "IB_QPS_PER_CONNECTION", 2);
extern int64_t rcclParamIbCastQpsPerP2p();
extern int64_t rcclParamIbCastGdrFlushGpuMemNoRelaxedOrdering();

// Calculate number of QPs based on P2P flag and device counts
static int IbCastCalculateNqps(int isP2p, int localNdevs, int remoteNdevs, const char* funcName) {
  auto qpMultiplier =
    (rcclParamIbCastQpsPerP2p() > 0 && isP2p) ? rcclParamIbCastQpsPerP2p() : ncclParamIbCastQpsPerConn();
  int localNqps = qpMultiplier * localNdevs;
  int remoteNqps = qpMultiplier * remoteNdevs;
  int maxNqps = (remoteNqps > localNqps) ? remoteNqps : localNqps;
  INFO(NCCL_NET, "NET/IB: %s Max Nqps=%d, localNqps=%d, remoteNqps=%d", funcName, maxNqps, localNqps, remoteNqps);
  return maxNqps;
}

#define NCCL_CTS_QP_SLOT_INVALID 0xFF
enum ncclIbChannelType {
  ncclIbChannelTypeCts = 0,
  ncclIbChannelTypeData = 1,
  ncclIbChannelTypeMax = 2
};

struct ncclChannelToUd {
  int channelId;
  bool udId;
  bool udAllocated;
};

static ncclChannelToUd nccl_channel_ud_map[MAX_IB_DEVS][MAXCHANNELS][ncclIbChannelTypeMax];
static bool nccl_channel_last_ud[MAX_IB_DEVS][ncclIbChannelTypeMax];

static inline bool IbCastIsCtsOffloadEnabled(int isP2p) {
  return IbCastOffloadEnabled && !(isP2p && rcclParamIbCastP2pDisableCts());
}

static int IbCastResolveRecvMatchingScheme(bool useCtsOffload) {
  // Order matters here:
  // BY_ORDER -> ctsoffload
  // BY_ID -> failover
  // BY_INDEX -> default or user requested

  if (useCtsOffload) {
    return BY_ORDER;
  }

  if (IbCastQpSharingEnabled()) {
    return BY_ID;
  }

  if (ncclParamIbCastOooRq() || (ncclParamIbCastResiliencyPortFailover() == 1)) {
    return BY_ID;
  }

  int64_t requested = ncclParamIbCastReceiverSideMatchingScheme();
  if (requested == -2 || requested == BY_ORDER) {
    return BY_INDEX;
  }
  return requested;
}

ncclResult_t IbCastInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base, void* cq_context, int64_t cqSize) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = IbCastDevs + ibDevN;
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (0 == ibDev->pdRefs++) {
      NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context));
    }
    base->pd = ibDev->pd;
  }

  if (ibDev->maxCqe > 0 && cqSize > ibDev->maxCqe) {
    WARN("NET/IB: %s: requested CQ size %ld exceeds device %s max_cqe %d, clamping",
         __func__, cqSize, ibDev->devName, ibDev->maxCqe);
    cqSize = ibDev->maxCqe;
  }

  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, (int)cqSize, cq_context, NULL, 0));

  return ncclSuccess;
}

ncclResult_t IbCastDestroyBase(struct ncclIbNetCommDevBase* base) {
  NCCLCHECK(wrap_ibv_destroy_cq(base->cq));

  std::lock_guard<std::mutex> lock(IbCastDevs[base->ibDevN].mutex);
  if (0 == --IbCastDevs[base->ibDevN].pdRefs) {
    NCCLCHECK(wrap_ibv_dealloc_pd(IbCastDevs[base->ibDevN].pd));
  }
  return ncclSuccess;
}

// GID Format
// global:  |              64b  - subnet-prefix                |                 64b - EUI                          |
// raw   :  | 10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix |                 64b - EUI                          |
static uint16_t IbCastExtractLocalSubnetPrefix(uint64_t subnet_prefix) {
  return (be64toh(subnet_prefix) & 0xffff);
}

static int IbCastExtractFlid(union ibv_gid* gid) {
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
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
  ;
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

static ncclResult_t IbCastRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version) {
  char gidRoceVerStr[16] = {0};
  char roceTypePath[PATH_MAX] = {0};
  snprintf(roceTypePath, sizeof(roceTypePath), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName,
           portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    WARN("NET/IB: open failed in IbCastRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    // In containerized environments, read could return EINVAL if the GID index is not mapped to the
    // container sysfs. In this case return ncclSuccess and let the caller move to next GID index.
    if (errno == EINVAL) return ncclSuccess;
    WARN("NET/IB: read failed in IbCastRoceGetVersionNum: %s", strerror(errno));
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
    NCCLCHECK(IbCastRoceGetVersionNum(deviceName, portNum, *gidIndex, &gidRoceVerNum));
    NCCLCHECK(IbCastRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return ncclSuccess;
}

ncclResult_t IbCastGetGidIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                               int* gidIndex) {
  int gidTblLen = portAttr->gid_tbl_len;

  // for IB, choose GID Index that will have routable FLID if present
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    union ibv_gid gid;
    int routableGidIndex = ncclParamIbCastRoutableFlidIbGidIndex();
    if (routableGidIndex < gidTblLen) {
      NCCLCHECK(wrap_ibv_query_gid(context, portNum, routableGidIndex, &gid));
      if (IbCastExtractFlid(&gid) != 0) {
        *gidIndex = routableGidIndex;
        return ncclSuccess;
      }
    }
    *gidIndex = 0;
    return ncclSuccess;
  }

  // for ROCE
  *gidIndex = ncclParamIbCastGidIndex();
  if (*gidIndex >= 0) {
    return ncclSuccess;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = ncclParamIbCastRoceVersionNum();
  int prefixlen;
  void* prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    NCCLCHECK(ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext,
                                 gidIndex));
  }

  return ncclSuccess;
}
ncclResult_t IbCastQpInit(struct ncclIbQp* qp) {
  struct ncclIbQpInitAttr* initAttr = &qp->initAttr;
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = initAttr->state;
  qpAttr.pkey_index = initAttr->pkeyIndex;
  qpAttr.port_num = initAttr->portNum;
  qpAttr.qp_access_flags = initAttr->qpAccessFlags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

static ncclResult_t ncclIbCreateQpMlx5(struct ncclIbQpCreateAttr* createQpAttrs, struct ncclIbQp* qp) {
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
  qpInitAttr.cap.max_inline_data = IbCastUseInline ? sizeof(struct ncclIbSendFifo) * NCCL_NET_IB_MAX_RECVS : 0;

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


static ncclResult_t ncclIbCreateQpIonic(struct ncclIbQpCreateAttr* createQpAttrs, struct ncclIbQp* qp) {
  struct ibv_qp_init_attr qpInitAttr;
  enum ncclIbChannelType channel_type = (createQpAttrs->isDataQp ? ncclIbChannelTypeData : ncclIbChannelTypeCts);
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.qp_context = createQpAttrs->qpContext;
  qpInitAttr.send_cq = createQpAttrs->cq;
  qpInitAttr.recv_cq = createQpAttrs->cq;
  qpInitAttr.qp_type = createQpAttrs->type;
  // Scale WR depths for QP sharing
  if (createQpAttrs->isQpSharingEnabled) {
    qpInitAttr.cap.max_recv_wr = createQpAttrs->maxRecvWorkRequest * createQpAttrs->cqDepthMultiplier;
    qpInitAttr.cap.max_send_wr = createQpAttrs->maxSendWorkRequest * createQpAttrs->cqDepthMultiplier;
  } else {
    qpInitAttr.cap.max_recv_wr = createQpAttrs->maxRecvWorkRequest;
    qpInitAttr.cap.max_send_wr = createQpAttrs->maxSendWorkRequest;
  }
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  if (createQpAttrs->isCtsEnabled) {
    qpInitAttr.cap.max_inline_data = MAX_INLINE_DATA_SIZE;
  } else {
    // for multi-receive scenarios, the inline payload will be a
    // multiple of 32B ncclIbSendFifoCtsInline elements and hence
    // effectively inline handling will be disabled. So limit the
    // max_inline_data to single request size.
    qpInitAttr.cap.max_inline_data = IbCastAinicCtsInlineData ? sizeof(struct ncclIbSendFifoCtsInline) : 0;
  }
  qpInitAttr.sq_sig_all |= (1 << 16);
  if (createQpAttrs->isDataQp) {
    qpInitAttr.sq_sig_all |= (1 << 17);
  } else {
    qpInitAttr.sq_sig_all &= (~(1 << 17));
  }
  qpInitAttr.sq_sig_all |= (1 << 18);
  if (createQpAttrs->isCtsEnabled) {
    qpInitAttr.sq_sig_all |= (1 << 19);
  } else {
    qpInitAttr.sq_sig_all &= (~(1 << 19));
  }

  if (createQpAttrs->isQpSharingEnabled && (createQpAttrs->qpSharingGroupIdx >= 0)) {
    // For Ionic with QP sharing, use groupIdx for UDMA mask selection
    uint8_t mask = (createQpAttrs->qpSharingGroupIdx % 2 == 0) ? IONIC_UDMA_MASK_LOW : IONIC_UDMA_MASK_HIGH;
    wrap_ionicdv_pd_set_udma_mask(createQpAttrs->pd, mask);
  } else {
    if (!nccl_channel_ud_map[createQpAttrs->ibDevN][createQpAttrs->channelId][channel_type].udAllocated) {
      bool lud = nccl_channel_last_ud[createQpAttrs->ibDevN][channel_type];
      nccl_channel_ud_map[createQpAttrs->ibDevN][createQpAttrs->channelId][channel_type].udId = lud;
      nccl_channel_ud_map[createQpAttrs->ibDevN][createQpAttrs->channelId][channel_type].udAllocated = true;
      nccl_channel_last_ud[createQpAttrs->ibDevN][channel_type] =
        !(nccl_channel_last_ud[createQpAttrs->ibDevN][channel_type]);
    }
    if (nccl_channel_ud_map[createQpAttrs->ibDevN][createQpAttrs->channelId][channel_type].udId) {
      wrap_ionicdv_pd_set_udma_mask(createQpAttrs->pd, IONIC_UDMA_MASK_HIGH);
    } else {
      wrap_ionicdv_pd_set_udma_mask(createQpAttrs->pd, IONIC_UDMA_MASK_LOW);
    }
  }

  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, createQpAttrs->pd, &qpInitAttr));
  NCCLCHECK(wrap_ionicdv_qp_set_gda(qp->qp, false, true));
  qp->ctsQpSlot = createQpAttrs->ctsQpSlot;
  return ncclSuccess;
}

// Build base QP creation attributes from comm context. Callers MUST set
// channelId and isDataQp from the saved ncclIbQp fields — these control
// AINIC driver behavior (UDMA load balancing and sq_sig_all feature flags)
// and differ between sender QPs (isDataQp=true) and receiver QPs (isDataQp=false).
void IbCastBuildDataQpCreateAttr(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQpCreateAttr* out) {
  memset(out, 0, sizeof(*out));
  out->type = IBV_QPT_RC;
  out->qpContext = (void*)&base->stats;
  struct ncclIbNetCommDevBase* devBase = IbCastGetNetCommDevBase(base, devIndex);
  out->cq = devBase->cq;
  out->pd = devBase->pd;
  out->ibDevN = devBase->ibDevN;
  out->useIonic = IbCastAinicRoce;
  if (base->isSend) {
    out->maxRecvWorkRequest = 0;
    out->maxSendWorkRequest = 2 * NET_IB_MAX_REQUESTS;
  } else {
    IbCastResiliencyDataRqSizeGet(base->resiliency, devIndex, &out->maxRecvWorkRequest);
    out->maxSendWorkRequest = NET_IB_MAX_REQUESTS;
  }
}

ncclResult_t IbCastQpCreate(struct ncclIbQp* qp, struct ncclIbQpCreateAttr* createQpAttrs) {
  qp->telQpStats = NULL;
  if (createQpAttrs->oooRq) {
    NCCLCHECK(ncclIbCreateQpMlx5(createQpAttrs, qp));
    return ncclSuccess;
  }
  if (createQpAttrs->useIonic && createQpAttrs->type != IBV_QPT_UD) {
    NCCLCHECK(ncclIbCreateQpIonic(createQpAttrs, qp));
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
  qpInitAttr.cap.max_inline_data = IbCastUseInline ? sizeof(struct ncclIbSendFifo) * NCCL_NET_IB_MAX_RECVS : 0;
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, createQpAttrs->pd, &qpInitAttr));
  return ncclSuccess;
}

ncclResult_t IbCastQpRtr(struct ncclIbQp* qp) {
  struct ncclIbQpRtrAttr* rtrAttr = &qp->rtrAttr;
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
    bool sameSubnet = (IbCastExtractLocalSubnetPrefix(rtrAttr->localGid.global.subnet_prefix) ==
                       IbCastExtractLocalSubnetPrefix(rtrAttr->remoteGid.global.subnet_prefix));
    bool needGlobal = !sameSubnet || (rtrAttr->localPortFlags & IBV_QPF_GRH_REQUIRED);
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = rtrAttr->remoteLid;
    if (needGlobal) {
      if (!sameSubnet) {
        uint16_t flid = IbCastExtractFlid(&rtrAttr->remoteGid);
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

ncclResult_t IbCastQpRts(struct ncclIbQp* qp) {
  struct ncclIbQpRtsAttr* rtsAttr = &qp->rtsAttr;
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

ncclResult_t IbCastQpReset(struct ncclIbQp* qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}

ncclResult_t IbCastQpError(struct ncclIbQp* qp) {
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

extern "C" int ncclIbCastTestGidSameSubnet(const uint8_t localGid[16], const uint8_t remoteGid[16], int prefixLen) {
  union ibv_gid l, r;
  memcpy(l.raw, localGid, 16);
  memcpy(r.raw, remoteGid, 16);
  return gidSameSubnet(&l, &r, prefixLen) ? 1 : 0;
}

extern "C" int ncclIbCastTestSubnetMatchesAny(const uint8_t localGid[16], const uint8_t* remoteGids, int nRemote,
                                              int prefixLen) {
  union ibv_gid l;
  memcpy(l.raw, localGid, 16);
  union ibv_gid r[NCCL_IB_MAX_DEVS_PER_NIC];
  if (nRemote < 0 || nRemote > NCCL_IB_MAX_DEVS_PER_NIC) return 0;
  for (int i = 0; i < nRemote; i++) memcpy(r[i].raw, remoteGids + (size_t)i * 16, 16);
  return subnetMatchesAny(&l, r, nRemote, prefixLen) ? 1 : 0;
}

// Given remote GIDs (one per PF on the remote side), find a local merged IB
// device that shares a subnet with any of them. Writes defaultDev to *foundDev
// if no better match is found, preserving existing behavior for single-subnet
// and IB deployments.
// Checks the default device first to preserve NIC Fusion when all PFs in
// the fused device can reach the peer (e.g., 2-node or switch setup).
static ncclResult_t IbCastFindDevBySubnet(union ibv_gid* remoteGids, int nRemoteGids, int defaultDev, int* foundDev) {
  *foundDev = defaultDev;

  int prefixLen = ncclParamIbCastSubnetPrefixLen();
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
  if (defaultDev >= 0 && defaultDev < IbCastNMergedDevs) {
    struct ncclIbMergedDev* mDev = IbCastMergedDevs + defaultDev;
    int checked = 0, matched = 0;
    for (int i = 0; i < mDev->vProps.ndevs; i++) {
      int ibDevN = mDev->vProps.devs[i];
      ncclIbDev* ibDev = IbCastDevs + ibDevN;
      if (ibDev->portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
      int gidIndex = 0;
      union ibv_gid localGid;
      memset(&localGid, 0, sizeof(localGid));
      if (IbCastGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &gidIndex) != ncclSuccess) continue;
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
  for (int devIdx = 0; devIdx < IbCastNMergedDevs; devIdx++) {
    if (devIdx == defaultDev) continue;
    struct ncclIbMergedDev* mDev = IbCastMergedDevs + devIdx;
    int checked = 0, matched = 0;
    for (int i = 0; i < mDev->vProps.ndevs; i++) {
      int ibDevN = mDev->vProps.devs[i];
      ncclIbDev* ibDev = IbCastDevs + ibDevN;
      if (ibDev->portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
      int gidIndex = 0;
      union ibv_gid localGid;
      memset(&localGid, 0, sizeof(localGid));
      if (IbCastGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &gidIndex) != ncclSuccess) continue;
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

ncclResult_t IbCastListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  struct ncclIbHandle* handle = (struct ncclIbHandle*)opaqueHandle;
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  memset(handle, 0, sizeof(struct ncclIbHandle));
  comm->dev = dev;
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &IbCastIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);

  // Embed GIDs of the first 2 RoCE PFs (handle-size limited) in the handle so
  // the connector can find a local NIC on the same subnet as one of our ports.
  // ncclIbHandle is bounded to 128 B, so at most 2 GIDs fit; a vNIC with more
  // than 2 RoCE PFs advertises only the first 2.
  if (ncclParamIbCastSubnetAwareRouting() && dev < IbCastNMergedDevs) {
    struct ncclIbMergedDev* mDev = IbCastMergedDevs + dev;
    int gidSlot = 0, roceDevs = 0;
    for (int i = 0; i < mDev->vProps.ndevs; i++) {
      int ibDevN = mDev->vProps.devs[i];
      ncclIbDev* ibDev = IbCastDevs + ibDevN;
      if (ibDev->portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
      roceDevs++;
      if (gidSlot >= 2) continue;
      int gidIndex;
      NCCLCHECKGOTO(IbCastGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &gidIndex), ret, fail);
      NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, gidIndex, &handle->listenGids[gidSlot]), ret,
                    fail);
      gidSlot++;
    }
    if (roceDevs > 2) {
      WARN("NET/IB: %s: subnet-aware routing device %d has %d RoCE PFs but only the first 2 are advertised "
           "(handle-size limited)",
           __func__, dev, roceDevs);
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

#define NCCL_IB_SL_DEFAULT 0
#define NCCL_IB_TC_DEFAULT 0

// The function creates and initializes QPs (modifies the QPs to INIT) on the
// sender side. Afterwards it populates the metadata structure, provided to the
// function (meta), with the QPs' information. Note that after the QPs'
// creation, the QPs are also queried for ECE support and the metadata structure
// is updated accordingly. The meta data structure is then expected to be
// delivered to the remote side (receiver) as part of the connection
// establishment process.
// depthMult is the sizing already applied to this comm's CQs (IbCastInitCommDevBase),
// clamped to what the device's max_cqe could actually support -- QP send/recv WR
// depth must scale by that same, possibly-reduced, value or the QP can post more
// work than its CQ can record completions for.
static ncclResult_t IbCastSenderQpsCreate(ncclIbSendComm* comm, struct ncclIbConnectionMetadata* meta, int channelId,
                                          int depthMult) {
  uint nqps = comm->base.nqps;
  struct ncclIbQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(struct ncclIbQpCreateAttr));
  qpCreateAttrs.type = IBV_QPT_RC;
  qpCreateAttrs.maxRecvWorkRequest = 0;
  // Send requests are sent using at most 2 messages (RDMA Write and RDMA Write with Immediate)
  qpCreateAttrs.maxSendWorkRequest = 2 * NET_IB_MAX_REQUESTS;
  qpCreateAttrs.isQpSharingEnabled = IbCastQpSharingEnabled();
  qpCreateAttrs.qpSharingGroupIdx = meta->sharedGroupIdx;
  qpCreateAttrs.cqDepthMultiplier = depthMult;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % comm->base.vProps.ndevs;
    ncclIbSendCommDev* commDev = &comm->devs[devIndex];
    ncclIbDev* ibDev = &IbCastDevs[commDev->base.ibDevN];
    ncclIbQp* localQp = &comm->base.qps[qpIndex];
    ncclIbQpInfo* localQpInfo = &meta->qpInfo[qpIndex];

    qpCreateAttrs.cq = commDev->base.cq;
    qpCreateAttrs.pd = commDev->base.pd;
    qpCreateAttrs.qpContext = &comm->base.stats;

    qpCreateAttrs.ctsQpSlot = NCCL_CTS_QP_SLOT_INVALID;
    qpCreateAttrs.isCtsEnabled = comm->useCtsOffload;
    qpCreateAttrs.isDataQp = true;
    qpCreateAttrs.channelId = channelId;
    qpCreateAttrs.ibDevN = commDev->base.ibDevN;
    qpCreateAttrs.useIonic = IbCastAinicRoce;

    if (ibDev->ibProvider == IB_PROVIDER_MLX5 && ncclParamIbCastOooRq()) {
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

    NCCLCHECK(IbCastQpCreate(localQp, &qpCreateAttrs));

    INFO(NCCL_NET,
         "NET/IB: %s: QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p oooRq=%d",
         __func__, ibDev->portNum, commDev->base.ibDevN, IbCastDevs[commDev->base.ibDevN].devName, IbCastNDevs,
         IbCastNMergedDevs, localQp->qp->qp_num, (uint16_t)ncclParamIbCastPkey(), commDev->base.pd,
         qpCreateAttrs.oooRq);
    localQp->devIndex = devIndex;
    localQp->channelId = channelId;
    localQp->isDataQp = qpCreateAttrs.isDataQp;

    // Populate the metadata that will be delivered to the remote peer
    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    // Transition the QP to INIT state
    struct ncclIbQpInitAttr* initAttr = &localQp->initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = ncclParamIbCastPkey();
    initAttr->portNum = ibDev->portNum;
    initAttr->qpAccessFlags = IBV_ACCESS_REMOTE_WRITE;
    NCCLCHECK(IbCastQpInit(localQp));

    if (ncclParamIbCastEceEnable()) {
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
    IbCastResiliencySenderCreateQps(comm->base.resiliency, &meta->resiliencyInfo);
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
static ncclResult_t IbCastSenderQpsToRts(ncclIbSendComm* comm, struct ncclIbConnectionMetadata* remMeta) {
  if (IbCastQpSharingEnabled()) {
    comm->remCommId = remMeta->commId;

    // If secondary shared QP, skip RTR/RTS -- QPs are already in RTS from primary
    if (IbCastCommIsSecondary(&comm->base)) {
      // Just assign remDevIdx from remote metadata
      uint nqps = comm->base.nqps;
      for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
        ncclIbQp* localQp = &comm->base.qps[qpIndex];
        ncclIbQpInfo* remQpInfo = &remMeta->qpInfo[qpIndex];
        localQp->remDevIdx = remQpInfo->devIndex;
      }
      if (comm->base.resiliency) {
        NCCLCHECK(IbCastResiliencySenderQpsToRts(comm->base.resiliency, remMeta));
      }
      return ncclSuccess;
    }
  }

  uint nqps = comm->base.nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    ncclIbQp* localQp = &comm->base.qps[qpIndex];
    ncclIbSendCommDev* commDev = &comm->devs[localQp->devIndex];
    ncclIbDev* ibDev = &IbCastDevs[commDev->base.ibDevN];
    ncclIbQpInfo* remQpInfo = &remMeta->qpInfo[qpIndex];
    ncclIbDevInfo* remDevInfo = &remMeta->devs[remQpInfo->devIndex];

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

    struct ncclIbQpRtrAttr* rtrAttr = &localQp->rtrAttr;
    rtrAttr->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu);
    remDevInfo->mtu = rtrAttr->mtu;
    rtrAttr->linkLayer = remDevInfo->link_layer;
    rtrAttr->tc = remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET ? remMeta->tc : -1;
    rtrAttr->sl = remMeta->sl;
    rtrAttr->remoteQpNum = remQpInfo->qpn;
    rtrAttr->remoteLid = remDevInfo->lid;
    rtrAttr->remoteGid = remDevInfo->gid;
    rtrAttr->localIbPort = ibDev->portNum;
    rtrAttr->localPortFlags = ibDev->portAttr.flags;
    rtrAttr->localGid = commDev->base.gidInfo.localGid;
    rtrAttr->localGidIndex = commDev->base.gidInfo.localGidIndex;
    NCCLCHECK(IbCastQpRtr(localQp));
    struct ncclIbQpRtsAttr* rtsAttr = &localQp->rtsAttr;
    rtsAttr->timeout = ncclParamIbCastTimeout();
    rtsAttr->retryCnt = ncclParamIbCastRetryCnt();
    NCCLCHECK(IbCastQpRts(localQp));
  }

  if (comm->base.resiliency) {
    NCCLCHECK(IbCastResiliencySenderQpsToRts(comm->base.resiliency, remMeta));
  }

  return ncclSuccess;
}

int IbCastGetTrafficClass(void* ctx) {
  ncclNetCommConfig_t* config = (ncclNetCommConfig_t*)ctx;
  if (ctx == NULL) return NCCL_NET_TRAFFIC_CLASS_UNDEF;
  return config->trafficClass;
}
void IbCastSetTrafficClass(void* ctx, int trafficClass) {
  ncclNetCommConfig_t* config = (ncclNetCommConfig_t*)ctx;
  if (config) config->trafficClass = trafficClass;
}

// Determine QP sharing role for sender and set up secondary if applicable.
// On return:
//   *outRole == QP_SHARING_SECONDARY: QPs assigned from pool, skip IbCastSenderQpsCreate
//   *outRole == QP_SHARING_PRIMARY:   proceed with QP creation, then call RegisterPrimary
//   *outRole == QP_SHARING_NONE:      sharing disabled/fallback, proceed normally
static ncclResult_t IbCastQpSharingSenderSetup(
    struct ncclIbSendComm* comm,
    struct ncclIbHandle* handle,
    struct ncclIbConnectionMetadata* meta,
    const ncclNetVDeviceProps_t* remoteVProps,
    int depthMult,
    enum IbCastQpSharingRole* outRole) {

  *outRole = QP_SHARING_NONE;

  if (!IbCastQpSharingEnabled() || handle->isRMA) {
    return ncclSuccess;
  }

  int ngroups = rcclParamIbCastCommNGroups();

  // Allocate commId for this comm
  comm->base.commId = IbCastAllocCommId(comm, true);
  if (comm->base.commId == 0) {
    // Fallback to non-sharing if pool exhausted
    return ncclSuccess;
  }

  // Build probe key from peer address
  IbCastSharedQpKey probeKey;
  memset(&probeKey, 0, sizeof(probeKey));
  memcpy(&probeKey.peerAddr, &handle->connectAddr, sizeof(union ncclSocketAddress));
  IbCastStripPort(&probeKey.peerAddr);

  // TODO - QP sharing
  //        handle for Fusion/Cast where comm->base.vProps.ndevs > 1
  //        handle for Fusion/Cast where remoteVProps->ndevs > 1
  int probeIbDevN = (comm->base.vProps.ndevs > 0) ? comm->base.vProps.devs[0] : 0;
  int probeRemIbDevIdx = remoteVProps->devs[0];

  // Count existing refs to determine groupIdx
  int totalRefs = IbCastCountPeerTotalRefcount(probeIbDevN, &probeKey.peerAddr, probeRemIbDevIdx, true);
  int groupIdx = totalRefs % ngroups;
  comm->base.sharedGroupIdx = groupIdx;
  comm->base.remIbDevIdx = probeRemIbDevIdx;

  probeKey.ibDevN = probeIbDevN;
  probeKey.remIbDevIdx = probeRemIbDevIdx;
  probeKey.isSend = true;
  probeKey.groupIdx = groupIdx;
  probeKey.qpIdx = 0;

  struct IbCastSharedQp* existingSlot = IbCastFindSharedQp(&probeKey);

  if (existingSlot != NULL) {
    // SECONDARY: reuse existing QPs from pool
    INFO(NCCL_NET, "NET/IB: %s: QP sharing SECONDARY sender commId=%u group=%d totalRefs=%d",
         __func__, comm->base.commId, groupIdx, totalRefs);

    comm->base.isSharedQpPrimary = false;
    int primaryNqps = IbCastCountGroupQpSlots(&probeKey.peerAddr, probeRemIbDevIdx, true, groupIdx);
    comm->base.sharedPrimaryNqps = primaryNqps;

    IbCastSharedQpKey key;
    memset(&key, 0, sizeof(key));
    memcpy(&key.peerAddr, &probeKey.peerAddr, sizeof(union ncclSocketAddress));
    key.isSend = true;
    key.groupIdx = groupIdx;

    int nqps = comm->base.nqps;
    for (int q = 0; q < nqps; q++) {
      int mappedQP = q % primaryNqps;
      key.ibDevN = comm->base.vProps.devs[mappedQP % comm->base.vProps.ndevs];
      key.remIbDevIdx = remoteVProps->devs[mappedQP % remoteVProps->ndevs];
      key.qpIdx = mappedQP;

      struct IbCastSharedQp* slot = IbCastFindSharedQp(&key);
      if (slot == NULL) {
        WARN("NET/IB: %s: QP sharing SECONDARY: could not find shared QP for qpIdx=%d group=%d", __func__, q, groupIdx);
        // Fallback: free commId and go non-sharing
        IbCastFreeCommId(comm->base.commId);
        comm->base.commId = 0;
        comm->base.sharedGroupIdx = -1;
        *outRole = QP_SHARING_NONE;
        return ncclSuccess;
      }

      // Copy QP info from shared pool to this comm
      comm->base.qps[q].qp = slot->qp;
      comm->base.qps[q].devIndex = slot->devIndex;
      comm->base.qps[q].ctsQpSlot = slot->ctsQpSlot;
      comm->base.activeQps[q] = &comm->base.qps[q];

      // Populate metadata with shared QP info
      meta->qpInfo[q].qpn = slot->qp->qp_num;
      meta->qpInfo[q].devIndex = slot->devIndex;

      slot->refcount++;
    }

    // Redirect CQs: destroy per-comm CQs and point to primary's CQs
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      NCCLCHECK(wrap_ibv_destroy_cq(comm->devs[i].base.cq));
      comm->devs[i].base.cq = existingSlot->primaryCq;

      // This comm's own PD ref, taken by IbCastInitCommDevBase, is unused: as a
      // SECONDARY it borrows the primary's QPs and CQ. Release it now so the
      // group's PD refcount reflects one owner (the PRIMARY), matching the
      // single release IbCastCleanupGroupCqs performs at the group's last close.
      int secondaryIbDevN = comm->base.vProps.devs[i];
      std::lock_guard<std::mutex> pdLock(IbCastDevs[secondaryIbDevN].mutex);
      if (0 == --IbCastDevs[secondaryIbDevN].pdRefs) {
        NCCLCHECK(wrap_ibv_dealloc_pd(IbCastDevs[secondaryIbDevN].pd));
      }
    }
    existingSlot->cqRefcount++;

    *outRole = QP_SHARING_SECONDARY;
  } else {
    // PRIMARY: create QPs with scaled depth, then register in pool
    INFO(NCCL_NET, "NET/IB: %s: QP sharing PRIMARY sender commId=%u group=%d depthMult=%d",
         __func__, comm->base.commId, groupIdx, depthMult);
    comm->base.isSharedQpPrimary = true;
    *outRole = QP_SHARING_PRIMARY;
  }

  return ncclSuccess;
}

// Register this primary sender's QPs in the shared pool.
// Called after IbCastSenderQpsCreate when role == QP_SHARING_PRIMARY.
static ncclResult_t IbCastQpSharingSenderRegisterPrimary(
    struct ncclIbSendComm* comm,
    struct ncclIbHandle* handle,
    const ncclNetVDeviceProps_t* remoteVProps) {

  union ncclSocketAddress peerAddr;
  memcpy(&peerAddr, &handle->connectAddr, sizeof(union ncclSocketAddress));
  IbCastStripPort(&peerAddr);

  int nqps = comm->base.nqps;
  for (int q = 0; q < nqps; q++) {
    IbCastSharedQpKey key;
    memset(&key, 0, sizeof(key));
    key.peerAddr = peerAddr;
    key.ibDevN = comm->base.vProps.devs[q % comm->base.vProps.ndevs];
    key.remIbDevIdx = remoteVProps->devs[q % remoteVProps->ndevs];
    key.groupIdx = comm->base.sharedGroupIdx;
    key.isSend = true;
    key.qpIdx = q;

    int devIdx = q % comm->base.vProps.ndevs;
    struct IbCastSharedQp* entry = IbCastRegisterSharedQp(&key,
        comm->base.qps[q].qp, comm->devs[devIdx].base.cq,
        comm->devs[devIdx].base.ibDevN, comm->base.qps[q].devIndex, 1);
    if (entry == NULL) {
      WARN("NET/IB: %s: QP sharing PRIMARY sender commId=%u group=%d: shared-QP pool exhausted "
           "registering qpIdx=%d/%d", __func__, comm->base.commId, comm->base.sharedGroupIdx, q, nqps);
      return ncclInternalError;
    }
    if (q == 0) {
      entry->cqRefcount = 1;
    }
    entry->ctsQpSlot = comm->base.qps[q].ctsQpSlot;
  }

  return ncclSuccess;
}

ncclResult_t IbCastConnectImpl(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                               ncclNetDeviceHandle_t** sendDevComm, int envTrafficClass) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbHandle* handle = (struct ncclIbHandle*)opaqueHandle;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  int ready;

  uint8_t link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  int channelId = 0;
  int isP2p = 0;
  *sendComm = NULL;

  // Subnet-aware device selection: use the listener's GIDs (embedded in the
  // handle) to find a local NIC on the same subnet as the remote peer.
  // For single-subnet or IB deployments, all GIDs are zero → dev stays unchanged.
  if (ncclParamIbCastSubnetAwareRouting()) NCCLCHECK(IbCastFindDevBySubnet(handle->listenGids, 2, dev, &dev));

  // The channel id only reaches the transport on the AINIC path; elsewhere the
  // fifth connect() arg carries the device handle, not the context, so all
  // channels share bucket 0 and telemetry reports num_channels as unknown.
  bool telChannelIdKnown = (IbCastAinicRoce && sendDevComm);
  if (telChannelIdKnown) {
    channelId = ((ncclNet_ctxt_t*)sendDevComm)->chId;
  }

  if (stage->state == ncclIbCommStateConnect) goto ib_connect_check;
  if (stage->state == ncclIbCommStateSendDevList) goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecvDevList) goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStateConnecting) goto ib_connect;
  if (stage->state == ncclIbCommStateConnected) goto ib_send_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }
  stage->buffer = NULL;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));
  NCCLCHECKGOTO(IbCastSendCommInit(comm), ret, fail);
  NCCLCHECKGOTO(IbCastStatsInit(&comm->base.stats), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1),
                ret, fail);
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECKGOTO(ncclSocketConnect(&comm->base.sock), ret, fail);

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECKGOTO(ncclSocketReady(&comm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;

  // IB Setup
  struct ncclIbMergedDev* mergedDev;
  if (dev >= IbCastNMergedDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", dev);
    return ncclInternalError;
  }

  mergedDev = IbCastMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  stage->state = ncclIbCommStateSendDevList;
  stage->offset = 0;
  struct ncclIbConnectionMetadata meta;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(meta)), ret, fail);
  memcpy(stage->buffer, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));

  struct ncclIbDevExtraProps exProps;
  exProps.oooRq = true;
  for (int i = 0; i < mergedDev->vProps.ndevs; i++) {
    int ibDevN = mergedDev->vProps.devs[i];
    exProps.oooRq = exProps.oooRq && IbCastDevs[ibDevN].oooRqSize;
  }
  comm->base.localOooRq = exProps.oooRq;
  memcpy((char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), &exProps, sizeof(struct ncclIbDevExtraProps));

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_send_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer,
                               sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;

  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer,
                               sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;
  stage->offset = 0;
  ncclNetVDeviceProps_t remoteVProps;
  int trafficClass;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  memcpy(&exProps, (char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), sizeof(exProps));
  comm->base.remOooRq = exProps.oooRq;

  mergedDev = IbCastMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  // Read isP2p from handle
  isP2p = handle->isP2p;
  comm->useCtsOffload = IbCastIsCtsOffloadEnabled(isP2p) && !handle->isRMA;
  comm->base.recvMatchingScheme = IbCastResolveRecvMatchingScheme(comm->useCtsOffload);

  INFO(NCCL_NET, "NET/IB: IbCastConnect isP2p=%d isRMA=%d useCtsOffload=%d recvMatchingScheme=%d", isP2p, handle->isRMA,
       comm->useCtsOffload, comm->base.recvMatchingScheme);
  comm->base.nqps = IbCastCalculateNqps(isP2p, comm->base.vProps.ndevs, remoteVProps.ndevs, __func__);
  if (handle->isRMA) {
    comm->base.nqps = 1;
  }

  comm->base.nDataQps = std::max(comm->base.vProps.ndevs, remoteVProps.ndevs);

  IbCastCommInitSharingFields(&comm->base);
  comm->remCommId = 0;

  if (comm->base.resiliency) {
    NCCLCHECK(IbCastResiliencyDeviceNumSet(comm->base.resiliency, comm->base.vProps.ndevs, remoteVProps.ndevs));
  }

  int depthMult;
  depthMult = IbCastQpSharingDepthMultiplier();

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  // Sender's CQ size needs to accomodate the upper bound of number of send
  // requests multiplied by the number of QPs used per request.
  int cqSize;
  cqSize = 3 * NET_IB_MAX_REQUESTS * ncclParamIbCastQpsPerConn();
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    int ibDevN = comm->base.vProps.devs[i];
    if (comm->base.resiliency) {
      IbCastResiliencyDataCqSizeGet(comm->base.resiliency, i, &cqSize);
    }
    if (IbCastDevs[ibDevN].maxCqe > 0) {
      cqSize = std::min(IbCastDevs[ibDevN].maxCqe, cqSize);
    }
    NCCLCHECKGOTO(IbCastInitCommDevBase(ibDevN, &comm->devs[i].base, &comm->base.stats, ((int64_t)cqSize * depthMult)), ret, fail);
    if (depthMult > 1) {
      // IbCastInitCommDevBase clamps the CQ to the device's max_cqe; if that clamped
      // us below what depthMult asked for, shrink depthMult to match so QP WR depth
      // and this group's shared-QP capacityUnits never exceed what the CQ can record.
      int achievedMult = comm->devs[i].base.cq->cqe / cqSize;
      if (achievedMult < depthMult) depthMult = std::max(1, achievedMult);
    }
    comm->ar = comm->ar && IbCastDevs[ibDevN].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
    if (comm->base.resiliency) {
      NCCLCHECKGOTO(IbCastResiliencyDevInit(comm->base.resiliency, i, &IbCastDevs[ibDevN]), ret, fail);
    }
  }

  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.vProps.ndevs;
  meta.isP2p = isP2p;
  meta.isRMA = handle->isRMA;
  meta.sharedGroupIdx = -1;
  meta.commId = 0;
  // TODO - QP sharing
  //        handle for Fusion/Cast where comm->base.vProps.ndevs > 1
  meta.senderIbDevIdx = (comm->base.vProps.ndevs > 0) ? comm->base.vProps.devs[0] : -1;

  // QP Sharing: determine role and set up secondary if applicable
  enum IbCastQpSharingRole sharingRole;
  NCCLCHECKGOTO(IbCastQpSharingSenderSetup(comm, handle, &meta, &remoteVProps, depthMult, &sharingRole), ret, fail);

  // Set sharedGroupIdx before IbCastSenderQpsCreate which needs it
  meta.sharedGroupIdx = comm->base.sharedGroupIdx;

  if (sharingRole != QP_SHARING_SECONDARY) {
    NCCLCHECKGOTO(IbCastSenderQpsCreate(comm, &meta, channelId, depthMult), ret, fail);
    comm->telChId = channelId;
    comm->telChStats = NULL;
  
    // Telemetry-only: skip when off. IbCastQpCreate() left every telQpStats NULL,
    // the untracked value the hooks expect.
    if (rcclTelemetryOn()) {
      for (int i = 0; i < comm->base.vProps.ndevs; i++) {
        int ibDevN = comm->base.vProps.devs[i];
        int numQpsForDev = 0;
        for (int q = 0; q < comm->base.nqps; q++)
          if (comm->base.qps[q].devIndex == i) numQpsForDev++;
        int numSlots = 0;
        int startSlot = rcclTelemetrySetupChannel(ibDevN, channelId, numQpsForDev, &numSlots);
        if (!telChannelIdKnown) rcclTelemetryMarkChannelsUnknown(ibDevN);
        int slotOffset = 0;
        for (int q = 0; q < comm->base.nqps && slotOffset < numSlots; q++) {
          if (comm->base.qps[q].devIndex != i) continue;
          // QPs past the granted slots keep telQpStats == NULL (untracked).
          int telSlot = startSlot + slotOffset++;
          rcclTelemetrySetQpRole(ibDevN, channelId, telSlot, comm->base.qps[q].isDataQp);
          comm->base.qps[q].telQpStats = rcclTelemetryResolveQp(ibDevN, channelId, telSlot);
        }
      }
      // Request completions are charged to the comm's first device.
      comm->telChStats = rcclTelemetryResolveChannel(comm->base.vProps.devs[0], channelId);
    }

    if (sharingRole == QP_SHARING_PRIMARY) {
      NCCLCHECKGOTO(IbCastQpSharingSenderRegisterPrimary(comm, handle, &remoteVProps), ret, fail);
    }
  }

  // Populate remaining QP sharing metadata
  meta.commId = comm->base.commId;
  // TODO - QP sharing
  //        handle for Fusion/Cast where comm->base.vProps.ndevs > 1
  meta.senderIbDevIdx = (comm->base.vProps.ndevs > 0) ? comm->base.vProps.devs[0] : -1;

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    ncclIbDev* ibDev = IbCastDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    ncclIbDevInfo* devInfo = meta.devs + i;
    devInfo->ib_port = ibDev->portNum;
    devInfo->mtu = ibDev->portAttr.active_mtu;
    devInfo->lid = ibDev->portAttr.lid;
    devInfo->ibv_dev_index = commDev->base.ibDevN;

    // Prepare GIN Put Signal scratchpad (for RDMA Atomic result)
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->putSignalScratchpadMr, commDev->base.pd, &comm->putSignalScratchpad,
                                  sizeof(comm->putSignalScratchpad), IBV_ACCESS_LOCAL_WRITE),
                  ret, fail);

    // Prepare my CTS FIFO
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->ctsFifoMr, commDev->base.pd, comm->ctsFifo, sizeof(comm->ctsFifo),
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    devInfo->rkey = commDev->ctsFifoMr->rkey;

    // Pack local GID info
    devInfo->link_layer = commDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    NCCLCHECKGOTO(IbCastGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr,
                                    &commDev->base.gidInfo.localGidIndex),
                  ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, commDev->base.gidInfo.localGidIndex,
                                     &commDev->base.gidInfo.localGid),
                  ret, fail);
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
               ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid,
               (uint64_t)devInfo->gid.global.subnet_prefix, IbCastExtractFlid(&devInfo->gid), commDev->ctsFifoMr->rkey,
               commDev->ctsFifoMr->lkey);
        } else { // RoCE
          INFO(
            NCCL_NET,
            "NET/IB: %s: %s %d IbDev %d Port %d qp_num %d mtu %d GID %ld (%lX/%lX) ctsFifoRkey=0x%x ctsFifoLkey=0x%x",
            __func__, comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev, commDev->base.ibDevN,
            ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, (int64_t)commDev->base.gidInfo.localGidIndex,
            (uint64_t)devInfo->gid.global.subnet_prefix, devInfo->gid.global.interface_id, commDev->ctsFifoMr->rkey,
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
           IbCastDevs[ibDev0].devName, IbCastDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }
  trafficClass = IbCastGetTrafficClass(ctx);
  meta.addr = (uint64_t)comm->ctsFifo;
  meta.sl = (ncclParamIbCastSl() != -1)                    ? ncclParamIbCastSl() :
            (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass :
                                                             NCCL_IB_SL_DEFAULT;
  meta.tc = (envTrafficClass != -1)                        ? envTrafficClass :
            (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass :
                                                             NCCL_IB_TC_DEFAULT;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta), &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(meta)) return ncclSuccess;

  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct ncclIbConnectionMetadata remMeta;
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclIbConnectionMetadata),
                                   &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(ncclIbConnectionMetadata));

  // ensure that the remote devices have the same link layer than the local devices used in the connection.
  if (comm->base.vProps.ndevs > 0) {
    int ibDev0 = comm->devs[0].base.ibDevN;
    link_layer = IbCastDevs[ibDev0].portAttr.link_layer;
    for (int i = 0; i < remMeta.ndevs; i++) {
      if (remMeta.devs[i].link_layer != link_layer) {
        WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one "
             "link type using NCCL_IB_HCA",
             NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, IbCastDevs[ibDev0].devName, IbCastDevs[ibDev0].portNum,
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
      NCCLCHECKGOTO(IbCastResiliencyRemoteCompletionRecordsSet(comm->base.resiliency, comm->remCmplsRecords.rkeys[i],
                                                               comm->remCmplsRecords.addr, i),
                    ret, fail);
    }
  }

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->cmplsRecordsMr, comm->devs[i].base.pd, &comm->remCmplsRecords.elems,
                                  sizeof(comm->remCmplsRecords.elems),
                                  IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    comm->devs[i].sge.lkey = comm->devs[i].cmplsRecordsMr->lkey;
  }

  NCCLCHECKGOTO(IbCastSenderQpsToRts(comm, &remMeta), ret, fail);

  comm->base.ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *sendComm = comm;
exit:
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  return ret;
fail:
  free(comm);
  goto exit;
}

ncclResult_t IbCastConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                           ncclNetDeviceHandle_t** sendDevComm) {
  return IbCastConnectImpl(ctx, dev, opaqueHandle, sendComm, sendDevComm, ncclParamIbCastTc());
}

NCCL_PARAM(IbCastWarnRailLocal, "IB_WARN_RAIL_LOCAL", 0);

ncclResult_t IbCastCheckVProps(ncclNetVDeviceProps_t* vProps1, ncclNetVDeviceProps_t* vProps2) {
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

  // In the case that at least one side has a fused NIC but there are no matching physical NICs, we should check if the user wants this
  if (ncclParamIbCastWarnRailLocal() && outVProps.ndevs < maxVProps->ndevs) {
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
// depthMult is the sizing already applied to this comm's CQs (IbCastInitCommDevBase),
// clamped to what the device's max_cqe could actually support -- QP send/recv WR
// depth must scale by that same, possibly-reduced, value or the QP can post more
// work than its CQ can record completions for.
static ncclResult_t IbCastReceiverQpsCreateToRts(ncclIbRecvComm* rComm, struct ncclIbConnectionMetadata* remMeta,
                                                 struct ncclIbConnectionMetadata* meta, int channelId,
                                                 int depthMult) {
  uint nqps = rComm->base.nqps;
  struct ncclIbQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(struct ncclIbQpCreateAttr));
  qpCreateAttrs.type = IBV_QPT_RC;
  qpCreateAttrs.maxRecvWorkRequest = NET_IB_MAX_REQUESTS;
  // CTS messages are posted using send work requests.
  // Note that because only specific CTS messages are signaled, the send queue
  // size needs to be double the number of max requests.
  // When resiliency is enabled, the number of send work requests is as the
  // number of max requests because every CTS message is signaled.
  qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS * (rComm->base.resiliency ? 1 : 2);

  qpCreateAttrs.isQpSharingEnabled = IbCastQpSharingEnabled();
  qpCreateAttrs.qpSharingGroupIdx = remMeta->sharedGroupIdx;
  qpCreateAttrs.cqDepthMultiplier = depthMult;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % rComm->base.vProps.ndevs;
    ncclIbRecvCommDev* rCommDev = &rComm->devs[devIndex];
    ncclIbDev* ibDev = &IbCastDevs[rCommDev->base.ibDevN];
    ncclIbQpInfo* remQpInfo = &remMeta->qpInfo[qpIndex];
    ncclIbQpInfo* localQpInfo = &meta->qpInfo[qpIndex];
    int remDevIndex = remQpInfo->devIndex;
    ncclIbDevInfo* remDevInfo = &remMeta->devs[remDevIndex];
    ncclIbQp* localQp = &rComm->base.qps[qpIndex];

    localQp->remDevIdx = remDevIndex;
    localQp->devIndex = devIndex;

    qpCreateAttrs.cq = rCommDev->base.cq;
    qpCreateAttrs.pd = rCommDev->base.pd;
    qpCreateAttrs.qpContext = &rComm->base.stats;

    qpCreateAttrs.ctsQpSlot = qpIndex;
    qpCreateAttrs.isCtsEnabled = rComm->useCtsOffload;
    qpCreateAttrs.isDataQp = false;
    qpCreateAttrs.channelId = channelId;
    qpCreateAttrs.ibDevN = rCommDev->base.ibDevN;
    qpCreateAttrs.useIonic = IbCastAinicRoce;

    if (rComm->base.resiliency) {
      IbCastResiliencyDataRqSizeGet(rComm->base.resiliency, devIndex, &qpCreateAttrs.maxRecvWorkRequest);
    }
    if (ibDev->ibProvider == IB_PROVIDER_MLX5 && ncclParamIbCastOooRq()) {
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

    NCCLCHECK(IbCastQpCreate(localQp, &qpCreateAttrs));
    localQp->channelId = channelId;
    localQp->isDataQp = qpCreateAttrs.isDataQp;

    INFO(NCCL_NET,
         "NET/IB: %s: QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p oooRq=%d",
         __func__, ibDev->portNum, rCommDev->base.ibDevN, IbCastDevs[rCommDev->base.ibDevN].devName, IbCastNDevs,
         IbCastNMergedDevs, localQp->qp->qp_num, (uint16_t)ncclParamIbCastPkey(), rCommDev->base.pd,
         qpCreateAttrs.oooRq);

    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    // Transition the QP to INIT state
    struct ncclIbQpInitAttr* initAttr = &localQp->initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = ncclParamIbCastPkey();
    initAttr->portNum = ibDev->portNum;
    // Remote Atomic operations are used for GIN! REMOTE_READ is required for GIN Get (RDMA READ).
    initAttr->qpAccessFlags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_READ;
    NCCLCHECK(IbCastQpInit(localQp));

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

    struct ncclIbQpRtrAttr* rtrAttr = &localQp->rtrAttr;
    rtrAttr->mtu = ibDev->portAttr.active_mtu;
    rtrAttr->linkLayer = remDevInfo->link_layer;
    rtrAttr->tc = (remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET && ncclParamIbCastFifoTc() != -1) ?
                    ncclParamIbCastFifoTc() :
                    remMeta->tc;
    rtrAttr->sl = remMeta->sl;
    rtrAttr->remoteQpNum = remQpInfo->qpn;
    rtrAttr->remoteLid = remDevInfo->lid;
    rtrAttr->remoteGid = remDevInfo->gid;
    rtrAttr->localIbPort = ibDev->portNum;
    rtrAttr->localPortFlags = ibDev->portAttr.flags;
    rtrAttr->localGid = rCommDev->base.gidInfo.localGid;
    rtrAttr->localGidIndex = rCommDev->base.gidInfo.localGidIndex;
    NCCLCHECK(IbCastQpRtr(localQp));
    struct ncclIbQpRtsAttr* rtsAttr = &localQp->rtsAttr;
    rtsAttr->timeout = ncclParamIbCastTimeout();
    rtsAttr->retryCnt = ncclParamIbCastRetryCnt();
    NCCLCHECK(IbCastQpRts(localQp));

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

  if (rComm->flushEnabled) {
    for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
      ncclIbRecvCommDev* rCommDev = &rComm->devs[i];
      ncclIbDev* ibDev = &IbCastDevs[rCommDev->base.ibDevN];

      struct ncclIbQpCreateAttr qpCreateAttrs;
      memset(&qpCreateAttrs, 0, sizeof(struct ncclIbQpCreateAttr));
      qpCreateAttrs.type = IBV_QPT_RC;
      qpCreateAttrs.cq = rCommDev->base.cq;
      qpCreateAttrs.pd = rCommDev->base.pd;
      qpCreateAttrs.maxRecvWorkRequest = 0;
      qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS;
      qpCreateAttrs.qpContext = &rComm->base.stats;
      qpCreateAttrs.ctsQpSlot = NCCL_CTS_QP_SLOT_INVALID;
      qpCreateAttrs.isCtsEnabled = rComm->useCtsOffload;
      qpCreateAttrs.isDataQp = true;
      qpCreateAttrs.channelId = channelId;
      qpCreateAttrs.ibDevN = rCommDev->base.ibDevN;
      qpCreateAttrs.useIonic = IbCastAinicRoce;
      IbCastQpCreateAttrInitSharing(&qpCreateAttrs);

      NCCLCHECK(IbCastQpCreate(&rCommDev->gpuFlush.qp, &qpCreateAttrs));
      rCommDev->gpuFlush.qp.channelId = channelId;
      rCommDev->gpuFlush.qp.isDataQp = qpCreateAttrs.isDataQp;

      INFO(NCCL_NET,
           "NET/IB: %s: Flush QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p",
           __func__, ibDev->portNum, rCommDev->base.ibDevN, IbCastDevs[rCommDev->base.ibDevN].devName, IbCastNDevs,
           IbCastNMergedDevs, rCommDev->gpuFlush.qp.qp->qp_num, (uint16_t)ncclParamIbCastPkey(), rCommDev->base.pd);

      ncclIbQp* flushQp = &rCommDev->gpuFlush.qp;

      // Transition the QP to INIT state
      struct ncclIbQpInitAttr* initAttr = &flushQp->initAttr;
      initAttr->state = IBV_QPS_INIT;
      initAttr->pkeyIndex = ncclParamIbCastPkey();
      initAttr->portNum = ibDev->portNum;
      initAttr->qpAccessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
      NCCLCHECK(IbCastQpInit(flushQp));

      struct ncclIbQpRtrAttr* rtrAttr = &flushQp->rtrAttr;
      rtrAttr->mtu = ibDev->portAttr.active_mtu;
      rtrAttr->linkLayer = ibDev->portAttr.link_layer;
      // TODO: Flush QP is a "loopback QP" (connected to itself), so it should
      // not use any information from the remote side during configuration.
      rtrAttr->tc = ibDev->portAttr.link_layer == IBV_LINK_LAYER_ETHERNET ? remMeta->tc : -1;
      rtrAttr->sl = remMeta->sl;
      rtrAttr->remoteQpNum = rCommDev->gpuFlush.qp.qp->qp_num;
      rtrAttr->remoteLid = ibDev->portAttr.lid;
      rtrAttr->remoteGid = rCommDev->base.gidInfo.localGid;
      rtrAttr->localIbPort = ibDev->portNum;
      rtrAttr->localPortFlags = ibDev->portAttr.flags;
      rtrAttr->localGid = rCommDev->base.gidInfo.localGid;
      rtrAttr->localGidIndex = rCommDev->base.gidInfo.localGidIndex;
      NCCLCHECK(IbCastQpRtr(flushQp));
      struct ncclIbQpRtsAttr* rtsAttr = &flushQp->rtsAttr;
      rtsAttr->timeout = ncclParamIbCastTimeout();
      rtsAttr->retryCnt = ncclParamIbCastRetryCnt();
      NCCLCHECK(IbCastQpRts(flushQp));
    }
  }

  if (rComm->base.resiliency) {
    NCCLCHECK(IbCastResiliencyReceiverQpsCreateToRts(rComm->base.resiliency, remMeta, &meta->resiliencyInfo));
  }

  return ncclSuccess;
}

ncclResult_t IbCastPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm* recvComm, ncclIbQp* dataQp) {
  uint32_t nRecvWorkRequestsPerQp = NET_IB_MAX_REQUESTS;
  if (recvComm->base.resiliency) {
    IbCastResiliencyDataRqSizeGet(recvComm->base.resiliency, dataQp->devIndex, &nRecvWorkRequestsPerQp);
  }
  INFO(NCCL_NET, "NET/IB: %s: Pre-posting %d Receive WQEs on QP (qp_num=%d, comm=%p)", __func__, nRecvWorkRequestsPerQp,
       dataQp->qp->qp_num, recvComm);
  for (int j = 0; j < nRecvWorkRequestsPerQp; j++) {
    NCCLCHECK(IbCastPostRecvWorkRequest(dataQp->qp, &recvComm->ibRecvWorkRequest));
  }
  return ncclSuccess;
}

ncclResult_t IbCastReceiverPrePostReceiveWorkRequests(struct ncclIbRecvComm* recvComm) {
  int nqps = recvComm->base.nqps;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(IbCastPostReceiveWorkRequestsOnQp(recvComm, &recvComm->base.qps[i]));
  }
  return ncclSuccess;
}

// Determine QP sharing role for receiver and set up secondary if applicable.
// On return:
//   *outRole == QP_SHARING_SECONDARY: QPs assigned from pool, skip IbCastReceiverQpsCreateToRts
//   *outRole == QP_SHARING_PRIMARY:   proceed with QP creation, then call RegisterPrimary
//   *outRole == QP_SHARING_NONE:      sharing disabled/fallback, proceed normally
static ncclResult_t IbCastQpSharingReceiverSetup(
    struct ncclIbRecvComm* rComm,
    const struct ncclIbConnectionMetadata* remMeta,
    struct ncclIbConnectionMetadata* meta,
    enum IbCastQpSharingRole* outRole) {

  *outRole = QP_SHARING_NONE;

  if (!IbCastQpSharingEnabled() || remMeta->isRMA || remMeta->sharedGroupIdx < 0) {
    return ncclSuccess;
  }

  int recvGroupIdx = remMeta->sharedGroupIdx;

  // Allocate commId for this comm
  rComm->base.commId = IbCastAllocCommId(rComm, false);
  if (rComm->base.commId == 0) {
    // Fallback to non-sharing if pool exhausted
    return ncclSuccess;
  }

  // Build probe key from peer address
  union ncclSocketAddress recvPeerAddr;
  ncclSocketGetAddr(&rComm->base.sock, &recvPeerAddr);
  IbCastStripPort(&recvPeerAddr);

  rComm->base.sharedGroupIdx = recvGroupIdx;
  rComm->base.remIbDevIdx = remMeta->senderIbDevIdx;
  int recvProbeIbDevN = (rComm->base.vProps.ndevs > 0) ? rComm->base.vProps.devs[0] : 0;

  IbCastSharedQpKey recvProbeKey;
  memset(&recvProbeKey, 0, sizeof(recvProbeKey));
  recvProbeKey.ibDevN = recvProbeIbDevN;
  recvProbeKey.peerAddr = recvPeerAddr;
  recvProbeKey.remIbDevIdx = remMeta->senderIbDevIdx;
  recvProbeKey.isSend = false;
  recvProbeKey.groupIdx = recvGroupIdx;
  recvProbeKey.qpIdx = 0;

  struct IbCastSharedQp* recvExistingSlot = IbCastFindSharedQp(&recvProbeKey);

  if (recvExistingSlot != NULL) {
    // SECONDARY receiver: reuse existing QPs
    INFO(NCCL_NET, "NET/IB: %s: QP sharing SECONDARY receiver commId=%u group=%d",
         __func__, rComm->base.commId, recvGroupIdx);

    rComm->base.isSharedQpPrimary = false;
    int primaryNqps = IbCastCountGroupQpSlots(&recvPeerAddr, remMeta->senderIbDevIdx, false, remMeta->sharedGroupIdx);
    rComm->base.sharedPrimaryNqps = primaryNqps;
    rComm->useCtsOffload = false;

    IbCastSharedQpKey recvKey;
    memset(&recvKey, 0, sizeof(recvKey));
    recvKey.peerAddr = recvPeerAddr;
    recvKey.isSend = false;
    recvKey.groupIdx = recvGroupIdx;

    int nqps = rComm->base.nqps;
    for (int q = 0; q < nqps; q++) {
      int mappedQ = q % primaryNqps;
      int localDevIdx = mappedQ % rComm->base.vProps.ndevs;
      recvKey.ibDevN = rComm->base.vProps.devs[localDevIdx];
      recvKey.remIbDevIdx = remMeta->senderIbDevIdx;
      recvKey.qpIdx = mappedQ;

      struct IbCastSharedQp* recvSlot = IbCastFindSharedQp(&recvKey);
      if (recvSlot == NULL) {
        WARN("NET/IB: %s: QP sharing SECONDARY recv: could not find shared QP for qpIdx=%d group=%d", __func__, q, recvGroupIdx);
        IbCastFreeCommId(rComm->base.commId);
        rComm->base.commId = 0;
        rComm->base.sharedGroupIdx = -1;
        *outRole = QP_SHARING_NONE;
        return ncclSuccess;
      }

      rComm->base.qps[q].qp = recvSlot->qp;
      rComm->base.qps[q].devIndex = recvSlot->devIndex;
      rComm->base.qps[q].ctsQpSlot = recvSlot->ctsQpSlot;
      // remDevIdx is normally set by IbCastReceiverQpsCreateToRts, which is
      // skipped for secondary comms; set it here or CTS rkey selection is wrong.
      rComm->base.qps[q].remDevIdx = remMeta->qpInfo[q].devIndex;
      rComm->base.activeQps[q] = &rComm->base.qps[q];

      meta->qpInfo[q].qpn = recvSlot->qp->qp_num;
      meta->qpInfo[q].devIndex = recvSlot->devIndex;

      recvSlot->refcount++;
    }

    // Redirect CQs: destroy per-comm CQs and point to primary's CQs
    for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
      NCCLCHECK(wrap_ibv_destroy_cq(rComm->devs[i].base.cq));
      rComm->devs[i].base.cq = recvExistingSlot->primaryCq;

      // This comm's own PD ref, taken by IbCastInitCommDevBase, is unused: as a
      // SECONDARY it borrows the primary's QPs and CQ. Release it now so the
      // group's PD refcount reflects one owner (the PRIMARY), matching the
      // single release IbCastCleanupGroupCqs performs at the group's last close.
      int secondaryIbDevN = rComm->base.vProps.devs[i];
      std::lock_guard<std::mutex> pdLock(IbCastDevs[secondaryIbDevN].mutex);
      if (0 == --IbCastDevs[secondaryIbDevN].pdRefs) {
        NCCLCHECK(wrap_ibv_dealloc_pd(IbCastDevs[secondaryIbDevN].pd));
      }
    }
    recvExistingSlot->cqRefcount++;

    // Share flush QPs from primary
    if (rComm->flushEnabled) {
      for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
        IbCastSharedQpKey flushKey;
        memset(&flushKey, 0, sizeof(flushKey));
        flushKey.ibDevN = rComm->base.vProps.devs[i];
        flushKey.peerAddr = recvPeerAddr;
        flushKey.remIbDevIdx = remMeta->senderIbDevIdx;
        flushKey.isSend = false;
        flushKey.groupIdx = recvGroupIdx;
        flushKey.qpIdx = IBCAST_FLUSH_QP_IDX;

        struct IbCastSharedQp* flushSlot = IbCastFindSharedQp(&flushKey);
        if (flushSlot) {
          rComm->devs[i].gpuFlush.qp.qp = flushSlot->qp;
          flushSlot->refcount++;
          INFO(NCCL_NET, "NET/IB: %s: SECONDARY recv sharing flush QP qpn=%u dev=%d group=%d refcount=%d commId=%u",
               __func__, flushSlot->qp->qp_num, i, recvGroupIdx, flushSlot->refcount, rComm->base.commId);
        } else {
          WARN("NET/IB: %s: SECONDARY recv could not find shared flush QP for dev=%d group=%d commId=%u",
               __func__, i, recvGroupIdx, rComm->base.commId);
        }
      }
    }

    *outRole = QP_SHARING_SECONDARY;
  } else {
    // PRIMARY receiver: create QPs with scaled depth, register in pool
    INFO(NCCL_NET, "NET/IB: %s: QP sharing PRIMARY receiver commId=%u group=%d",
         __func__, rComm->base.commId, recvGroupIdx);
    rComm->base.isSharedQpPrimary = true;
    rComm->useCtsOffload = false; // sender useCtsOffload is also false: IbCastOffloadEnabled=false at init (init.cc)
    *outRole = QP_SHARING_PRIMARY;
  }

  return ncclSuccess;
}

// Register this primary receiver's QPs (and flush QPs) in the shared pool.
// Called after IbCastReceiverQpsCreateToRts when role == QP_SHARING_PRIMARY.
static ncclResult_t IbCastQpSharingReceiverRegisterPrimary(
    struct ncclIbRecvComm* rComm,
    const struct ncclIbConnectionMetadata* remMeta) {

  union ncclSocketAddress recvPeerAddr;
  ncclSocketGetAddr(&rComm->base.sock, &recvPeerAddr);
  IbCastStripPort(&recvPeerAddr);

  IbCastSharedQpKey recvKey;
  memset(&recvKey, 0, sizeof(recvKey));
  recvKey.peerAddr = recvPeerAddr;
  recvKey.remIbDevIdx = remMeta->senderIbDevIdx;
  recvKey.isSend = false;
  recvKey.groupIdx = rComm->base.sharedGroupIdx;

  int nqps = rComm->base.nqps;
  for (int q = 0; q < nqps; q++) {
    recvKey.ibDevN = rComm->base.vProps.devs[q % rComm->base.vProps.ndevs];
    recvKey.qpIdx = q;

    int devIdx = q % rComm->base.vProps.ndevs;
    struct IbCastSharedQp* entry = IbCastRegisterSharedQp(&recvKey,
        rComm->base.qps[q].qp, rComm->devs[devIdx].base.cq,
        rComm->devs[devIdx].base.ibDevN, rComm->base.qps[q].devIndex, 1);
    if (entry == NULL) {
      WARN("NET/IB: %s: QP sharing PRIMARY receiver commId=%u group=%d: shared-QP pool exhausted "
           "registering qpIdx=%d/%d", __func__, rComm->base.commId, rComm->base.sharedGroupIdx, q, nqps);
      return ncclInternalError;
    }
    entry->ctsQpSlot = rComm->base.qps[q].ctsQpSlot;
    if (q == 0) {
      entry->cqRefcount = 1;
    }
  }

  // Register flush QPs in shared pool (primary only)
  if (rComm->flushEnabled) {
    for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
      IbCastSharedQpKey flushKey;
      memset(&flushKey, 0, sizeof(flushKey));
      flushKey.ibDevN = rComm->base.vProps.devs[i];
      flushKey.peerAddr = recvPeerAddr;
      flushKey.remIbDevIdx = remMeta->senderIbDevIdx;
      flushKey.isSend = false;
      flushKey.groupIdx = rComm->base.sharedGroupIdx;
      flushKey.qpIdx = IBCAST_FLUSH_QP_IDX;

      struct IbCastSharedQp* flushEntry = IbCastRegisterSharedQp(&flushKey, rComm->devs[i].gpuFlush.qp.qp,
          rComm->devs[i].base.cq, rComm->devs[i].base.ibDevN, i, 1);
      if (flushEntry == NULL) {
        WARN("NET/IB: %s: QP sharing PRIMARY recv: shared-QP pool exhausted registering flush QP "
             "dev=%d group=%d commId=%u, secondaries will not be able to share it",
             __func__, i, rComm->base.sharedGroupIdx, rComm->base.commId);
        continue;
      }
      INFO(NCCL_NET, "NET/IB: %s: PRIMARY recv registered flush QP qpn=%u dev=%d group=%d commId=%u",
           __func__, rComm->devs[i].gpuFlush.qp.qp->qp_num, i,
           rComm->base.sharedGroupIdx, rComm->base.commId);
    }
  }

  return ncclSuccess;
}

ncclResult_t IbCastAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = lComm->stage;
  if (stage == NULL) {
    NCCLCHECK(ncclCalloc(&lComm->stage, 1));
    stage = lComm->stage;
  }
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  int channelId = 0;
  bool useDmaBuf = false;
  *recvComm = NULL;

  // See IbCastConnect(): the channel id only reaches the transport on AINIC.
  bool telChannelIdKnown = (IbCastAinicRoce && recvDevComm);
  if (telChannelIdKnown) {
    channelId = ((ncclNet_ctxt_t*)recvDevComm)->chId;
  }

  if (stage->state == ncclIbCommStateAccept) goto ib_accept_check;
  if (stage->state == ncclIbCommStateRecvDevList) goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSendDevList) goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecv) goto ib_recv;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStatePendingReady) goto ib_recv_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));
  NCCLCHECKGOTO(IbCastRecvCommInit(rComm), ret, fail);
  NCCLCHECKGOTO(IbCastStatsInit(&rComm->base.stats), ret, fail);
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  NCCLCHECKGOTO(ncclSocketInit(&rComm->base.sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketAccept(&rComm->base.sock, &lComm->sock), ret, fail);

  // Alloc stage->buffer here to be used for all following steps
  struct ncclIbConnectionMetadata remMeta;
  struct ncclIbDevExtraProps exProps;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)));

ib_accept_check:
  NCCLCHECKGOTO(ncclSocketReady(&rComm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;
  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer,
                               sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  if (lComm->dev >= IbCastNMergedDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", lComm->dev);
    return ncclInternalError;
  }

  memcpy(&exProps, (char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), sizeof(exProps));
  rComm->base.remOooRq = exProps.oooRq;

  // Reduce the physical device list and store in the connection base
  struct ncclIbMergedDev* mergedDev;
  mergedDev = IbCastMergedDevs + lComm->dev;
  NCCLCHECK(IbCastCheckVProps(&mergedDev->vProps, &remoteVProps));
  rComm->base.vProps = mergedDev->vProps;
  memcpy(stage->buffer, &rComm->base.vProps, sizeof(ncclNetVDeviceProps_t));
  if (rComm->base.resiliency) {
    NCCLCHECK(IbCastResiliencyDeviceNumSet(rComm->base.resiliency, rComm->base.vProps.ndevs, remoteVProps.ndevs));
  }

  stage->offset = 0;
  stage->state = ncclIbCommStateSendDevList;

  exProps.oooRq = true;
  for (int i = 0; i < mergedDev->vProps.ndevs; i++) {
    int ibDevN = mergedDev->vProps.devs[i];
    exProps.oooRq = exProps.oooRq && IbCastDevs[ibDevN].oooRqSize;
  }
  rComm->base.localOooRq = exProps.oooRq;
  memcpy((char*)stage->buffer + sizeof(ncclNetVDeviceProps_t), &exProps, sizeof(struct ncclIbDevExtraProps));

ib_send_dev_list:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer,
                                   sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset),
                ret, fail);
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStateRecv;

ib_recv:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta), &stage->offset),
                ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct ncclIbConnectionMetadata));

  rComm->useCtsOffload = IbCastIsCtsOffloadEnabled(remMeta.isP2p) && !remMeta.isRMA;
  rComm->base.recvMatchingScheme = IbCastResolveRecvMatchingScheme(rComm->useCtsOffload);
  INFO(NCCL_NET, "NET/IB: ncclIbAccept isP2p=%d isRMA=%d useCtsOffload=%d (IbP2pDisableCts=%ld) recvMatchingScheme=%d",
       remMeta.isP2p, remMeta.isRMA, rComm->useCtsOffload, rcclParamIbCastP2pDisableCts(),
       rComm->base.recvMatchingScheme);
  rComm->base.nqps = IbCastCalculateNqps(remMeta.isP2p, rComm->base.vProps.ndevs, remMeta.ndevs, __func__);
  if (remMeta.isRMA) {
    rComm->base.nqps = 1;
  }
  rComm->base.nDataQps = std::max(rComm->base.vProps.ndevs, remMeta.ndevs);

  // Subnet-aware device selection: use the remote sender's GIDs to find a local
  // NIC on the same subnet. Override lComm->dev and update vProps if a
  // better device is found.
  if (ncclParamIbCastSubnetAwareRouting() && remMeta.ndevs > 0) {
    union ibv_gid remoteGids[NCCL_IB_MAX_DEVS_PER_NIC];
    int nRemoteGids = 0;
    for (int i = 0; i < remMeta.ndevs && i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      if (remMeta.devs[i].link_layer == IBV_LINK_LAYER_ETHERNET) {
        remoteGids[nRemoteGids++] = remMeta.devs[i].gid;
      }
    }
    int effectiveDev = lComm->dev;
    NCCLCHECKGOTO(IbCastFindDevBySubnet(remoteGids, nRemoteGids, lComm->dev, &effectiveDev), ret, fail);
    if (effectiveDev != lComm->dev) {
      lComm->dev = effectiveDev;
      rComm->base.vProps = IbCastMergedDevs[effectiveDev].vProps;
    }
  }

  IbCastCommInitSharingFields(&rComm->base);

  // IB setup
  struct ncclIbDev* ibDev;
  int ibDevN;
  struct ncclIbRecvCommDev* rCommDev;

  mergedDev = IbCastMergedDevs + lComm->dev;

  if (remMeta.ndevs != rComm->base.vProps.ndevs) {
    INFO(NCCL_NET, "NET/IB : Local mergedDev %s has a different number of devices=%d as remote %s %d",
         mergedDev->devName, rComm->base.vProps.ndevs, remMeta.devName, remMeta.ndevs);
  }

  // Metadata to send back to requestor (sender)
  struct ncclIbConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));

  int recvDepthMult;
  recvDepthMult = IbCastQpSharingDepthMultiplier();

  // Receiver's CQ size needs to accomodate receive requests that can generate
  // up to 2 completions (one for the CTS message and one for the completion
  // of a receive request) per QP, in the worst case.
  int cqSize;
  cqSize = 3 * NET_IB_MAX_REQUESTS * ncclParamIbCastQpsPerConn();
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rComm->base.vProps.devs[i];
    if (rComm->base.resiliency) {
      IbCastResiliencyDataCqSizeGet(rComm->base.resiliency, i, &cqSize);
    }
    if (IbCastDevs[ibDevN].maxCqe > 0) {
      cqSize = std::min(IbCastDevs[ibDevN].maxCqe, cqSize);
    }
    NCCLCHECKGOTO(IbCastInitCommDevBase(ibDevN, &rCommDev->base, &rComm->base.stats, ((int64_t)cqSize * recvDepthMult)), ret, fail);
    if (recvDepthMult > 1) {
      // IbCastInitCommDevBase clamps the CQ to the device's max_cqe; if that clamped
      // us below what recvDepthMult asked for, shrink it to match so QP WR depth and
      // this group's shared-QP capacityUnits never exceed what the CQ can record.
      int achievedMult = rCommDev->base.cq->cqe / cqSize;
      if (achievedMult < recvDepthMult) recvDepthMult = std::max(1, achievedMult);
    }
    if (rComm->base.resiliency) {
      NCCLCHECKGOTO(IbCastResiliencyDevInit(rComm->base.resiliency, i, &IbCastDevs[ibDevN]), ret, fail);
    }
    ibDev = IbCastDevs + ibDevN;
    NCCLCHECKGOTO(IbCastGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr,
                                    &rCommDev->base.gidInfo.localGidIndex),
                  ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, rCommDev->base.gidInfo.localGidIndex,
                                     &rCommDev->base.gidInfo.localGid),
                  ret, fail);
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = ibDev->portAttr.link_layer;
    if (link_layer != ibDev->portAttr.link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of "
           "only one link type using NCCL_IB_HCA",
           ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0,
           IbCastDevs[ibDev0].devName, IbCastDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
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
           NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, IbCastDevs[ibDev0].devName, IbCastDevs[ibDev0].portNum,
           NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Store the number of remote devices provided by the remote peer
  rComm->base.nRemDevs = remMeta.ndevs;

  // Store the remote GID information per-device provided by the remote peer
  for (int i = 0; i < rComm->base.nRemDevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id = rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].gid.global.subnet_prefix;
  }

  // Determine if Flush is enabled for this Comm. Must be done before creating
  // QPs. If Flush is enabled, extra QPs will be created for Flush operations.
  useDmaBuf = (IbCastDmaBufSupport(lComm->dev) == ncclSuccess && ncclParamDmaBufEnable());
  rComm->flushEnabled = (((IbCastGdrSupport() == ncclSuccess || useDmaBuf) && (!IbCastOffloadEnabled) &&
                          (ncclParamIbCastGdrFlushDisable() == 0)) ||
                         remMeta.isRMA) ?
                          1 :
                          0;

  // QP Sharing: determine role and set up secondary if applicable
  enum IbCastQpSharingRole sharingRole;
  NCCLCHECKGOTO(IbCastQpSharingReceiverSetup(rComm, &remMeta, &meta, &sharingRole), ret, fail);

  if (sharingRole != QP_SHARING_SECONDARY) {
    NCCLCHECKGOTO(IbCastReceiverQpsCreateToRts(rComm, &remMeta, &meta, channelId, recvDepthMult), ret, fail);
    if (sharingRole == QP_SHARING_PRIMARY) {
      NCCLCHECKGOTO(IbCastQpSharingReceiverRegisterPrimary(rComm, &remMeta), ret, fail);
    }
  }

  // Populate QP sharing metadata in response
  meta.sharedGroupIdx = rComm->base.sharedGroupIdx;
  meta.commId = rComm->base.commId;

  if (rComm->prepostReceiveWorkRequests) {
    NCCLCHECKGOTO(IbCastReceiverPrePostReceiveWorkRequests(rComm), ret, fail);
  }

  rComm->telChId = channelId;
  rComm->telChStats = NULL;

  // See IbCastConnect(): telemetry-only work, skipped when telemetry is off.
  if (rcclTelemetryOn()) {
    for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
      int telIbDevN = rComm->base.vProps.devs[i];
      int numQpsForDev = 0;
      for (int q = 0; q < rComm->base.nqps; q++)
        if (rComm->base.qps[q].devIndex == i) numQpsForDev++;
      int numSlots = 0;
      int startSlot = rcclTelemetrySetupChannel(telIbDevN, channelId, numQpsForDev, &numSlots);
      if (!telChannelIdKnown) rcclTelemetryMarkChannelsUnknown(telIbDevN);
      int slotOffset = 0;
      for (int q = 0; q < rComm->base.nqps && slotOffset < numSlots; q++) {
        if (rComm->base.qps[q].devIndex != i) continue;
        // QPs past the granted slots keep telQpStats == NULL (untracked).
        int telSlot = startSlot + slotOffset++;
        // isDataQp is always false on receiver QPs; classify by CTS role.
        rcclTelemetrySetQpRole(telIbDevN, channelId, telSlot, !IbCastRecvCommIsCtsQp(rComm, q));
        rComm->base.qps[q].telQpStats = rcclTelemetryResolveQp(telIbDevN, channelId, telSlot);
      }
    }
    rComm->telChStats = rcclTelemetryResolveChannel(rComm->base.vProps.devs[0], channelId);
  }

  // Store the remote CTS FIFO info provided by the remote peer
  rComm->remCtsFifo.addr = remMeta.addr;
  for (int i = 0; i < rComm->base.nRemDevs; i++) {
    rComm->remCtsFifo.rkeys[i] = remMeta.devs[i].rkey;
  }

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;

    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->ctsFifoMr, rCommDev->base.pd, &rComm->remCtsFifo.elems,
                                  sizeof(rComm->remCtsFifo.elems),
                                  IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    rCommDev->sge.lkey = rCommDev->ctsFifoMr->lkey;

    // Register completion records
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->cmplsRecordsMr, rCommDev->base.pd, &rComm->cmplsRecords,
                                  sizeof(rComm->cmplsRecords),
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    meta.devs[i].rkey = rCommDev->cmplsRecordsMr->rkey;
  }
  if (IbCastUseInline) rComm->remCtsFifo.flags = IBV_SEND_INLINE;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = IbCastDevs + rCommDev->base.ibDevN;

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
      if (rcclParamIbCastGdrFlushGpuMemNoRelaxedOrdering()) {
#if defined(HIP_UNCACHED_MEMORY)
        const unsigned int gpuFlushFlags = hipDeviceMallocUncached;
#else
        const unsigned int gpuFlushFlags = hipDeviceMallocFinegrained;
#endif
        // RCCL: allocate the GDR flush buffer directly via HIP (never cuMem/VMM)
        // so hsa_amd_portable_export_dmabuf can export it. cuMem/VMM allocations
        // fail to export through the HSA portable exporter on some ROCm/NIC stacks.
        CUDACHECKGOTO(hipExtMallocWithFlags((void**)&rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int), gpuFlushFlags),
                      ret, fail);
        CUDACHECKGOTO(hipMemset(rCommDev->gpuFlush.gpuFlushGpuMem, 0, sizeof(int)), ret, fail);

        if (useDmaBuf) {
          uint64_t exportOffset = 0;
          void* aligned_ptr = NULL;
          size_t alignedSize = 0;
          get_aligned_ptr_and_size(rCommDev->gpuFlush.gpuFlushGpuMem, sizeof(int) /*devicebuffersize*/, &aligned_ptr,
                                   &alignedSize);
          hsa_status_t exportStatus =
            pfn_hsa_amd_portable_export_dmabuf(aligned_ptr, alignedSize, &rCommDev->gpuFlush.dmabufFd, &exportOffset);
          if (rCommDev->gpuFlush.dmabufFd < 0 || exportStatus != HSA_STATUS_SUCCESS) {
            WARN("Failed to export DMA BUF");
            goto fail;
          }
          NCCLCHECKGOTO(
            wrap_ibv_reg_dmabuf_mr(&rCommDev->gpuFlush.gpuMr, rCommDev->base.pd, exportOffset, sizeof(int),
                                   (uint64_t)rCommDev->gpuFlush.gpuFlushGpuMem /*iova*/, rCommDev->gpuFlush.dmabufFd,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
            ret, fail);
        } else {
          rCommDev->gpuFlush.dmabufFd = -1;
          NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->gpuFlush.gpuMr, rCommDev->base.pd, rCommDev->gpuFlush.gpuFlushGpuMem,
                                        sizeof(int),
                                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ),
                        ret, fail);
        }
      } else {
        rCommDev->gpuFlush.gpuFlushGpuMem = nullptr;
        rCommDev->gpuFlush.gpuMr = nullptr;
        rCommDev->gpuFlush.dmabufFd = -1;
      }
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->gpuFlush.hostMr, rCommDev->base.pd, &rComm->gpuFlushHostMem, sizeof(int),
                                    IBV_ACCESS_LOCAL_WRITE),
                    ret, fail);
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
    }

    // Fill Handle
    meta.devs[i].lid = ibDev->portAttr.lid;
    meta.devs[i].link_layer = rCommDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    meta.devs[i].ib_port = ibDev->portNum;
    meta.devs[i].gid.global.subnet_prefix = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].gid.global.interface_id = rCommDev->base.gidInfo.localGid.global.interface_id;
    meta.devs[i].mtu = ibDev->portAttr.active_mtu;
    meta.devs[i].ibv_dev_index = rCommDev->base.ibDevN;
  }
  meta.addr = (uint64_t)rComm->cmplsRecords;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

  meta.ndevs = rComm->base.vProps.ndevs;
  meta.isP2p = remMeta.isP2p;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) {
    free(stage->buffer);
    stage->buffer = NULL;
  }
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)), ret, fail);
  memcpy(stage->buffer, &meta, sizeof(struct ncclIbConnectionMetadata));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer,
                                   sizeof(struct ncclIbConnectionMetadata), &stage->offset),
                ret, fail);
  if (stage->offset < sizeof(struct ncclIbConnectionMetadata)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStatePendingReady;

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
  free(rComm);
  goto exit;
}

// Per-device MR cleanup for sender comms (shared between shared/non-shared paths)
static ncclResult_t IbCastCloseSendCommDevResources(struct ncclIbSendComm* comm, int devIndex) {
  struct ncclIbSendCommDev* commDev = comm->devs + devIndex;
  if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
  if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
  if (commDev->putSignalScratchpadMr != NULL)
    NCCLCHECK(wrap_ibv_dereg_mr(commDev->putSignalScratchpadMr));
  if (comm->base.resiliency) {
    NCCLCHECK(IbCastResiliencyDevDestroy(comm->base.resiliency, devIndex));
  }
  return ncclSuccess;
}

ncclResult_t IbCastCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    bool isSharing = IbCastCommIsSharing(&comm->base);
    if (comm->base.vProps.ndevs > 0)
      rcclTelemetryAddCqPolls(comm->base.vProps.devs[0], comm->base.telCqPollCount);
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    // Acquire QP sharing mutex only when this comm participates in sharing
    std::unique_lock<std::mutex> lock(g_IbCastSharedQpMutex, std::defer_lock);
    if (isSharing) lock.lock();

    // QP teardown: refcount-based for shared, direct destroy for non-shared
    struct IbCastSharedQp* slot0 = NULL;
    for (int q = 0; q < comm->base.nqps; q++) {
      if (comm->base.qps[q].qp == NULL) continue;
      if (isSharing) {
        struct IbCastSharedQp* slot = IbCastFindSharedQpByQpn(comm->base.qps[q].qp->qp_num, true);
        if (slot) {
          if (q == 0) slot0 = slot;
          slot->refcount--;
          if (slot->refcount <= 0 && slot->qp) {
            INFO(NCCL_NET, "IB CAST TEARDOWN: destroying shared send QP qpn=%u group=%d qpIdx=%d",
                 slot->qp->qp_num, slot->key.groupIdx, slot->key.qpIdx);
            wrap_ibv_destroy_qp(slot->qp);
            slot->qp = NULL;
          }
        }
      } else {
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));
      }
    }

    if (comm->base.resiliency) {
      NCCLCHECK(IbCastResiliencyClose(comm->base.resiliency));
    }

    if (isSharing) {
      IbCastFreeCommIdLocked(comm->base.commId);
    }

    // Per-device resource cleanup
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      NCCLCHECK(IbCastCloseSendCommDevResources(comm, i));
      if (!isSharing) {
        NCCLCHECK(IbCastDestroyBase(&comm->devs[i].base));
      }
    }

    // Group CQs are torn down last: they must outlive every QP still bound
    // to them, or ibv_destroy_cq fails with EBUSY.
    if (isSharing && slot0) {
      slot0->cqRefcount--;
      if (slot0->cqRefcount <= 0) {
        IbCastCleanupGroupCqs(slot0);
      }
    }

    if (comm->base.resiliency) {
      NCCLCHECK(IbCastResiliencyDestroy(&comm->base.resiliency));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

// Per-device flush memory cleanup (common between shared/non-shared paths).
// Releases GPU memory, MRs, and dma-buf FDs. Does NOT destroy the flush QP.
static ncclResult_t IbCastCloseRecvFlushMem(struct ncclIbRecvCommDev* commDev) {
  if (commDev->gpuFlush.gpuFlushGpuMem != nullptr) {
    CUDACHECK(hipFree(commDev->gpuFlush.gpuFlushGpuMem));
    commDev->gpuFlush.gpuFlushGpuMem = nullptr;
    if (commDev->gpuFlush.gpuMr != nullptr) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.gpuMr));
    commDev->gpuFlush.gpuMr = nullptr;
    if (commDev->gpuFlush.dmabufFd > 0) { close(commDev->gpuFlush.dmabufFd); }
  }
  return ncclSuccess;
}

// Per-device MR cleanup for receiver comms (shared between shared/non-shared paths)
static ncclResult_t IbCastCloseRecvCommDevResources(struct ncclIbRecvComm* comm, int devIndex) {
  struct ncclIbRecvCommDev* commDev = comm->devs + devIndex;
  if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
  if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
  if (comm->base.resiliency) {
    NCCLCHECK(IbCastResiliencyDevDestroy(comm->base.resiliency, devIndex));
  }
  return ncclSuccess;
}

ncclResult_t IbCastCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    bool isSharing = IbCastCommIsSharing(&comm->base);
    if (comm->base.vProps.ndevs > 0)
      rcclTelemetryAddCqPolls(comm->base.vProps.devs[0], comm->base.telCqPollCount);
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    // Acquire QP sharing mutex only when this comm participates in sharing
    std::unique_lock<std::mutex> lock(g_IbCastSharedQpMutex, std::defer_lock);
    if (isSharing) lock.lock();

    // Data QP teardown: refcount-based for shared, direct destroy for non-shared
    struct IbCastSharedQp* slot0 = NULL;
    for (int q = 0; q < comm->base.nqps; q++) {
      if (comm->base.qps[q].qp == NULL) continue;
      if (isSharing) {
        struct IbCastSharedQp* slot = IbCastFindSharedQpByQpn(comm->base.qps[q].qp->qp_num, false);
        if (slot) {
          if (q == 0) slot0 = slot;
          slot->refcount--;
          if (slot->refcount <= 0 && slot->qp) {
            INFO(NCCL_NET, "IB CAST TEARDOWN: destroying shared recv QP qpn=%u group=%d qpIdx=%d",
                 slot->qp->qp_num, slot->key.groupIdx, slot->key.qpIdx);
            wrap_ibv_destroy_qp(slot->qp);
            slot->qp = NULL;
          }
        }
      } else {
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));
      }
    }

    if (comm->base.resiliency) {
      NCCLCHECK(IbCastResiliencyClose(comm->base.resiliency));
    }

    if (isSharing) {
      IbCastFreeCommIdLocked(comm->base.commId);
    }

    // Per-device resource cleanup: flush QP/memory and MR teardown
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbRecvCommDev* commDev = comm->devs + i;
      if (comm->flushEnabled) {
        // Flush memory is always per-comm
        NCCLCHECK(IbCastCloseRecvFlushMem(commDev));
        // Flush QP teardown: refcount-based for shared, direct destroy for non-shared
        if (commDev->gpuFlush.qp.qp != NULL) {
          if (isSharing) {
            struct IbCastSharedQp* flushSlot = IbCastFindSharedQpByQpn(
                commDev->gpuFlush.qp.qp->qp_num, false);
            if (flushSlot) {
              flushSlot->refcount--;
              if (flushSlot->refcount <= 0 && flushSlot->qp) {
                INFO(NCCL_NET, "IB CAST TEARDOWN: destroying shared flush QP qpn=%u group=%d",
                     flushSlot->qp->qp_num, flushSlot->key.groupIdx);
                wrap_ibv_destroy_qp(flushSlot->qp);
                flushSlot->qp = NULL;
                IbCastUnregisterSharedQpLocked(flushSlot);
              }
            } else {
              // Not in pool (pool-exhaustion fallback) — destroy directly
              NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
            }
          } else {
            NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
          }
          commDev->gpuFlush.qp.qp = NULL;
        }
        if (commDev->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.hostMr));
      }
      NCCLCHECK(IbCastCloseRecvCommDevResources(comm, i));
      if (!isSharing) {
        NCCLCHECK(IbCastDestroyBase(&commDev->base));
      }
    }

    // Group CQs are torn down last: they must outlive every QP still bound
    // to them (data QPs and flush QPs above), or ibv_destroy_cq fails with EBUSY.
    if (isSharing && slot0) {
      slot0->cqRefcount--;
      if (slot0->cqRefcount <= 0) {
        IbCastCleanupGroupCqs(slot0);
      }
    }

    if (comm->base.resiliency) {
      NCCLCHECK(IbCastResiliencyDestroy(&comm->base.resiliency));
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t IbCastCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t rcclCastNetP2pPolicy(void* handle, int isP2p) {
  if (!handle) return ncclInvalidArgument;
  struct ncclIbHandle* ibHandle = (struct ncclIbHandle*)handle;
  if (ibHandle->magic != NCCL_SOCKET_MAGIC) return ncclInvalidArgument;
  ibHandle->isP2p = isP2p;
  return ncclSuccess;
}
