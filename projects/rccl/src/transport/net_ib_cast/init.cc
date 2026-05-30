/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common_cast.h"
#include "p2p_resiliency_recovery_cast.h"

extern int64_t ncclParamIbCastQpsPerConn();
RCCL_PARAM(IbCastQpsPerP2p, "IB_QPS_PER_P2P", 0);
extern int64_t ncclParamIbCastResiliencyPortFailover();
extern int64_t ncclParamIbCastResiliencyPortRecovery();
extern int64_t ncclParamIbCastGdrFlushDisable();
// AMD AINIC
RCCL_PARAM(IbCastCtsOffloadEnabled, "CTS_OFFLOAD_ENABLED", -1);
RCCL_PARAM(IbCastP2pDisableCts, "IB_P2P_DISABLE_CTS", 1);

bool IbCastAinicRoce = 0;
bool IbCastOffloadEnabled = 0;
bool IbCastUseInline = 0;
int IbCastGdrFlushDisable = 0;
extern int64_t rcclParamAinicRoce();
extern int64_t ncclParamIbCastUseInline();
static int IbCastGetNumaNodeFromPath(const char* pciPath);
static ncclResult_t IbCastGetPciRootFromPath(const char* pciPath, char* root, size_t rootLen);

// Forward declaration for the 2-arg pciPathToInt64 defined in src/misc/utils.cc.
// Matches the resolution pattern used by net_ib/init.cc (via common.h).
ncclResult_t pciPathToInt64(char* path, int64_t* id);

NCCL_PARAM(IbCastPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbCastAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbCastDataDirect,"IB_DATA_DIRECT",1);

// default to 0 to disable ooo rq, if set to 1, ooo rq will be enabled or failed
NCCL_PARAM(IbCastOooRq,"IB_OOO_RQ", 0)

static std::mutex IbCastMutex;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// rather than once for all communicators. However, the internal plugin implementation
// still assumes the plugin is initialized only once across all communicators. The ref
// counter makes sure the plugin internally initializes only once. When per communicator
// context support is added to the plugin the ref counter can be removed.
static int netRefCount;

NCCL_PARAM(IbCastDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbCastMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbCastMergeNics, "IB_MERGE_NICS", 1);
NCCL_PARAM(IbCastDevicePciOrder, "IB_DEVICE_PCI_ORDER", 1);

extern int64_t ncclParamIbCastArThreshold();

// Returns 0 if this is the path of two VFs of the same physical device
static int IbCastMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (ncclParamIbCastMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

static int IbCastCompareDevs(const void* dev1, const void* dev2) {
  // Compare devices using the last component of the PCI path.
  // Note: fullPciPath is never NULL but empty if not found by realpath.
  char* path1 = ((struct ncclIbDev*)dev1)->fullPciPath;
  char* path2 = ((struct ncclIbDev*)dev2)->fullPciPath;

  // if a path is empty, order the devices to the back of the list
  if (strlen(path1) == 0 || strlen(path2) == 0) return strlen(path2) - strlen(path1);

  int64_t id1, id2;
  pciPathToInt64(path1, &id1);
  pciPathToInt64(path2, &id2);

  return (id1 < id2) ? -1 : ((id1 == id2) ? 0 : 1);
}

static ncclResult_t IbCastGetPciPath(char* devName, char** path, char* fullPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  // set fullPath to empty if realpath returned NULL
  snprintf(fullPath, PATH_MAX, "%s", p ? p : "");
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbCastMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
  }
  if (path) {
    *path = p;
  } else {
    free(p);
  }
  return ncclSuccess;
}

static ncclResult_t IbCastGetRealPort(char* pciPath, int* realPort, int devIdx) {
  *realPort = 0;
  if (pciPath == NULL) return ncclSuccess;
  // Keep the real port aside (the ibv port is always 1 on recent cards)
  // Count only devices before the current device index to assign unique port numbers
  for (int d = 0; d < devIdx; d++) {
    if (IbCastMatchVfPath(pciPath, IbCastDevs[d].pciPath)) (*realPort)++;
  }
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000, /* NDR */
  200000  /* XDR */
};

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int IbCastWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int IbCastSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int IbCastRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbCastPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

static bool ncclMlx5dvDmaBufCapable(ibv_context *context){
  ncclResult_t res;
  int dev_fail = 0;

  struct ibv_pd* pd;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, context), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  return true;
failure:
  return false;
}

extern int64_t ncclParamIbCastPrepostReceiveWorkRequests();
extern int64_t ncclParamIbCastReceiverSideMatchingScheme();

static ncclResult_t IbCastQueryOooRqSize(struct ibv_context* ibvCtx, const char *devName, uint32_t* oooRqSize) {
  ncclResult_t ret;
  if (!oooRqSize) return ncclInvalidArgument;
  *oooRqSize = 0;

  if (ncclParamIbCastOooRq() == 0) return ncclSuccess;

  // out-of-order recv prerequisite: device capability
  struct mlx5dv_context dvCtx;
  *oooRqSize = 0;
  dvCtx.comp_mask = MLX5DV_CONTEXT_MASK_OOO_RECV_WRS;
  NCCLCHECKGOTO(wrap_mlx5dv_query_device(ibvCtx, &dvCtx), ret, fail);
  if ((dvCtx.comp_mask & MLX5DV_CONTEXT_MASK_OOO_RECV_WRS) && dvCtx.ooo_recv_wrs_caps.max_rc > 0) {
    *oooRqSize = dvCtx.ooo_recv_wrs_caps.max_rc;
  }


  return ncclSuccess;
fail:
  return ncclInternalError;
}

ncclResult_t IbCastMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  if (ncclParamIbCastMergeNics() == 0 && props->ndevs > 1) {
    WARN("NET/IB : Skipping makeVDevice, Please set NCCL_IB_MERGE_NICS=1");
    return ncclInvalidUsage;
  }

  if (props->ndevs == 0) {
      WARN("NET/IB : Can't make virtual NIC with 0 devices");
      return ncclInvalidUsage;
  }

  if (IbCastNMergedDevs == MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  // Always count up number of merged devices
  ncclIbMergedDev* mDev = IbCastMergedDevs + IbCastNMergedDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  for (int i = 0; i < props->ndevs; i++) {
    ncclIbDev* dev = IbCastDevs + props->devs[i];
    if (mDev->vProps.ndevs == NCCL_IB_MAX_DEVS_PER_NIC) return ncclInvalidUsage;
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName), sizeof(mDev->devName) - strlen(mDev->devName), "+%s", dev->devName);
    // First time, copy the plain name
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
    }
  }

  // Check link layers
  ncclIbDev* dev0 = IbCastDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= IbCastNDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], IbCastNDevs);
      return ncclInvalidUsage;
    }
    ncclIbDev* dev = IbCastDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
        props->devs[0], dev0->devName, dev0->portNum, NCCL_IB_LLSTR(dev0->link), props->devs[i], dev->devName, dev->portNum, NCCL_IB_LLSTR(dev->link));
      return ncclInvalidUsage;
    }
  }

  int numa0 = IbCastGetNumaNodeFromPath(dev0->pciPath);
  //format -> 0000:00 
  char root0[8]; 
  IbCastGetPciRootFromPath(dev0->pciPath, root0, sizeof(root0));
  for (int i = 1; i < props->ndevs; i++) {
    ncclIbDev* dev = IbCastDevs + props->devs[i];
    int numaI = IbCastGetNumaNodeFromPath(dev->pciPath);
    if (numa0 >= 0 && numaI >= 0 && numaI != numa0) {
      WARN("NET/IB : Merging NICs across NUMA nodes (%s numa=%d, %s numa=%d). "
           "This may significantly reduce performance.",
           dev0->devName, numa0, dev->devName, numaI);
      break;
    }

    char root_i[8];
    IbCastGetPciRootFromPath(dev->pciPath, root_i, sizeof(root_i));
    if (strcmp(root_i, root0) != 0) {
      WARN("NET/IB : Merging NICs across PCIe Root Complexes "
           "(%s root=%s, %s root=%s). "
           "GPUDirect RDMA and bandwidth aggregation may be impacted.",
           dev0->devName, root0, dev->devName, root_i);
      break;
    }
  }

  // CTS Offload and CTS Inline are not yet compatible with NIC Fusion
  // (NCCL_IB_MERGE_NICS). Disable them when a multi-NIC vNIC is created.
  if (props->ndevs > 1) {
    if (IbCastOffloadEnabled) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : NIC Fusion (ndevs=%d) - disabling CTS Offload (not yet supported with merge)", props->ndevs);
      IbCastOffloadEnabled = false;
    }
  }

  *d = IbCastNMergedDevs++;
  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return ncclSuccess;
}

ncclResult_t IbCastMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(IbCastMutex);
  ncclResult_t res = IbCastMakeVDeviceInternal(d, props);
  return res;

}

ncclResult_t IbCastSetNetAttr(void *ctx, ncclNetAttr_t *netAttr) {
  (void)ctx;
  (void)netAttr;
  return ncclSuccess;
}

const char* ibCastProviderName[] = {
  "None",
  "Mlx5",
};

ncclResult_t IbCastFinalizeDevices(void) {
  netRefCount--;
  return ncclSuccess;
}

extern int64_t IbCastArThreshold;
ncclResult_t IbCastInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  if (netRefCount++) return ret;
  IbCastProfilerFunction = profFunction;
  if (ncclParamIbCastDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if(wrap_mlx5dv_symbols() != ncclSuccess) { INFO(NCCL_NET, "NET/IB : Failed to open mlx5dv symbols. Advance features like CX-8 Direct-NIC will be disabled."); }
  if(wrap_ionicdv_symbols() != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB : Failed to open ionicdv symbols. Advance features like AINIC UD load balancing will be disabled.");
    return ncclInternalError;
  }

  // Detect IB cards
  int nIbDevs = 0;
  struct ibv_device** devices = NULL;
  IbCastAinicRoce = rcclUseAinic();

  if (IbCastNDevs == -1) {
    std::lock_guard<std::mutex> lock(IbCastMutex);
    wrap_ibv_fork_init();
    if (IbCastNDevs == -1) {
      int nIpIfs = 0;
      IbCastNDevs = 0;
      IbCastNMergedDevs = 0;
      NCCLCHECK(ncclFindInterfaces(IbCastIfName, &IbCastIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Check if user defined which IB device:port to use
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }

      for (int d=0; d<nIbDevs && IbCastNDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context = NULL;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        char dataDirectDevicePath[PATH_MAX] = "/sys";
        int devCount = /*undefined*/-1, devOffset = 0;

        uint32_t oooRqSize = 0;
        enum ncclIbProvider ibProvider = wrap_mlx5dv_is_supported(devices[d]) ? IB_PROVIDER_MLX5 : IB_PROVIDER_NONE;
        if (ibProvider == IB_PROVIDER_MLX5 && ncclParamIbCastOooRq()) {
          NCCLCHECKGOTO(IbCastQueryOooRqSize(context, devices[d]->name, &oooRqSize), ret, fail);
        }

        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
            struct ibv_port_attr portAttr;
            if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

            // check against user specified HCAs/ports
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }

            // check for mlx5 data direct support only once for a each device
            if (devCount == -1) {
              devCount = 1;
              devOffset = 0;
              if (ncclParamIbCastDataDirect() > 0 && ibProvider == IB_PROVIDER_MLX5 && ncclMlx5dvDmaBufCapable(context)) {
                int pathLen = strlen(dataDirectDevicePath);
                ncclResult_t res = wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + pathLen, sizeof(dataDirectDevicePath) - pathLen);
                if (res == ncclSuccess) {
                  // data direct devices are exposed twice: with the C2C + PCIe link and with the data direct link
                  devCount = 2;
                  // by default only expose the data direct NIC (devOffset = 1), unless set to 2 by the user
                  devOffset = (ncclParamIbCastDataDirect() == 2) ? 0 : 1;
                  INFO(NCCL_INIT | NCCL_NET, "NET/IB: Data Direct DMA Interface is detected for device %s", devices[d]->name);
                } else if (res == ncclInvalidArgument) {
                  TRACE(NCCL_NET, "NET/IB: Device %s does not support Data Direct DMA.", devices[d]->name);
                } else {
                  WARN("NET/IB: Error in mlx5dv_get_data_direct_sysfs_path with device %s", devices[d]->name);
                  return res;
                }
              }
            }
            for (int dev = devOffset; dev < devCount; ++dev) {
              IbCastDevs[IbCastNDevs].device = d;
              IbCastDevs[IbCastNDevs].ibProvider = ibProvider;
              IbCastDevs[IbCastNDevs].guid = devAttr.sys_image_guid;
              IbCastDevs[IbCastNDevs].portAttr = portAttr;
              IbCastDevs[IbCastNDevs].portNum = port_num;
              IbCastDevs[IbCastNDevs].link = portAttr.link_layer;
              if (portAttr.active_speed_ex) {
                // A non-zero active_speed_ex indicates XDR rate (0x100) or higher
                IbCastDevs[IbCastNDevs].speed = IbCastSpeed(portAttr.active_speed_ex) * IbCastWidth(portAttr.active_width);
              } else {
                IbCastDevs[IbCastNDevs].speed = IbCastSpeed(portAttr.active_speed) * IbCastWidth(portAttr.active_width);
              }
              IbCastDevs[IbCastNDevs].context = context;
              IbCastDevs[IbCastNDevs].pdRefs = 0;
              IbCastDevs[IbCastNDevs].pd = NULL;
              // for dev==1 (data direct device), pciPath is given by mlx5
              strncpy(IbCastDevs[IbCastNDevs].devName, devices[d]->name, MAXNAMESIZE);
              NCCLCHECKGOTO(IbCastGetPciPath(IbCastDevs[IbCastNDevs].devName, (dev == 1) ? NULL : &IbCastDevs[IbCastNDevs].pciPath, IbCastDevs[IbCastNDevs].fullPciPath), ret, fail);
              if (dev == 1) {
                snprintf(IbCastDevs[IbCastNDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
                NCCLCHECK(ncclCalloc(&IbCastDevs[IbCastNDevs].pciPath, PATH_MAX));
                strncpy(IbCastDevs[IbCastNDevs].pciPath, dataDirectDevicePath, PATH_MAX);
                IbCastDevs[IbCastNDevs].capsProvider.mlx5.dataDirect = 1;
              }

              IbCastDevs[IbCastNDevs].maxQp = devAttr.max_qp;
              IbCastDevs[IbCastNDevs].oooRqSize = oooRqSize;
              IbCastDevs[IbCastNDevs].mrCache.capacity = 0;
              IbCastDevs[IbCastNDevs].mrCache.population = 0;
              IbCastDevs[IbCastNDevs].mrCache.slots = NULL;
              NCCLCHECK(IbCastStatsInit(&IbCastDevs[IbCastNDevs].stats));

              // Enable ADAPTIVE_ROUTING by default on IB networks
              // But allow it to be overloaded by an env parameter
              IbCastDevs[IbCastNDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
              if (ncclParamIbCastAdaptiveRouting() != -2) IbCastDevs[IbCastNDevs].ar = ncclParamIbCastAdaptiveRouting();


              INFO(NCCL_NET, "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d oooRqSize=%d", d, devices[d]->name, devices[d]->dev_name,
                   IbCastDevs[IbCastNDevs].portNum, NCCL_IB_LLSTR(portAttr.link_layer), ibCastProviderName[IbCastDevs[IbCastNDevs].ibProvider], IbCastDevs[IbCastNDevs].speed, context,
                   IbCastDevs[IbCastNDevs].pciPath, IbCastDevs[IbCastNDevs].ar, IbCastDevs[IbCastNDevs].oooRqSize);

              IbCastAsyncThread = std::thread(IbCastAsyncThreadMain, IbCastDevs + IbCastNDevs);
              ncclSetThreadName(IbCastAsyncThread.native_handle(), "NCCL IbAsync %2d", IbCastNDevs);
              IbCastAsyncThread.detach();

              IbCastNDevs++;
              nPorts++;
            }
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      if (devices && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; }
    }
    if (IbCastNDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }
    // Determine whether RELAXED_ORDERING is enabled and possible
    IbCastRelaxedOrderingEnabled = IbCastRelaxedOrderingCapable();

    // Default value for IbCastArThreshold is 8192
    if (ncclParamIbCastArThreshold() != -2) {
      if (ncclParamIbCastOooRq()) {
        INFO(NCCL_NET, "NET/IB: OOO RQ is enabled, AR threshold will be ignored.");
      } else {
        IbCastArThreshold = ncclParamIbCastArThreshold();  // set explicitly by user
      }
    }
    // sort devices to ensure a consistent order across nodes
    if (ncclParamIbCastDevicePciOrder()) qsort(IbCastDevs, IbCastNDevs, sizeof(struct ncclIbDev), IbCastCompareDevs);
    // Once sorted, get the realPort ID and create the virtual devices.
    // Doing it after sorting ensures that devices will have consistent realPort ids across nodes.
    char line[2048] = "";
    for (int d = 0; d < IbCastNDevs; d++) {
      NCCLCHECKGOTO(IbCastGetRealPort(IbCastDevs[d].pciPath, &IbCastDevs[d].realPort, d), ret, fail);
      snprintf(line + strlen(line), sizeof(line) - strlen(line), " [%d]%s:%d/%s", d, IbCastDevs[d].devName, IbCastDevs[d].portNum, NCCL_IB_LLSTR(IbCastDevs[d].link));

      // Add this plain physical device to the list of virtual devices (after sorting)
      int vDev;
      ncclNetVDeviceProps_t vProps = {0};
      vProps.ndevs = 1;
      vProps.devs[0] = d;
      NCCLCHECK(IbCastMakeVDeviceInternal(&vDev, &vProps));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT | NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, IbCastRelaxedOrderingEnabled ? "[RO]" : "", IbCastIfName, ncclSocketToString(&IbCastIfAddr, addrline));


    IbCastUseInline = ncclParamIbCastUseInline();
    IbCastGdrFlushDisable = ncclParamIbCastGdrFlushDisable(); 

    if (IbCastAinicRoce) {
      // for AINIC, these params are defaulted to enabled unless user forces it to disable(0).
      IbCastOffloadEnabled = ((rcclParamIbCastCtsOffloadEnabled() == 0) ? false : true);

      // CTS Offload and CTS Inline are mutually dependent — both must be
      // enabled for either to function. Disable both if either is missing.
      if (IbCastOffloadEnabled && rcclParamIbCastQpSchedEnable()) {
        INFO(NCCL_INIT|NCCL_NET, "NET/IB : CAST enabled - disabling CTS Inline Data and CTS Offload (not yet supported with CAST)");
        IbCastOffloadEnabled = false;
      }
      // for AINIC IbUseInline is enabled by default always
      IbCastUseInline = true;
      // for AINIC GDR flush is disabled by default
      IbCastGdrFlushDisable = 1;
  
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : AINIC RoCEv2 optimizations enabled: CTS Inline Data: %s; CTS Offload: %s; "
           "IB Use Inline: enabled; GDR Flush: disabled", IbCastUseInline ? "Enabled": "Disabled",
           IbCastOffloadEnabled ? "Enabled": "Disabled");
    }
  }
exit:
  if (ret == ncclSuccess)
    ret = IbCastQpSchedInitParms(&castGlobalQpSchedParms);
  if (ret == ncclSuccess && castGlobalQpSchedParms.enable &&
      (ncclParamIbCastResiliencyPortFailover() || ncclParamIbCastResiliencyPortRecovery())) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : PORT_FAILOVER/RECOVERY enabled - disabling QP scheduler "
         "(load balancer integration with resiliency is pending)");
    castGlobalQpSchedParms.enable = false;
  }
  if (ret == ncclSuccess && IbCastOffloadEnabled &&
      (ncclParamIbCastResiliencyPortFailover() || ncclParamIbCastResiliencyPortRecovery())) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : PORT_FAILOVER/RECOVERY enabled - disabling CTS offload "
         "(not compatible with resiliency)");
    IbCastOffloadEnabled = false;
  }
  return ret;
fail:
  if(devices && (ncclSuccess != wrap_ibv_free_device_list(devices))){WARN("NET/IB : Unable to free device list");}
  goto exit;
}

ncclResult_t IbCastInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(IbCastInitDevices(logFunction, profFunction));
  NCCLCHECK(IbCastPortRecoveryThreadStart());
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = config->trafficClass;
  *ctx = (void *)netCommConfig;
  return ret;
}

ncclResult_t IbCastDevices(int* ndev) {
  *ndev = IbCastNMergedDevs;
  return ncclSuccess;
}

ncclResult_t IbCastGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = IbCastDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (IbCastGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (IbCastDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  if (IbCastOffloadEnabled && !rcclParamIbCastP2pDisableCts()) {
    props->maxRecvs = 1; 
  } else {
    props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  }
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

ncclResult_t IbCastGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev >= IbCastNMergedDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, IbCastNMergedDevs);
    return ncclInvalidUsage;
  }
  struct ncclIbMergedDev* mergedDev = IbCastMergedDevs + dev;
  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  NCCLCHECK(IbCastGetPhysProperties(mergedDev->vProps.devs[0], props));
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

ncclResult_t IbCastFinalize(void* ctx) {
  free(ctx);
  NCCLCHECK(IbCastPortRecoveryThreadStop());
  return IbCastFinalizeDevices();
}


/**
 * Extract the PCIe root complex (domain:bus) from a Linux sysfs PCI device path.
 *
 * This function scans a sysfs PCI device path (e.g.
 *   "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0")
 * and extracts the first occurrence of the PCI root host bridge identifier
 * of the form "pciDDDD:BB", where:
 *   - DDDD is the PCI domain (hex)
 *   - BB   is the PCI bus number (hex)
 *
 * The extracted root is returned as a string formatted as "DDDD:BB".
 *
 * This helper is typically used to determine whether multiple NICs or devices
 * reside under the same PCIe root complex, which is important for validating
 * NIC merging and avoiding cross-root PCIe traffic.
 *
 * @param pciPath  NUL-terminated sysfs PCI device path
 * @param root     Output buffer receiving the "DDDD:BB" PCI root identifier
 * @param rootLen  Size of the output buffer; must be >= 8
 *
 * @return ncclSuccess on success
 * @return ncclInvalidUsage if inputs are invalid or no PCI root is found
 *
 * Notes:
 * - The function scans for the first valid "pciDDDD:BB" pattern in the path.
 * - The loop is bounded by the length of the string and cannot hang.
 * - This function does not allocate memory.
 */
static ncclResult_t IbCastGetPciRootFromPath(
    const char* pciPath,
    char* root,
    size_t rootLen
) {
    if (pciPath == NULL || root == NULL || rootLen < 8){
        return ncclInvalidUsage;
    }
    const char* p = strstr(pciPath, "pci");
    while (p != NULL) {
        int domain, bus;
        int charsRead = 0;
        if (sscanf(p, "pci%4x:%2x%n", &domain, &bus, &charsRead) == 2 &&
            charsRead == 10) {
            snprintf(root, rootLen, "%04x:%02x", domain, bus);
            return ncclSuccess;
        }
        p = strstr(p + 1, "pci");
    }
    return ncclInvalidUsage;
}
/**
 * Determine the NUMA node associated with a PCI device from its sysfs path.
 *
 * This function reads the "numa_node" attribute from a PCI device's sysfs
 * directory (e.g. "<pciPath>/numa_node") and returns the NUMA node ID.
 *
 * Linux sysfs convention:
 *   - A non-negative integer indicates the NUMA node the device is local to
 *   - "-1" indicates that the device has no specific NUMA affinity
 *
 * This helper is used to validate NUMA locality when merging NICs, ensuring
 * that merged devices do not silently span NUMA nodes, which could negatively
 * impact performance.
 *
 * @param pciPath  NUL-terminated sysfs PCI device path
 *
 * @return NUMA node ID (>= 0) on success
 * @return -1 if the NUMA node cannot be determined, the file is missing,
 *         unreadable, or contains invalid data
 *
 * Notes:
 * - Uses open/read instead of stdio to avoid buffering and locale issues.
 * - Uses strtol for robust numeric parsing and overflow detection.
 * - A return value of -1 may indicate either "no NUMA affinity" or an error;
 *   callers should treat it as "unknown or unspecified".
*/
static int IbCastGetNumaNodeFromPath(const char* pciPath) {
    if (pciPath == NULL) {
        return -1;
    }
    char numaPath[PATH_MAX];
    if (snprintf(numaPath, sizeof(numaPath), "%s/numa_node", pciPath) >= PATH_MAX) {
        return -1; 
    }

    int fd = open(numaPath, O_RDONLY);
    if (fd < 0) return -1;

    char buf[32];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (n <= 0) return -1;
    buf[n] = '\0';
    
    char* endptr;
    errno = 0;
    long numa = strtol(buf, &endptr, 10);
    if (endptr == buf || errno == ERANGE) {
        return -1;
    }
    return (int)numa;
}
