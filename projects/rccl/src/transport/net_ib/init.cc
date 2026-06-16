/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "p2p_resiliency_recovery.h"

ncclResult_t pciPathToInt64(char* path, int64_t* id);

NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbDataDirect,"IB_DATA_DIRECT",1);

// default to 0 to disable ooo rq, if set to 1, ooo rq will be enabled or failed
NCCL_PARAM(IbOooRq,"IB_OOO_RQ", 0)

static std::mutex ncclIbMutex;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// rather than once for all communicators. However, the internal plugin implementation
// still assumes the plugin is initialized only once across all communicators. The ref
// counter makes sure the plugin internally initializes only once. When per communicator
// context support is added to the plugin the ref counter can be removed.
static int netRefCount;

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);
NCCL_PARAM(IbDevicePciOrder, "IB_DEVICE_PCI_ORDER", 1);

extern int64_t ncclParamIbArThreshold();

// Returns 1 if this is the path of two VFs of the same physical device
static int ncclIbMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (ncclParamIbMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

/**
 * Assumes PCIe path ends with xxxx:xx:xx.x
 */
 static void ncclIbNormalizePciPath(const char* in, char* out, size_t out_size) {
  if (!in || !out || out_size == 0) return;
  // Safe copy with truncation
  size_t len = strnlen(in, out_size - 1);
  memmove(out, in, len);
  out[len] = '\0';
  if (len < 4) return;
  // Merge multi-port NICs (.1/.2/.3 -> .0)
  out[len - 1] = '0';
  // Merge VFs if enabled
  if (ncclParamIbMergeVfs()) {
    out[len - 3] = '0';
    out[len - 4] = '0';
  }
}

/**
 * Extract the PCIe root complex (domain:bus) from a Linux sysfs PCI device path.
 */
static ncclResult_t ncclIbGetPciRootFromPath(
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
        int chars_read = 0;
        if (sscanf(p, "pci%4x:%2x%n", &domain, &bus, &chars_read) == 2 &&
            chars_read == 10) {
            snprintf(root, rootLen, "%04x:%02x", domain, bus);
            return ncclSuccess;
        }
        p = strstr(p + 1, "pci");
    }
    return ncclInvalidUsage;
}

/**
 * Determine the NUMA node associated with a PCI device from its sysfs path.
 */
static int ncclIbGetNumaNodeFromPath(const char* pciPath) {
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

static int ncclIbCompareDevs(const void* dev1, const void* dev2) {
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

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, char* fullPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  // set fullPath to the raw real path (used for PCI ordering); empty if realpath returned NULL
  snprintf(fullPath, PATH_MAX, "%s", p ? p : "");
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
    if (path) *path = p;
  } else {
    // Store the normalized path so multi-port NICs and VFs share the same pciPath,
    // which is what ncclIbMatchVfPath / ncclIbMakeVDeviceInternal rely on.
    char* normalized = (char*)malloc(PATH_MAX);
    if (normalized == NULL) return ncclSystemError;
    ncclIbNormalizePciPath(p, normalized, PATH_MAX);
    free(p);
    if (path) *path = normalized;
  }
  return ncclSuccess;
}

static ncclResult_t ncclIbGetRealPort(char* pciPath, int* realPort, int devIdx) {
  *realPort = 0;
  if (pciPath == NULL) return ncclSuccess;
  // Keep the real port aside (the ibv port is always 1 on recent cards)
  // Count only devices before the current device index to assign unique port numbers
  for (int d = 0; d < devIdx; d++) {
    if (ncclIbMatchVfPath(pciPath, ncclIbDevs[d].pciPath)) (*realPort)++;
  }
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
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
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
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
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  return true;
failure:
  return false;
}

extern int64_t ncclParamIbPrepostReceiveWorkRequests();
extern int64_t ncclParamIbReceiverSideMatchingScheme();

static ncclResult_t ncclIbQueryOooRqSize(struct ibv_context* ibvCtx, const char *devName, uint32_t* oooRqSize) {
  ncclResult_t ret;
  if (!oooRqSize) return ncclInvalidArgument;
  *oooRqSize = 0;

  if (ncclParamIbOooRq() == 0) return ncclSuccess;

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

ncclResult_t ncclIbMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  if (ncclParamIbMergeNics() == 0 && props->ndevs > 1) {
    WARN("NET/IB : Skipping makeVDevice, Please set NCCL_IB_MERGE_NICS=1");
    return ncclInvalidUsage;
  }

  if (props->ndevs == 0) {
      WARN("NET/IB : Can't make virtual NIC with 0 devices");
      return ncclInvalidUsage;
  }

  if (ncclNMergedIbDevs == MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  // Always count up number of merged devices
  ncclIbMergedDev tmp;
  memset(&tmp,0,sizeof(tmp));
  bool used[MAX_IB_DEVS] = {0};

  for (int i = 0; i < props->ndevs; i++) {
    if( props->devs[i]  < 0 || props->devs[i] >= ncclNIbDevs ) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], ncclNIbDevs);
      return ncclInvalidUsage;
    }
    if(used[props->devs[i]]) continue;
    const ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (tmp.vProps.ndevs == NCCL_IB_MAX_DEVS_PER_NIC) return ncclInvalidUsage;
    tmp.vProps.devs[tmp.vProps.ndevs++] = props->devs[i];
    tmp.speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (tmp.vProps.ndevs > 1) {
      size_t off = strlen(tmp.devName);
      snprintf(tmp.devName + off, sizeof(tmp.devName) - off, "+%s", dev->devName);
    // First time, copy the plain name
    } else {
      strncpy(tmp.devName, dev->devName, MAXNAMESIZE-1);
      tmp.devName[MAXNAMESIZE-1] = '\0';
    }
    used[props->devs[i]] = true;
  }

  // Check link layers
  const ncclIbDev* dev0 = ncclIbDevs + tmp.vProps.devs[0];
  for (int i = 1; i < tmp.vProps.ndevs; i++) {
    const ncclIbDev* dev = ncclIbDevs + tmp.vProps.devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
        tmp.vProps.devs[0], dev0->devName, dev0->portNum, NCCL_IB_LLSTR(dev0->link),tmp.vProps.devs[i], dev->devName, dev->portNum, NCCL_IB_LLSTR(dev->link));
      return ncclInvalidUsage;
    }
  }

  int numa0 = ncclIbGetNumaNodeFromPath(dev0->pciPath);
  //format -> 0000:00
  char root0[8];
  ncclIbGetPciRootFromPath(dev0->pciPath, root0, sizeof(root0));
  for (int i = 1; i < tmp.vProps.ndevs; i++) {
    const ncclIbDev* dev = ncclIbDevs + tmp.vProps.devs[i];
    int numa_i = ncclIbGetNumaNodeFromPath(dev->pciPath);
    if (numa0 >= 0 && numa_i >= 0 && numa_i != numa0) {
      WARN("NET/IB : Merging NICs across NUMA nodes (%s numa=%d, %s numa=%d). "
           "This may significantly reduce performance.",
           dev0->devName, numa0, dev->devName, numa_i);
      break;
    }

    char root_i[8];
    ncclIbGetPciRootFromPath(dev->pciPath, root_i, sizeof(root_i));
    if (strcmp(root_i, root0) != 0) {
      WARN("NET/IB : Merging NICs across PCIe Root Complexes "
           "(%s root=%s, %s root=%s). "
           "GPUDirect RDMA and bandwidth aggregation may be impacted.",
           dev0->devName, root0, dev->devName, root_i);
      break;
    }
  }
  ncclIbMergedDevs[ncclNMergedIbDevs] = tmp;
  *d = ncclNMergedIbDevs++;
  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, tmp.devName, tmp.speed, tmp.vProps.ndevs);
  return ncclSuccess;
}

ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(ncclIbMutex);
  ncclResult_t res = ncclIbMakeVDeviceInternal(d, props);
  return res;

}

ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr) {
  (void)ctx;
  (void)netAttr;
  return ncclSuccess;
}

const char* ibProviderName[] = {
  "None",
  "Mlx5",
};

ncclResult_t ncclIbFinalizeDevices(void) {
  netRefCount--;
  return ncclSuccess;
}

extern int64_t ncclIbArThreshold;
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  struct ibv_device** devices = NULL;
  if (netRefCount++) return ret;
  ncclProfilerFunction = profFunction;
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if(wrap_mlx5dv_symbols() != ncclSuccess) { INFO(NCCL_NET, "NET/IB : Failed to open mlx5dv symbols. Advance features like CX-8 Direct-NIC will be disabled."); }

  if (ncclNIbDevs == -1) {
    std::lock_guard<std::mutex> lock(ncclIbMutex);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      int nIpIfs = 0;
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs = 0;

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

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context = NULL;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        char dataDirectDevicePath[PATH_MAX] = "/sys";
        int devCount = /*undefined*/-1, devOffset = 0;

        uint32_t oooRqSize = 0;
        enum ncclIbProvider ibProvider = wrap_mlx5dv_is_supported(devices[d]) ? IB_PROVIDER_MLX5 : IB_PROVIDER_NONE;
        if (ibProvider == IB_PROVIDER_MLX5 && ncclParamIbOooRq()) {
          NCCLCHECKGOTO(ncclIbQueryOooRqSize(context, devices[d]->name, &oooRqSize), ret, fail);
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
              if (ncclParamIbDataDirect() > 0 && ibProvider == IB_PROVIDER_MLX5 && ncclMlx5dvDmaBufCapable(context)) {
                int pathLen = strlen(dataDirectDevicePath);
                ncclResult_t res = wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + pathLen, sizeof(dataDirectDevicePath) - pathLen);
                if (res == ncclSuccess) {
                  // data direct devices are exposed twice: with the C2C + PCIe link and with the data direct link
                  devCount = 2;
                  // by default only expose the data direct NIC (devOffset = 1), unless set to 2 by the user
                  devOffset = (ncclParamIbDataDirect() == 2) ? 0 : 1;
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
              ncclIbDevs[ncclNIbDevs].device = d;
              ncclIbDevs[ncclNIbDevs].ibProvider = ibProvider;
              ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
              ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
              ncclIbDevs[ncclNIbDevs].portNum = port_num;
              ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
              if (portAttr.active_speed_ex) {
                // A non-zero active_speed_ex indicates XDR rate (0x100) or higher
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed_ex) * ncclIbWidth(portAttr.active_width);
              } else {
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
              }
              ncclIbDevs[ncclNIbDevs].context = context;
              ncclIbDevs[ncclNIbDevs].pdRefs = 0;
              ncclIbDevs[ncclNIbDevs].pd = NULL;
              // for dev==1 (data direct device), pciPath is given by mlx5
              strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
              NCCLCHECKGOTO(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, (dev == 1) ? NULL : &ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].fullPciPath), ret, fail);
              if (dev == 1) {
                snprintf(ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
                NCCLCHECK(ncclCalloc(&ncclIbDevs[ncclNIbDevs].pciPath, PATH_MAX));
                strncpy(ncclIbDevs[ncclNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
                ncclIbDevs[ncclNIbDevs].capsProvider.mlx5.dataDirect = 1;
              }

              ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
              ncclIbDevs[ncclNIbDevs].oooRqSize = oooRqSize;
              ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
              NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

              // Enable ADAPTIVE_ROUTING by default on IB networks
              // But allow it to be overloaded by an env parameter
              ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
              if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();


              INFO(NCCL_NET, "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d oooRqSize=%d", d, devices[d]->name, devices[d]->dev_name,
                   ncclIbDevs[ncclNIbDevs].portNum, NCCL_IB_LLSTR(portAttr.link_layer), ibProviderName[ncclIbDevs[ncclNIbDevs].ibProvider], ncclIbDevs[ncclNIbDevs].speed, context,
                   ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar, ncclIbDevs[ncclNIbDevs].oooRqSize);

              ncclIbAsyncThread = std::thread(ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs);
              ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
              ncclIbAsyncThread.detach();

              ncclNIbDevs++;
              nPorts++;
            }
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      if (devices && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; }
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }
    // Determine whether RELAXED_ORDERING is enabled and possible
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();

    // Default value for ncclIbArThreshold is 8192
    if (ncclParamIbArThreshold() != -2) {
      if (ncclParamIbOooRq()) {
        INFO(NCCL_NET, "NET/IB: OOO RQ is enabled, AR threshold will be ignored.");
      } else {
        ncclIbArThreshold = ncclParamIbArThreshold();  // set explicitly by user
      }
    }
    // sort devices to ensure a consistent order across nodes
    if (ncclParamIbDevicePciOrder()) qsort(ncclIbDevs, ncclNIbDevs, sizeof(struct ncclIbDev), ncclIbCompareDevs);
    // Once sorted, get the realPort ID and create the virtual devices.
    // Doing it after sorting ensures that devices will have consistent realPort ids across nodes.
    char line[2048] = "";
    for (int d = 0; d < ncclNIbDevs; d++) {
      NCCLCHECKGOTO(ncclIbGetRealPort(ncclIbDevs[d].pciPath, &ncclIbDevs[d].realPort, d), ret, fail);
      snprintf(line + strlen(line), sizeof(line) - strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName, ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link));

      // Add this plain physical device to the list of virtual devices (after sorting)
      int vDev;
      ncclNetVDeviceProps_t vProps = {0};
      vProps.ndevs = 1;
      vProps.devs[0] = d;
      NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT | NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "", ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));
  }
exit:
  return ret;
fail:
  if(devices != NULL && ncclSuccess != wrap_ibv_free_device_list(devices)){WARN("NET/IB : Unable to free device list");}
  goto exit;
}

ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclIbInitDevices(logFunction, profFunction));
  NCCLCHECK(ncclIbPortRecoveryThreadStart());
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = config->trafficClass;
  *ctx = (void *)netCommConfig;
  return ret;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;
  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  NCCLCHECK(ncclIbGetPhysProperties(mergedDev->vProps.devs[0], props));
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

ncclResult_t ncclIbFinalize(void* ctx) {
  free(ctx);
  NCCLCHECK(ncclIbPortRecoveryThreadStop());
  return ncclIbFinalizeDevices();
}
