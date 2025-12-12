/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "graph/xml.h"

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int ncclIbGdrModuleLoaded = 0; // 1 = true, 0 = false

// Introduce RCCL_FORCE_ENABLE_GDRDMA to force load GPU-NIC RDMA module
// Use ONLY for debugging!
RCCL_PARAM(ForceEnableGdrdma, "FORCE_ENABLE_GDRDMA", -1);

extern int64_t ncclParamIbPciRelaxedOrdering();

static void ibGdrSupportInitOnce() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (rcclParamForceEnableGdrdma() == 1) {
    // RCCL_FORCE_ENABLE_GDRDMA=1 enables GPU-NIC RDMA only from RCCL-side
    // Requires support from NIC driver modules
    // Use ONLY for debugging!
    ncclIbGdrModuleLoaded = 1;
    INFO(NCCL_INIT, "RCCL_FORCE_ENABLE_GDRDMA = 1, so explicitly setting ncclIbGdrModuleLoaded = 1");
  }

  if (ncclIbGdrModuleLoaded == 0) {
    // Check for `memory_peers` directory containing `amdkfd/version`
    // This `memory_peers` directory is created by NIC-GPU driver interaction
    // On Linux kernel 5.15.0 (e.g. Ubuntu 22.04), `memory_peers` is created under `/sys/kernel/mm/`
    // However, on newer kernels like Ubuntu 24.04.1 (Linux kernel 6.8.0) or Ubuntu 22.04.4 HWE (Linux kernel 6.5.0),
    // this `memory_peers` directory is either not created (go to else-if condition)
    // or created under a different path like `/sys/kernel/` or `/sys/` (depending on your ib_peer_mem module)
    const char* memory_peers_paths[] = {"/sys/kernel/mm/memory_peers/amdkfd/version",
                                  "/sys/kernel/memory_peers/amdkfd/version",
                                  "/sys/memory_peers/amdkfd/version",
                                  NULL};
    int i = 0;

    while (memory_peers_paths[i]) {
      if (access(memory_peers_paths[i], F_OK) == 0) {
        ncclIbGdrModuleLoaded = 1;
        INFO(NCCL_INIT,"Found %s", memory_peers_paths[i]);
        break;
      } else {
        ncclIbGdrModuleLoaded = 0;
      }
      ++i;
    }

    char strValue[MAX_STR_LEN];
    (void)ncclTopoGetStrFromSys("/sys/devices/virtual/dmi/id", "bios_version", strValue);
    if (strncmp("Hyper-V UEFI Release", strValue, 20) == 0) {
      int roMode = ncclParamIbPciRelaxedOrdering();
      (void)ncclTopoGetStrFromSys("/proc/sys/kernel", "numa_balancing", strValue);
      if (strcmp(strValue, "1") == 0 && roMode == 0)
        ncclIbGdrModuleLoaded = 0;
    }

    if (ncclIbGdrModuleLoaded == 0) {
      // Check for `ib_register_peer_memory_client` symbol in `/proc/kallsyms`
      // if your system uses native OS ib_peer module
      char buf[256];
      FILE *fp = NULL;
      fp = fopen("/proc/kallsyms", "r");

      if (fp == NULL) {
        INFO(NCCL_INIT,"Could not open /proc/kallsyms");
      } else {
        while (fgets(buf, sizeof(buf), fp) != NULL) {
          if (strstr(buf, "t ib_register_peer_memory_client") != NULL ||
              strstr(buf, "T ib_register_peer_memory_client") != NULL) {
            ncclIbGdrModuleLoaded = 1;
            INFO(NCCL_INIT,"Found ib_register_peer_memory_client in /proc/kallsyms");
            break;
          }
        }
      }
    }
  }
#else
  // Check for the nv_peer_mem module being loaded
  ncclIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
                          KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
                          KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
#endif
}

// Returns ncclSuccess if any of the peermem modules are loaded.
ncclResult_t ncclIbGdrSupport() {
  static std::once_flag once;
  std::call_once(once, ibGdrSupportInitOnce);
  if (!ncclIbGdrModuleLoaded)
    return ncclSystemError;
  return ncclSuccess;
}

static int ncclIbPeerMemModuleLoaded = 0; // 1 = true, 0 = false
static void ibPeerMemSupportInitOnce() {
  ncclIbPeerMemModuleLoaded = KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

// Returns ncclSuccess if nvidia_peermem module is loaded. Does not check legacy implementations of nv_peer_mem (e.g. nv_mem, nv_mem_nc)
ncclResult_t ncclIbPeerMemSupport() {
  static std::once_flag once;
  std::call_once(once, ibPeerMemSupportInitOnce);
  if (!ncclIbPeerMemModuleLoaded)
    return ncclSystemError;
  return ncclSuccess;
}

static thread_local int ibDmaSupportInitDev; // which device to init, must be thread local
static void ibDmaBufSupportInitOnce() {
  ncclResult_t res;
  int dev_fail = 0;

  // This is a physical device, not a virtual one, so select from ibDevs
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  struct ibv_pd* pd;
  struct ibv_context* ctx = ibDev->context;
  res = rocmLibraryInit();
  if (res != ncclSuccess) goto failure;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  ibDev->dmaBufSupported = 1;
  return;
failure:
  ibDev->dmaBufSupported = -1;
  return;
}
// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t ncclIbDmaBufSupport(int dev) {
  static std::once_flag onces[MAX_IB_DEVS];
  // init the device only once
  ibDmaSupportInitDev = dev;
  std::call_once(onces[dev], ibDmaBufSupportInitOnce);
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  int dmaBufSupported = ibDev->dmaBufSupported;
  if (dmaBufSupported == 1) return ncclSuccess;
  return ncclSystemError;
}
