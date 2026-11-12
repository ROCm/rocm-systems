/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_GLOBALS_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_GLOBALS_H_

#include <cstdlib>
#include <cstring>
#include <new>

#include "net_ib/common.h"

// Puts the device tables and counters src/transport/net_ib/common.cc owns back to their load-time values. Each
// net_ib fixture's TearDown calls this.
inline void NetIbResetGlobals() {
  for (ncclIbDev& dev : ncclIbDevs) {
    std::free(dev.pciPath);
    std::free(dev.mrCache.slots);
    dev.~ncclIbDev();
    new (&dev) ncclIbDev();
  }
  std::memset(ncclIbMergedDevs, 0, sizeof(ncclIbMergedDevs));
  ncclNIbDevs = -1;
  ncclNMergedIbDevs = -1;
  std::memset(ncclIbIfName, 0, sizeof(ncclIbIfName));
  std::memset(&ncclIbIfAddr, 0, sizeof(ncclIbIfAddr));
  ncclIbRelaxedOrderingEnabled = 0;
  ncclIbSpeedChangeCounter = 0;
  ncclProfilerFunction = nullptr;
}

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_GLOBALS_H_
