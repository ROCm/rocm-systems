/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_MULTISEG_NET_IB_MSEG_GLOBALS_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_MULTISEG_NET_IB_MSEG_GLOBALS_H_

#include <cstdlib>
#include <cstring>
#include <new>

#include "net_ib_multiseg/common_mseg.h"

// Puts the device tables and counters src/transport/net_ib_multiseg/common.cc owns back to their load-time values. Each
// net_ib fixture's TearDown calls this.
inline void NetIbMsegResetGlobals() {
  for (IbMsegDev& dev : IbMsegDevs) {
    std::free(dev.pciPath);
    std::free(dev.mrCache.slots);
    dev.~IbMsegDev();
    new (&dev) IbMsegDev();
  }
  std::memset(IbMsegMergedDevs, 0, sizeof(IbMsegMergedDevs));
  IbMsegNDevs = -1;
  IbMsegNMergedDevs = -1;
  std::memset(IbMsegIfName, 0, sizeof(IbMsegIfName));
  std::memset(&IbMsegIfAddr, 0, sizeof(IbMsegIfAddr));
  IbMsegRelaxedOrderingEnabled = 0;
  IbMsegSpeedChangeCounter = 0;
  IbMsegProfilerFunction = nullptr;
}

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_MULTISEG_NET_IB_MSEG_GLOBALS_H_
