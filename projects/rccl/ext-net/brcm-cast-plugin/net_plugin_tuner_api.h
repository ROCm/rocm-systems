/*************************************************************************
 * Copyright (c) 2025-2026 Broadcom. The term "Broadcom" refers solely to the Broadcom Inc. corporate affiliate that distributes this software. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once 

#include <stdint.h>

#define NET_PLUGIN_API	"ncclTunerPluginNotify"

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

struct tunerSchedParms {
  ncclFunc_t collType;
  size_t msgSz;
  bool match;
  bool enableValid;
  bool wrrEnableValid;
  bool updateIntervalValid;
  bool resetIntervalValid;
  bool weightNewValid;
  bool splitDataValid;
  bool splitDataMinValid;
  bool collResetRtt;
  bool msgSzResetRtt;
  struct ncclIbQpSchedParms parms;
};

typedef void (*netTunerPluginNotifyApi)(struct tunerSchedParms *tunerParms);

