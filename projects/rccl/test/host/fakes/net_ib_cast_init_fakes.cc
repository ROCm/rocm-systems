/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/init.cc fakes. See
// net_ib_cast_init_fakes.h.

#include "net_ib_cast_init_fakes.h"

#include "common_cast.h"
#include "nccl_fakes.h"

bool IbCastAinicRoce = 0;
bool IbCastMultiplaneEnable = false;
bool IbCastOffloadEnabled = 0;
bool IbCastUseInline = 0;
bool IbCastAinicCtsInlineData = 0;
int IbCastGdrFlushDisable = 0;

int64_t rcclParamIbCastQpsPerP2p() { return g_loadParam("RCCL_IB_QPS_PER_P2P", 0); }
int64_t rcclParamIbCastP2pDisableCts() { return g_loadParam("RCCL_IB_P2P_DISABLE_CTS", 1); }
int64_t ncclParamIbCastOooRq() { return g_loadParam("IB_OOO_RQ", 0); }

void ResetNetIbCastInitFakes() {
  IbCastAinicRoce = 0;
  IbCastMultiplaneEnable = false;
  IbCastOffloadEnabled = 0;
  IbCastUseInline = 0;
  IbCastAinicCtsInlineData = 0;
  IbCastGdrFlushDisable = 0;
}
