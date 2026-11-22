/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// net_ib's parameter definitions, which IB-MSEG declares extern; the fork binaries link no net_ib unit.
// Hand-copied from src/transport/net_ib/*.cc; NetIbMsegParamCopies (connect-test.cc) fails when they drift.

#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

// src/transport/net_ib/connect.cc
DEFINE_NCCL_PARAM(
  ncclParamIbGidIndex, int32_t, NCCL_IB_GID_INDEX, NCCL_PARAM_VAL_AUTO,
  NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED,
  ncclParamCombo(ncclParamBounded(0),
                 makeOptions(makeOption("AUTO", NCCL_PARAM_VAL_AUTO, "NCCL will automatically determine the GID"),
                             makeOption("-1", NCCL_PARAM_VAL_AUTO, "Legacy value of AUTO"))),
  "Global ID index used in RoCE mode. If set to AUTO, NCCL will automatically "
  "determine the GID.");
DEFINE_NCCL_PARAM(ncclParamIbRoutableFlidIbGidIndex, int32_t, NCCL_IB_ROUTABLE_FLID_GID_INDEX, 1,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "GID index for routable FLID");
DEFINE_NCCL_PARAM(ncclParamIbRoceVersionNum, int32_t, NCCL_IB_ROCE_VERSION_NUM, 2,
                  NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT,
                  "RoCE version for GID dynamically selected by NCCL when IB_GID_INDEX is unset.");
DEFINE_NCCL_PARAM(ncclParamIbTimeout, uint8_t, NCCL_IB_TIMEOUT, 20, NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED,
                  ncclParamBounded<uint8_t>(0),
                  "InfiniBand Verbs timeout. Computed as 4.096us * 2^timeout. Values 0-31. "
                  "Value of 0 or >=32 results in infinite timeout value. "
                  "For example, the default NCCL_IB_TIMEOUT=20 will set the IB verbs timeout as "
                  "4.096 * 2^20 = 4,294,967.296us (or ~4.3s).");
DEFINE_NCCL_PARAM(ncclParamIbRetryCnt, uint8_t, NCCL_IB_RETRY_CNT, 7,
                  NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED, ncclParamBounded<uint8_t>(0, 7),
                  "InfiniBand retry count. Valid values 0-7.");
DEFINE_NCCL_PARAM(ncclParamIbPkey, uint16_t, NCCL_IB_PKEY, 0, NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "InfiniBand partition key");
NCCL_PARAM(IbPkeyValue, "IB_PKEY_VALUE", -1);
DEFINE_NCCL_PARAM(ncclParamIbUseInline, bool, NCCL_IB_USE_INLINE, false, NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "Use inline data for IB sends");
DEFINE_NCCL_PARAM(
  ncclParamIbSl, int32_t, NCCL_IB_SL, NCCL_PARAM_VAL_AUTO, NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED,
  ncclParamCombo(ncclParamBounded(0, 15), makeOptions(makeOption("AUTO", NCCL_PARAM_VAL_AUTO,
                                                                 "Use comm's Traffic Class or 0 as the Service Level"),
                                                      makeOption("-1", NCCL_PARAM_VAL_AUTO, "Legacy value of AUTO"))),
  "InfiniBand Service Level. "
  "If set to AUTO, NCCL will use comm's Traffic Class or 0 as the Service Level.");
DEFINE_NCCL_PARAM(ncclParamIbTc, int32_t, NCCL_IB_TC, NCCL_PARAM_VAL_AUTO,
                  NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED,
                  ncclParamCombo(ncclParamBounded(0, 0xFF),
                                 makeOptions(makeOption("AUTO", NCCL_PARAM_VAL_AUTO, "Use comm's Traffic Class or 0"),
                                             makeOption("-1", NCCL_PARAM_VAL_AUTO, "Legacy value of AUTO"))),
                  "InfiniBand Traffic Class. "
                  "If set to AUTO, NCCL will use comm's Traffic Class or 0.");
DEFINE_NCCL_PARAM(ncclParamIbFifoTc, int32_t, NCCL_IB_FIFO_TC, NCCL_PARAM_VAL_AUTO,
                  NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED,
                  ncclParamCombo(ncclParamBounded(0, 0xFF),
                                 makeOptions(makeOption("AUTO", NCCL_PARAM_VAL_AUTO, "Use sender side's NCCL_IB_TC"),
                                             makeOption("-1", NCCL_PARAM_VAL_AUTO, "Legacy value of AUTO"))),
                  "InfiniBand Traffic Class for control messages. "
                  "If set to AUTO, NCCL will use sender side's Traffic Class which is controlled by "
                  "NCCL_IB_TC.");
DEFINE_NCCL_PARAM(ncclParamIbEceEnable, bool, NCCL_IB_ECE_ENABLE, true,
                  NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT,
                  "Enable querying Enhanced Connection Establishment (ECE) capabilities on IB/RoCE, "
                  "and initialize ECE metadata to send to the remote.");
DEFINE_NCCL_PARAM(ncclParamIbQpsPerConn, int32_t, NCCL_IB_QPS_PER_CONNECTION, 1,
                  NCCL_PARAM_FLAG_CACHED | NCCL_PARAM_FLAG_PUBLISHED, ncclParamBounded(1, 128),
                  "Number of IB queue pairs per connection. Useful on multi-level fabrics for "
                  "routing entropy.");
RCCL_PARAM(IbQpsPerP2p, "IB_QPS_PER_P2P", 0);
RCCL_PARAM(IbGdrFlushFlag, "IB_GDR_FLUSH_FLAG", 0);
NCCL_PARAM(IbSubnetAwareRouting, "IB_SUBNET_AWARE_ROUTING", 0);
NCCL_PARAM(IbSubnetPrefixLen, "IB_SUBNET_PREFIX_LEN", 24);
DEFINE_NCCL_PARAM(ncclParamIbWarnRailLocal, bool, NCCL_IB_WARN_RAIL_LOCAL, false, NCCL_PARAM_FLAG_CACHED,
                  NCCL_PARAM_DEFAULT, "Warn when using local rail for IB");
DEFINE_NCCL_PARAM(ncclParamIbGdrFlushDisable, bool, NCCL_GDR_FLUSH_DISABLE, false, NCCL_PARAM_FLAG_CACHED,
                  NCCL_PARAM_DEFAULT, "If set, disables GPU Direct RDMA flush");
