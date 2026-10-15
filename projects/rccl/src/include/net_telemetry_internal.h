/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* Internal to net_telemetry.cc, exposed for the unit test. */

#ifndef RCCL_NET_TELEMETRY_INTERNAL_H_
#define RCCL_NET_TELEMETRY_INTERNAL_H_

#include "net_telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Longest `ethtool -S` statistic name we ever ask for. The kernel bounds a
 * statistic name by ETH_GSTRING_LEN (32), and the per-priority names we format
 * stay well inside that, so this has room to spare. */
#define RCCL_TEL_ETHTOOL_KEY_MAX 64

typedef struct {
  char     key[RCCL_TEL_ETHTOOL_KEY_MAX];
  int64_t* target;
} RcclEthtoolWantedEx;

/* First match in `wanted` order wins, so a primary key queued before its
 * fallback is preferred. */
void rcclTelemetryEthtoolResolve(const char* names, size_t nameStride, const uint64_t* values, unsigned n,
                                 RcclEthtoolWantedEx* wanted, int nwanted);

/* Root of the sysfs/debugfs tree telemetry reads, "/sys" outside tests. */
extern const char* rcclTelemetrySysRoot;

#ifdef __cplusplus
}
#endif

#endif /* RCCL_NET_TELEMETRY_INTERNAL_H_ */
