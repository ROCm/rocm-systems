/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_ANVIL_CONN_CHECK_H_
#define GIN_ANVIL_CONN_CHECK_H_

#ifdef ENABLE_ROCSHMEM_GIN

#include <hip/hip_runtime.h>

// [GIN-CONN-CHECK] Device launchers for the LSA signal connectivity self-test.
// Implemented in gin_anvil_conn_check_device.cc (HIP). Host unit tests stub these
// in gin_anvil_plugin_test_stubs.cc.
extern "C" int ginAnvilConnWrite(void* remoteAddrsDev, int nRanks, int selfRank,
                                 unsigned long long stamp, hipStream_t stream);
extern "C" int ginAnvilConnCheck(void* localSignals, int nRanks, unsigned long long stamp,
                                 int* missingDev, hipStream_t stream);

#endif  // ENABLE_ROCSHMEM_GIN

#endif  // GIN_ANVIL_CONN_CHECK_H_
