/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_GLOBALS_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_GLOBALS_H_

// Puts the device tables and counters src/transport/net_ib/common.cc owns back to their load-time values. Every
// net_ib suite shares them in one binary, so each fixture's TearDown calls this. Defined in common-test.cc, the TU
// that compiles common.cc.
void NetIbResetGlobals();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_GLOBALS_H_
