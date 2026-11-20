/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_NET_IB_MSEG_INSPECT_HPP_
#define RCCL_TEST_NET_IB_MSEG_INSPECT_HPP_

#ifdef MPI_TESTS_ENABLED

#include <cstdint>

/*
 * Re-export the shared struct and function declarations from the library's
 * internal net_ib_multiseg headers.  Using the same headers in both the library
 * and the tests guarantees that the ABI can never silently diverge.
 */
#include "net_ib_multiseg/net_ib_gid_inspect_mseg.h"
#include "net_ib_multiseg/net_ib_wqe_lat_inspect_mseg.h"

// net_ib_multiseg/connect.cc; the fork has no header for it, as net_ib's copy lives in net.h.
bool IbMsegGdrFlushFlagOrdered(int64_t oooRq, int64_t portFailover, int64_t matchingScheme, int nDevs,
                               const int* adaptiveRouting);

#endif /* MPI_TESTS_ENABLED */

#endif /* RCCL_TEST_NET_IB_MSEG_INSPECT_HPP_ */
