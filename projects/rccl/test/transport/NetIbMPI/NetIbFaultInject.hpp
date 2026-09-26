/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_NET_IB_FAULT_INJECT_HPP_
#define RCCL_TEST_NET_IB_FAULT_INJECT_HPP_

#if defined(MPI_TESTS_ENABLED) && defined(ENABLE_FAULT_INJECTION)

/*
 * Re-export the shared fault injection API declarations from the library's
 * internal headers. Using the same headers in both the library and the tests
 * guarantees that the ABI can never silently diverge.
 *   - net_ib_fault_inject.h        : CAST transport (src/transport/net_ib_cast)
 *   - net_ib_flush_fault_inject.h  : base GDR flush   (src/transport/net_ib)
 */
#include "net_ib_fault_inject.h"
#include "net_ib/net_ib_flush_fault_inject.h"

#endif /* MPI_TESTS_ENABLED && ENABLE_FAULT_INJECTION */

#endif /* RCCL_TEST_NET_IB_FAULT_INJECT_HPP_ */
