/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_NET_IB_QP_SHARING_INSPECT_HPP_
#define RCCL_TEST_NET_IB_QP_SHARING_INSPECT_HPP_

#ifdef MPI_TESTS_ENABLED

/*
 * Re-export the shared struct and function declarations from the library's
 * internal header.  Using the same header in both the library (qp_sharing.cc)
 * and the tests guarantees that the ABI can never silently diverge.
 */
#include "net_ib_qp_sharing_inspect.h"

#endif /* MPI_TESTS_ENABLED */

#endif /* RCCL_TEST_NET_IB_QP_SHARING_INSPECT_HPP_ */
