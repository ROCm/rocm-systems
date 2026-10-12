/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_GIN_ROCSHMEM_GDA_INSPECT_H_
#define RCCL_GIN_ROCSHMEM_GDA_INSPECT_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Test-only wrappers over static helpers in gin_rocshmem_gda_factory.cc. Plain ints (holding
 * enum ibv_mtu values) avoid pulling in <infiniband/verbs.h> here. */

int ginTestIbvMtuToInt(int mtu);

/* Returns -1 without touching *out_mtu if remote_mtu is invalid. */
int ginTestNegotiatePathMtu(int local_mtu, int remote_mtu, int* out_mtu);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_GIN_ROCSHMEM_GDA_INSPECT_H_ */
