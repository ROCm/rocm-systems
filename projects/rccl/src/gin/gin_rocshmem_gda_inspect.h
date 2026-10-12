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

/*
 * Test-only wrappers over internal static helpers in gin_rocshmem_gda_factory.cc
 * (host-only, no HW). Mirrors the pattern in
 * src/transport/net_ib_cast/net_ib_cast_inspect.h.
 *
 * MTU values are passed as plain ints holding enum ibv_mtu values
 * (IBV_MTU_256=1 .. IBV_MTU_4096=5) so this header does not need to pull in
 * <infiniband/verbs.h>.
 */

/* Returns the MTU in bytes for a valid enum ibv_mtu value, else 0. */
int ginTestIbvMtuToInt(int mtu);

/* Negotiates path MTU as min(local_mtu, remote_mtu). Returns 0 and sets
 * *out_mtu (in bytes) on success, -1 if remote_mtu is not a valid enum
 * ibv_mtu value. Caller must have already validated local_mtu. */
int ginTestNegotiatePathMtu(int local_mtu, int remote_mtu, int* out_mtu);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_GIN_ROCSHMEM_GDA_INSPECT_H_ */
