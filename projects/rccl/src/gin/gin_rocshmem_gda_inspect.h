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

/* Arrays describe n query_gid_table() entries (gid_type: 0=IB, 1=RoCEv1, 2=RoCEv2). Returns the
 * gid_index that gin_select_gid_pos() selects for `port`, or -1 if no entry is usable. */
int ginTestSelectGidIndex(const unsigned char (*gid_raw)[16], const unsigned* gid_index, const unsigned* port_num,
                          const unsigned* gid_type, int n, int port, int routable_gid_index);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_GIN_ROCSHMEM_GDA_INSPECT_H_ */
