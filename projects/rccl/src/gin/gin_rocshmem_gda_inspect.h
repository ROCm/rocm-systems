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

/* Arrays describe n query_gid_table() entries; returns the gid_index that gin_select_ib_gid_pos()
 * would select for `port`. first_ib_pos is the array position of the first IB-type entry. */
int ginTestSelectIbGidIndex(const unsigned char (*gid_raw)[16], const unsigned* gid_index, const unsigned* port_num,
                            int n, int port, int first_ib_pos, int routable_gid_index);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_GIN_ROCSHMEM_GDA_INSPECT_H_ */
