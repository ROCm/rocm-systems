/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_ANVIL_SDMA_FACTORY_H_
#define GIN_ANVIL_SDMA_FACTORY_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle: standalone SDMA queue table (per local peer x channels) +
 * sdmaDirty bitmask. Independent of PGAS runtime init; uses AnvilLib device paths.
 */
typedef struct gin_anvil_sdma_opaque* gin_anvil_sdma_handle_t;

/** @return 1 if SDMA Anvil path is compiled in and HIP reports at least one device, else 0 */
int gin_anvil_sdma_probe(void);

/**
 * Create SDMA queues for all local peers.
 *
 * The caller supplies a bootstrap allgather callback:
 *   int allgather(void* ctx, void* buf, size_t bytes_per_rank)
 * On entry, buf must point to nRanks * sizeof(int) bytes. The caller sets
 * buf[myRank] = my_device_id before invoking allgather; after the collective,
 * buf[r] is the HIP device ordinal for rank r.
 *
 * @param my_device_id  HIP ordinal for this rank (typically hipGetDevice).
 * @param num_channels  SDMA channels per peer pair (clamped to [1,8]).
 * @param out_gpu_handles Device pointer to array of nRanks*num_channels pointers to
 *                        SdmaQueueDeviceHandle (layout: local_pe * num_channels + ch).
 * @param out_sdma_dirty  Device pointer to a single uint64_t dirty bitmask (device memory).
 * @return 0 on success, -1 on failure.
 */
int gin_anvil_sdma_create(
    int nRanks, int myRank, int my_device_id,
    int (*allgather)(void* ctx, void* buf, size_t bytes_per_rank), void* allgather_ctx,
    int num_channels, gin_anvil_sdma_handle_t* out_handle, void** out_gpu_handles,
    uint64_t** out_sdma_dirty);

void gin_anvil_sdma_destroy(gin_anvil_sdma_handle_t handle);

/** Fields stored in the opaque handle for RCCL plugin / device code */
int gin_anvil_sdma_get_n_ranks(gin_anvil_sdma_handle_t handle);
int gin_anvil_sdma_get_num_channels(gin_anvil_sdma_handle_t handle);
/** 1 = spread wavefronts across channels (NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS, default on) */
int gin_anvil_sdma_get_channel_stride(gin_anvil_sdma_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // GIN_ANVIL_SDMA_FACTORY_H_
