/******************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef GIN_ANVIL_SDMA_FACTORY_H_
#define GIN_ANVIL_SDMA_FACTORY_H_

#include <stddef.h>
#include <stdint.h>

#define GIN_ANVIL_SDMA_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle: standalone SDMA queue table (per local peer × channels) +
 * sdmaDirty bitmask. Independent of PGAS runtime init; uses AnvilLib device paths.
 */
typedef struct gin_anvil_sdma_opaque* gin_anvil_sdma_handle_t;

/** @return 1 if SDMA Anvil path is compiled in and HIP reports at least one device, else 0 */
GIN_ANVIL_SDMA_API int gin_anvil_sdma_probe(void);

/**
 * Bootstrap allgather: int allgather(void* ctx, void* buf, size_t bytes_per_rank)
 * Each rank fills buf[myRank * elem_size ..] before the collective; result is full table.
 *
 * On entry, `buf` must point to `nRanks * sizeof(int)` bytes. The caller sets
 * `buf[myRank] = my_device_id` before invoking `allgather`; after the collective,
 * `buf[r]` is the HIP device ordinal for rank `r`.
 *
 * @param my_device_id  HIP ordinal for this rank (typically hipGetDevice).
 * @param num_channels  SDMA channels per peer pair (clamped to [1,8]).
 * @param out_gpu_handles Device pointer to array of nRanks*num_channels pointers to
 *                        SdmaQueueDeviceHandle (layout: local_pe * num_channels + ch).
 * @param out_sdma_dirty  Device pointer to a single uint64_t dirty bitmask (device memory).
 */
GIN_ANVIL_SDMA_API int gin_anvil_sdma_create(
    int nRanks, int myRank, int my_device_id,
    int (*allgather)(void* ctx, void* buf, size_t bytes_per_rank), void* allgather_ctx,
    int num_channels, gin_anvil_sdma_handle_t* out_handle, void** out_gpu_handles,
    uint64_t** out_sdma_dirty);

GIN_ANVIL_SDMA_API void gin_anvil_sdma_destroy(gin_anvil_sdma_handle_t handle);

/** Fields stored in the opaque handle for RCCL plugin / device code */
GIN_ANVIL_SDMA_API int gin_anvil_sdma_get_n_ranks(gin_anvil_sdma_handle_t handle);
GIN_ANVIL_SDMA_API int gin_anvil_sdma_get_num_channels(gin_anvil_sdma_handle_t handle);
/** 1 = spread wavefronts across channels (NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS, default on) */
GIN_ANVIL_SDMA_API int gin_anvil_sdma_get_channel_stride(gin_anvil_sdma_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // GIN_ANVIL_SDMA_FACTORY_H_
