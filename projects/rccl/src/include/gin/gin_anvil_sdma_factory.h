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

typedef struct gin_anvil_sdma_opaque* gin_anvil_sdma_handle_t;

int gin_anvil_sdma_probe(void);

int gin_anvil_sdma_create(
    int nRanks, int myRank, int my_device_id,
    int (*allgather)(void* ctx, void* buf, size_t bytes_per_rank), void* allgather_ctx,
    int num_channels, gin_anvil_sdma_handle_t* out_handle, void** out_gpu_handles,
    uint64_t** out_sdma_dirty);

void gin_anvil_sdma_destroy(gin_anvil_sdma_handle_t handle);

int gin_anvil_sdma_get_n_ranks(gin_anvil_sdma_handle_t handle);
int gin_anvil_sdma_get_num_channels(gin_anvil_sdma_handle_t handle);
int gin_anvil_sdma_get_channel_stride(gin_anvil_sdma_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // GIN_ANVIL_SDMA_FACTORY_H_
