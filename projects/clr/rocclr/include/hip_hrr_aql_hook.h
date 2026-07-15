/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/*
 * ROCclr → HRR: record Cijk_* kernarg bytes immediately before AQL dispatch.
 * Implemented in hipamd/src/hrr/hip_capture.cpp.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void hip_hrr_record_dispatch_kernarg(const char* kernel_name, uint64_t kernarg_dev_va,
                                     const void* kernarg_host, size_t kernarg_size,
                                     uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                     uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                     uint32_t shared_mem);

#ifdef __cplusplus
}
#endif
