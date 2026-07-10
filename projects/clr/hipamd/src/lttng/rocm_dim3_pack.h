/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * dim3 packing macro for LTTng curated tracepoints.
 *
 * Lane layout (64-bit packed value):
 *   bits  0..31  : x  (full 32 bits)
 *   bits 32..47  : y  (16-bit lane, saturates to 0xFFFF on overflow)
 *   bits 48..62  : z  (15-bit lane, saturates to 0x7FFF on overflow)
 *   bit  63      : overflow flag (set iff y or z exceeded its lane;
 *                  z >= 0x8000 is treated as overflow because the lane is
 *                  intentionally 15 bits to keep bit 63 unambiguous)
 *
 * Branch-light: only saturating arithmetic, no abort path. Overflow is a
 * degraded-data signal: consumers MUST treat any value with bit 63 set as
 * "true y and/or z is unknown but >= the lane-saturated value".
 */
#ifndef ROCM_DIM3_PACK_H_
#define ROCM_DIM3_PACK_H_

#include <stdint.h>
#include <hip/hip_runtime_api.h>  /* for dim3 */

#define ROCM_DIM3_OVERFLOW_BIT (1ULL << 63)
#define ROCM_DIM3_Z_MAX        (0x7FFFu)   /* 15-bit lane max */

static inline uint64_t ROCM_DIM3_PACK(dim3 d) {
    const uint64_t x = (uint64_t)d.x;
    const uint32_t y_raw = d.y;
    const uint32_t z_raw = d.z;
    const uint64_t y = (y_raw > 0xFFFFu) ? 0xFFFFu : y_raw;
    const uint64_t z = (z_raw > ROCM_DIM3_Z_MAX) ? ROCM_DIM3_Z_MAX : z_raw;
    const uint64_t overflow = ((y_raw > 0xFFFFu) || (z_raw > ROCM_DIM3_Z_MAX))
                                  ? ROCM_DIM3_OVERFLOW_BIT : 0ULL;
    return x | (y << 32) | (z << 48) | overflow;
}

#endif  /* ROCM_DIM3_PACK_H_ */
