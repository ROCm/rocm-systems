/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_LL128_FLAG_H_
#define RCCL_LL128_FLAG_H_

#include <stdint.h>

#include "device.h"

/* Experimental: bytes of flag at the end of each LL128 line: 8 (default), 4, 2 or 1. Set at build time (CMake
 * RCCL_LL128_FLAG_BYTES) so that device code and the net proxy, which checks LL128 flags for host-memory buffers,
 * agree. With 8 every macro below reduces to the original LL128 flag.
 *
 * A narrower flag holds (step + 1) truncated to its width in the top bytes of the line's last word, and wraps.
 * Every (RCCL_LL128_CLEAN_MASK + 1) steps of a slot the sender stamps the current flag on every line of the step and
 * forwards the whole step, so no line's flag is a whole flag period old, as in LL. RCCL_LL128_FLAG_TEST=1 / 2 (CMake
 * option of the same name, testing only) wraps a narrow flag every 256 / 32 steps to exercise this. */
#ifndef RCCL_LL128_FLAG_BYTES
#define RCCL_LL128_FLAG_BYTES 8
#endif
static_assert(RCCL_LL128_FLAG_BYTES == 8 || RCCL_LL128_FLAG_BYTES == 4 || RCCL_LL128_FLAG_BYTES == 2 ||
                RCCL_LL128_FLAG_BYTES == 1,
              "Invalid RCCL_LL128_FLAG_BYTES value");
#define RCCL_LL128_NARROW_FLAG (RCCL_LL128_FLAG_BYTES < 8)
#define RCCL_LL128_FLAG_SHIFT (64 - 8 * RCCL_LL128_FLAG_BYTES)
#if !RCCL_LL128_NARROW_FLAG
#define RCCL_LL128_CLEAN_MASK 0
#define RCCL_LL128_FLAG_OF(step1) ((uint64_t)(step1))
#elif RCCL_LL128_FLAG_TEST == 2
#define RCCL_LL128_CLEAN_MASK 0x008
#define RCCL_LL128_FLAG_OF(step1) ((uint64_t)(step1) % 0x20)
#elif RCCL_LL128_FLAG_TEST || RCCL_LL128_FLAG_BYTES == 1
#define RCCL_LL128_CLEAN_MASK 0x078
#define RCCL_LL128_FLAG_OF(step1) ((uint64_t)(step1) % 0x100)
#elif RCCL_LL128_FLAG_BYTES == 2
#define RCCL_LL128_CLEAN_MASK 0x7ff8
#define RCCL_LL128_FLAG_OF(step1) ((uint64_t)(step1) % 0x10000)
#else
#define RCCL_LL128_CLEAN_MASK 0x7ffffff8
#define RCCL_LL128_FLAG_OF(step1) ((uint64_t)(uint32_t)(step1))
#endif
static_assert(RCCL_LL128_CLEAN_MASK % NCCL_STEPS == 0, "Invalid RCCL_LL128_CLEAN_MASK value");
// True when step's slot is stamped and forwarded whole.
#define RCCL_LL128_CLEAN_STEP(step) \
  (RCCL_LL128_NARROW_FLAG && ((step) & RCCL_LL128_CLEAN_MASK) == RCCL_LL128_CLEAN_MASK)
// word is the last 64-bit word of a line, flag a value of RCCL_LL128_FLAG_OF.
#define RCCL_LL128_FLAG_MATCH(word, flag) (((word) >> RCCL_LL128_FLAG_SHIFT) == (flag))

#endif // RCCL_LL128_FLAG_H_
